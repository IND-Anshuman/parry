//! Bounded, TTL-aware response cache keyed by a blake3 hash of the
//! canonicalised request.
//!
//! Values are opaque strings (parry stores serialised SDK responses), so the
//! cache never holds Python objects and never needs the GIL to evict.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::tokenize::{NormalizeError, normalize_json};

/// Default number of entries. The cache is always bounded.
pub const DEFAULT_MAX_ENTRIES: u64 = 10_000;
/// Default time-to-live in seconds.
pub const DEFAULT_TTL_SECONDS: f64 = 3600.0;

#[derive(Debug)]
struct Entry {
    value: String,
    tokens: u64,
    cost_usd: f64,
}

/// Build a cache key: `blake3(namespace || 0x00 || canonical_json(payload))`.
pub fn make_key(namespace: &str, payload: &str) -> Result<String, NormalizeError> {
    let canonical = normalize_json(payload)?;
    let mut hasher = blake3::Hasher::new();
    hasher.update(namespace.as_bytes());
    hasher.update(&[0]);
    hasher.update(canonical.as_bytes());
    Ok(hasher.finalize().to_hex().to_string())
}

pub struct ResponseCache {
    inner: moka::sync::Cache<String, Arc<Entry>>,
    hits: AtomicU64,
    misses: AtomicU64,
    tokens_saved: AtomicU64,
    // f64 stored as bits so it can be updated atomically.
    cost_saved_bits: AtomicU64,
    max_entries: u64,
    ttl: Option<Duration>,
}

impl ResponseCache {
    pub fn new(max_entries: u64, ttl: Option<Duration>) -> Self {
        let mut builder = moka::sync::Cache::builder().max_capacity(max_entries);
        if let Some(ttl) = ttl {
            builder = builder.time_to_live(ttl);
        }
        Self {
            inner: builder.build(),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            tokens_saved: AtomicU64::new(0),
            cost_saved_bits: AtomicU64::new(0f64.to_bits()),
            max_entries,
            ttl,
        }
    }

    /// Look up a key, updating hit/miss counters. Returns the value together
    /// with the tokens and cost the original call consumed.
    pub fn lookup(&self, key: &str) -> Option<(String, u64, f64)> {
        match self.inner.get(key) {
            Some(entry) => {
                self.hits.fetch_add(1, Ordering::Relaxed);
                self.tokens_saved.fetch_add(entry.tokens, Ordering::Relaxed);
                add_f64(&self.cost_saved_bits, entry.cost_usd);
                Some((entry.value.clone(), entry.tokens, entry.cost_usd))
            }
            None => {
                self.misses.fetch_add(1, Ordering::Relaxed);
                None
            }
        }
    }

    pub fn insert(&self, key: String, value: String, tokens: u64, cost_usd: f64) {
        self.inner.insert(
            key,
            Arc::new(Entry {
                value,
                tokens,
                cost_usd,
            }),
        );
    }

    pub fn contains(&self, key: &str) -> bool {
        self.inner.contains_key(key)
    }

    pub fn remove(&self, key: &str) -> bool {
        self.inner.remove(key).is_some()
    }

    pub fn clear(&self) {
        self.inner.invalidate_all();
        self.inner.run_pending_tasks();
    }

    pub fn len(&self) -> u64 {
        self.inner.run_pending_tasks();
        self.inner.entry_count()
    }

    pub fn hits(&self) -> u64 {
        self.hits.load(Ordering::Relaxed)
    }

    pub fn misses(&self) -> u64 {
        self.misses.load(Ordering::Relaxed)
    }

    pub fn tokens_saved(&self) -> u64 {
        self.tokens_saved.load(Ordering::Relaxed)
    }

    pub fn cost_saved(&self) -> f64 {
        f64::from_bits(self.cost_saved_bits.load(Ordering::Relaxed))
    }

    pub fn reset_stats(&self) {
        self.hits.store(0, Ordering::Relaxed);
        self.misses.store(0, Ordering::Relaxed);
        self.tokens_saved.store(0, Ordering::Relaxed);
        self.cost_saved_bits
            .store(0f64.to_bits(), Ordering::Relaxed);
    }
}

pub(crate) fn add_f64(cell: &AtomicU64, delta: f64) {
    let mut current = cell.load(Ordering::Relaxed);
    loop {
        let next = (f64::from_bits(current) + delta).to_bits();
        match cell.compare_exchange_weak(current, next, Ordering::Relaxed, Ordering::Relaxed) {
            Ok(_) => return,
            Err(actual) => current = actual,
        }
    }
}

/// In-memory LRU response cache with a TTL.
///
/// >>> cache = Cache(max_entries=1000, ttl_seconds=600)
/// >>> key = cache.make_key("openai", '{"model": "gpt-4o", "messages": []}')
/// >>> cache.set(key, "serialised response")
/// >>> cache.get(key)
/// 'serialised response'
#[pyclass(name = "Cache", module = "parry._parry_native", frozen)]
pub struct PyCache {
    inner: ResponseCache,
}

#[pymethods]
impl PyCache {
    #[new]
    #[pyo3(signature = (max_entries = DEFAULT_MAX_ENTRIES, ttl_seconds = Some(DEFAULT_TTL_SECONDS)))]
    fn new(max_entries: u64, ttl_seconds: Option<f64>) -> PyResult<Self> {
        if max_entries == 0 {
            return Err(PyValueError::new_err(
                "max_entries must be at least 1; pass cache=False to disable caching",
            ));
        }
        let ttl = match ttl_seconds {
            None => None,
            Some(s) if s.is_finite() && s > 0.0 => Some(Duration::from_secs_f64(s)),
            Some(s) => {
                return Err(PyValueError::new_err(format!(
                    "ttl_seconds must be a positive number or None, got {s}"
                )));
            }
        };
        Ok(Self {
            inner: ResponseCache::new(max_entries, ttl),
        })
    }

    /// Hash a JSON request payload into a cache key. `namespace` separates
    /// providers/endpoints/base URLs so they never share entries.
    #[staticmethod]
    fn make_key(py: Python<'_>, namespace: &str, payload_json: &str) -> PyResult<String> {
        Ok(py.detach(|| make_key(namespace, payload_json))?)
    }

    /// Return the cached value, or None. Counts as a hit or a miss.
    fn get(&self, key: &str) -> Option<String> {
        self.inner.lookup(key).map(|(value, _, _)| value)
    }

    /// Like `get`, but returns `(value, tokens, cost_usd)` recorded at insert time.
    fn lookup(&self, key: &str) -> Option<(String, u64, f64)> {
        self.inner.lookup(key)
    }

    /// Store a value. `tokens` and `cost_usd` describe what the original call
    /// cost and feed the "saved" statistics on later hits.
    #[pyo3(signature = (key, value, tokens = 0, cost_usd = 0.0))]
    fn set(&self, key: String, value: String, tokens: u64, cost_usd: f64) {
        self.inner.insert(key, value, tokens, cost_usd);
    }

    /// Remove one key. Returns True if it was present.
    fn invalidate(&self, key: &str) -> bool {
        self.inner.remove(key)
    }

    /// Drop every entry (statistics are kept).
    fn clear(&self) {
        self.inner.clear();
    }

    fn reset_stats(&self) {
        self.inner.reset_stats();
    }

    #[getter]
    fn hits(&self) -> u64 {
        self.inner.hits()
    }

    #[getter]
    fn misses(&self) -> u64 {
        self.inner.misses()
    }

    #[getter]
    fn tokens_saved(&self) -> u64 {
        self.inner.tokens_saved()
    }

    #[getter]
    fn cost_saved_usd(&self) -> f64 {
        self.inner.cost_saved()
    }

    #[getter]
    fn max_entries(&self) -> u64 {
        self.inner.max_entries
    }

    #[getter]
    fn ttl_seconds(&self) -> Option<f64> {
        self.inner.ttl.map(|d| d.as_secs_f64())
    }

    fn stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        d.set_item("hits", self.inner.hits())?;
        d.set_item("misses", self.inner.misses())?;
        d.set_item("entries", self.inner.len())?;
        d.set_item("tokens_saved", self.inner.tokens_saved())?;
        d.set_item("cost_saved_usd", self.inner.cost_saved())?;
        Ok(d)
    }

    fn __len__(&self) -> usize {
        usize::try_from(self.inner.len()).unwrap_or(usize::MAX)
    }

    fn __contains__(&self, key: &str) -> bool {
        self.inner.contains(key)
    }

    fn __repr__(&self) -> String {
        let ttl = self
            .inner
            .ttl
            .map_or_else(|| "None".to_owned(), |d| d.as_secs_f64().to_string());
        format!(
            "Cache(entries={}, max_entries={}, ttl_seconds={}, hits={}, misses={})",
            self.inner.len(),
            self.inner.max_entries,
            ttl,
            self.inner.hits(),
            self.inner.misses()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_requests_share_a_key() {
        let a = make_key(
            "openai",
            r#"{"model":"gpt-4o","messages":[{"role":"user","content":"hi"}]}"#,
        )
        .unwrap();
        let b = make_key(
            "openai",
            r#"{"messages":[{"content":" hi ","role":"user"}],"model":"gpt-4o"}"#,
        )
        .unwrap();
        assert_eq!(a, b);
        assert_eq!(a.len(), 64);
    }

    #[test]
    fn namespace_separates_keys() {
        let payload = r#"{"model":"gpt-4o"}"#;
        assert_ne!(
            make_key("openai", payload).unwrap(),
            make_key("azure", payload).unwrap()
        );
    }

    #[test]
    fn different_parameters_do_not_collide() {
        let a = make_key("x", r#"{"temperature":0}"#).unwrap();
        let b = make_key("x", r#"{"temperature":1}"#).unwrap();
        assert_ne!(a, b);
    }

    #[test]
    fn hit_and_miss_accounting() {
        let cache = ResponseCache::new(10, None);
        assert!(cache.lookup("k").is_none());
        cache.insert("k".into(), "v".into(), 42, 0.5);
        let (v, tokens, cost) = cache.lookup("k").unwrap();
        assert_eq!((v.as_str(), tokens), ("v", 42));
        assert!((cost - 0.5).abs() < 1e-12);
        assert_eq!(cache.hits(), 1);
        assert_eq!(cache.misses(), 1);
        assert_eq!(cache.tokens_saved(), 42);
        assert!((cache.cost_saved() - 0.5).abs() < 1e-12);
    }

    #[test]
    fn capacity_is_bounded() {
        let cache = ResponseCache::new(5, None);
        for i in 0..100 {
            cache.insert(format!("k{i}"), "v".into(), 0, 0.0);
        }
        assert!(cache.len() <= 5, "len = {}", cache.len());
    }

    #[test]
    fn entries_expire() {
        let cache = ResponseCache::new(10, Some(Duration::from_millis(50)));
        cache.insert("k".into(), "v".into(), 0, 0.0);
        assert!(cache.contains("k"));
        std::thread::sleep(Duration::from_millis(120));
        assert!(cache.lookup("k").is_none());
    }
}
