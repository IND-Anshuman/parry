//! Exponential backoff with jitter, and retryable-error classification.
//!
//! Only transient failures are retried: rate limits (429), timeouts,
//! connection errors, overload (529) and 5xx. Client errors such as 400, 401,
//! 403, 404 and 422 are never retried — retrying a bad key is just spam.

use std::collections::hash_map::RandomState;
use std::hash::BuildHasher;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

/// Exception class names (OpenAI / Anthropic SDKs, httpx, stdlib) that mean
/// "transient, try again" when no HTTP status is available.
const RETRYABLE_NAMES: &[&str] = &[
    "RateLimitError",
    "APITimeoutError",
    "APIConnectionError",
    "InternalServerError",
    "OverloadedError",
    "ServiceUnavailableError",
    "TimeoutError",
    "ConnectionError",
    "ConnectError",
    "ConnectTimeout",
    "ReadTimeout",
    "WriteTimeout",
    "PoolTimeout",
    "RemoteProtocolError",
];

pub fn is_retryable_status(status: u16) -> bool {
    matches!(status, 408 | 409 | 425 | 429 | 500..=599)
}

pub fn is_retryable(name: &str, status: Option<u16>) -> bool {
    match status {
        Some(code) => is_retryable_status(code),
        None => RETRYABLE_NAMES.contains(&name),
    }
}

/// Small, dependency-free PRNG (splitmix64) for jitter. Not cryptographic.
struct Jitter {
    seed: u64,
    counter: AtomicU64,
}

impl Jitter {
    fn new() -> Self {
        Self {
            seed: RandomState::new().hash_one(0x5eed_u64),
            counter: AtomicU64::new(0),
        }
    }

    /// Uniform float in [0, 1).
    fn next(&self) -> f64 {
        let n = self.counter.fetch_add(1, Ordering::Relaxed);
        let mut z = self
            .seed
            .wrapping_add(n.wrapping_mul(0x9E37_79B9_7F4A_7C15));
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        (z >> 11) as f64 / (1u64 << 53) as f64
    }
}

pub struct Policy {
    pub max_attempts: u32,
    pub base_delay: f64,
    pub max_delay: f64,
    pub multiplier: f64,
    pub jitter: bool,
    rng: Jitter,
}

impl Policy {
    pub fn new(
        max_attempts: u32,
        base_delay: f64,
        max_delay: f64,
        multiplier: f64,
        jitter: bool,
    ) -> Self {
        Self {
            max_attempts,
            base_delay,
            max_delay,
            multiplier,
            jitter,
            rng: Jitter::new(),
        }
    }

    /// Seconds to wait after the `attempt`-th failure (1-based). A server
    /// supplied `Retry-After` wins over the computed backoff, capped at
    /// `max_delay`.
    pub fn delay(&self, attempt: u32, retry_after: Option<f64>) -> f64 {
        if let Some(ra) = retry_after.filter(|r| r.is_finite() && *r >= 0.0) {
            return ra.min(self.max_delay);
        }
        let exp = i32::try_from(attempt.saturating_sub(1)).unwrap_or(i32::MAX);
        let raw = (self.base_delay * self.multiplier.powi(exp)).min(self.max_delay);
        if self.jitter {
            // "Equal jitter": half fixed, half random. Spreads retries out
            // without ever collapsing to an immediate retry.
            raw / 2.0 + self.rng.next() * raw / 2.0
        } else {
            raw
        }
    }

    pub fn should_retry(&self, attempt: u32, name: &str, status: Option<u16>) -> bool {
        attempt < self.max_attempts && is_retryable(name, status)
    }
}

fn status_of(err: &Bound<'_, PyAny>) -> Option<u16> {
    err.getattr("status_code").ok()?.extract::<u16>().ok()
}

/// Read `retry-after-ms` / `retry-after` (seconds) from `exc.response.headers`.
fn retry_after_of(err: &Bound<'_, PyAny>) -> Option<f64> {
    let headers = err.getattr("response").ok()?.getattr("headers").ok()?;
    let get = |name: &str| -> Option<f64> {
        let v = headers.call_method1("get", (name,)).ok()?;
        if v.is_none() {
            return None;
        }
        v.extract::<String>().ok()?.trim().parse::<f64>().ok()
    };
    get("retry-after-ms")
        .map(|ms| ms / 1000.0)
        .or_else(|| get("retry-after"))
}

/// Exponential backoff with jitter.
///
/// >>> retry = Retry(max_attempts=3, base_delay=0.5)
/// >>> retry.call(flaky_function, arg1, kw=2)
#[pyclass(name = "Retry", module = "parry._parry_native", frozen)]
pub struct PyRetry {
    inner: Policy,
    retries: AtomicU64,
}

#[pymethods]
impl PyRetry {
    #[new]
    #[pyo3(signature = (max_attempts = 3, base_delay = 0.5, max_delay = 30.0, multiplier = 2.0, jitter = true))]
    fn new(
        max_attempts: u32,
        base_delay: f64,
        max_delay: f64,
        multiplier: f64,
        jitter: bool,
    ) -> PyResult<Self> {
        if max_attempts == 0 {
            return Err(PyValueError::new_err("max_attempts must be at least 1"));
        }
        let valid = |x: f64| x.is_finite() && x >= 0.0;
        if !valid(base_delay) || !valid(max_delay) || !(multiplier.is_finite() && multiplier >= 1.0)
        {
            return Err(PyValueError::new_err(
                "delays must be non-negative and multiplier must be >= 1",
            ));
        }
        Ok(Self {
            inner: Policy::new(max_attempts, base_delay, max_delay, multiplier, jitter),
            retries: AtomicU64::new(0),
        })
    }

    /// Seconds to wait after the `attempt`-th failure (1-based).
    #[pyo3(signature = (attempt, retry_after = None))]
    fn delay(&self, attempt: u32, retry_after: Option<f64>) -> f64 {
        self.inner.delay(attempt, retry_after)
    }

    /// Whether an error with this class name / HTTP status is transient.
    #[staticmethod]
    #[pyo3(signature = (name, status = None))]
    fn is_retryable(name: &str, status: Option<u16>) -> bool {
        is_retryable(name, status)
    }

    /// Whether to retry after the `attempt`-th failure (1-based) with this error.
    #[pyo3(signature = (attempt, name, status = None))]
    fn should_retry(&self, attempt: u32, name: &str, status: Option<u16>) -> bool {
        self.inner.should_retry(attempt, name, status)
    }

    /// Classify a Python exception: `(retryable, retry_after_seconds)`.
    fn classify(&self, exc: &Bound<'_, PyAny>) -> PyResult<(bool, Option<f64>)> {
        let name = exc.get_type().name()?.to_string();
        Ok((is_retryable(&name, status_of(exc)), retry_after_of(exc)))
    }

    /// Count one retry in `retries` (used by callers that run their own loop).
    fn note_retry(&self) {
        self.retries.fetch_add(1, Ordering::Relaxed);
    }

    /// Call `fn(*args, **kwargs)`, retrying transient failures. Sleeps with
    /// the GIL released. Non-retryable errors are raised immediately.
    #[pyo3(signature = (func, *args, **kwargs))]
    fn call<'py>(
        &self,
        py: Python<'py>,
        func: &Bound<'py, PyAny>,
        args: &Bound<'py, PyTuple>,
        kwargs: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let mut attempt = 1u32;
        loop {
            match func.call(args, kwargs) {
                Ok(v) => return Ok(v),
                Err(err) => {
                    let value = err.value(py);
                    let name = value.get_type().name()?.to_string();
                    if !self.inner.should_retry(attempt, &name, status_of(value)) {
                        return Err(err);
                    }
                    let wait = self.inner.delay(attempt, retry_after_of(value));
                    self.retries.fetch_add(1, Ordering::Relaxed);
                    py.detach(|| std::thread::sleep(Duration::from_secs_f64(wait)));
                    py.check_signals()?;
                    attempt += 1;
                }
            }
        }
    }

    #[getter]
    fn max_attempts(&self) -> u32 {
        self.inner.max_attempts
    }

    #[getter]
    fn retries(&self) -> u64 {
        self.retries.load(Ordering::Relaxed)
    }

    fn __repr__(&self) -> String {
        format!(
            "Retry(max_attempts={}, base_delay={}, max_delay={}, multiplier={}, jitter={})",
            self.inner.max_attempts,
            self.inner.base_delay,
            self.inner.max_delay,
            self.inner.multiplier,
            if self.inner.jitter { "True" } else { "False" }
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classification() {
        for code in [408, 409, 429, 500, 502, 503, 504, 529] {
            assert!(is_retryable_status(code), "{code}");
        }
        for code in [400, 401, 403, 404, 422] {
            assert!(!is_retryable_status(code), "{code}");
        }
        assert!(is_retryable("APITimeoutError", None));
        assert!(is_retryable("APIConnectionError", None));
        assert!(!is_retryable("AuthenticationError", None));
        assert!(!is_retryable("BadRequestError", Some(400)));
        assert!(!is_retryable("ValueError", None));
    }

    #[test]
    fn backoff_without_jitter() {
        let p = Policy::new(5, 0.5, 4.0, 2.0, false);
        assert_eq!(p.delay(1, None), 0.5);
        assert_eq!(p.delay(2, None), 1.0);
        assert_eq!(p.delay(3, None), 2.0);
        assert_eq!(p.delay(4, None), 4.0);
        assert_eq!(p.delay(10, None), 4.0);
    }

    #[test]
    fn jitter_stays_in_range() {
        let p = Policy::new(5, 1.0, 30.0, 2.0, true);
        let samples: Vec<f64> = (0..1000).map(|_| p.delay(3, None)).collect();
        assert!(samples.iter().all(|d| (2.0..=4.0).contains(d)));
        let distinct = samples
            .iter()
            .map(|d| d.to_bits())
            .collect::<std::collections::HashSet<_>>();
        assert!(distinct.len() > 900);
    }

    #[test]
    fn retry_after_wins_but_is_capped() {
        let p = Policy::new(3, 0.5, 10.0, 2.0, true);
        assert_eq!(p.delay(1, Some(2.0)), 2.0);
        assert_eq!(p.delay(1, Some(120.0)), 10.0);
    }

    #[test]
    fn attempt_budget() {
        let p = Policy::new(3, 0.0, 0.0, 2.0, false);
        assert!(p.should_retry(1, "RateLimitError", Some(429)));
        assert!(p.should_retry(2, "RateLimitError", Some(429)));
        assert!(!p.should_retry(3, "RateLimitError", Some(429)));
        assert!(!p.should_retry(1, "AuthenticationError", Some(401)));
    }
}
