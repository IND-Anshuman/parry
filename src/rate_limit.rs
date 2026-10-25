//! Token-bucket rate limiting per `(provider, model)`.
//!
//! Limits are configured as *rules* and instantiated as *buckets*. A rule
//! such as `"gpt-4o"` or `"openai:*"` is a template; every concrete
//! `provider:model` that matches it gets its own bucket, because providers
//! enforce limits per model. Lookup order for `(provider, model)`:
//!
//! 1. `provider:model`
//! 2. `model`
//! 3. `provider:*`
//! 4. `*`
//! 5. built-in default for the provider (see `DEFAULT_LIMITS`)
//!
//! The limiter never sleeps on its own in `try_acquire`: it returns how long
//! the caller should wait, so sync code can `time.sleep` and async code can
//! `await asyncio.sleep`.

use std::collections::HashMap;
use std::num::NonZeroU32;
use std::sync::Arc;
use std::time::Duration;

use governor::clock::{Clock, DefaultClock};
use governor::{DefaultDirectRateLimiter, Quota, RateLimiter};
use parking_lot::RwLock;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// Conservative defaults: OpenAI usage tier 1 and Anthropic tier 1 request
/// limits, as published by each provider. Override with `set_limit`.
const DEFAULT_LIMITS: &[(&str, u32)] = &[("openai", 500), ("anthropic", 50)];

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Rule {
    pub requests_per_minute: Option<u32>,
    pub tokens_per_minute: Option<u32>,
    pub burst: Option<u32>,
}

struct Bucket {
    requests: Option<DefaultDirectRateLimiter>,
    tokens: Option<DefaultDirectRateLimiter>,
    token_capacity: u32,
}

fn quota(per_minute: u32, burst: Option<u32>) -> Option<(Quota, u32)> {
    let rate = NonZeroU32::new(per_minute)?;
    let burst = burst.and_then(NonZeroU32::new).unwrap_or(rate);
    Some((Quota::per_minute(rate).allow_burst(burst), burst.get()))
}

impl Bucket {
    fn new(rule: Rule) -> Self {
        let requests = rule
            .requests_per_minute
            .and_then(|rpm| quota(rpm, rule.burst))
            .map(|(q, _)| RateLimiter::direct(q));
        let (tokens, token_capacity) = match rule.tokens_per_minute.and_then(|tpm| quota(tpm, None))
        {
            Some((q, cap)) => (Some(RateLimiter::direct(q)), cap),
            None => (None, 0),
        };
        Self {
            requests,
            tokens,
            token_capacity,
        }
    }

    /// Try to take one request (and `tokens` tokens). Returns the wait time if
    /// the bucket is empty. Tokens are checked first so a request slot is not
    /// burned when the token budget is what is exhausted.
    fn try_acquire(&self, tokens: u32) -> Option<Duration> {
        let clock = DefaultClock::default();
        if let Some(limiter) = &self.tokens {
            // A single request larger than the whole per-minute budget could
            // never pass; let it through as a full bucket rather than hang.
            let n = tokens.clamp(1, self.token_capacity.max(1));
            if let Some(n) = NonZeroU32::new(n) {
                match limiter.check_n(n) {
                    Ok(Ok(())) => {}
                    Ok(Err(not_until)) => return Some(not_until.wait_time_from(clock.now())),
                    Err(_insufficient) => {}
                }
            }
        }
        if let Some(limiter) = &self.requests
            && let Err(not_until) = limiter.check()
        {
            return Some(not_until.wait_time_from(clock.now()));
        }
        None
    }
}

#[derive(Default)]
pub struct Limiter {
    rules: RwLock<HashMap<String, Rule>>,
    buckets: RwLock<HashMap<String, Arc<Bucket>>>,
}

impl Limiter {
    pub fn set_rule(&self, key: &str, rule: Rule) {
        self.rules.write().insert(key.to_owned(), rule);
        // Rules changed; rebuild buckets lazily on next use.
        self.buckets.write().clear();
    }

    pub fn remove_rule(&self, key: &str) -> bool {
        let removed = self.rules.write().remove(key).is_some();
        if removed {
            self.buckets.write().clear();
        }
        removed
    }

    pub fn resolve(&self, provider: &str, model: &str) -> Rule {
        let rules = self.rules.read();
        let candidates = [
            format!("{provider}:{model}"),
            model.to_owned(),
            format!("{provider}:*"),
            "*".to_owned(),
        ];
        for key in &candidates {
            if let Some(rule) = rules.get(key) {
                return *rule;
            }
        }
        let rpm = DEFAULT_LIMITS
            .iter()
            .find(|(p, _)| *p == provider)
            .map(|(_, rpm)| *rpm);
        Rule {
            requests_per_minute: rpm,
            tokens_per_minute: None,
            burst: None,
        }
    }

    fn bucket(&self, provider: &str, model: &str) -> Arc<Bucket> {
        let key = format!("{provider}:{model}");
        if let Some(bucket) = self.buckets.read().get(&key) {
            return Arc::clone(bucket);
        }
        let rule = self.resolve(provider, model);
        let mut buckets = self.buckets.write();
        Arc::clone(
            buckets
                .entry(key)
                .or_insert_with(|| Arc::new(Bucket::new(rule))),
        )
    }

    /// Returns `None` if the request may proceed now, else how long to wait.
    pub fn try_acquire(&self, provider: &str, model: &str, tokens: u32) -> Option<Duration> {
        self.bucket(provider, model).try_acquire(tokens)
    }

    pub fn reset(&self) {
        self.buckets.write().clear();
    }
}

fn rule_from_args(
    requests_per_minute: Option<u32>,
    tokens_per_minute: Option<u32>,
    burst: Option<u32>,
) -> PyResult<Rule> {
    if requests_per_minute == Some(0) || tokens_per_minute == Some(0) || burst == Some(0) {
        return Err(PyValueError::new_err(
            "limits must be positive; pass None for 'unlimited'",
        ));
    }
    Ok(Rule {
        requests_per_minute,
        tokens_per_minute,
        burst,
    })
}

/// Per-model token-bucket rate limiter.
///
/// >>> rl = RateLimit()
/// >>> rl.set_limit("gpt-4o", requests_per_minute=60)
/// >>> rl.try_acquire("openai", "gpt-4o")   # 0.0 means "go"
/// 0.0
#[pyclass(name = "RateLimit", module = "parry._parry_native", frozen)]
#[derive(Default)]
pub struct PyRateLimit {
    inner: Limiter,
}

#[pymethods]
impl PyRateLimit {
    #[new]
    fn new() -> Self {
        Self::default()
    }

    /// Configure a limit. `key` is `"provider:model"`, `"model"`,
    /// `"provider:*"` or `"*"`. Passing no limits makes the key unlimited.
    #[pyo3(signature = (key, requests_per_minute = None, tokens_per_minute = None, burst = None))]
    fn set_limit(
        &self,
        key: &str,
        requests_per_minute: Option<u32>,
        tokens_per_minute: Option<u32>,
        burst: Option<u32>,
    ) -> PyResult<()> {
        let rule = rule_from_args(requests_per_minute, tokens_per_minute, burst)?;
        self.inner.set_rule(key, rule);
        Ok(())
    }

    /// Remove a configured rule, falling back to the next match. Returns True
    /// if the rule existed.
    fn remove_limit(&self, key: &str) -> bool {
        self.inner.remove_rule(key)
    }

    /// Non-blocking acquire. Returns 0.0 when the request may proceed, or the
    /// number of seconds to wait before trying again.
    #[pyo3(signature = (provider, model, tokens = 0))]
    fn try_acquire(&self, provider: &str, model: &str, tokens: u32) -> f64 {
        self.inner
            .try_acquire(provider, model, tokens)
            .map_or(0.0, |d| d.as_secs_f64().max(1e-6))
    }

    /// Blocking acquire with the GIL released. Returns False if no slot became
    /// free within `timeout` seconds (None waits forever).
    #[pyo3(signature = (provider, model, tokens = 0, timeout = None))]
    fn acquire(
        &self,
        py: Python<'_>,
        provider: &str,
        model: &str,
        tokens: u32,
        timeout: Option<f64>,
    ) -> bool {
        let deadline =
            timeout.map(|t| std::time::Instant::now() + Duration::from_secs_f64(t.max(0.0)));
        py.detach(|| {
            loop {
                let Some(wait) = self.inner.try_acquire(provider, model, tokens) else {
                    return true;
                };
                let wait = match deadline {
                    Some(deadline) => {
                        let now = std::time::Instant::now();
                        if now + wait > deadline {
                            return false;
                        }
                        wait
                    }
                    None => wait,
                };
                std::thread::sleep(wait);
            }
        })
    }

    /// The limit that applies to `(provider, model)`, as a dict.
    fn limit_for<'py>(
        &self,
        py: Python<'py>,
        provider: &str,
        model: &str,
    ) -> PyResult<Bound<'py, PyDict>> {
        let rule = self.inner.resolve(provider, model);
        let d = PyDict::new(py);
        d.set_item("requests_per_minute", rule.requests_per_minute)?;
        d.set_item("tokens_per_minute", rule.tokens_per_minute)?;
        d.set_item("burst", rule.burst)?;
        Ok(d)
    }

    /// Refill every bucket (rules are kept).
    fn reset(&self) {
        self.inner.reset();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rpm(n: u32) -> Rule {
        Rule {
            requests_per_minute: Some(n),
            tokens_per_minute: None,
            burst: None,
        }
    }

    #[test]
    fn blocks_after_quota() {
        let l = Limiter::default();
        l.set_rule("gpt-4o", rpm(60));
        for i in 0..60 {
            assert!(
                l.try_acquire("openai", "gpt-4o", 0).is_none(),
                "request {i} blocked"
            );
        }
        let wait = l
            .try_acquire("openai", "gpt-4o", 0)
            .expect("61st must wait");
        assert!(wait <= Duration::from_secs(1) + Duration::from_millis(50));
    }

    #[test]
    fn buckets_are_per_model() {
        let l = Limiter::default();
        l.set_rule("openai:*", rpm(1));
        assert!(l.try_acquire("openai", "a", 0).is_none());
        assert!(l.try_acquire("openai", "b", 0).is_none());
        assert!(l.try_acquire("openai", "a", 0).is_some());
    }

    #[test]
    fn lookup_precedence() {
        let l = Limiter::default();
        l.set_rule("*", rpm(1));
        l.set_rule("openai:*", rpm(2));
        l.set_rule("gpt-4o", rpm(3));
        l.set_rule("azure:gpt-4o", rpm(4));
        assert_eq!(l.resolve("azure", "gpt-4o").requests_per_minute, Some(4));
        assert_eq!(l.resolve("openai", "gpt-4o").requests_per_minute, Some(3));
        assert_eq!(l.resolve("openai", "o3").requests_per_minute, Some(2));
        assert_eq!(l.resolve("anthropic", "x").requests_per_minute, Some(1));
    }

    #[test]
    fn provider_defaults_apply() {
        let l = Limiter::default();
        assert_eq!(l.resolve("openai", "gpt-4o").requests_per_minute, Some(500));
        assert_eq!(
            l.resolve("anthropic", "claude").requests_per_minute,
            Some(50)
        );
        assert_eq!(l.resolve("other", "m").requests_per_minute, None);
        assert!(l.try_acquire("other", "m", 0).is_none());
    }

    #[test]
    fn unlimited_rule_never_blocks() {
        let l = Limiter::default();
        l.set_rule(
            "gpt-4o",
            Rule {
                requests_per_minute: None,
                tokens_per_minute: None,
                burst: None,
            },
        );
        for _ in 0..10_000 {
            assert!(l.try_acquire("openai", "gpt-4o", 0).is_none());
        }
    }

    #[test]
    fn token_budget() {
        let l = Limiter::default();
        l.set_rule(
            "m",
            Rule {
                requests_per_minute: None,
                tokens_per_minute: Some(1000),
                burst: None,
            },
        );
        assert!(l.try_acquire("p", "m", 600).is_none());
        assert!(l.try_acquire("p", "m", 600).is_some());
        // Oversized requests are clamped instead of waiting forever.
        l.reset();
        assert!(l.try_acquire("p", "m", 50_000).is_none());
    }

    #[test]
    fn changing_rules_rebuilds_buckets() {
        let l = Limiter::default();
        l.set_rule("m", rpm(1));
        assert!(l.try_acquire("p", "m", 0).is_none());
        assert!(l.try_acquire("p", "m", 0).is_some());
        l.set_rule("m", rpm(100));
        assert!(l.try_acquire("p", "m", 0).is_none());
    }
}
