//! Static pricing table and a thread-safe spend ledger.

use std::collections::HashMap;

use parking_lot::{Mutex, RwLock};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

/// A pricing tier that scales the whole call. Standard is the on-demand rate;
/// Batch is the asynchronous 50%-off queue; Fast is the priority/fast-mode
/// surcharge (OpenAI ~1.75×, Anthropic 2×).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tier {
    Standard,
    Batch,
    Fast,
}

impl Tier {
    fn parse(s: &str) -> Option<Tier> {
        match s {
            "standard" => Some(Tier::Standard),
            "batch" => Some(Tier::Batch),
            "fast" => Some(Tier::Fast),
            _ => None,
        }
    }
}

/// A token breakdown for a single call. The cache fields default to zero, so a
/// plain `(input, output)` call costs exactly what it always did.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Usage {
    /// Standard (uncached) input tokens.
    pub input_tokens: u64,
    pub output_tokens: u64,
    /// Input tokens served from a prompt cache (a cache *hit*).
    pub cached_input_tokens: u64,
    /// Input tokens written to a 5-minute prompt cache.
    pub cache_write_5m_tokens: u64,
    /// Input tokens written to a 1-hour prompt cache.
    pub cache_write_1h_tokens: u64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Pricing {
    pub input_per_million: f64,
    pub output_per_million: f64,
    /// Prompts larger than this many input tokens are billed at the
    /// `long_*` rates (for models with a long-context price tier).
    pub long_context_threshold: Option<u64>,
    pub long_input_per_million: f64,
    pub long_output_per_million: f64,
    /// Prompt-caching rates (per million input tokens). `None` means the tier
    /// is not modelled for this entry, and those tokens fall back to the
    /// standard input rate — an upper bound, so parry never undercharges.
    pub cached_input_per_million: Option<f64>,
    pub cache_write_5m_per_million: Option<f64>,
    pub cache_write_1h_per_million: Option<f64>,
    /// Whole-call tier multipliers. `None` means the tier is not modelled and
    /// is billed at the standard rate (multiplier 1.0).
    pub batch_multiplier: Option<f64>,
    pub fast_multiplier: Option<f64>,
}

impl Pricing {
    const fn flat(input: f64, output: f64) -> Self {
        Self {
            input_per_million: input,
            output_per_million: output,
            long_context_threshold: None,
            long_input_per_million: input,
            long_output_per_million: output,
            cached_input_per_million: None,
            cache_write_5m_per_million: None,
            cache_write_1h_per_million: None,
            batch_multiplier: None,
            fast_multiplier: None,
        }
    }

    const fn tiered(input: f64, output: f64, threshold: u64, long_in: f64, long_out: f64) -> Self {
        Self {
            input_per_million: input,
            output_per_million: output,
            long_context_threshold: Some(threshold),
            long_input_per_million: long_in,
            long_output_per_million: long_out,
            cached_input_per_million: None,
            cache_write_5m_per_million: None,
            cache_write_1h_per_million: None,
            batch_multiplier: None,
            fast_multiplier: None,
        }
    }

    /// Every field spelled out. Used by the generated pricing table
    /// (`src/pricing_generated.rs`), where the sync tool knows each model's
    /// exact cache and batch rates from LiteLLM's data.
    #[allow(dead_code)]
    #[allow(clippy::too_many_arguments)]
    pub const fn full(
        input_per_million: f64,
        output_per_million: f64,
        long_context_threshold: Option<u64>,
        long_input_per_million: f64,
        long_output_per_million: f64,
        cached_input_per_million: Option<f64>,
        cache_write_5m_per_million: Option<f64>,
        cache_write_1h_per_million: Option<f64>,
        batch_multiplier: Option<f64>,
        fast_multiplier: Option<f64>,
    ) -> Self {
        Self {
            input_per_million,
            output_per_million,
            long_context_threshold,
            long_input_per_million,
            long_output_per_million,
            cached_input_per_million,
            cache_write_5m_per_million,
            cache_write_1h_per_million,
            batch_multiplier,
            fast_multiplier,
        }
    }

    /// Fill in the tier rates that aren't hand-entered, from provider-standard
    /// ratios off the base input rate: cache read is 10% of input for both
    /// providers; Anthropic adds a 5-minute cache write at 1.25× and a 1-hour
    /// write at 2×; batch is 50% off for both; fast is 1.75× (OpenAI) or 2×
    /// (Anthropic). Later Phase 0 work replaces these ratios with the exact
    /// per-model rates synced from LiteLLM's table.
    pub fn with_provider_tiers(mut self, provider: &str) -> Self {
        let cache_read = self.input_per_million * 0.10;
        match provider {
            "anthropic" => {
                self.cached_input_per_million.get_or_insert(cache_read);
                self.cache_write_5m_per_million
                    .get_or_insert(self.input_per_million * 1.25);
                self.cache_write_1h_per_million
                    .get_or_insert(self.input_per_million * 2.00);
                self.batch_multiplier.get_or_insert(0.50);
                self.fast_multiplier.get_or_insert(2.00);
            }
            "openai" => {
                // OpenAI caches prompts automatically, so there is no separate
                // cache-write charge; only the cache-read discount applies.
                self.cached_input_per_million.get_or_insert(cache_read);
                self.batch_multiplier.get_or_insert(0.50);
                self.fast_multiplier.get_or_insert(1.75);
            }
            _ => {}
        }
        self
    }

    /// Standard-tier cost of a plain input/output call (no caching).
    pub fn cost(&self, input_tokens: u64, output_tokens: u64) -> f64 {
        self.cost_detailed(
            &Usage {
                input_tokens,
                output_tokens,
                ..Usage::default()
            },
            Tier::Standard,
        )
    }

    /// Cost of a call described by a full token breakdown at a given tier.
    pub fn cost_detailed(&self, u: &Usage, tier: Tier) -> f64 {
        let total_input = u.input_tokens
            + u.cached_input_tokens
            + u.cache_write_5m_tokens
            + u.cache_write_1h_tokens;
        let long = self.long_context_threshold.is_some_and(|t| total_input > t);
        let (in_rate, out_rate) = if long {
            (self.long_input_per_million, self.long_output_per_million)
        } else {
            (self.input_per_million, self.output_per_million)
        };
        // Unknown cache rates fall back to the standard input rate (upper bound).
        let read_rate = self.cached_input_per_million.unwrap_or(in_rate);
        let write_5m_rate = self.cache_write_5m_per_million.unwrap_or(in_rate);
        let write_1h_rate = self.cache_write_1h_per_million.unwrap_or(in_rate);
        let raw = u.input_tokens as f64 * in_rate
            + u.cached_input_tokens as f64 * read_rate
            + u.cache_write_5m_tokens as f64 * write_5m_rate
            + u.cache_write_1h_tokens as f64 * write_1h_rate
            + u.output_tokens as f64 * out_rate;
        let mult = match tier {
            Tier::Standard => 1.0,
            Tier::Batch => self.batch_multiplier.unwrap_or(1.0),
            Tier::Fast => self.fast_multiplier.unwrap_or(1.0),
        };
        raw * mult / 1_000_000.0
    }
}

pub struct ModelPricing {
    pub provider: &'static str,
    pub model: &'static str,
    pub pricing: Pricing,
}

const fn p(provider: &'static str, model: &'static str, pricing: Pricing) -> ModelPricing {
    ModelPricing {
        provider,
        model,
        pricing,
    }
}

// Pricing last verified 2026-10-25 from https://developers.openai.com/api/docs/pricing
// (standard tier, short-context rates) and https://docs.claude.com/en/docs/about-claude/pricing.
// Open an issue if anything here is outdated. Entries are matched by longest
// prefix on a `-`, `@` or `:` boundary, so dated snapshots resolve to their
// family (`gpt-4o-2024-08-06` -> `gpt-4o`) unless they have their own row.
// Prompt-caching discounts and batch/flex tiers are not modelled: recorded
// cost is an upper bound for those requests.
pub static PRICING: &[ModelPricing] = &[
    // --- OpenAI -----------------------------------------------------------
    p("openai", "gpt-6-astra", Pricing::flat(10.00, 50.00)),
    p("openai", "gpt-6.1-sol", Pricing::flat(2.00, 10.00)),
    p("openai", "gpt-6-sol", Pricing::flat(2.00, 10.00)),
    p("openai", "gpt-6-luna", Pricing::flat(0.10, 0.50)),
    p("openai", "gpt-5.6-sol", Pricing::flat(4.00, 20.00)),
    p("openai", "gpt-5.6-terra", Pricing::flat(2.00, 12.00)),
    p("openai", "gpt-5.6-luna", Pricing::flat(0.20, 1.20)),
    p("openai", "gpt-5.6-cyber", Pricing::flat(12.50, 75.00)),
    p("openai", "gpt-5.5", Pricing::flat(5.00, 30.00)),
    p("openai", "gpt-5.5-pro", Pricing::flat(30.00, 180.00)),
    p("openai", "gpt-5.5-cyber", Pricing::flat(12.50, 75.00)),
    p("openai", "gpt-5.4", Pricing::flat(2.50, 15.00)),
    p("openai", "gpt-5.4-pro", Pricing::flat(30.00, 180.00)),
    p("openai", "gpt-5.4-mini", Pricing::flat(0.75, 4.50)),
    p("openai", "gpt-5.4-nano", Pricing::flat(0.20, 1.25)),
    p("openai", "gpt-5.3-codex", Pricing::flat(1.75, 14.00)),
    p("openai", "gpt-5.2", Pricing::flat(1.75, 14.00)),
    p("openai", "gpt-5.2-pro", Pricing::flat(21.00, 168.00)),
    p("openai", "gpt-5.1", Pricing::flat(1.25, 10.00)),
    p("openai", "gpt-5", Pricing::flat(1.25, 10.00)),
    p("openai", "gpt-5-mini", Pricing::flat(0.25, 2.00)),
    p("openai", "gpt-5-nano", Pricing::flat(0.05, 0.40)),
    p("openai", "gpt-5-pro", Pricing::flat(15.00, 120.00)),
    p("openai", "gpt-5-search-api", Pricing::flat(1.25, 10.00)),
    p("openai", "chat-latest", Pricing::flat(5.00, 30.00)),
    p("openai", "gpt-4.1", Pricing::flat(2.00, 8.00)),
    p("openai", "gpt-4.1-mini", Pricing::flat(0.40, 1.60)),
    p("openai", "gpt-4.1-nano", Pricing::flat(0.10, 0.40)),
    p("openai", "gpt-4o", Pricing::flat(2.50, 10.00)),
    p("openai", "gpt-4o-2024-05-13", Pricing::flat(5.00, 15.00)),
    p("openai", "gpt-4o-mini", Pricing::flat(0.15, 0.60)),
    p("openai", "gpt-4-turbo", Pricing::flat(10.00, 30.00)),
    p("openai", "gpt-4", Pricing::flat(30.00, 60.00)),
    p("openai", "gpt-3.5-turbo", Pricing::flat(0.50, 1.50)),
    p("openai", "o1", Pricing::flat(15.00, 60.00)),
    p("openai", "o1-pro", Pricing::flat(150.00, 600.00)),
    p("openai", "o3", Pricing::flat(2.00, 8.00)),
    p("openai", "o3-pro", Pricing::flat(20.00, 80.00)),
    p("openai", "o3-mini", Pricing::flat(1.10, 4.40)),
    p("openai", "o4-mini", Pricing::flat(1.10, 4.40)),
    // --- Anthropic --------------------------------------------------------
    p("anthropic", "claude-fable-5-1", Pricing::flat(10.00, 50.00)),
    p("anthropic", "claude-fable-5", Pricing::flat(10.00, 50.00)),
    p(
        "anthropic",
        "claude-mythos-5-1",
        Pricing::flat(10.00, 50.00),
    ),
    p("anthropic", "claude-opus-5-5", Pricing::flat(4.00, 20.00)),
    p("anthropic", "claude-opus-5", Pricing::flat(5.00, 25.00)),
    p("anthropic", "claude-opus-4-8", Pricing::flat(5.00, 25.00)),
    p("anthropic", "claude-opus-4-7", Pricing::flat(5.00, 25.00)),
    p("anthropic", "claude-opus-4-6", Pricing::flat(5.00, 25.00)),
    p("anthropic", "claude-opus-4-5", Pricing::flat(5.00, 25.00)),
    p("anthropic", "claude-opus-4-1", Pricing::flat(15.00, 75.00)),
    p("anthropic", "claude-opus-4", Pricing::flat(15.00, 75.00)),
    p("anthropic", "claude-sonnet-5-5", Pricing::flat(2.00, 10.00)),
    p("anthropic", "claude-sonnet-5", Pricing::flat(2.00, 10.00)),
    p("anthropic", "claude-sonnet-4-6", Pricing::flat(3.00, 15.00)),
    p("anthropic", "claude-sonnet-4-5", Pricing::flat(3.00, 15.00)),
    p("anthropic", "claude-sonnet-4", Pricing::flat(3.00, 15.00)),
    p("anthropic", "claude-3-7-sonnet", Pricing::flat(3.00, 15.00)),
    p("anthropic", "claude-3-5-sonnet", Pricing::flat(3.00, 15.00)),
    p(
        "anthropic",
        "claude-haiku-5-5",
        Pricing::tiered(0.10, 0.50, 100_000, 0.50, 2.50),
    ),
    p("anthropic", "claude-haiku-4-5", Pricing::flat(1.00, 5.00)),
    p("anthropic", "claude-3-5-haiku", Pricing::flat(0.80, 4.00)),
    p("anthropic", "claude-3-haiku", Pricing::flat(0.25, 1.25)),
];

/// Reduce provider-specific spellings to the bare model name:
/// `ft:gpt-4o-mini-2024-07-18:org::id` -> `gpt-4o-mini-2024-07-18`,
/// `anthropic.claude-opus-5-5` (Bedrock) -> `claude-opus-5-5`,
/// `openai/gpt-4o` (routers) -> `gpt-4o`.
pub fn canonical_model(model: &str) -> &str {
    let mut m = model.trim();
    if let Some(rest) = m.strip_prefix("ft:") {
        m = rest.split(':').next().unwrap_or(rest);
    }
    if let Some((_, tail)) = m.rsplit_once('/') {
        m = tail;
    }
    for prefix in [
        "anthropic.",
        "us.anthropic.",
        "eu.anthropic.",
        "apac.anthropic.",
        "global.anthropic.",
    ] {
        if let Some(rest) = m.strip_prefix(prefix) {
            m = rest;
        }
    }
    m
}

pub(crate) fn prefix_matches(model: &str, prefix: &str) -> bool {
    match model.strip_prefix(prefix) {
        Some("") => true,
        Some(rest) => rest.starts_with(['-', '@', ':']),
        None => false,
    }
}

/// Best static match as `(matched prefix length, pricing)`. Hand-curated rows
/// in `PRICING` are searched alongside the LiteLLM-synced rows in `GENERATED`;
/// on an equal-length prefix the hand-curated row wins, so a sync can add
/// breadth without ever overwriting a row parry maintains more accurately.
fn best_match<'a>(
    m: &str,
    hand: &'a [ModelPricing],
    generated: &'a [ModelPricing],
) -> Option<&'a ModelPricing> {
    let longest = |rows: &'a [ModelPricing]| {
        rows.iter()
            .filter(|row| prefix_matches(m, row.model))
            .max_by_key(|row| row.model.len())
    };
    match (longest(hand), longest(generated)) {
        // Equal-length prefix: the hand-curated row wins.
        (Some(h), Some(g)) => Some(if h.model.len() >= g.model.len() { h } else { g }),
        (Some(h), None) => Some(h),
        (None, Some(g)) => Some(g),
        (None, None) => None,
    }
}

fn best_static(model: &str) -> Option<(usize, Pricing)> {
    let m = canonical_model(model);
    best_match(m, PRICING, crate::pricing_generated::GENERATED).map(|row| {
        (
            row.model.len(),
            row.pricing.with_provider_tiers(row.provider),
        )
    })
}

#[cfg(test)]
pub fn lookup_static(model: &str) -> Option<Pricing> {
    best_static(model).map(|(_, p)| p)
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Totals {
    pub requests: u64,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cost_usd: f64,
    pub unpriced_requests: u64,
    pub estimated_requests: u64,
}

impl Totals {
    fn add(&mut self, input: u64, output: u64, cost: Option<f64>, estimated: bool) {
        self.requests += 1;
        self.input_tokens += input;
        self.output_tokens += output;
        match cost {
            Some(c) => self.cost_usd += c,
            None => self.unpriced_requests += 1,
        }
        if estimated {
            self.estimated_requests += 1;
        }
    }
}

#[derive(Default)]
pub struct Ledger {
    overrides: RwLock<HashMap<String, Pricing>>,
    state: Mutex<(Totals, HashMap<String, Totals>)>,
}

impl Ledger {
    /// Overrides and the static table compete on prefix length, overrides
    /// winning ties, so overriding `gpt-4o` does not reprice `gpt-4o-mini`.
    pub fn pricing(&self, model: &str) -> Option<Pricing> {
        let fallback = best_static(model);
        let overrides = self.overrides.read();
        if overrides.is_empty() {
            return fallback.map(|(_, p)| p);
        }
        if let Some(exact) = overrides.get(model) {
            return Some(*exact);
        }
        let m = canonical_model(model);
        let custom = overrides
            .iter()
            .filter(|(k, _)| prefix_matches(m, k))
            .max_by_key(|(k, _)| k.len())
            .map(|(k, v)| (k.len(), *v));
        match (custom, fallback) {
            (Some((cl, c)), Some((sl, _))) if cl >= sl => Some(c),
            (Some(_), Some((_, s))) => Some(s),
            (Some((_, c)), None) => Some(c),
            (None, s) => s.map(|(_, p)| p),
        }
    }

    pub fn set_pricing(&self, model: &str, pricing: Pricing) {
        self.overrides.write().insert(model.to_owned(), pricing);
    }

    pub fn compute(&self, model: &str, input: u64, output: u64) -> Option<f64> {
        self.pricing(model).map(|p| p.cost(input, output))
    }

    pub fn compute_tiered(&self, model: &str, usage: &Usage, tier: Tier) -> Option<f64> {
        self.pricing(model).map(|p| p.cost_detailed(usage, tier))
    }

    /// Record one completed request with a full token breakdown and tier.
    /// Cache tokens are folded into the input-token totals for accounting, but
    /// billed at their own tier rates. Returns the cost charged (None if the
    /// model is not in the pricing table).
    pub fn record_tiered(
        &self,
        provider: &str,
        model: &str,
        usage: &Usage,
        tier: Tier,
        estimated: bool,
    ) -> Option<f64> {
        let cost = self.compute_tiered(model, usage, tier);
        let total_in = usage.input_tokens
            + usage.cached_input_tokens
            + usage.cache_write_5m_tokens
            + usage.cache_write_1h_tokens;
        let mut state = self.state.lock();
        state.0.add(total_in, usage.output_tokens, cost, estimated);
        state
            .1
            .entry(format!("{provider}:{model}"))
            .or_default()
            .add(total_in, usage.output_tokens, cost, estimated);
        cost
    }

    /// Record one completed request. Returns the cost charged (None if the
    /// model is not in the pricing table).
    pub fn record(
        &self,
        provider: &str,
        model: &str,
        input: u64,
        output: u64,
        estimated: bool,
    ) -> Option<f64> {
        let cost = self.compute(model, input, output);
        let mut state = self.state.lock();
        state.0.add(input, output, cost, estimated);
        state
            .1
            .entry(format!("{provider}:{model}"))
            .or_default()
            .add(input, output, cost, estimated);
        cost
    }

    pub fn totals(&self) -> Totals {
        self.state.lock().0.clone()
    }

    pub fn per_model(&self) -> Vec<(String, Totals)> {
        let state = self.state.lock();
        let mut rows: Vec<_> = state
            .1
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        rows.sort_by(|a, b| {
            b.1.cost_usd
                .total_cmp(&a.1.cost_usd)
                .then_with(|| a.0.cmp(&b.0))
        });
        rows
    }

    pub fn reset(&self) {
        *self.state.lock() = Default::default();
    }
}

fn totals_dict<'py>(py: Python<'py>, t: &Totals) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("requests", t.requests)?;
    d.set_item("input_tokens", t.input_tokens)?;
    d.set_item("output_tokens", t.output_tokens)?;
    d.set_item("cost_usd", t.cost_usd)?;
    d.set_item("unpriced_requests", t.unpriced_requests)?;
    d.set_item("estimated_requests", t.estimated_requests)?;
    Ok(d)
}

/// Pricing lookup plus a running spend ledger.
///
/// >>> cost = Cost()
/// >>> cost.compute("gpt-4o", input_tokens=1_000_000, output_tokens=0)
/// 2.5
#[pyclass(name = "Cost", module = "parry._parry_native", frozen)]
#[derive(Default)]
pub struct PyCost {
    inner: Ledger,
}

#[pymethods]
impl PyCost {
    #[new]
    fn new() -> Self {
        Self::default()
    }

    /// `(input_per_million, output_per_million)` for a model, or None.
    fn price(&self, model: &str) -> Option<(f64, f64)> {
        self.inner
            .pricing(model)
            .map(|p| (p.input_per_million, p.output_per_million))
    }

    /// Per-tier rates for a model, or None if the model is unknown. Keys:
    /// `cached_input_per_million`, `cache_write_5m_per_million`,
    /// `cache_write_1h_per_million`, `batch_multiplier`, `fast_multiplier`
    /// (each may itself be None when the tier is not modelled).
    fn tier_rates<'py>(
        &self,
        py: Python<'py>,
        model: &str,
    ) -> PyResult<Option<Bound<'py, PyDict>>> {
        let Some(p) = self.inner.pricing(model) else {
            return Ok(None);
        };
        let d = PyDict::new(py);
        d.set_item("cached_input_per_million", p.cached_input_per_million)?;
        d.set_item("cache_write_5m_per_million", p.cache_write_5m_per_million)?;
        d.set_item("cache_write_1h_per_million", p.cache_write_1h_per_million)?;
        d.set_item("batch_multiplier", p.batch_multiplier)?;
        d.set_item("fast_multiplier", p.fast_multiplier)?;
        Ok(Some(d))
    }

    /// Cost in USD of a call, or None if the model is unknown.
    #[pyo3(signature = (model, input_tokens, output_tokens = 0))]
    fn compute(&self, model: &str, input_tokens: u64, output_tokens: u64) -> Option<f64> {
        self.inner.compute(model, input_tokens, output_tokens)
    }

    /// Cost in USD of a call with a full token breakdown and pricing tier, or
    /// None if the model is unknown. `tier` is "standard", "batch" or "fast".
    #[pyo3(signature = (
        model,
        input_tokens,
        output_tokens = 0,
        *,
        cached_input_tokens = 0,
        cache_write_5m_tokens = 0,
        cache_write_1h_tokens = 0,
        tier = "standard",
    ))]
    #[allow(clippy::too_many_arguments)]
    fn compute_tiered(
        &self,
        model: &str,
        input_tokens: u64,
        output_tokens: u64,
        cached_input_tokens: u64,
        cache_write_5m_tokens: u64,
        cache_write_1h_tokens: u64,
        tier: &str,
    ) -> PyResult<Option<f64>> {
        let tier = Tier::parse(tier)
            .ok_or_else(|| PyValueError::new_err("tier must be 'standard', 'batch' or 'fast'"))?;
        let usage = Usage {
            input_tokens,
            output_tokens,
            cached_input_tokens,
            cache_write_5m_tokens,
            cache_write_1h_tokens,
        };
        Ok(self.inner.compute_tiered(model, &usage, tier))
    }

    /// Override or add pricing for a model (exact name or family prefix).
    #[pyo3(signature = (model, input_per_million, output_per_million))]
    fn set_pricing(
        &self,
        model: &str,
        input_per_million: f64,
        output_per_million: f64,
    ) -> PyResult<()> {
        if !(input_per_million.is_finite() && output_per_million.is_finite())
            || input_per_million < 0.0
            || output_per_million < 0.0
        {
            return Err(PyValueError::new_err(
                "prices must be finite and non-negative",
            ));
        }
        self.inner
            .set_pricing(model, Pricing::flat(input_per_million, output_per_million));
        Ok(())
    }

    /// Record a completed call with a full token breakdown and pricing tier.
    /// Returns the cost charged (None if the model is unpriced).
    #[pyo3(signature = (
        provider,
        model,
        input_tokens,
        output_tokens,
        cached_input_tokens = 0,
        cache_write_5m_tokens = 0,
        cache_write_1h_tokens = 0,
        tier = "standard",
        estimated = false,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn record_tiered(
        &self,
        provider: &str,
        model: &str,
        input_tokens: u64,
        output_tokens: u64,
        cached_input_tokens: u64,
        cache_write_5m_tokens: u64,
        cache_write_1h_tokens: u64,
        tier: &str,
        estimated: bool,
    ) -> PyResult<Option<f64>> {
        let tier = Tier::parse(tier)
            .ok_or_else(|| PyValueError::new_err("tier must be 'standard', 'batch' or 'fast'"))?;
        let usage = Usage {
            input_tokens,
            output_tokens,
            cached_input_tokens,
            cache_write_5m_tokens,
            cache_write_1h_tokens,
        };
        Ok(self
            .inner
            .record_tiered(provider, model, &usage, tier, estimated))
    }

    /// Record a completed call and return the cost charged (None if unpriced).
    #[pyo3(signature = (provider, model, input_tokens, output_tokens, estimated = false))]
    fn record(
        &self,
        provider: &str,
        model: &str,
        input_tokens: u64,
        output_tokens: u64,
        estimated: bool,
    ) -> Option<f64> {
        self.inner
            .record(provider, model, input_tokens, output_tokens, estimated)
    }

    #[getter]
    fn total_cost_usd(&self) -> f64 {
        self.inner.totals().cost_usd
    }

    /// Session totals as a dict.
    fn totals<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        totals_dict(py, &self.inner.totals())
    }

    /// Per `provider:model` totals, most expensive first.
    fn by_model<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        for (k, t) in self.inner.per_model() {
            d.set_item(k, totals_dict(py, &t)?)?;
        }
        Ok(d)
    }

    fn reset(&self) {
        self.inner.reset();
    }

    /// The built-in pricing table as a list of dicts.
    #[staticmethod]
    fn pricing_table<'py>(py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let list = PyList::empty(py);
        for row in PRICING.iter().chain(crate::pricing_generated::GENERATED) {
            let pricing = row.pricing.with_provider_tiers(row.provider);
            let d = PyDict::new(py);
            d.set_item("provider", row.provider)?;
            d.set_item("model", row.model)?;
            d.set_item("input_per_million", pricing.input_per_million)?;
            d.set_item("output_per_million", pricing.output_per_million)?;
            d.set_item("long_context_threshold", pricing.long_context_threshold)?;
            d.set_item("cached_input_per_million", pricing.cached_input_per_million)?;
            d.set_item(
                "cache_write_5m_per_million",
                pricing.cache_write_5m_per_million,
            )?;
            d.set_item(
                "cache_write_1h_per_million",
                pricing.cache_write_1h_per_million,
            )?;
            d.set_item("batch_multiplier", pricing.batch_multiplier)?;
            d.set_item("fast_multiplier", pricing.fast_multiplier)?;
            list.append(d)?;
        }
        Ok(list)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    #[test]
    fn longest_prefix_wins() {
        assert_eq!(lookup_static("gpt-4o").unwrap().input_per_million, 2.50);
        assert_eq!(
            lookup_static("gpt-4o-mini").unwrap().input_per_million,
            0.15
        );
        assert_eq!(
            lookup_static("gpt-4o-2024-08-06")
                .unwrap()
                .input_per_million,
            2.50
        );
        assert_eq!(
            lookup_static("gpt-4o-2024-05-13")
                .unwrap()
                .input_per_million,
            5.00
        );
        assert_eq!(
            lookup_static("gpt-4o-mini-2024-07-18")
                .unwrap()
                .input_per_million,
            0.15
        );
        assert_eq!(
            lookup_static("gpt-4.1-mini").unwrap().input_per_million,
            0.40
        );
        assert_eq!(lookup_static("o3-mini").unwrap().input_per_million, 1.10);
        assert_eq!(
            lookup_static("claude-opus-4-1-20250805")
                .unwrap()
                .input_per_million,
            15.0
        );
    }

    #[test]
    fn prefix_requires_boundary() {
        // "gpt-4" must not swallow "gpt-4.1" or "gpt-4o"; "o1" must not match "o10".
        assert_eq!(lookup_static("gpt-4.1").unwrap().input_per_million, 2.00);
        assert!(lookup_static("o10").is_none());
        assert!(lookup_static("my-model").is_none());
    }

    #[test]
    fn provider_spellings_resolve() {
        assert_eq!(
            canonical_model("ft:gpt-4o-mini-2024-07-18:acme::abc"),
            "gpt-4o-mini-2024-07-18"
        );
        assert_eq!(
            canonical_model("anthropic.claude-opus-5-5"),
            "claude-opus-5-5"
        );
        assert_eq!(
            canonical_model("us.anthropic.claude-sonnet-4-5"),
            "claude-sonnet-4-5"
        );
        assert_eq!(canonical_model("openai/gpt-4o"), "gpt-4o");
        assert!(lookup_static("ft:gpt-4o-mini-2024-07-18:acme::abc").is_some());
    }

    #[test]
    fn computes_cost() {
        let l = Ledger::default();
        // 1000 in, 500 out on gpt-4o: 1000*2.5/1e6 + 500*10/1e6
        assert!(close(l.compute("gpt-4o", 1000, 500).unwrap(), 0.0075));
        assert!(l.compute("unknown-model", 1, 1).is_none());
    }

    #[test]
    fn tiered_pricing() {
        let haiku = lookup_static("claude-haiku-5-5").unwrap();
        assert!(close(haiku.cost(100_000, 0), 0.01));
        assert!(close(haiku.cost(100_001, 0), 100_001.0 * 0.5 / 1e6));
    }

    #[test]
    fn ledger_accumulates() {
        let l = Ledger::default();
        for _ in 0..5 {
            l.record("openai", "gpt-4o-mini", 1000, 200, false);
        }
        l.record("x", "mystery", 10, 10, true);
        let t = l.totals();
        assert_eq!(t.requests, 6);
        assert_eq!(t.unpriced_requests, 1);
        assert_eq!(t.estimated_requests, 1);
        assert!(close(
            t.cost_usd,
            5.0 * (1000.0 * 0.15 + 200.0 * 0.60) / 1e6
        ));
        assert_eq!(l.per_model()[0].0, "openai:gpt-4o-mini");
    }

    #[test]
    fn generated_rows_merge_with_hand_rows() {
        let hand = [p("openai", "gpt-4o", Pricing::flat(2.50, 10.00))];
        let generated = [
            // Same model as a hand row: hand must win the tie.
            p("openai", "gpt-4o", Pricing::flat(99.0, 99.0)),
            // A model parry doesn't hand-maintain: filled from the sync.
            p("google", "gemini-3-pro", Pricing::flat(2.00, 12.00)),
            // Longer prefix than any hand row: the sync row wins.
            p("openai", "gpt-4o-audio", Pricing::flat(40.0, 80.0)),
        ];
        // Hand wins the tie on gpt-4o.
        assert_eq!(
            best_match("gpt-4o", &hand, &generated)
                .unwrap()
                .pricing
                .input_per_million,
            2.50
        );
        // Gap filled from the generated table.
        assert_eq!(
            best_match("gemini-3-pro", &hand, &generated)
                .unwrap()
                .pricing
                .input_per_million,
            2.00
        );
        // Longer generated prefix beats the shorter hand prefix.
        assert_eq!(
            best_match("gpt-4o-audio-preview", &hand, &generated)
                .unwrap()
                .pricing
                .input_per_million,
            40.0
        );
        // No match anywhere.
        assert!(best_match("no-such-model", &hand, &generated).is_none());
    }

    #[test]
    fn provider_tiers_preserve_explicit_synced_rates() {
        // A synced row carries exact LiteLLM rates; enrichment must not clobber
        // them. 0.99 is deliberately not the derived 0.10x input, to prove it.
        let pr = Pricing::full(
            3.0,
            15.0,
            None,
            3.0,
            15.0,
            Some(0.99),
            None,
            None,
            Some(0.40),
            None,
        );
        let enriched = pr.with_provider_tiers("anthropic");
        assert_eq!(enriched.cached_input_per_million, Some(0.99)); // explicit kept
        assert_eq!(enriched.batch_multiplier, Some(0.40)); // explicit kept (not 0.50)
        assert_eq!(enriched.cache_write_5m_per_million, Some(3.0 * 1.25)); // gap filled
        assert_eq!(enriched.fast_multiplier, Some(2.0)); // gap filled
    }

    #[test]
    fn provider_tiers_are_derived() {
        let oa = lookup_static("gpt-4o").unwrap();
        assert_eq!(oa.cached_input_per_million, Some(0.25)); // 2.50 * 0.10
        assert_eq!(oa.batch_multiplier, Some(0.50));
        assert_eq!(oa.fast_multiplier, Some(1.75));
        assert_eq!(oa.cache_write_5m_per_million, None); // OpenAI: auto-cached

        let an = lookup_static("claude-sonnet-5-5").unwrap();
        assert_eq!(an.cached_input_per_million, Some(0.20)); // 2.00 * 0.10
        assert_eq!(an.cache_write_5m_per_million, Some(2.50)); // 2.00 * 1.25
        assert_eq!(an.cache_write_1h_per_million, Some(4.00)); // 2.00 * 2.00
        assert_eq!(an.fast_multiplier, Some(2.00));
    }

    #[test]
    fn plain_cost_unchanged_by_tier_fields() {
        let p = lookup_static("gpt-4o").unwrap();
        assert!(close(
            p.cost(1000, 500),
            (1000.0 * 2.5 + 500.0 * 10.0) / 1e6
        ));
    }

    #[test]
    fn batch_tier_halves_cost_and_fast_raises_it() {
        let l = Ledger::default();
        let std = l.compute("gpt-4o", 1000, 500).unwrap();
        let u = Usage {
            input_tokens: 1000,
            output_tokens: 500,
            ..Usage::default()
        };
        assert!(close(
            l.compute_tiered("gpt-4o", &u, Tier::Batch).unwrap(),
            std * 0.50
        ));
        assert!(close(
            l.compute_tiered("gpt-4o", &u, Tier::Fast).unwrap(),
            std * 1.75
        ));
    }

    #[test]
    fn cache_read_is_cheaper_than_standard_input() {
        let l = Ledger::default();
        let cached = Usage {
            cached_input_tokens: 1000,
            ..Usage::default()
        };
        let standard = Usage {
            input_tokens: 1000,
            ..Usage::default()
        };
        let c = l
            .compute_tiered("claude-sonnet-5-5", &cached, Tier::Standard)
            .unwrap();
        let s = l
            .compute_tiered("claude-sonnet-5-5", &standard, Tier::Standard)
            .unwrap();
        assert!(close(c, s * 0.10));
    }

    #[test]
    fn unknown_tier_rates_fall_back_to_standard_rate() {
        // A custom override carries no tier rates, so cached tokens bill at the
        // standard input rate and batch is a no-op (never silently free).
        let l = Ledger::default();
        l.set_pricing("my-model", Pricing::flat(3.0, 6.0));
        let cached = Usage {
            cached_input_tokens: 1_000_000,
            ..Usage::default()
        };
        assert!(close(
            l.compute_tiered("my-model", &cached, Tier::Standard)
                .unwrap(),
            3.0
        ));
        let io = Usage {
            input_tokens: 1_000_000,
            ..Usage::default()
        };
        assert!(close(
            l.compute_tiered("my-model", &io, Tier::Batch).unwrap(),
            3.0
        ));
    }

    #[test]
    fn record_tiered_bills_cache_and_folds_token_totals() {
        let l = Ledger::default();
        // 1000 standard + 500 cache-read input, 200 output on claude-sonnet-5-5.
        let u = Usage {
            input_tokens: 1000,
            output_tokens: 200,
            cached_input_tokens: 500,
            ..Usage::default()
        };
        let cost = l
            .record_tiered("anthropic", "claude-sonnet-5-5", &u, Tier::Standard, false)
            .unwrap();
        let expected = (1000.0 * 2.00 + 500.0 * 0.20 + 200.0 * 10.00) / 1e6;
        assert!(close(cost, expected));
        let t = l.totals();
        assert_eq!(t.input_tokens, 1500); // cache tokens fold into the input total
        assert_eq!(t.output_tokens, 200);
        assert_eq!(t.requests, 1);
    }

    #[test]
    fn overrides_take_precedence() {
        let l = Ledger::default();
        l.set_pricing("gpt-4o", Pricing::flat(1.0, 1.0));
        l.set_pricing("my-finetune", Pricing::flat(7.0, 7.0));
        assert!(close(
            l.compute("gpt-4o-2024-08-06", 1_000_000, 0).unwrap(),
            1.0
        ));
        assert!(close(l.compute("my-finetune", 1_000_000, 0).unwrap(), 7.0));
        // Unrelated models still use the table.
        assert!(close(l.compute("gpt-4o-mini", 1_000_000, 0).unwrap(), 0.15));
    }
}
