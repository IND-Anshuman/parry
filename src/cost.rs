//! Static pricing table and a thread-safe spend ledger.

use std::collections::HashMap;

use parking_lot::{Mutex, RwLock};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Pricing {
    pub input_per_million: f64,
    pub output_per_million: f64,
    /// Prompts larger than this many input tokens are billed at the
    /// `long_*` rates (for models with a long-context price tier).
    pub long_context_threshold: Option<u64>,
    pub long_input_per_million: f64,
    pub long_output_per_million: f64,
}

impl Pricing {
    const fn flat(input: f64, output: f64) -> Self {
        Self {
            input_per_million: input,
            output_per_million: output,
            long_context_threshold: None,
            long_input_per_million: input,
            long_output_per_million: output,
        }
    }

    const fn tiered(input: f64, output: f64, threshold: u64, long_in: f64, long_out: f64) -> Self {
        Self {
            input_per_million: input,
            output_per_million: output,
            long_context_threshold: Some(threshold),
            long_input_per_million: long_in,
            long_output_per_million: long_out,
        }
    }

    pub fn cost(&self, input_tokens: u64, output_tokens: u64) -> f64 {
        let long = self
            .long_context_threshold
            .is_some_and(|t| input_tokens > t);
        let (i, o) = if long {
            (self.long_input_per_million, self.long_output_per_million)
        } else {
            (self.input_per_million, self.output_per_million)
        };
        (input_tokens as f64 * i + output_tokens as f64 * o) / 1_000_000.0
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

fn prefix_matches(model: &str, prefix: &str) -> bool {
    match model.strip_prefix(prefix) {
        Some("") => true,
        Some(rest) => rest.starts_with(['-', '@', ':']),
        None => false,
    }
}

/// Best static match as `(matched prefix length, pricing)`.
fn best_static(model: &str) -> Option<(usize, Pricing)> {
    let m = canonical_model(model);
    PRICING
        .iter()
        .filter(|row| prefix_matches(m, row.model))
        .max_by_key(|row| row.model.len())
        .map(|row| (row.model.len(), row.pricing))
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

    /// Cost in USD of a call, or None if the model is unknown.
    #[pyo3(signature = (model, input_tokens, output_tokens = 0))]
    fn compute(&self, model: &str, input_tokens: u64, output_tokens: u64) -> Option<f64> {
        self.inner.compute(model, input_tokens, output_tokens)
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
        for row in PRICING {
            let d = PyDict::new(py);
            d.set_item("provider", row.provider)?;
            d.set_item("model", row.model)?;
            d.set_item("input_per_million", row.pricing.input_per_million)?;
            d.set_item("output_per_million", row.pricing.output_per_million)?;
            d.set_item("long_context_threshold", row.pricing.long_context_threshold)?;
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
