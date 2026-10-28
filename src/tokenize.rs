//! Request normalisation (for cache keys) and token counting (for cost
//! estimates and token-per-minute limits).
//!
//! Token counts use tiktoken. OpenAI models get their real encoding; every
//! other model (Anthropic included, whose tokenizer is not public) gets
//! `o200k_base` as an approximation. Counts are only ever used for
//! *estimates*: real usage reported by the provider always wins.
//!
//! Anthropic's newer tokenizer (Claude 4.7+ and the Mythos preview) produces
//! roughly 30% more tokens than `o200k_base` for the same text, per Anthropic's
//! pricing page. Since that tokenizer is not public either, counts for those
//! models are scaled by [`ANTHROPIC_NEW_TOKENIZER_MULTIPLIER`]. Without this,
//! the token-bucket limiter and the pre-flight `max_cost_usd` check both
//! under-enforce by ~30% on those models.

use std::fmt::Write as _;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use serde_json::Value;
use tiktoken_rs::CoreBPE;

use crate::cost::{canonical_model, prefix_matches};

/// Keys whose string values are prompt text worth counting.
const TEXT_KEYS: &[&str] = &[
    "content",
    "text",
    "system",
    "instructions",
    "input",
    "arguments",
    "description",
    "prompt",
];

/// Approximate per-message framing overhead (role markers etc.).
const TOKENS_PER_MESSAGE: usize = 4;

/// Claude 4.7+ and the Mythos preview use a newer tokenizer that produces
/// about 30% more tokens than the previous one for the same text. parry has no
/// public tokenizer for these models, so it scales the `o200k_base` estimate by
/// this factor. Source: Anthropic pricing page, verified 2026-10-26.
const ANTHROPIC_NEW_TOKENIZER_MULTIPLIER: f64 = 1.30;

/// Models (by family prefix) that use Anthropic's newer tokenizer. Claude
/// Sonnet 4.6 / Opus 4.6 and earlier use the previous tokenizer, so they are
/// deliberately absent. Dated snapshots and provider spellings resolve to these
/// prefixes via [`canonical_model`] + [`prefix_matches`].
const ANTHROPIC_NEW_TOKENIZER_MODELS: &[&str] = &[
    "claude-opus-4-7",
    "claude-opus-4-8",
    "claude-opus-5",
    "claude-opus-5-5",
    "claude-sonnet-5",
    "claude-sonnet-5-5",
    "claude-mythos-5-1",
    "claude-fable-5",
    "claude-fable-5-1",
];

/// The tokenizer multiplier for a model: `1.30` for Claude 4.7+/Mythos, else
/// `1.0`. Matching mirrors the pricing table so that `claude-opus-5-5`,
/// `claude-opus-5-5-20260101` and `anthropic.claude-opus-5-5` all resolve alike.
fn token_multiplier(model: &str) -> f64 {
    let m = canonical_model(model);
    if ANTHROPIC_NEW_TOKENIZER_MODELS
        .iter()
        .any(|prefix| prefix_matches(m, prefix))
    {
        ANTHROPIC_NEW_TOKENIZER_MULTIPLIER
    } else {
        1.0
    }
}

/// Scale a raw `o200k_base` count by the model's tokenizer multiplier.
fn scale_for_model(raw: usize, model: &str) -> usize {
    let mult = token_multiplier(model);
    if mult == 1.0 {
        raw
    } else {
        (raw as f64 * mult).round() as usize
    }
}

#[derive(Debug, thiserror::Error)]
pub enum NormalizeError {
    #[error("request payload is not valid JSON: {0}")]
    InvalidJson(#[from] serde_json::Error),
}

impl From<NormalizeError> for PyErr {
    fn from(err: NormalizeError) -> Self {
        PyValueError::new_err(err.to_string())
    }
}

/// Canonicalise a JSON document so that semantically identical requests hash
/// identically: object keys are sorted, CRLF becomes LF and leading/trailing
/// whitespace of every string is trimmed. Internal whitespace is preserved.
pub fn normalize_json(payload: &str) -> Result<String, NormalizeError> {
    let value: Value = serde_json::from_str(payload)?;
    let mut out = String::with_capacity(payload.len());
    write_canonical(&value, &mut out);
    Ok(out)
}

fn write_canonical(value: &Value, out: &mut String) {
    match value {
        Value::Null | Value::Bool(_) | Value::Number(_) => {
            // `Display` for these variants is already canonical JSON.
            let _ = write!(out, "{value}");
        }
        Value::String(s) => write_string(&normalize_text(s), out),
        Value::Array(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_canonical(item, out);
            }
            out.push(']');
        }
        Value::Object(map) => {
            // Sort explicitly: serde_json's map is only sorted when the
            // `preserve_order` feature is off, and another crate may enable it.
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort_unstable();
            out.push('{');
            for (i, key) in keys.into_iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_string(key, out);
                out.push(':');
                if let Some(v) = map.get(key) {
                    write_canonical(v, out);
                }
            }
            out.push('}');
        }
    }
}

fn write_string(s: &str, out: &mut String) {
    // `Value::String` display performs correct JSON escaping.
    let _ = write!(out, "{}", Value::String(s.to_owned()));
}

fn normalize_text(s: &str) -> String {
    let trimmed = s.trim();
    if trimmed.contains('\r') {
        trimmed.replace("\r\n", "\n")
    } else {
        trimmed.to_owned()
    }
}

/// Pick the tiktoken encoding for a model name.
fn encoder_for(model: &str) -> &'static CoreBPE {
    let m = model.rsplit('/').next().unwrap_or(model);
    let legacy = (m.starts_with("gpt-4") && !m.starts_with("gpt-4o") && !m.starts_with("gpt-4.1"))
        || m.starts_with("gpt-3.5")
        || m.starts_with("text-embedding-ada");
    if legacy {
        tiktoken_rs::cl100k_base_singleton()
    } else {
        tiktoken_rs::o200k_base_singleton()
    }
}

/// Count tokens in a piece of text for the given model.
pub fn count_text_tokens(text: &str, model: &str) -> usize {
    if text.is_empty() {
        return 0;
    }
    let raw = encoder_for(model).encode_ordinary(text).len();
    scale_for_model(raw, model)
}

/// Estimate prompt tokens for a JSON request payload by counting every prompt
/// text field plus a small per-message overhead.
pub fn estimate_payload_tokens(payload: &str, model: &str) -> Result<usize, NormalizeError> {
    let value: Value = serde_json::from_str(payload)?;
    let bpe = encoder_for(model);
    let mut total = 0usize;
    walk(&value, None, bpe, &mut total);
    Ok(scale_for_model(total, model))
}

fn walk(value: &Value, key: Option<&str>, bpe: &CoreBPE, total: &mut usize) {
    match value {
        Value::String(s) => {
            if key.is_some_and(|k| TEXT_KEYS.contains(&k)) {
                *total += bpe.encode_ordinary(s).len();
            }
        }
        Value::Array(items) => {
            let is_messages = key == Some("messages");
            for item in items {
                if is_messages {
                    *total += TOKENS_PER_MESSAGE;
                }
                // Array elements inherit the parent key, so
                // `"content": ["a", "b"]` counts both strings.
                walk(item, key, bpe, total);
            }
        }
        Value::Object(map) => {
            for (k, v) in map {
                walk(v, Some(k.as_str()), bpe, total);
            }
        }
        Value::Null | Value::Bool(_) | Value::Number(_) => {}
    }
}

/// Count the tokens in `text` using the encoding for `model`.
#[pyfunction]
#[pyo3(signature = (text, model = "gpt-4o"))]
pub fn count_tokens(py: Python<'_>, text: &str, model: &str) -> usize {
    py.detach(|| count_text_tokens(text, model))
}

/// Estimate the prompt tokens of a JSON-encoded request payload.
#[pyfunction]
#[pyo3(signature = (payload_json, model = "gpt-4o"))]
pub fn estimate_tokens(py: Python<'_>, payload_json: &str, model: &str) -> PyResult<usize> {
    Ok(py.detach(|| estimate_payload_tokens(payload_json, model))?)
}

/// Canonicalise a JSON-encoded request payload (sorted keys, trimmed strings).
#[pyfunction]
pub fn normalize(py: Python<'_>, payload_json: &str) -> PyResult<String> {
    Ok(py.detach(|| normalize_json(payload_json))?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn key_order_does_not_matter() {
        let a = normalize_json(r#"{"b":1,"a":{"y":2,"x":3}}"#).unwrap();
        let b = normalize_json(r#"{"a":{"x":3,"y":2},"b":1}"#).unwrap();
        assert_eq!(a, b);
        assert_eq!(a, r#"{"a":{"x":3,"y":2},"b":1}"#);
    }

    #[test]
    fn strings_are_trimmed_and_crlf_normalised() {
        let a = normalize_json(r#"{"content":"  hello\r\nworld \n"}"#).unwrap();
        assert_eq!(a, r#"{"content":"hello\nworld"}"#);
    }

    #[test]
    fn internal_whitespace_is_preserved() {
        let a = normalize_json(r#"{"content":"def f():\n    return 1"}"#).unwrap();
        let b = normalize_json(r#"{"content":"def f():\n return 1"}"#).unwrap();
        assert_ne!(a, b);
    }

    #[test]
    fn escaping_round_trips() {
        let a = normalize_json(r#"{"content":"quote \" and \\ and é"}"#).unwrap();
        let v: Value = serde_json::from_str(&a).unwrap();
        assert_eq!(v["content"], "quote \" and \\ and é");
    }

    #[test]
    fn invalid_json_is_an_error() {
        assert!(normalize_json("{nope").is_err());
    }

    #[test]
    fn counts_tokens() {
        assert_eq!(count_text_tokens("", "gpt-4o"), 0);
        let n = count_text_tokens("Hello, world!", "gpt-4o");
        assert!((2..=6).contains(&n), "unexpected count {n}");
        let legacy = count_text_tokens("Hello, world!", "gpt-4");
        assert!((2..=6).contains(&legacy));
    }

    #[test]
    fn new_anthropic_tokenizer_scales_up_by_30pct() {
        let text = "The quick brown fox jumps over the lazy dog, repeatedly and often.";
        // Old tokenizer models share the o200k_base baseline (multiplier 1.0).
        let base = count_text_tokens(text, "claude-sonnet-4-6");
        let scaled = count_text_tokens(text, "claude-sonnet-5");
        assert!(scaled > base, "expected scale-up, got {scaled} vs {base}");
        assert_eq!(scaled, (base as f64 * 1.30).round() as usize);
    }

    #[test]
    fn openai_and_pre_47_claude_are_unscaled() {
        let text = "hello world, this is a plain tokenizer counting sanity check";
        let gpt = count_text_tokens(text, "gpt-4o");
        // gpt-4o, Sonnet 4.6 and Opus 4.6 all use the o200k_base baseline here.
        assert_eq!(count_text_tokens(text, "claude-sonnet-4-6"), gpt);
        assert_eq!(count_text_tokens(text, "claude-opus-4-6"), gpt);
        assert_eq!(count_text_tokens(text, "claude-opus-4-5"), gpt);
    }

    #[test]
    fn multiplier_resolves_dated_and_provider_spellings() {
        assert_eq!(token_multiplier("claude-opus-5-5-20260101"), 1.30);
        assert_eq!(token_multiplier("anthropic.claude-opus-5-5"), 1.30);
        assert_eq!(token_multiplier("us.anthropic.claude-sonnet-5"), 1.30);
        assert_eq!(token_multiplier("claude-fable-5-1"), 1.30);
        assert_eq!(token_multiplier("claude-mythos-5-1"), 1.30);
        // Explicitly not affected.
        assert_eq!(token_multiplier("claude-opus-4-6"), 1.0);
        assert_eq!(token_multiplier("claude-sonnet-4-6"), 1.0);
        assert_eq!(token_multiplier("claude-haiku-5-5"), 1.0);
        assert_eq!(token_multiplier("gpt-4o"), 1.0);
    }

    #[test]
    fn payload_estimate_scales_for_new_tokenizer() {
        let payload = r#"{"model":"claude-opus-5-5","messages":[
            {"role":"user","content":"Summarise the attached report in three bullet points."}
        ]}"#;
        let old = estimate_payload_tokens(payload, "claude-opus-4-6").unwrap();
        let new = estimate_payload_tokens(payload, "claude-opus-5-5").unwrap();
        assert_eq!(new, (old as f64 * 1.30).round() as usize);
    }

    #[test]
    fn estimates_message_payloads() {
        let payload = r#"{"model":"gpt-4o","messages":[
            {"role":"system","content":"You are terse."},
            {"role":"user","content":[{"type":"text","text":"Say hi"}]}
        ]}"#;
        let n = estimate_payload_tokens(payload, "gpt-4o").unwrap();
        // Two messages of overhead plus a handful of text tokens; the model
        // name and roles must not be counted.
        assert!((2 * TOKENS_PER_MESSAGE + 3..=2 * TOKENS_PER_MESSAGE + 12).contains(&n));
    }
}
