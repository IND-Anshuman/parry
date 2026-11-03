# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
uses [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.2.0] - 2026-11-03

This release is about **cost-model correctness** and **provider reach**: parry's
numbers now line up with the provider bill across cache, batch and fast tiers,
it refuses to silently undercharge an unpriced model, and `install()` guards six
more SDKs.

### Added

- **Pricing tiers.** Cost now models cached input, 5-minute and 1-hour cache
  writes, batch and fast/priority tiers, not just the standard tier. New
  `Cost.compute_tiered()`, `Cost.record_tiered()` and `Cost.tier_rates()`, and a
  `parry pricing --tiers` view. OpenAI cached input bills at 10%, batch at 50%,
  priority at 1.75×; Anthropic cache reads at 10%, 5m writes at 1.25×, 1h writes
  at 2×, batch at 50%, fast at 2×.
- **Cache-aware usage.** Adapters split cache-read and cache-write tokens out of
  the reported usage (`UsageDetail`), so a cached Anthropic or OpenAI call is
  billed at the cache rate instead of the full input rate.
- **Fail-closed unpriced models.** A new `on_unpriced` option (default
  `"raise"`) makes a call to a model with no known price raise `UnpricedModel`
  rather than slip past `max_cost_usd` at $0. Pass `on_unpriced="ignore"` for the
  old behaviour.
- **Anthropic 4.7+ tokenizer multiplier.** The newer Anthropic tokenizer
  produces ~30% more tokens; nine affected models now apply a 1.30× multiplier in
  pre-flight token and budget estimates.
- **More providers, same `install()`.** Groq and Together (OpenAI-shaped,
  covered by one generic duck-typed patcher), Google `google-genai`
  (`generate_content` + streaming, with cached-content and thinking tokens),
  Ollama, Mistral and Cohere. Each has an `install()` flag and no-ops when its
  SDK is absent.
- **AWS Bedrock Converse** via `parry.integrations.bedrock.guard(client)`
  (per-client, because boto3 builds methods dynamically).
- **Pricing sync from LiteLLM.** `scripts/sync_pricing.py` maps LiteLLM's live
  price database into parry's table (fail-closed: a model without a finite input
  *and* output price is skipped), with a `--check` mode for CI.
- Guides for the OpenAI Agents SDK and Pydantic AI (both covered by the existing
  SDK hooks; documentation only).

### Changed

- `parry.install()` gained per-provider flags: `openai`, `anthropic`, `groq`,
  `together`, `google`, `ollama`, `mistral`, `cohere` (all default on).
- The patcher is now method- and stream-aware, so it can hook surfaces beyond
  `create` (e.g. Google's `generate_content_stream`, Mistral's `complete`).
- The docs site reads its version from `pyproject.toml` instead of hardcoding it.

### Fixed

- Release workflow: the package version is bumped to 0.2.0 (publishing 0.1.0
  again collided with the existing PyPI release), and PyPI publish now uses
  `skip-existing` so a re-run or a race between wheel jobs can't fail the release.

## [0.1.0] - 2026-10-25

### Added

- Published on PyPI as `llm-parry`; imported as `parry`.
- `parry.install()` hooks OpenAI chat completions, the OpenAI Responses API,
  Anthropic messages and Anthropic beta messages, on sync and async clients,
  including `.with_raw_response` and `stream=True`.
- `Parry` guard: response cache (blake3 key, bounded LRU with TTL), per-model
  token-bucket rate limiting with tier-1 defaults, cost tracking with a dated
  pricing table, a `max_cost_usd` budget, and retry with exponential backoff,
  jitter and `Retry-After`.
- Scoping: `with Parry():`, `parry.wrap(client)` and `parry.bypass()`.
- Optional SQLite persistence for the cache and the spend ledger, plus the
  `parry stats` / `parry pricing` CLI.
- LangChain `ParryCallback` and LlamaIndex `ParryHandler`.
- Standalone Rust primitives: `Cache`, `RateLimit`, `Cost`, `Retry`,
  `count_tokens` and `estimate_tokens`.
