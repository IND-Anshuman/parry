# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
uses [Semantic Versioning](https://semver.org/).

## [Unreleased]

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
