# Contributing to parry

Thanks for helping! Bug reports, pricing updates and small, focused PRs are
all welcome. The repository is
[github.com/IND-Anshuman/parry](https://github.com/IND-Anshuman/parry), and
issues go to [its issue tracker](https://github.com/IND-Anshuman/parry/issues).

## Development setup

You need Rust (the version is pinned in `rust-toolchain.toml`, and rustup
picks it up automatically) and Python 3.10+.

```bash
git clone https://github.com/IND-Anshuman/parry.git
cd parry
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install maturin
pip install -e ".[dev,integrations]"   # builds the Rust extension in debug mode
```

After changing Rust code, rebuild with `maturin develop` (add `--release`
before benchmarking).

## Checks

```bash
cargo fmt --check && cargo clippy --all-targets -- -D warnings && cargo test
ruff check python tests && ruff format --check python tests
pytest
python benches/overhead_bench.py
```

The tests use an in-process mock server that speaks the OpenAI and Anthropic
wire formats (`tests/conftest.py`). **No test may call a real provider.** If
you need a new response shape, teach the mock about it.

## Guidelines

- Keep `unwrap()` / `expect()` out of non-test Rust code.
- Never log request or response bodies. Log metadata only: model, tokens,
  cost, latency.
- New SDK hook targets go in `TARGETS` in `python/parry/_patch.py`, with an
  adapter in `_adapters.py` and tests in `tests/test_patch.py`.
- For pricing changes, update `src/cost.rs`, bump the "last verified" date,
  and link the provider's pricing page in the PR description.
- Add a line to `CHANGELOG.md` under *Unreleased*.

## Releasing

1. Bump the version in `Cargo.toml`, `pyproject.toml` and
   `python/parry/__init__.py`, and move the *Unreleased* changelog entries
   under the new version.
2. Tag and publish a GitHub release. `.github/workflows/release.yml` builds
   the wheels and publishes them to PyPI through trusted publishing.
