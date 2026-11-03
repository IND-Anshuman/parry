//! Pricing rows synced from LiteLLM's `model_prices_and_context_window.json`.
//!
//! GENERATED FILE — do not edit by hand. Regenerate with:
//!
//!     python scripts/sync_pricing.py            # fetch the live LiteLLM table
//!     python scripts/sync_pricing.py --file x.json
//!
//! The sync maps LiteLLM's per-token rates (input/output, cache read/creation,
//! batch) into parry's per-million schema and skips any model that lacks a
//! finite input or output price (fail closed — never ship a half-priced model).
//! Hand-curated rows in `src/cost.rs` win on an equal-length prefix, so the
//! models parry prices most accurately are never overwritten by a sync.
//!
//! It ships empty until the sync is pointed at a source.

use crate::cost::ModelPricing;

pub static GENERATED: &[ModelPricing] = &[
    // SYNC:START
    // SYNC:END
];
