//! Native core of parry. Exposed to Python as `parry._parry_native`.
//!
//! Every type here is thread-safe and holds no Python objects, so the Python
//! layer can share one instance across threads and asyncio tasks.

use pyo3::prelude::*;

mod cache;
mod cost;
mod rate_limit;
mod retry;
mod tokenize;

#[pymodule]
fn _parry_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<cache::PyCache>()?;
    m.add_class::<rate_limit::PyRateLimit>()?;
    m.add_class::<cost::PyCost>()?;
    m.add_class::<retry::PyRetry>()?;
    m.add_function(wrap_pyfunction!(tokenize::count_tokens, m)?)?;
    m.add_function(wrap_pyfunction!(tokenize::estimate_tokens, m)?)?;
    m.add_function(wrap_pyfunction!(tokenize::normalize, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("DEFAULT_MAX_ENTRIES", cache::DEFAULT_MAX_ENTRIES)?;
    m.add("DEFAULT_TTL_SECONDS", cache::DEFAULT_TTL_SECONDS)?;
    Ok(())
}
