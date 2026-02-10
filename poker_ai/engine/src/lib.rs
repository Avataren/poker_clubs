pub mod actions;
pub mod card;
pub mod game_state;
pub mod hand_eval;
pub mod hand_eval_features;
pub mod pot;
pub mod python_bindings;

use pyo3::prelude::*;

/// Python module definition.
#[pymodule]
fn engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<python_bindings::PokerEnv>()?;
    m.add_class::<python_bindings::BatchPokerEnv>()?;
    Ok(())
}
