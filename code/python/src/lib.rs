use numpy::{IntoPyArray, PyArray1};
use pyo3::prelude::*;
use simulation_project::models::MultiExponentialHawkes;
use simulation_project::simulation::simulate as rs_simulate;

#[pyclass(name = "MultiExponentialHawkes")]
#[derive(Clone)]
pub struct PyMultiExponentialHawkes {
    pub inner: MultiExponentialHawkes,
}

#[pymethods]
impl PyMultiExponentialHawkes {
    #[new]
    fn new(mu: f64, alpha: Vec<f64>, beta: Vec<f64>) -> Self {
        Self { inner: MultiExponentialHawkes::new(mu, alpha, beta) }
    }

    #[staticmethod]
    fn with_stationary_state(mu: f64, alpha: Vec<f64>, beta: Vec<f64>) -> Self {
        let base = MultiExponentialHawkes::new(mu, alpha.clone(), beta.clone());
        let inner = MultiExponentialHawkes::new_with_state(
            base.stationary_state(), mu, alpha, beta,
        );
        Self { inner }
    }

    fn stationary_state<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.stationary_state().into_pyarray_bound(py)
    }

    fn m(&self) -> usize {
        self.inner.m()
    }
}

/// Simulate the Hawkes process up to `t_max`. Returns the event times as
/// a 1-D numpy array of f64 (dim info is dropped — Hawkes is single-dim).
#[pyfunction]
fn simulate_hawkes<'py>(
    py: Python<'py>,
    hawkes: &PyMultiExponentialHawkes,
    t_max: f64,
    seed: Option<u64>,
) -> Bound<'py, PyArray1<f64>> {
    let result = rs_simulate(&hawkes.inner, t_max, seed);
    let times: Vec<f64> = result.events.iter().map(|e| e.time).collect();
    times.into_pyarray_bound(py)
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", "0.1.0")?;
    m.add_class::<PyMultiExponentialHawkes>()?;
    m.add_function(wrap_pyfunction!(simulate_hawkes, m)?)?;
    Ok(())
}
