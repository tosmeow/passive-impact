use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use simulation_project::models::{
    MultiExponentialHawkes,
    AffineQueueProcess, AffineBidAskQueueProcess, BidAskAffineParams, AffineIntensityParams,
    MarkovianProcess, MultivariateSimulationResult,
};
use simulation_project::simulation::simulate as rs_simulate;
use simulation_project::simulation::simulate_with_externals as rs_simulate_with_externals;
use simulation_project::simulation::ConditionalSimulationContext;
use simulation_project::simulation_helpers::{
    hawkes_to_market_orders as rs_hawkes_to_market_orders,
    merge_events as rs_merge_events,
    create_meta_orders as rs_create_meta_orders,
    events_to_dim as rs_events_to_dim,
    extract_events_by_dim as rs_extract_events_by_dim,
    sample_queue_at_times as rs_sample_queue_at_times,
};
use simulation_project::conditional_impact::{TailImpact, AggressiveImpactPath};

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

/// Opaque wrapper for a MultivariateSimulationResult (a list of
/// (time, dim) event tuples). Pass between functions; convert to
/// numpy via .times() / .dims() when needed.
#[pyclass(name = "SimulationResult")]
#[derive(Clone)]
pub struct PySimulationResult {
    pub inner: MultivariateSimulationResult,
}

#[pymethods]
impl PySimulationResult {
    fn times<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        let v: Vec<f64> = self.inner.events.iter().map(|e| e.time).collect();
        v.into_pyarray_bound(py)
    }
    fn dims<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<usize>> {
        let v: Vec<usize> = self.inner.events.iter().map(|e| e.dim).collect();
        v.into_pyarray_bound(py)
    }
    fn __len__(&self) -> usize {
        self.inner.events.len()
    }
}

/// Opaque wrapper for a MarkovianProcess (the queue process produced
/// by AffineQueueProcess::new_queue). Treated as black-box.
#[pyclass(name = "QueueProcess")]
pub struct PyQueueProcess {
    pub inner: MarkovianProcess,
}

#[pymethods]
impl PyQueueProcess {
    fn dim(&self) -> usize {
        use simulation_project::models::MultivariateMarkovianIntensity;
        self.inner.dim()
    }
}

#[pyclass(name = "AffineQueueProcess")]
pub struct PyAffineQueueProcess;

#[pymethods]
impl PyAffineQueueProcess {
    /// Decoupled single-queue process (state = [q]). Market orders dim=2 must
    /// be supplied as externals.
    #[staticmethod]
    fn new_queue(q0: f64, a_l: f64, b_l: f64, a_c: f64, b_c: f64) -> PyQueueProcess {
        PyQueueProcess { inner: AffineQueueProcess::new_queue(q0, a_l, b_l, a_c, b_c) }
    }

    /// Static helper: c_lambda = b_c - b_l.
    #[staticmethod]
    fn c_lambda(b_l: f64, b_c: f64) -> f64 {
        AffineQueueProcess::c_lambda(b_l, b_c)
    }
}

#[pyclass(name = "AffineBidAskQueueProcess")]
pub struct PyAffineBidAskQueueProcess;

#[pymethods]
impl PyAffineBidAskQueueProcess {
    /// Decoupled double-queue process (state = [q_a, q_b]).
    /// Limit/cancel intensities are affine in (q_a, q_b) per side.
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    fn new_queue(
        q0_a: f64, q0_b: f64,
        // ask side: λ^L_a = a + b_aa*q_a + b_ab*q_b, similarly for cancel
        l_a_const: f64, l_a_self: f64, l_a_cross: f64,
        c_a_const: f64, c_a_self: f64, c_a_cross: f64,
        // bid side
        l_b_const: f64, l_b_self: f64, l_b_cross: f64,
        c_b_const: f64, c_b_self: f64, c_b_cross: f64,
    ) -> PyQueueProcess {
        let params = BidAskAffineParams {
            lambda_l_a: AffineIntensityParams::new(l_a_const, l_a_self, l_a_cross),
            lambda_c_a: AffineIntensityParams::new(c_a_const, c_a_self, c_a_cross),
            lambda_l_b: AffineIntensityParams::new(l_b_const, l_b_self, l_b_cross),
            lambda_c_b: AffineIntensityParams::new(c_b_const, c_b_self, c_b_cross),
        };
        PyQueueProcess { inner: AffineBidAskQueueProcess::new_queue(q0_a, q0_b, params) }
    }
}

/// Simulate process with external events (e.g. market orders driven by Hawkes).
#[pyfunction]
fn simulate_with_externals(
    process: &PyQueueProcess,
    t_max: f64,
    externals: &PySimulationResult,
    seed: Option<u64>,
) -> PySimulationResult {
    PySimulationResult {
        inner: rs_simulate_with_externals(&process.inner, t_max, &externals.inner, seed),
    }
}

/// Simulate Hawkes and return as a SimulationResult marked dim=2 (market orders).
#[pyfunction]
fn simulate_hawkes_as_market_orders(
    hawkes: &PyMultiExponentialHawkes,
    t_max: f64,
    seed: Option<u64>,
) -> PySimulationResult {
    let result = rs_simulate(&hawkes.inner, t_max, seed);
    PySimulationResult { inner: rs_hawkes_to_market_orders(&result) }
}

#[pyfunction]
fn merge_events(a: &PySimulationResult, b: &PySimulationResult) -> PySimulationResult {
    PySimulationResult { inner: rs_merge_events(&a.inner, &b.inner) }
}

/// Build an evenly-spaced metaorder block of n orders from t_start to t_end.
#[pyfunction]
fn create_meta_orders(n: u32, t_start: f64, t_end: f64) -> PySimulationResult {
    PySimulationResult { inner: rs_create_meta_orders(n, t_start, t_end) }
}

/// Build a metaorder from an explicit list of times; tagged at target_dim.
#[pyfunction]
fn create_meta_orders_from_times(
    times: PyReadonlyArray1<f64>,
    target_dim: usize,
    total_dims: usize,
) -> PySimulationResult {
    use simulation_project::models::{MultivariateEvent};
    let mut result = MultivariateSimulationResult::new(total_dims);
    for &t in times.as_slice().unwrap() {
        result.push(MultivariateEvent { time: t, dim: target_dim });
    }
    PySimulationResult { inner: result }
}

#[pyfunction]
fn events_to_dim(
    events: &PySimulationResult,
    target_dim: usize,
    total_dims: usize,
) -> PySimulationResult {
    PySimulationResult { inner: rs_events_to_dim(&events.inner, target_dim, total_dims) }
}

/// Returns a list of f64 numpy arrays per dim.
#[pyfunction]
fn extract_events_by_dim<'py>(
    py: Python<'py>,
    result: &PySimulationResult,
    total_dims: usize,
    exclude_dim: Option<usize>,
) -> Vec<Bound<'py, PyArray1<f64>>> {
    let by_dim = rs_extract_events_by_dim(&result.inner, total_dims, exclude_dim);
    by_dim.into_iter().map(|v| v.into_pyarray_bound(py)).collect()
}

#[pyfunction]
fn sample_queue_at_times<'py>(
    py: Python<'py>,
    queue_path_events: &PySimulationResult,
    initial_q: u32,
    times: PyReadonlyArray1<f64>,
) -> Bound<'py, PyArray1<u32>> {
    let q_path = AffineQueueProcess::result_to_queue_path(&queue_path_events.inner, initial_q);
    let samples = rs_sample_queue_at_times(&q_path, times.as_slice().unwrap());
    samples.into_pyarray_bound(py)
}

/// Conditional simulation context. Owns its inputs (conditioning events + externals + process)
/// and rebuilds the borrowed Rust context on each call.
#[pyclass(name = "ConditionalSimulationContext")]
pub struct PyConditionalSimulationContext {
    process: Py<PyQueueProcess>,
    cond_events_by_dim: Vec<Vec<f64>>,
    cond_externals: Option<MultivariateSimulationResult>,
    new_externals: Option<MultivariateSimulationResult>,
    t_max: f64,
}

#[pymethods]
impl PyConditionalSimulationContext {
    #[new]
    #[pyo3(signature = (process, cond_events_by_dim, t_max, *, cond_externals=None, new_externals=None))]
    fn new(
        process: Py<PyQueueProcess>,
        cond_events_by_dim: Vec<Vec<f64>>,
        t_max: f64,
        cond_externals: Option<&PySimulationResult>,
        new_externals: Option<&PySimulationResult>,
    ) -> Self {
        Self {
            process,
            cond_events_by_dim,
            cond_externals: cond_externals.map(|r| r.inner.clone()),
            new_externals: new_externals.map(|r| r.inner.clone()),
            t_max,
        }
    }

    /// Memory-efficient queue sampling at specified times. Returns a 1-D numpy
    /// array of u32 queue values aligned with `times`.
    fn simulate_queue_at_times<'py>(
        &self,
        py: Python<'py>,
        times: PyReadonlyArray1<f64>,
        initial_queue_size: u32,
        seed: Option<u64>,
    ) -> Bound<'py, PyArray1<u32>> {
        let process_borrow = self.process.borrow(py);
        let ctx = ConditionalSimulationContext::new(
            &process_borrow.inner,
            &self.cond_events_by_dim,
            self.cond_externals.as_ref(),
            self.new_externals.as_ref(),
            self.t_max,
        );
        let samples = ctx.simulate_queue_at_times(
            times.as_slice().unwrap(),
            initial_queue_size,
            None,
            seed,
        );
        samples.into_pyarray_bound(py)
    }

    /// Single-shot conditional simulate; returns the resulting event stream.
    fn simulate(
        &self,
        py: Python,
        seed: Option<u64>,
    ) -> PySimulationResult {
        let process_borrow = self.process.borrow(py);
        let ctx = ConditionalSimulationContext::new(
            &process_borrow.inner,
            &self.cond_events_by_dim,
            self.cond_externals.as_ref(),
            self.new_externals.as_ref(),
            self.t_max,
        );
        PySimulationResult { inner: ctx.simulate(None, seed) }
    }
}

#[pyclass(name = "TailImpact")]
pub struct PyTailImpact {
    pub inner: TailImpact,
}

#[pymethods]
impl PyTailImpact {
    /// Build TailImpact from affine-queue parameters.
    #[staticmethod]
    fn from_affine_queue(
        mu: f64,
        alpha: Vec<f64>,
        beta: Vec<f64>,
        b_l: f64,
        b_c: f64,
        events: Vec<f64>,
    ) -> Self {
        Self { inner: TailImpact::from_affine_queue(mu, alpha, beta, b_l, b_c, events) }
    }
}

#[pyclass(name = "AggressiveImpactPath")]
pub struct PyAggressiveImpactPath {
    pub impact_path: Vec<f64>,
}

#[pymethods]
impl PyAggressiveImpactPath {
    fn impact<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.impact_path.clone().into_pyarray_bound(py)
    }
}

/// Compute aggressive impact path from pre-sampled queues.
/// `kappa` is a Python callable f64 -> f64 invoked at each evaluation time.
#[pyfunction]
fn aggressive_impact_from_queue_samples(
    py: Python,
    q_samples: PyReadonlyArray1<u32>,
    q_bar_samples: PyReadonlyArray1<u32>,
    eval_times: PyReadonlyArray1<f64>,
    is_market_order: Vec<bool>,
    hawkes: &PyMultiExponentialHawkes,
    kappa: PyObject,
) -> PyResult<PyAggressiveImpactPath> {
    let kappa_clone = kappa.clone_ref(py);
    let path = AggressiveImpactPath::from_queue_samples(
        q_samples.as_slice().unwrap(),
        q_bar_samples.as_slice().unwrap(),
        eval_times.as_slice().unwrap(),
        &is_market_order,
        &hawkes.inner,
        move |q: f64| -> f64 {
            Python::with_gil(|py| {
                let res = kappa_clone.call1(py, (q,)).unwrap();
                res.extract::<f64>(py).unwrap()
            })
        },
    );
    Ok(PyAggressiveImpactPath { impact_path: path.impact_path })
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", "0.1.0")?;
    m.add_class::<PyMultiExponentialHawkes>()?;
    m.add_function(wrap_pyfunction!(simulate_hawkes, m)?)?;
    m.add_class::<PySimulationResult>()?;
    m.add_class::<PyQueueProcess>()?;
    m.add_class::<PyAffineQueueProcess>()?;
    m.add_class::<PyAffineBidAskQueueProcess>()?;
    m.add_function(wrap_pyfunction!(simulate_with_externals, m)?)?;
    m.add_function(wrap_pyfunction!(simulate_hawkes_as_market_orders, m)?)?;
    m.add_function(wrap_pyfunction!(merge_events, m)?)?;
    m.add_function(wrap_pyfunction!(create_meta_orders, m)?)?;
    m.add_function(wrap_pyfunction!(create_meta_orders_from_times, m)?)?;
    m.add_function(wrap_pyfunction!(events_to_dim, m)?)?;
    m.add_function(wrap_pyfunction!(extract_events_by_dim, m)?)?;
    m.add_function(wrap_pyfunction!(sample_queue_at_times, m)?)?;
    m.add_class::<PyConditionalSimulationContext>()?;
    m.add_class::<PyTailImpact>()?;
    m.add_class::<PyAggressiveImpactPath>()?;
    m.add_function(wrap_pyfunction!(aggressive_impact_from_queue_samples, m)?)?;
    Ok(())
}
