use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use simulation_project::conditional_impact::{
    AggressiveImpactPath, BidAskImpactPath, BidAskTailImpact, ImpactPath, SymmetricCMatrix,
    TailImpact,
};
use simulation_project::experiments::impact_cost::{
    build_execution_latency_grid as rs_build_execution_latency_grid,
    select_first_limit_every as rs_select_first_limit_every,
    select_limit_indices as rs_select_limit_indices,
    select_random_limit_fraction as rs_select_random_limit_fraction,
    simulate_anchored_affine_queue as rs_simulate_anchored_affine_queue,
    track_passive_fills as rs_track_passive_fills, AffineQueueIntensity, AnchoredQueueInput,
    CancellationPolicy, ExecutionLatencyGridInput, PassiveFillTrackerInput,
};
use simulation_project::models::{
    AffineBidAskQueueProcess, AffineIntensityParams, AffineQueueProcess, BidAskAffineParams,
    MarkovianProcess, MultiExponentialHawkes, MultivariateSimulationResult,
};
use simulation_project::simulation::simulate as rs_simulate;
use simulation_project::simulation::simulate_with_externals as rs_simulate_with_externals;
use simulation_project::simulation::{ConditionalSimulationContext, SimulationConfig};
use simulation_project::simulation_helpers::{
    create_meta_orders as rs_create_meta_orders, events_to_dim as rs_events_to_dim,
    extract_events_by_dim as rs_extract_events_by_dim,
    hawkes_to_market_orders as rs_hawkes_to_market_orders, merge_events as rs_merge_events,
    sample_queue_at_times as rs_sample_queue_at_times,
};

#[pyclass(name = "MultiExponentialHawkes")]
#[derive(Clone)]
pub struct PyMultiExponentialHawkes {
    pub inner: MultiExponentialHawkes,
}

#[pymethods]
impl PyMultiExponentialHawkes {
    #[new]
    fn new(mu: f64, alpha: Vec<f64>, beta: Vec<f64>) -> Self {
        Self {
            inner: MultiExponentialHawkes::new(mu, alpha, beta),
        }
    }

    #[staticmethod]
    fn with_stationary_state(mu: f64, alpha: Vec<f64>, beta: Vec<f64>) -> Self {
        let base = MultiExponentialHawkes::new(mu, alpha.clone(), beta.clone());
        let inner =
            MultiExponentialHawkes::new_with_state(base.stationary_state(), mu, alpha, beta);
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

/// Simulate the Hawkes process up to `t_max`, preserving the full
/// SimulationResult wrapper rather than dropping dimension metadata.
#[pyfunction]
fn simulate_hawkes_result(
    hawkes: &PyMultiExponentialHawkes,
    t_max: f64,
    seed: Option<u64>,
) -> PySimulationResult {
    PySimulationResult {
        inner: rs_simulate(&hawkes.inner, t_max, seed),
    }
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

#[pyclass(name = "AffineCountingProcess")]
pub struct PyAffineCountingProcess {
    pub inner: MarkovianProcess,
    a: f64,
    b: f64,
}

fn affine_counting_process(a: f64, b: f64) -> MarkovianProcess {
    MarkovianProcess::new(
        1,
        vec![0.0],
        move |state: &[f64], _t: f64, _t_last: f64| vec![b + a * state[0]],
        move |state: &[f64], event, _t: f64, _t_prev: f64| {
            let mut next = state.to_vec();
            if event.dim == 0 {
                next[0] += 1.0;
            }
            next
        },
    )
}

#[pymethods]
impl PyAffineCountingProcess {
    #[new]
    #[pyo3(signature = (a, b))]
    fn new(a: f64, b: f64) -> PyResult<Self> {
        if !a.is_finite() || !b.is_finite() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "a and b must be finite",
            ));
        }
        if a < 0.0 || b < 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "a and b must be non-negative",
            ));
        }
        Ok(Self {
            inner: affine_counting_process(a, b),
            a,
            b,
        })
    }

    fn a(&self) -> f64 {
        self.a
    }

    fn b(&self) -> f64 {
        self.b
    }

    fn dim(&self) -> usize {
        1
    }
}

#[pyfunction]
#[pyo3(signature = (process, t_max, seed=None))]
fn simulate_affine_counting_process(
    process: &PyAffineCountingProcess,
    t_max: f64,
    seed: Option<u64>,
) -> PySimulationResult {
    PySimulationResult {
        inner: rs_simulate(&process.inner, t_max, seed),
    }
}

#[pyfunction]
#[pyo3(signature = (process, t_max, externals, seed=None))]
fn simulate_affine_counting_process_with_externals(
    process: &PyAffineCountingProcess,
    t_max: f64,
    externals: &PySimulationResult,
    seed: Option<u64>,
) -> PySimulationResult {
    PySimulationResult {
        inner: rs_simulate_with_externals(&process.inner, t_max, &externals.inner, seed),
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
        PyQueueProcess {
            inner: AffineQueueProcess::new_queue(q0, a_l, b_l, a_c, b_c),
        }
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
        q0_a: f64,
        q0_b: f64,
        // ask side: λ^L_a = a + b_aa*q_a + b_ab*q_b, similarly for cancel
        l_a_const: f64,
        l_a_self: f64,
        l_a_cross: f64,
        c_a_const: f64,
        c_a_self: f64,
        c_a_cross: f64,
        // bid side
        l_b_const: f64,
        l_b_self: f64,
        l_b_cross: f64,
        c_b_const: f64,
        c_b_self: f64,
        c_b_cross: f64,
    ) -> PyQueueProcess {
        let params = BidAskAffineParams {
            lambda_l_a: AffineIntensityParams::new(l_a_const, l_a_self, l_a_cross),
            lambda_c_a: AffineIntensityParams::new(c_a_const, c_a_self, c_a_cross),
            lambda_l_b: AffineIntensityParams::new(l_b_const, l_b_self, l_b_cross),
            lambda_c_b: AffineIntensityParams::new(c_b_const, c_b_self, c_b_cross),
        };
        PyQueueProcess {
            inner: AffineBidAskQueueProcess::new_queue(q0_a, q0_b, params),
        }
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
    PySimulationResult {
        inner: rs_hawkes_to_market_orders(&result),
    }
}

/// Simulate Hawkes while forcing an external event stream into its state.
#[pyfunction]
fn simulate_hawkes_with_externals(
    hawkes: &PyMultiExponentialHawkes,
    t_max: f64,
    externals: &PySimulationResult,
    seed: Option<u64>,
) -> PySimulationResult {
    PySimulationResult {
        inner: rs_simulate_with_externals(&hawkes.inner, t_max, &externals.inner, seed),
    }
}

#[pyfunction]
fn merge_events(a: &PySimulationResult, b: &PySimulationResult) -> PySimulationResult {
    PySimulationResult {
        inner: rs_merge_events(&a.inner, &b.inner),
    }
}

/// Build an evenly-spaced metaorder block of n orders from t_start to t_end.
#[pyfunction]
fn create_meta_orders(n: u32, t_start: f64, t_end: f64) -> PySimulationResult {
    PySimulationResult {
        inner: rs_create_meta_orders(n, t_start, t_end),
    }
}

/// Build a metaorder from an explicit list of times; tagged at target_dim.
#[pyfunction]
fn create_meta_orders_from_times(
    times: PyReadonlyArray1<f64>,
    target_dim: usize,
    total_dims: usize,
) -> PySimulationResult {
    use simulation_project::models::MultivariateEvent;
    let mut result = MultivariateSimulationResult::new(total_dims);
    for &t in times.as_slice().unwrap() {
        result.push(MultivariateEvent {
            time: t,
            dim: target_dim,
        });
    }
    PySimulationResult { inner: result }
}

#[pyfunction]
fn events_to_dim(
    events: &PySimulationResult,
    target_dim: usize,
    total_dims: usize,
) -> PySimulationResult {
    PySimulationResult {
        inner: rs_events_to_dim(&events.inner, target_dim, total_dims),
    }
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
    by_dim
        .into_iter()
        .map(|v| v.into_pyarray_bound(py))
        .collect()
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

#[pyfunction]
fn sample_bidask_queue_at_times<'py>(
    py: Python<'py>,
    queue_path_events: &PySimulationResult,
    initial_q_a: u32,
    initial_q_b: u32,
    times: PyReadonlyArray1<f64>,
) -> PyResult<Bound<'py, PyDict>> {
    let paths = AffineBidAskQueueProcess::result_to_queue_paths(
        &queue_path_events.inner,
        initial_q_a,
        initial_q_b,
    );
    let time_slice = times.as_slice()?;
    let ask_samples = rs_sample_queue_at_times(&paths.ask, time_slice);
    let bid_samples = rs_sample_queue_at_times(&paths.bid, time_slice);

    let out = PyDict::new_bound(py);
    out.set_item("ask", ask_samples.into_pyarray_bound(py))?;
    out.set_item("bid", bid_samples.into_pyarray_bound(py))?;
    Ok(out)
}

/// Experiment-local anchored conditional queue simulation.
///
/// The empirical queue snapshots define q_bar. Rust simulates dq = q - q_bar
/// and returns sampled queues/offsets on `sample_times`.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (
    event_times,
    event_dims,
    event_qtys,
    bar_q_pre,
    bar_q_post,
    passive_flags,
    sample_times,
    initial_q,
    horizon_seconds,
    n_simulations,
    a_l,
    b_l,
    a_c,
    b_c,
    seed,
    own_qtys = None
))]
fn simulate_anchored_affine_queue<'py>(
    py: Python<'py>,
    event_times: PyReadonlyArray1<f64>,
    event_dims: PyReadonlyArray1<i32>,
    event_qtys: PyReadonlyArray1<u32>,
    bar_q_pre: PyReadonlyArray1<f64>,
    bar_q_post: PyReadonlyArray1<f64>,
    passive_flags: PyReadonlyArray1<bool>,
    sample_times: PyReadonlyArray1<f64>,
    initial_q: f64,
    horizon_seconds: f64,
    n_simulations: usize,
    a_l: f64,
    b_l: f64,
    a_c: f64,
    b_c: f64,
    seed: Option<u64>,
    own_qtys: Option<PyReadonlyArray1<u32>>,
) -> PyResult<Bound<'py, PyDict>> {
    let event_dims_slice = event_dims.as_slice()?;
    let event_qtys_slice = event_qtys.as_slice()?;
    let passive_flags_slice = passive_flags.as_slice()?;
    let own_qtys_vec = match own_qtys {
        Some(values) => values.as_slice()?.to_vec(),
        None => event_dims_slice
            .iter()
            .zip(event_qtys_slice.iter())
            .zip(passive_flags_slice.iter())
            .map(|((&dim, &qty), &flag)| {
                if flag && dim == 0 {
                    qty
                } else {
                    0
                }
            })
            .collect(),
    };
    let input = AnchoredQueueInput {
        event_times: event_times.as_slice()?.to_vec(),
        event_dims: event_dims_slice.to_vec(),
        event_qtys: event_qtys_slice.to_vec(),
        bar_q_pre: bar_q_pre.as_slice()?.to_vec(),
        bar_q_post: bar_q_post.as_slice()?.to_vec(),
        passive_flags: passive_flags_slice.to_vec(),
        own_qtys: own_qtys_vec,
        sample_times: sample_times.as_slice()?.to_vec(),
        initial_q,
        horizon_seconds,
        n_simulations,
        seed,
        intensity: AffineQueueIntensity { a_l, b_l, a_c, b_c },
    };

    let result = rs_simulate_anchored_affine_queue(input)
        .map_err(pyo3::exceptions::PyValueError::new_err)?;

    let out = PyDict::new_bound(py);
    out.set_item("n_times", result.n_times)?;
    out.set_item("n_simulations", result.n_simulations)?;
    out.set_item("factual_queue", result.factual_queue.into_pyarray_bound(py))?;
    out.set_item(
        "mechanical_queue",
        result.mechanical_queue.into_pyarray_bound(py),
    )?;
    out.set_item("queue_samples", result.queue_samples.into_pyarray_bound(py))?;
    out.set_item(
        "offset_samples",
        result.offset_samples.into_pyarray_bound(py),
    )?;
    out.set_item("event_times", result.event_times.into_pyarray_bound(py))?;
    out.set_item("event_dims", result.event_dims.into_pyarray_bound(py))?;
    out.set_item("event_qtys", result.event_qtys.into_pyarray_bound(py))?;
    out.set_item(
        "event_simulations",
        result.event_simulations.into_pyarray_bound(py),
    )?;
    Ok(out)
}

#[pyfunction]
fn select_limit_flags_first_every<'py>(
    py: Python<'py>,
    event_times: PyReadonlyArray1<f64>,
    event_dims: PyReadonlyArray1<i32>,
    every_seconds: f64,
) -> PyResult<Bound<'py, PyArray1<bool>>> {
    let flags = rs_select_first_limit_every(
        event_times.as_slice()?,
        event_dims.as_slice()?,
        every_seconds,
    )
    .map_err(pyo3::exceptions::PyValueError::new_err)?;
    Ok(flags.into_pyarray_bound(py))
}

#[pyfunction]
fn select_limit_flags_indices<'py>(
    py: Python<'py>,
    event_dims: PyReadonlyArray1<i32>,
    indices: Vec<usize>,
    index_base: usize,
) -> PyResult<Bound<'py, PyArray1<bool>>> {
    let flags = rs_select_limit_indices(event_dims.as_slice()?, &indices, index_base)
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
    Ok(flags.into_pyarray_bound(py))
}

#[pyfunction]
fn select_limit_flags_random_fraction<'py>(
    py: Python<'py>,
    event_dims: PyReadonlyArray1<i32>,
    fraction: f64,
    seed: Option<u64>,
) -> PyResult<Bound<'py, PyArray1<bool>>> {
    let flags = rs_select_random_limit_fraction(event_dims.as_slice()?, fraction, seed)
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
    Ok(flags.into_pyarray_bound(py))
}

#[pyfunction]
#[pyo3(signature = (
    event_times,
    event_dims,
    event_qtys,
    queue_post,
    passive_flags,
    cancellation_policy,
    theta,
    seed,
    cap_position_by_queue_post = false,
    own_qtys = None
))]
fn track_passive_fills<'py>(
    py: Python<'py>,
    event_times: PyReadonlyArray1<f64>,
    event_dims: PyReadonlyArray1<i32>,
    event_qtys: PyReadonlyArray1<u32>,
    queue_post: PyReadonlyArray1<f64>,
    passive_flags: PyReadonlyArray1<bool>,
    cancellation_policy: String,
    theta: f64,
    seed: Option<u64>,
    cap_position_by_queue_post: bool,
    own_qtys: Option<PyReadonlyArray1<u32>>,
) -> PyResult<Bound<'py, PyDict>> {
    let policy = CancellationPolicy::from_name(&cancellation_policy, theta)
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
    let event_dims_slice = event_dims.as_slice()?;
    let event_qtys_slice = event_qtys.as_slice()?;
    let passive_flags_slice = passive_flags.as_slice()?;
    let own_qtys_vec = match own_qtys {
        Some(values) => values.as_slice()?.to_vec(),
        None => event_dims_slice
            .iter()
            .zip(event_qtys_slice.iter())
            .zip(passive_flags_slice.iter())
            .map(|((&dim, &qty), &flag)| {
                if flag && dim == 0 {
                    qty
                } else {
                    0
                }
            })
            .collect(),
    };
    let input = PassiveFillTrackerInput {
        event_times: event_times.as_slice()?.to_vec(),
        event_dims: event_dims_slice.to_vec(),
        event_qtys: event_qtys_slice.to_vec(),
        queue_post: queue_post.as_slice()?.to_vec(),
        passive_flags: passive_flags_slice.to_vec(),
        own_qtys: own_qtys_vec,
        cancellation_policy: policy,
        cap_position_by_queue_post,
        seed,
    };
    let result = rs_track_passive_fills(input).map_err(pyo3::exceptions::PyValueError::new_err)?;

    let out = PyDict::new_bound(py);
    out.set_item("order_ids", result.order_ids.into_pyarray_bound(py))?;
    out.set_item("order_row_pos", result.order_row_pos.into_pyarray_bound(py))?;
    out.set_item("order_times", result.order_times.into_pyarray_bound(py))?;
    out.set_item("initial_qtys", result.initial_qtys.into_pyarray_bound(py))?;
    out.set_item("executed_qtys", result.executed_qtys.into_pyarray_bound(py))?;
    out.set_item(
        "remaining_qtys",
        result.remaining_qtys.into_pyarray_bound(py),
    )?;
    out.set_item("canceled_qtys", result.canceled_qtys.into_pyarray_bound(py))?;
    out.set_item(
        "final_position_qtys",
        result.final_position_qtys.into_pyarray_bound(py),
    )?;
    out.set_item(
        "final_top_qtys",
        result.final_top_qtys.into_pyarray_bound(py),
    )?;
    out.set_item(
        "completed_times",
        result.completed_times.into_pyarray_bound(py),
    )?;
    out.set_item(
        "fill_order_ids",
        result.fill_order_ids.into_pyarray_bound(py),
    )?;
    out.set_item(
        "fill_order_row_pos",
        result.fill_order_row_pos.into_pyarray_bound(py),
    )?;
    out.set_item(
        "fill_event_row_pos",
        result.fill_event_row_pos.into_pyarray_bound(py),
    )?;
    out.set_item("fill_times", result.fill_times.into_pyarray_bound(py))?;
    out.set_item("fill_qtys", result.fill_qtys.into_pyarray_bound(py))?;
    Ok(out)
}

/// Experiment-local minute-by-minute passive execution latency grid.
///
/// For each `minute_start`, this selects up to `n_orders` own limit orders by
/// taking the first limit event in each `order_spacing_seconds` slot, then
/// tracks fills over `tracking_horizon_seconds`.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (
    event_times,
    event_dims,
    event_qtys,
    queue_post,
    source_row_pos,
    minute_starts,
    n_orders,
    order_spacing_seconds,
    tracking_horizon_seconds,
    cancellation_policy,
    theta,
    seed,
    cap_position_by_queue_post = false
))]
fn simulate_execution_latency_grid<'py>(
    py: Python<'py>,
    event_times: PyReadonlyArray1<f64>,
    event_dims: PyReadonlyArray1<i32>,
    event_qtys: PyReadonlyArray1<u32>,
    queue_post: PyReadonlyArray1<f64>,
    source_row_pos: PyReadonlyArray1<i64>,
    minute_starts: PyReadonlyArray1<f64>,
    n_orders: usize,
    order_spacing_seconds: f64,
    tracking_horizon_seconds: f64,
    cancellation_policy: String,
    theta: f64,
    seed: Option<u64>,
    cap_position_by_queue_post: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let event_times = event_times.as_slice()?;
    let event_dims = event_dims.as_slice()?;
    let event_qtys = event_qtys.as_slice()?;
    let queue_post = queue_post.as_slice()?;
    let source_row_pos = source_row_pos.as_slice()?;
    let minute_starts = minute_starts.as_slice()?;

    let policy = CancellationPolicy::from_name(&cancellation_policy, theta)
        .map_err(pyo3::exceptions::PyValueError::new_err)?;

    let source_row_positions: Vec<usize> = source_row_pos
        .iter()
        .map(|&row_pos| {
            if row_pos < 0 {
                Err(pyo3::exceptions::PyValueError::new_err(
                    "source_row_pos must be non-negative",
                ))
            } else {
                Ok(row_pos as usize)
            }
        })
        .collect::<PyResult<_>>()?;

    let result = rs_build_execution_latency_grid(ExecutionLatencyGridInput {
        event_times: event_times.to_vec(),
        event_dims: event_dims.to_vec(),
        event_qtys: event_qtys.to_vec(),
        queue_post: queue_post.to_vec(),
        source_row_positions,
        minute_starts: minute_starts.to_vec(),
        tracking_horizon_seconds,
        n_orders,
        order_spacing_seconds,
        cancellation_policy: policy,
        cap_position_by_queue_post,
        seed,
    })
    .map_err(pyo3::exceptions::PyValueError::new_err)?;

    let mut orders_filled_by_window = vec![0_u32; minute_starts.len()];
    for row in &result.order_rows {
        if row.remaining_qty == 0 {
            orders_filled_by_window[row.minute_index] += 1;
        }
    }

    let mut order_ids = Vec::<usize>::with_capacity(result.order_rows.len());
    let mut order_window_ids = Vec::<usize>::with_capacity(result.order_rows.len());
    let mut order_minute_starts = Vec::<f64>::with_capacity(result.order_rows.len());
    let mut order_slots = Vec::<usize>::with_capacity(result.order_rows.len());
    let mut order_row_pos = Vec::<usize>::with_capacity(result.order_rows.len());
    let mut order_source_row_pos = Vec::<i64>::with_capacity(result.order_rows.len());
    let mut order_times = Vec::<f64>::with_capacity(result.order_rows.len());
    let mut initial_qtys = Vec::<u32>::with_capacity(result.order_rows.len());
    let mut executed_qtys = Vec::<u32>::with_capacity(result.order_rows.len());
    let mut remaining_qtys = Vec::<u32>::with_capacity(result.order_rows.len());
    let mut final_position_qtys = Vec::<f64>::with_capacity(result.order_rows.len());
    let mut final_top_qtys = Vec::<u32>::with_capacity(result.order_rows.len());
    let mut completed_times = Vec::<f64>::with_capacity(result.order_rows.len());
    let mut latencies = Vec::<f64>::with_capacity(result.order_rows.len());
    let mut order_key_to_global =
        std::collections::HashMap::<(usize, usize), (usize, usize, i64)>::new();

    for row in &result.order_rows {
        let global_order_id = order_ids.len();
        let source_row_pos_i64 = row.source_row_pos as i64;
        order_key_to_global.insert(
            (row.minute_index, row.order_id),
            (global_order_id, row.event_row_pos, source_row_pos_i64),
        );

        order_ids.push(global_order_id);
        order_window_ids.push(row.minute_index);
        order_minute_starts.push(row.minute_start_time);
        order_slots.push(row.order_slot);
        order_row_pos.push(row.event_row_pos);
        order_source_row_pos.push(source_row_pos_i64);
        order_times.push(row.post_time);
        initial_qtys.push(row.initial_qty);
        executed_qtys.push(row.executed_qty);
        remaining_qtys.push(row.remaining_qty);
        final_position_qtys.push(row.final_position_qty);
        final_top_qtys.push(row.final_top_qty);
        completed_times.push(row.completed_time.unwrap_or(f64::NAN));
        latencies.push(row.latency_seconds.unwrap_or(f64::NAN));
    }

    let mut fill_order_ids = Vec::<usize>::with_capacity(result.fill_rows.len());
    let mut fill_window_ids = Vec::<usize>::with_capacity(result.fill_rows.len());
    let mut fill_order_row_pos = Vec::<usize>::with_capacity(result.fill_rows.len());
    let mut fill_order_source_row_pos = Vec::<i64>::with_capacity(result.fill_rows.len());
    let mut fill_event_row_pos = Vec::<usize>::with_capacity(result.fill_rows.len());
    let mut fill_event_source_row_pos = Vec::<i64>::with_capacity(result.fill_rows.len());
    let mut fill_times = Vec::<f64>::with_capacity(result.fill_rows.len());
    let mut fill_qtys = Vec::<u32>::with_capacity(result.fill_rows.len());

    for row in &result.fill_rows {
        let (global_order_id, order_event_row_pos, order_source_row_pos) = order_key_to_global
            .get(&(row.minute_index, row.order_id))
            .copied()
            .ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(
                    "native grid returned a fill for an unknown order id",
                )
            })?;

        fill_order_ids.push(global_order_id);
        fill_window_ids.push(row.minute_index);
        fill_order_row_pos.push(order_event_row_pos);
        fill_order_source_row_pos.push(order_source_row_pos);
        fill_event_row_pos.push(row.fill_event_row_pos);
        fill_event_source_row_pos.push(row.fill_source_row_pos as i64);
        fill_times.push(row.fill_time);
        fill_qtys.push(row.fill_qty);
    }

    let out = PyDict::new_bound(py);
    out.set_item("n_windows", minute_starts.len())?;
    out.set_item("n_orders", n_orders)?;
    out.set_item("order_spacing_seconds", order_spacing_seconds)?;
    out.set_item("tracking_horizon_seconds", tracking_horizon_seconds)?;
    out.set_item(
        "minute_starts",
        minute_starts.to_vec().into_pyarray_bound(py),
    )?;
    out.set_item(
        "orders_requested_by_window",
        vec![n_orders as u32; minute_starts.len()].into_pyarray_bound(py),
    )?;
    out.set_item(
        "orders_posted_by_window",
        result
            .posted_counts
            .iter()
            .map(|&count| count as u32)
            .collect::<Vec<_>>()
            .into_pyarray_bound(py),
    )?;
    out.set_item(
        "orders_filled_by_window",
        orders_filled_by_window.into_pyarray_bound(py),
    )?;
    out.set_item("order_ids", order_ids.into_pyarray_bound(py))?;
    out.set_item("order_window_ids", order_window_ids.into_pyarray_bound(py))?;
    out.set_item(
        "order_minute_starts",
        order_minute_starts.into_pyarray_bound(py),
    )?;
    out.set_item("order_slots", order_slots.into_pyarray_bound(py))?;
    out.set_item("order_row_pos", order_row_pos.into_pyarray_bound(py))?;
    out.set_item(
        "order_source_row_pos",
        order_source_row_pos.into_pyarray_bound(py),
    )?;
    out.set_item("order_times", order_times.into_pyarray_bound(py))?;
    out.set_item("initial_qtys", initial_qtys.into_pyarray_bound(py))?;
    out.set_item("executed_qtys", executed_qtys.into_pyarray_bound(py))?;
    out.set_item("remaining_qtys", remaining_qtys.into_pyarray_bound(py))?;
    out.set_item(
        "final_position_qtys",
        final_position_qtys.into_pyarray_bound(py),
    )?;
    out.set_item("final_top_qtys", final_top_qtys.into_pyarray_bound(py))?;
    out.set_item("completed_times", completed_times.into_pyarray_bound(py))?;
    out.set_item("latencies", latencies.into_pyarray_bound(py))?;
    out.set_item("fill_order_ids", fill_order_ids.into_pyarray_bound(py))?;
    out.set_item("fill_window_ids", fill_window_ids.into_pyarray_bound(py))?;
    out.set_item(
        "fill_order_row_pos",
        fill_order_row_pos.into_pyarray_bound(py),
    )?;
    out.set_item(
        "fill_order_source_row_pos",
        fill_order_source_row_pos.into_pyarray_bound(py),
    )?;
    out.set_item(
        "fill_event_row_pos",
        fill_event_row_pos.into_pyarray_bound(py),
    )?;
    out.set_item(
        "fill_event_source_row_pos",
        fill_event_source_row_pos.into_pyarray_bound(py),
    )?;
    out.set_item("fill_times", fill_times.into_pyarray_bound(py))?;
    out.set_item("fill_qtys", fill_qtys.into_pyarray_bound(py))?;
    Ok(out)
}

/// Conditional simulation context for one-dimensional MultiExponentialHawkes paths.
/// Owns its inputs and rebuilds the borrowed Rust context on each call.
#[pyclass(name = "ConditionalHawkesSimulationContext")]
pub struct PyConditionalHawkesSimulationContext {
    hawkes: Py<PyMultiExponentialHawkes>,
    cond_events_by_dim: Vec<Vec<f64>>,
    cond_externals: Option<MultivariateSimulationResult>,
    new_externals: Option<MultivariateSimulationResult>,
    t_max: f64,
}

#[pymethods]
impl PyConditionalHawkesSimulationContext {
    #[new]
    #[pyo3(signature = (hawkes, cond_events_by_dim, t_max, *, cond_externals=None, new_externals=None))]
    fn new(
        hawkes: Py<PyMultiExponentialHawkes>,
        cond_events_by_dim: Vec<Vec<f64>>,
        t_max: f64,
        cond_externals: Option<&PySimulationResult>,
        new_externals: Option<&PySimulationResult>,
    ) -> Self {
        Self {
            hawkes,
            cond_events_by_dim,
            cond_externals: cond_externals.map(|r| r.inner.clone()),
            new_externals: new_externals.map(|r| r.inner.clone()),
            t_max,
        }
    }

    /// Single-shot conditional Hawkes simulation; returns the resulting event stream.
    fn simulate(&self, py: Python, seed: Option<u64>) -> PySimulationResult {
        let hawkes_borrow = self.hawkes.borrow(py);
        let ctx = ConditionalSimulationContext::new(
            &hawkes_borrow.inner,
            &self.cond_events_by_dim,
            self.cond_externals.as_ref(),
            self.new_externals.as_ref(),
            self.t_max,
        );
        PySimulationResult {
            inner: ctx.simulate(None, seed),
        }
    }

    /// Batched conditional Hawkes simulation. This avoids per-path Python calls
    /// while preserving variable-length event streams.
    #[pyo3(signature = (n_simulations, base_seed=None, shared_acceptance=false))]
    fn simulate_many(
        &self,
        py: Python,
        n_simulations: usize,
        base_seed: Option<u64>,
        shared_acceptance: bool,
    ) -> Vec<PySimulationResult> {
        let hawkes_borrow = self.hawkes.borrow(py);
        let ctx = ConditionalSimulationContext::new(
            &hawkes_borrow.inner,
            &self.cond_events_by_dim,
            self.cond_externals.as_ref(),
            self.new_externals.as_ref(),
            self.t_max,
        );

        if shared_acceptance {
            let configs: Vec<SimulationConfig<'_, Vec<f64>>> = (0..n_simulations)
                .map(|_| SimulationConfig::new(self.new_externals.as_ref(), None))
                .collect();
            return ctx
                .simulate_multiple(&configs, base_seed.unwrap_or(0))
                .into_iter()
                .map(|inner| PySimulationResult { inner })
                .collect();
        }

        (0..n_simulations)
            .map(|sim_idx| {
                let seed = base_seed.map(|s| s.wrapping_add(sim_idx as u64));
                PySimulationResult {
                    inner: ctx.simulate(None, seed),
                }
            })
            .collect()
    }

    /// Batched conditional Hawkes simulation returning ragged time arrays.
    #[pyo3(signature = (n_simulations, base_seed=None, shared_acceptance=false))]
    fn simulate_many_times<'py>(
        &self,
        py: Python<'py>,
        n_simulations: usize,
        base_seed: Option<u64>,
        shared_acceptance: bool,
    ) -> PyResult<Bound<'py, PyList>> {
        let results = self.simulate_many(py, n_simulations, base_seed, shared_acceptance);
        let out = PyList::empty_bound(py);
        for result in results {
            let times: Vec<f64> = result.inner.events.iter().map(|e| e.time).collect();
            out.append(times.into_pyarray_bound(py))?;
        }
        Ok(out)
    }
}

/// Conditional simulation context for one-dimensional affine counting paths.
/// The intensity is lambda(N_t) = b + a * N_t.
#[pyclass(name = "ConditionalAffineCountingSimulationContext")]
pub struct PyConditionalAffineCountingSimulationContext {
    process: Py<PyAffineCountingProcess>,
    cond_events_by_dim: Vec<Vec<f64>>,
    cond_externals: Option<MultivariateSimulationResult>,
    new_externals: Option<MultivariateSimulationResult>,
    t_max: f64,
}

#[pymethods]
impl PyConditionalAffineCountingSimulationContext {
    #[new]
    #[pyo3(signature = (process, cond_events_by_dim, t_max, *, cond_externals=None, new_externals=None))]
    fn new(
        process: Py<PyAffineCountingProcess>,
        cond_events_by_dim: Vec<Vec<f64>>,
        t_max: f64,
        cond_externals: Option<&PySimulationResult>,
        new_externals: Option<&PySimulationResult>,
    ) -> PyResult<Self> {
        if cond_events_by_dim.len() != 1 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "cond_events_by_dim must contain exactly one event-time list",
            ));
        }
        if !t_max.is_finite() || t_max <= 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "t_max must be positive and finite",
            ));
        }
        Ok(Self {
            process,
            cond_events_by_dim,
            cond_externals: cond_externals.map(|r| r.inner.clone()),
            new_externals: new_externals.map(|r| r.inner.clone()),
            t_max,
        })
    }

    /// Single-shot conditional affine counting simulation.
    fn simulate(&self, py: Python, seed: Option<u64>) -> PySimulationResult {
        let process_borrow = self.process.borrow(py);
        let ctx = ConditionalSimulationContext::new(
            &process_borrow.inner,
            &self.cond_events_by_dim,
            self.cond_externals.as_ref(),
            self.new_externals.as_ref(),
            self.t_max,
        );
        PySimulationResult {
            inner: ctx.simulate(None, seed),
        }
    }

    /// Batched conditional affine counting simulation.
    #[pyo3(signature = (n_simulations, base_seed=None, shared_acceptance=false))]
    fn simulate_many(
        &self,
        py: Python,
        n_simulations: usize,
        base_seed: Option<u64>,
        shared_acceptance: bool,
    ) -> Vec<PySimulationResult> {
        let process_borrow = self.process.borrow(py);
        let ctx = ConditionalSimulationContext::new(
            &process_borrow.inner,
            &self.cond_events_by_dim,
            self.cond_externals.as_ref(),
            self.new_externals.as_ref(),
            self.t_max,
        );

        if shared_acceptance {
            let configs: Vec<SimulationConfig<'_, Vec<f64>>> = (0..n_simulations)
                .map(|_| SimulationConfig::new(self.new_externals.as_ref(), None))
                .collect();
            return ctx
                .simulate_multiple(&configs, base_seed.unwrap_or(0))
                .into_iter()
                .map(|inner| PySimulationResult { inner })
                .collect();
        }

        (0..n_simulations)
            .map(|sim_idx| {
                let seed = base_seed.map(|s| s.wrapping_add(sim_idx as u64));
                PySimulationResult {
                    inner: ctx.simulate(None, seed),
                }
            })
            .collect()
    }

    /// Batched conditional affine counting simulation returning ragged time arrays.
    #[pyo3(signature = (n_simulations, base_seed=None, shared_acceptance=false))]
    fn simulate_many_times<'py>(
        &self,
        py: Python<'py>,
        n_simulations: usize,
        base_seed: Option<u64>,
        shared_acceptance: bool,
    ) -> PyResult<Bound<'py, PyList>> {
        let results = self.simulate_many(py, n_simulations, base_seed, shared_acceptance);
        let out = PyList::empty_bound(py);
        for result in results {
            let times: Vec<f64> = result.inner.events.iter().map(|e| e.time).collect();
            out.append(times.into_pyarray_bound(py))?;
        }
        Ok(out)
    }
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
        let samples =
            ctx.simulate_queue_at_times(times.as_slice().unwrap(), initial_queue_size, None, seed);
        samples.into_pyarray_bound(py)
    }

    /// Memory-efficient bid-ask queue sampling at specified times. Returns a
    /// dict with "ask" and "bid" arrays aligned with `times`.
    fn simulate_bidask_queue_at_times<'py>(
        &self,
        py: Python<'py>,
        times: PyReadonlyArray1<f64>,
        initial_ask_queue_size: u32,
        initial_bid_queue_size: u32,
        seed: Option<u64>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let process_borrow = self.process.borrow(py);
        let ctx = ConditionalSimulationContext::new(
            &process_borrow.inner,
            &self.cond_events_by_dim,
            self.cond_externals.as_ref(),
            self.new_externals.as_ref(),
            self.t_max,
        );
        let (ask_samples, bid_samples) = ctx.simulate_bidask_queue_at_times(
            times.as_slice()?,
            initial_ask_queue_size,
            initial_bid_queue_size,
            None,
            seed,
        );

        let out = PyDict::new_bound(py);
        out.set_item("ask", ask_samples.into_pyarray_bound(py))?;
        out.set_item("bid", bid_samples.into_pyarray_bound(py))?;
        Ok(out)
    }

    /// Single-shot conditional simulate; returns the resulting event stream.
    fn simulate(&self, py: Python, seed: Option<u64>) -> PySimulationResult {
        let process_borrow = self.process.borrow(py);
        let ctx = ConditionalSimulationContext::new(
            &process_borrow.inner,
            &self.cond_events_by_dim,
            self.cond_externals.as_ref(),
            self.new_externals.as_ref(),
            self.t_max,
        );
        PySimulationResult {
            inner: ctx.simulate(None, seed),
        }
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
        Self {
            inner: TailImpact::from_affine_queue(mu, alpha, beta, b_l, b_c, events),
        }
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

/// Compute aggressive impact path from pre-sampled queues using the hybrid model.
/// `kappa` is a Python callable f64 -> f64; `bar_kappa` is a scalar f64 for the
/// metaorder component.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn aggressive_impact_from_queue_samples(
    py: Python,
    q_samples: PyReadonlyArray1<u32>,
    q_bar_samples: PyReadonlyArray1<u32>,
    eval_times: PyReadonlyArray1<f64>,
    is_market_order: Vec<bool>,
    hawkes: &PyMultiExponentialHawkes,
    kappa: PyObject,
    bar_kappa: f64,
) -> PyResult<PyAggressiveImpactPath> {
    let kappa_clone = kappa.clone_ref(py);
    let path = AggressiveImpactPath::from_queue_samples(
        q_samples.as_slice().unwrap(),
        q_bar_samples.as_slice().unwrap(),
        eval_times.as_slice().unwrap(),
        &is_market_order,
        &hawkes.inner,
        move |q: f64| -> f64 {
            Python::with_gil(|py| match kappa_clone.call1(py, (q,)) {
                Ok(r) => r.extract::<f64>(py).unwrap_or_else(|e| {
                    eprintln!("kappa(q={}) returned non-f64: {:?}", q, e);
                    f64::NAN
                }),
                Err(e) => {
                    eprintln!("kappa(q={}) raised: {:?}", q, e);
                    f64::NAN
                }
            })
        },
        bar_kappa,
    );
    Ok(PyAggressiveImpactPath {
        impact_path: path.impact_path,
    })
}

/// Compute the impact path I(t) for a (q, bar_q) pair via the affine-queue model.
///
/// q_events / bar_q_events are the full event streams of the two queue processes
/// (e.g. as returned by merge_events of conditional simulations + market orders).
/// initial_q is the starting queue size used for both paths.
#[pyfunction]
fn compute_impact_path<'py>(
    py: Python<'py>,
    q_events: &PySimulationResult,
    bar_q_events: &PySimulationResult,
    initial_q: u32,
    tail_impact: &PyTailImpact,
) -> Bound<'py, PyArray1<f64>> {
    let q_path = AffineQueueProcess::result_to_queue_path(&q_events.inner, initial_q);
    let bar_q_path = AffineQueueProcess::result_to_queue_path(&bar_q_events.inner, initial_q);
    let impact = ImpactPath::new(q_path, bar_q_path, &tail_impact.inner);
    impact.impact_path.into_pyarray_bound(py)
}

/// Compute passive flow-imbalance impact from queues sampled at market times.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn passive_flow_impact_from_queue_samples<'py>(
    py: Python<'py>,
    q_samples: PyReadonlyArray1<u32>,
    q_bar_samples: PyReadonlyArray1<u32>,
    market_times: PyReadonlyArray1<f64>,
    mu: f64,
    alpha: Vec<f64>,
    beta: Vec<f64>,
    b_l: f64,
    b_c: f64,
    c_kappa: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let q = q_samples.as_slice()?;
    let q_bar = q_bar_samples.as_slice()?;
    let events = market_times.as_slice()?;
    if q.len() != q_bar.len() || q.len() != events.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "q_samples, q_bar_samples, and market_times must have matching lengths",
        ));
    }
    if alpha.len() != beta.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "alpha and beta must have matching lengths",
        ));
    }

    let tail = TailImpact::from_affine_queue(mu, alpha, beta, b_l, b_c, events.to_vec());
    let mut impact = ImpactPath::from_queue_samples(q, q_bar, &tail).impact_path;
    for value in &mut impact {
        *value *= c_kappa;
    }
    Ok(impact.into_pyarray_bound(py))
}

/// Compute bid-ask passive flow-imbalance impact from queues sampled at ask and
/// bid market-order times. Inputs are ordered as (q, q_bar), where q is the
/// no-metaorder baseline and q_bar is the queue with the passive metaorder.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn bidask_passive_impact_from_queue_samples<'py>(
    py: Python<'py>,
    q_a_at_ask: PyReadonlyArray1<u32>,
    q_b_at_ask: PyReadonlyArray1<u32>,
    q_bar_a_at_ask: PyReadonlyArray1<u32>,
    q_bar_b_at_ask: PyReadonlyArray1<u32>,
    q_a_at_bid: PyReadonlyArray1<u32>,
    q_b_at_bid: PyReadonlyArray1<u32>,
    q_bar_a_at_bid: PyReadonlyArray1<u32>,
    q_bar_b_at_bid: PyReadonlyArray1<u32>,
    ask_market_times: PyReadonlyArray1<f64>,
    bid_market_times: PyReadonlyArray1<f64>,
    mu: f64,
    alpha: Vec<f64>,
    beta: Vec<f64>,
    b_l_own: f64,
    b_l_cross: f64,
    b_c_own: f64,
    b_c_cross: f64,
    c_kappa_effective: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let q_a_ask = q_a_at_ask.as_slice()?;
    let q_b_ask = q_b_at_ask.as_slice()?;
    let q_bar_a_ask = q_bar_a_at_ask.as_slice()?;
    let q_bar_b_ask = q_bar_b_at_ask.as_slice()?;
    let q_a_bid = q_a_at_bid.as_slice()?;
    let q_b_bid = q_b_at_bid.as_slice()?;
    let q_bar_a_bid = q_bar_a_at_bid.as_slice()?;
    let q_bar_b_bid = q_bar_b_at_bid.as_slice()?;
    let ask_events = ask_market_times.as_slice()?;
    let bid_events = bid_market_times.as_slice()?;

    if alpha.len() != beta.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "alpha and beta must have matching lengths",
        ));
    }
    if q_a_ask.len() != ask_events.len()
        || q_b_ask.len() != ask_events.len()
        || q_bar_a_ask.len() != ask_events.len()
        || q_bar_b_ask.len() != ask_events.len()
    {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "ask-side queue sample arrays must match ask_market_times length",
        ));
    }
    if q_a_bid.len() != bid_events.len()
        || q_b_bid.len() != bid_events.len()
        || q_bar_a_bid.len() != bid_events.len()
        || q_bar_b_bid.len() != bid_events.len()
    {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "bid-side queue sample arrays must match bid_market_times length",
        ));
    }

    let c_matrix = SymmetricCMatrix::from_affine_symmetric(b_l_own, b_l_cross, b_c_own, b_c_cross);
    let tail_impact = BidAskTailImpact::new_symmetric_hawkes(
        mu,
        alpha,
        beta,
        c_matrix,
        ask_events.to_vec(),
        bid_events.to_vec(),
    );
    let mut impact = BidAskImpactPath::from_queue_samples(
        q_a_ask,
        q_b_ask,
        q_bar_a_ask,
        q_bar_b_ask,
        q_a_bid,
        q_b_bid,
        q_bar_a_bid,
        q_bar_b_bid,
        &tail_impact,
    );
    for value in &mut impact.ask_impact {
        *value *= c_kappa_effective;
    }
    for value in &mut impact.bid_impact {
        *value *= c_kappa_effective;
    }

    let out = PyDict::new_bound(py);
    out.set_item("ask_impact", impact.ask_impact.into_pyarray_bound(py))?;
    out.set_item("bid_impact", impact.bid_impact.into_pyarray_bound(py))?;
    Ok(out)
}

/// Compute passive flow-imbalance impact using direct fitted propagator tails.
///
/// This is the propagator-input analogue of `passive_flow_impact_from_queue_samples`.
/// The returned path is a single-queue contribution; callers that need signed
/// bid/ask price impact should apply their side sign convention outside Rust.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn passive_tail_propagator_impact_from_queue_samples<'py>(
    py: Python<'py>,
    q_samples: PyReadonlyArray1<u32>,
    q_bar_samples: PyReadonlyArray1<u32>,
    market_times: PyReadonlyArray1<f64>,
    propagator_kappa: f64,
    propagator_weights: Vec<f64>,
    propagator_beta: Vec<f64>,
    b_l: f64,
    b_c: f64,
    queue_sensitivity: f64,
    zeta: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let q = q_samples.as_slice()?;
    let q_bar = q_bar_samples.as_slice()?;
    let events = market_times.as_slice()?;
    if q.len() != q_bar.len() || q.len() != events.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "q_samples, q_bar_samples, and market_times must have matching lengths",
        ));
    }
    if propagator_weights.len() != propagator_beta.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "propagator_weights and propagator_beta must have matching lengths",
        ));
    }

    let tail = TailImpact::from_tail_propagator(
        propagator_kappa,
        propagator_weights,
        propagator_beta,
        AffineQueueProcess::c_lambda(b_l, b_c),
        zeta,
        events.to_vec(),
    )
    .map_err(pyo3::exceptions::PyValueError::new_err)?;
    let mut impact = ImpactPath::from_queue_samples(q, q_bar, &tail).impact_path;
    for value in &mut impact {
        *value *= queue_sensitivity;
    }
    Ok(impact.into_pyarray_bound(py))
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", "0.1.0")?;
    m.add_class::<PyMultiExponentialHawkes>()?;
    m.add_function(wrap_pyfunction!(simulate_hawkes, m)?)?;
    m.add_function(wrap_pyfunction!(simulate_hawkes_result, m)?)?;
    m.add_class::<PySimulationResult>()?;
    m.add_class::<PyAffineCountingProcess>()?;
    m.add_function(wrap_pyfunction!(simulate_affine_counting_process, m)?)?;
    m.add_function(wrap_pyfunction!(
        simulate_affine_counting_process_with_externals,
        m
    )?)?;
    m.add_class::<PyQueueProcess>()?;
    m.add_class::<PyAffineQueueProcess>()?;
    m.add_class::<PyAffineBidAskQueueProcess>()?;
    m.add_function(wrap_pyfunction!(simulate_with_externals, m)?)?;
    m.add_function(wrap_pyfunction!(simulate_hawkes_as_market_orders, m)?)?;
    m.add_function(wrap_pyfunction!(simulate_hawkes_with_externals, m)?)?;
    m.add_function(wrap_pyfunction!(merge_events, m)?)?;
    m.add_function(wrap_pyfunction!(create_meta_orders, m)?)?;
    m.add_function(wrap_pyfunction!(create_meta_orders_from_times, m)?)?;
    m.add_function(wrap_pyfunction!(events_to_dim, m)?)?;
    m.add_function(wrap_pyfunction!(extract_events_by_dim, m)?)?;
    m.add_function(wrap_pyfunction!(sample_queue_at_times, m)?)?;
    m.add_function(wrap_pyfunction!(sample_bidask_queue_at_times, m)?)?;
    m.add_function(wrap_pyfunction!(simulate_anchored_affine_queue, m)?)?;
    m.add_function(wrap_pyfunction!(select_limit_flags_first_every, m)?)?;
    m.add_function(wrap_pyfunction!(select_limit_flags_indices, m)?)?;
    m.add_function(wrap_pyfunction!(select_limit_flags_random_fraction, m)?)?;
    m.add_function(wrap_pyfunction!(track_passive_fills, m)?)?;
    m.add_function(wrap_pyfunction!(simulate_execution_latency_grid, m)?)?;
    m.add_class::<PyConditionalHawkesSimulationContext>()?;
    m.add_class::<PyConditionalAffineCountingSimulationContext>()?;
    m.add_class::<PyConditionalSimulationContext>()?;
    m.add_class::<PyTailImpact>()?;
    m.add_class::<PyAggressiveImpactPath>()?;
    m.add_function(wrap_pyfunction!(aggressive_impact_from_queue_samples, m)?)?;
    m.add_function(wrap_pyfunction!(compute_impact_path, m)?)?;
    m.add_function(wrap_pyfunction!(passive_flow_impact_from_queue_samples, m)?)?;
    m.add_function(wrap_pyfunction!(
        bidask_passive_impact_from_queue_samples,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        passive_tail_propagator_impact_from_queue_samples,
        m
    )?)?;
    Ok(())
}
