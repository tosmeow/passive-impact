//! Anchored conditional queue simulation.
//!
//! The observed queue snapshots define the factual path `bar_q`. The simulator
//! samples only a displacement `dq = q - bar_q`, which lets empirical queue
//! resets or non-replayable snapshot jumps stay in the conditioning path rather
//! than being forced into the core queue model.

mod conditional_simulator;
pub(crate) mod events;

pub use conditional_simulator::{
    AnchoredConditionalSimulationContext, AnchoredOffsetPath, AnchoredSimulatedEvent,
};
pub use events::{valid_dim, AffineQueueIntensity, CANCEL_DIM, LIMIT_DIM, MARKET_DIM};

/// Input for anchored affine queue simulation.
///
/// All event arrays are row-aligned. `bar_q_pre` and `bar_q_post` are the
/// empirical queue immediately before and after each row; `own_qtys` marks the
/// owned part of each limit/cancel row that should be present in `bar_q` but
/// removed from the no-us queue. `passive_flags` is retained for older callers
/// and is interpreted as full-row ownership for limit rows when `own_qtys` is
/// zero.
#[derive(Clone, Debug)]
pub struct AnchoredQueueInput {
    /// Event row times in seconds from the chosen origin.
    pub event_times: Vec<f64>,
    /// Event dimensions: limit, cancel, market, or ignored.
    pub event_dims: Vec<i32>,
    /// Row sizes. A size-`n` row is handled without expanding into units.
    pub event_qtys: Vec<u32>,
    /// Empirical queue just before each row.
    pub bar_q_pre: Vec<f64>,
    /// Empirical queue just after each row.
    pub bar_q_post: Vec<f64>,
    /// Rows representing our passive limit orders.
    pub passive_flags: Vec<bool>,
    /// Owned quantity per row. Supports partial ownership of sized limit/cancel rows.
    pub own_qtys: Vec<u32>,
    /// Times at which queue paths should be sampled.
    pub sample_times: Vec<f64>,
    /// Empirical queue before the first input row.
    pub initial_q: f64,
    /// Simulation horizon in seconds.
    pub horizon_seconds: f64,
    /// Number of independent no-us paths to simulate.
    pub n_simulations: usize,
    /// Optional base seed; path `i` uses `seed + i`.
    pub seed: Option<u64>,
    /// Limit/cancel affine intensities used for counterfactual deviations.
    pub intensity: AffineQueueIntensity,
}

impl AnchoredQueueInput {
    /// Borrow the row-aligned empirical conditioning path.
    pub fn conditioning_path(&self) -> AnchoredConditioningPath<'_> {
        AnchoredConditioningPath {
            event_times: &self.event_times,
            event_dims: &self.event_dims,
            event_qtys: &self.event_qtys,
            bar_q_pre: &self.bar_q_pre,
            bar_q_post: &self.bar_q_post,
            passive_flags: &self.passive_flags,
            own_qtys: &self.own_qtys,
            initial_q: self.initial_q,
        }
    }
}

/// Borrowed row-aligned empirical path used by the anchored simulator.
#[derive(Clone, Copy, Debug)]
pub struct AnchoredConditioningPath<'a> {
    /// Event row times in seconds.
    pub event_times: &'a [f64],
    /// Event dimensions.
    pub event_dims: &'a [i32],
    /// Row sizes.
    pub event_qtys: &'a [u32],
    /// Empirical pre-row queue snapshots.
    pub bar_q_pre: &'a [f64],
    /// Empirical post-row queue snapshots.
    pub bar_q_post: &'a [f64],
    /// Passive own-limit flags.
    pub passive_flags: &'a [bool],
    /// Owned row quantities for limit/cancel rows.
    pub own_qtys: &'a [u32],
    /// Empirical queue before the first row.
    pub initial_q: f64,
}

impl<'a> AnchoredConditioningPath<'a> {
    /// Return the owned quantity for a row, capped by the observed row size.
    pub fn own_qty_at(&self, row_idx: usize, dim: usize, qty: u32) -> u32 {
        let explicit = self.own_qtys[row_idx].min(qty);
        if explicit > 0 {
            return explicit;
        }
        if self.passive_flags[row_idx] && dim == LIMIT_DIM {
            return qty;
        }
        0
    }
}

/// Sampled output of anchored queue simulation.
#[derive(Clone, Debug)]
pub struct AnchoredQueueSimulation {
    /// Number of sample times.
    pub n_times: usize,
    /// Number of simulated no-us paths.
    pub n_simulations: usize,
    /// Empirical factual queue sampled on `sample_times`.
    pub factual_queue: Vec<f64>,
    /// Deterministic no-us queue from removing flagged passive limit rows only.
    pub mechanical_queue: Vec<f64>,
    /// Flattened sampled no-us queues, indexed as `time * n_simulations + sim`.
    pub queue_samples: Vec<f64>,
    /// Flattened sampled offsets `q - bar_q`.
    pub offset_samples: Vec<f64>,
    /// Times of simulated counterfactual events for diagnostics.
    pub event_times: Vec<f64>,
    /// Dimensions of simulated counterfactual events.
    pub event_dims: Vec<usize>,
    /// Sizes of simulated counterfactual events.
    pub event_qtys: Vec<u32>,
    /// Simulation index for each simulated event.
    pub event_simulations: Vec<usize>,
}

/// Simulate anchored no-us queue paths for an empirical conditioning path.
pub fn simulate_anchored_affine_queue(
    input: AnchoredQueueInput,
) -> Result<AnchoredQueueSimulation, String> {
    validate_input(&input)?;
    let ctx = AnchoredConditionalSimulationContext::new(
        input.conditioning_path(),
        input.intensity,
        input.horizon_seconds,
    );
    Ok(ctx.simulate_many(&input.sample_times, input.n_simulations, input.seed))
}

fn validate_input(input: &AnchoredQueueInput) -> Result<(), String> {
    let n = input.event_times.len();
    let same_len = input.event_dims.len() == n
        && input.event_qtys.len() == n
        && input.bar_q_pre.len() == n
        && input.bar_q_post.len() == n
        && input.passive_flags.len() == n
        && input.own_qtys.len() == n;
    if !same_len {
        return Err("anchored queue event arrays must have matching lengths".to_string());
    }
    for row_idx in 0..n {
        let own_qty = input.own_qtys[row_idx];
        if own_qty == 0 {
            continue;
        }
        if own_qty > input.event_qtys[row_idx] {
            return Err("own_qtys must be less than or equal to event_qtys".to_string());
        }
        match valid_dim(input.event_dims[row_idx]) {
            Some(LIMIT_DIM) | Some(CANCEL_DIM) => {}
            _ => return Err("own_qtys may only be positive on limit/cancel rows".to_string()),
        }
    }
    if input.sample_times.is_empty() {
        return Err("sample_times must not be empty".to_string());
    }
    if input.n_simulations == 0 {
        return Err("n_simulations must be positive".to_string());
    }
    if input.horizon_seconds < 0.0 {
        return Err("horizon_seconds must be non-negative".to_string());
    }
    if input.event_times.windows(2).any(|w| w[0] > w[1]) {
        return Err("event_times must be sorted".to_string());
    }
    if input.sample_times.windows(2).any(|w| w[0] > w[1]) {
        return Err("sample_times must be sorted".to_string());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base_input() -> AnchoredQueueInput {
        AnchoredQueueInput {
            event_times: vec![0.0, 1.0, 2.0],
            event_dims: vec![0, 2, 1],
            event_qtys: vec![1, 1, 1],
            bar_q_pre: vec![10.0, 20.0, 19.0],
            bar_q_post: vec![20.0, 19.0, 21.0],
            passive_flags: vec![false, false, false],
            own_qtys: vec![0, 0, 0],
            sample_times: vec![0.0, 0.5, 1.0, 2.0],
            initial_q: 10.0,
            horizon_seconds: 2.0,
            n_simulations: 2,
            seed: Some(7),
            intensity: AffineQueueIntensity {
                a_l: 10.0,
                b_l: 0.0,
                a_c: 10.0,
                b_c: 0.0,
            },
        }
    }

    #[test]
    fn no_passive_identity_follows_raw_snapshots() {
        let out = simulate_anchored_affine_queue(base_input()).unwrap();
        assert_eq!(out.factual_queue, vec![20.0, 20.0, 19.0, 21.0]);
        assert_eq!(out.mechanical_queue, out.factual_queue);
        assert_eq!(
            out.queue_samples,
            vec![20.0, 20.0, 20.0, 20.0, 19.0, 19.0, 21.0, 21.0]
        );
        assert!(out.offset_samples.iter().all(|x| *x == 0.0));
    }

    #[test]
    fn passive_limit_removal_creates_negative_offset_but_keeps_market_common() {
        let mut input = base_input();
        input.passive_flags[0] = true;
        input.n_simulations = 1;
        let out = simulate_anchored_affine_queue(input).unwrap();
        assert_eq!(out.mechanical_queue, vec![19.0, 19.0, 18.0, 20.0]);
        assert_eq!(out.queue_samples, out.mechanical_queue);
        assert_eq!(out.offset_samples, vec![-1.0, -1.0, -1.0, -1.0]);
    }

    #[test]
    fn common_cancel_can_change_offset_at_zero_boundary() {
        let mut input = base_input();
        input.event_times = vec![0.0];
        input.event_dims = vec![1];
        input.event_qtys = vec![1];
        input.bar_q_pre = vec![1.0];
        input.bar_q_post = vec![0.0];
        input.passive_flags = vec![false];
        input.own_qtys = vec![0];
        input.sample_times = vec![0.0];
        input.initial_q = 1.0;
        input.n_simulations = 1;
        let out = simulate_anchored_affine_queue(input).unwrap();
        assert_eq!(out.queue_samples, vec![0.0]);
        assert_eq!(out.offset_samples, vec![0.0]);
    }

    #[test]
    fn partial_limit_ownership_removes_only_owned_row_quantity() {
        let mut input = base_input();
        input.event_times = vec![0.0];
        input.event_dims = vec![0];
        input.event_qtys = vec![5];
        input.bar_q_pre = vec![10.0];
        input.bar_q_post = vec![15.0];
        input.passive_flags = vec![false];
        input.own_qtys = vec![2];
        input.sample_times = vec![0.0];
        input.initial_q = 10.0;
        input.n_simulations = 1;

        let out = simulate_anchored_affine_queue(input).unwrap();
        assert_eq!(out.factual_queue, vec![15.0]);
        assert_eq!(out.mechanical_queue, vec![13.0]);
        assert_eq!(out.queue_samples, vec![13.0]);
        assert_eq!(out.offset_samples, vec![-2.0]);
    }

    #[test]
    fn partial_cancel_ownership_removes_only_owned_cancel_from_no_us() {
        let mut input = base_input();
        input.event_times = vec![0.0];
        input.event_dims = vec![1];
        input.event_qtys = vec![5];
        input.bar_q_pre = vec![10.0];
        input.bar_q_post = vec![5.0];
        input.passive_flags = vec![false];
        input.own_qtys = vec![2];
        input.sample_times = vec![0.0];
        input.initial_q = 10.0;
        input.n_simulations = 1;

        let out = simulate_anchored_affine_queue(input).unwrap();
        assert_eq!(out.factual_queue, vec![5.0]);
        assert_eq!(out.mechanical_queue, vec![7.0]);
        assert_eq!(out.queue_samples, vec![7.0]);
        assert_eq!(out.offset_samples, vec![2.0]);
    }

    #[test]
    fn context_simulate_many_matches_wrapper() {
        let input = base_input();
        let ctx = AnchoredConditionalSimulationContext::new(
            input.conditioning_path(),
            input.intensity,
            input.horizon_seconds,
        );
        let from_context = ctx.simulate_many(&input.sample_times, input.n_simulations, input.seed);
        let from_wrapper = simulate_anchored_affine_queue(input).unwrap();

        assert_eq!(from_context.factual_queue, from_wrapper.factual_queue);
        assert_eq!(from_context.mechanical_queue, from_wrapper.mechanical_queue);
        assert_eq!(from_context.queue_samples, from_wrapper.queue_samples);
        assert_eq!(from_context.offset_samples, from_wrapper.offset_samples);
        assert_eq!(from_context.event_times, from_wrapper.event_times);
        assert_eq!(from_context.event_dims, from_wrapper.event_dims);
        assert_eq!(from_context.event_qtys, from_wrapper.event_qtys);
        assert_eq!(
            from_context.event_simulations,
            from_wrapper.event_simulations
        );
    }
}
