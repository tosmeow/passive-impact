use crate::simulation_helpers::{create_rng, sample_exponential, sample_uniform};

use super::events::{
    apply_q_only_event, apply_sized_event_to_queue, clamp_offset, sample_bar_queue,
    sample_step_values, valid_dim, AffineQueueIntensity, CANCEL_DIM, LIMIT_DIM, MARKET_DIM,
};
use super::{AnchoredConditioningPath, AnchoredQueueSimulation};

const EPSILON: f64 = 1e-12;

/// Reusable simulation context for one anchored conditioning path.
///
/// This context borrows the empirical path and can generate either a single
/// offset path or sampled batches on a requested output grid.
pub struct AnchoredConditionalSimulationContext<'a> {
    /// Empirical queue path and event rows.
    pub conditioning_path: AnchoredConditioningPath<'a>,
    /// Limit/cancel intensities for counterfactual deviations.
    pub intensity: AffineQueueIntensity,
    /// Simulation horizon in seconds.
    pub t_max: f64,
}

impl<'a> AnchoredConditionalSimulationContext<'a> {
    /// Create a new anchored simulation context.
    pub fn new(
        conditioning_path: AnchoredConditioningPath<'a>,
        intensity: AffineQueueIntensity,
        t_max: f64,
    ) -> Self {
        Self {
            conditioning_path,
            intensity,
            t_max,
        }
    }

    /// Simulate one offset path `dq = q - bar_q`.
    ///
    /// Input market rows are common to factual and no-us paths. Flagged passive
    /// limit rows are removed from the no-us queue but remain in `bar_q`.
    pub fn simulate(&self, seed: Option<u64>) -> AnchoredOffsetPath {
        let mut rng = create_rng(seed);
        let mut offset_times = vec![0.0];
        let mut offset_values = vec![0.0];
        let mut events = Vec::new();

        let mut dq = 0.0;
        let mut t = 0.0;
        let mut row_idx = 0;
        let mut bar_current = self.conditioning_path.initial_q;

        while t < self.t_max + EPSILON {
            let next_row_time = self
                .conditioning_path
                .event_times
                .get(row_idx)
                .copied()
                .unwrap_or(f64::INFINITY);
            let t_limit = next_row_time.min(self.t_max);

            while let Some((independent_time, independent_dim)) =
                self.next_independent_event(&mut rng, t, t_limit, bar_current, dq)
            {
                t = independent_time;
                dq = apply_q_only_event(dq, bar_current, independent_dim);
                offset_times.push(t);
                offset_values.push(dq);
                events.push(AnchoredSimulatedEvent {
                    time: t,
                    dim: independent_dim,
                    qty: 1,
                });
            }

            if row_idx >= self.conditioning_path.event_times.len() || next_row_time > self.t_max {
                break;
            }

            t = next_row_time;
            let bar_model = self.conditioning_path.bar_q_pre[row_idx].max(0.0);
            let dim_raw = self.conditioning_path.event_dims[row_idx];
            let qty = self.conditioning_path.event_qtys[row_idx];

            if let Some(dim) = valid_dim(dim_raw) {
                let own_qty = self.conditioning_path.own_qty_at(row_idx, dim, qty);
                let background_qty = qty - own_qty;
                let q_model = (bar_model + dq).max(0.0);
                match dim {
                    LIMIT_DIM | CANCEL_DIM => {
                        let accepted_qty = if background_qty == 0 {
                            0
                        } else {
                            let cond_int = self.intensity.intensity(bar_model, dim);
                            let new_int = self.intensity.intensity(q_model, dim);
                            accepted_count(&mut rng, background_qty, cond_int, new_int)
                        };
                        let bar_after = apply_sized_event_to_queue(bar_model, dim, qty);
                        let q_after = apply_sized_event_to_queue(q_model, dim, accepted_qty);
                        dq = q_after - bar_after;
                        if accepted_qty > 0 {
                            events.push(AnchoredSimulatedEvent {
                                time: t,
                                dim,
                                qty: accepted_qty,
                            });
                        }
                    }
                    MARKET_DIM => {
                        let bar_after = apply_sized_event_to_queue(bar_model, dim, qty);
                        let q_after = apply_sized_event_to_queue(q_model, dim, qty);
                        dq = q_after - bar_after;
                        if qty > 0 {
                            events.push(AnchoredSimulatedEvent { time: t, dim, qty });
                        }
                    }
                    _ => {}
                }
            }

            bar_current = self.conditioning_path.bar_q_post[row_idx].max(0.0);
            dq = clamp_offset(dq, bar_current);
            offset_times.push(t);
            offset_values.push(dq);
            row_idx += 1;
        }

        AnchoredOffsetPath {
            offset_times,
            offset_values,
            events,
        }
    }

    /// Sample the empirical factual queue on `sample_times`.
    pub fn factual_queue_at_times(&self, sample_times: &[f64]) -> Vec<f64> {
        sample_bar_queue(
            self.conditioning_path.event_times,
            self.conditioning_path.bar_q_post,
            sample_times,
            self.conditioning_path.initial_q,
        )
    }

    /// Sample deterministic no-us offsets from removing flagged passive limits.
    pub fn mechanical_offsets_at_times(&self, sample_times: &[f64]) -> Vec<f64> {
        let mut step_times = vec![0.0];
        let mut step_values = vec![0.0];
        let mut dq = 0.0;

        for row_idx in 0..self.conditioning_path.event_times.len() {
            let bar_model = self.conditioning_path.bar_q_pre[row_idx].max(0.0);
            if let Some(dim) = valid_dim(self.conditioning_path.event_dims[row_idx]) {
                let qty = self.conditioning_path.event_qtys[row_idx];
                let q_model = (bar_model + dq).max(0.0);
                let own_qty = self.conditioning_path.own_qty_at(row_idx, dim, qty);
                let accepted_qty = qty - own_qty;
                let bar_after = apply_sized_event_to_queue(bar_model, dim, qty);
                let q_after = apply_sized_event_to_queue(q_model, dim, accepted_qty);
                dq = q_after - bar_after;
            }
            dq = clamp_offset(dq, self.conditioning_path.bar_q_post[row_idx].max(0.0));
            step_times.push(self.conditioning_path.event_times[row_idx]);
            step_values.push(dq);
        }

        sample_step_values(&step_times, &step_values, sample_times)
    }

    /// Sample the deterministic no-us queue from passive limit removal only.
    pub fn mechanical_queue_at_times(&self, sample_times: &[f64]) -> Vec<f64> {
        let factual_queue = self.factual_queue_at_times(sample_times);
        let mechanical_offsets = self.mechanical_offsets_at_times(sample_times);
        factual_queue
            .iter()
            .zip(mechanical_offsets.iter())
            .map(|(&q_bar, &dq)| (q_bar + dq).max(0.0))
            .collect()
    }

    /// Simulate many independent no-us queue paths and sample them on a grid.
    pub fn simulate_many(
        &self,
        sample_times: &[f64],
        n_simulations: usize,
        seed: Option<u64>,
    ) -> AnchoredQueueSimulation {
        let factual_queue = self.factual_queue_at_times(sample_times);
        let mechanical_queue = self.mechanical_queue_at_times(sample_times);

        let n_times = sample_times.len();
        let mut queue_samples = vec![0.0; n_times * n_simulations];
        let mut offset_samples = vec![0.0; n_times * n_simulations];
        let mut all_event_times = Vec::new();
        let mut all_event_dims = Vec::new();
        let mut all_event_qtys = Vec::new();
        let mut all_event_simulations = Vec::new();

        for sim_idx in 0..n_simulations {
            let path_seed = seed.map(|s| s + sim_idx as u64);
            let path = self.simulate(path_seed);
            let offsets = path.offsets_at_times(sample_times);
            for time_idx in 0..n_times {
                let flat_idx = time_idx * n_simulations + sim_idx;
                offset_samples[flat_idx] = offsets[time_idx];
                queue_samples[flat_idx] = (factual_queue[time_idx] + offsets[time_idx]).max(0.0);
            }

            for event in path.events {
                all_event_times.push(event.time);
                all_event_dims.push(event.dim);
                all_event_qtys.push(event.qty);
                all_event_simulations.push(sim_idx);
            }
        }

        AnchoredQueueSimulation {
            n_times,
            n_simulations,
            factual_queue,
            mechanical_queue,
            queue_samples,
            offset_samples,
            event_times: all_event_times,
            event_dims: all_event_dims,
            event_qtys: all_event_qtys,
            event_simulations: all_event_simulations,
        }
    }

    fn next_independent_event<R: rand::Rng>(
        &self,
        rng: &mut R,
        t: f64,
        t_limit: f64,
        bar_q: f64,
        dq: f64,
    ) -> Option<(f64, usize)> {
        if t_limit <= t {
            return None;
        }

        let q = (bar_q + dq).max(0.0);
        let limit_rate = (self.intensity.intensity(q, LIMIT_DIM)
            - self.intensity.intensity(bar_q, LIMIT_DIM))
        .max(0.0);
        let cancel_rate = (self.intensity.intensity(q, CANCEL_DIM)
            - self.intensity.intensity(bar_q, CANCEL_DIM))
        .max(0.0);
        let total_rate = limit_rate + cancel_rate;
        if total_rate <= EPSILON {
            return None;
        }

        let tau = sample_exponential(rng, total_rate);
        if t + tau >= t_limit {
            return None;
        }

        let dim = if sample_uniform(rng) * total_rate <= limit_rate {
            LIMIT_DIM
        } else {
            CANCEL_DIM
        };
        Some((t + tau, dim))
    }
}

/// One simulated offset path plus accepted counterfactual events.
#[derive(Clone, Debug)]
pub struct AnchoredOffsetPath {
    /// Step times for the offset path.
    pub offset_times: Vec<f64>,
    /// Offset values aligned to `offset_times`.
    pub offset_values: Vec<f64>,
    /// Accepted simulated limit/cancel/market rows for diagnostics.
    pub events: Vec<AnchoredSimulatedEvent>,
}

impl AnchoredOffsetPath {
    /// Sample the offset path on `sample_times`.
    pub fn offsets_at_times(&self, sample_times: &[f64]) -> Vec<f64> {
        sample_step_values(&self.offset_times, &self.offset_values, sample_times)
    }
}

/// Simulated event row emitted by anchored no-us simulation.
#[derive(Clone, Debug)]
pub struct AnchoredSimulatedEvent {
    /// Event time in seconds from the chosen origin.
    pub time: f64,
    /// Event dimension.
    pub dim: usize,
    /// Event size.
    pub qty: u32,
}

fn accepted_count<R: rand::Rng>(
    rng: &mut R,
    qty: u32,
    cond_intensity: f64,
    new_intensity: f64,
) -> u32 {
    if qty == 0 {
        return 0;
    }
    if cond_intensity <= EPSILON {
        return if new_intensity + EPSILON >= cond_intensity {
            qty
        } else {
            0
        };
    }
    let mut accepted = 0;
    for _ in 0..qty {
        if sample_uniform(rng) * cond_intensity <= new_intensity {
            accepted += 1;
        }
    }
    accepted
}
