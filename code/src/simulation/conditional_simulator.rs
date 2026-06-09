use crate::models::{
    MultivariateEvent, MultivariateMarkovianIntensity, MultivariateSimulationResult,
};
use crate::simulation_helpers::{create_rng, sample_exponential, sample_uniform};

pub struct SimulationConfig<'a, S> {
    // External events for this simulation path, None will use no externals.
    pub external_events: Option<&'a MultivariateSimulationResult>,
    // Initial state override, None will use the default process initial state.
    pub initial_state: Option<S>,
}

impl<'a, S> SimulationConfig<'a, S> {
    pub fn new(
        external_events: Option<&'a MultivariateSimulationResult>,
        initial_state: Option<S>,
    ) -> Self {
        Self {
            external_events,
            initial_state,
        }
    }
}

pub struct ConditionalSimulationContext<'a, P: MultivariateMarkovianIntensity> {
    pub process: &'a P,

    pub conditioning_events_by_dim: &'a [Vec<f64>],

    pub conditioning_external_events: Option<&'a MultivariateSimulationResult>,

    pub new_external_events: Option<&'a MultivariateSimulationResult>,

    pub t_max: f64,
}

impl<'a, P: MultivariateMarkovianIntensity> ConditionalSimulationContext<'a, P> {
    pub fn new(
        process: &'a P,
        conditioning_events_by_dim: &'a [Vec<f64>],
        conditioning_external_events: Option<&'a MultivariateSimulationResult>,
        new_external_events: Option<&'a MultivariateSimulationResult>,
        t_max: f64,
    ) -> Self {
        Self {
            process,
            conditioning_events_by_dim,
            conditioning_external_events,
            new_external_events,
            t_max,
        }
    }

    pub fn new_without_externals(
        process: &'a P,
        conditioning_events_by_dim: &'a [Vec<f64>],
        t_max: f64,
    ) -> Self {
        Self {
            process,
            conditioning_events_by_dim,
            conditioning_external_events: None,
            new_external_events: None,
            t_max,
        }
    }

    pub fn simulate(
        &self,
        new_initial_state: Option<P::State>,
        seed: Option<u64>,
    ) -> MultivariateSimulationResult {
        let mut rng = create_rng(seed);
        let k = self.process.dim();

        let mut result = MultivariateSimulationResult::new(k);

        let mut cond_state = self.process.initial_state();
        let mut new_state = new_initial_state.unwrap_or_else(|| self.process.initial_state());

        let mut t_last_cond = 0.0;
        let mut t_last_new = 0.0;

        let mut t = 0.0;

        let mut cond_indices: Vec<usize> = vec![0; k]; // Index of each internal event type over which we condition.
        let mut cond_ext_idx = 0; // Index for conditioning external events
        let mut new_ext_idx = 0; // Index for new external events

        let mut next_cond_times: Vec<f64> = self
            .conditioning_events_by_dim
            .iter()
            .map(|events| events.first().copied().unwrap_or(f64::INFINITY))
            .collect(); // Pre-compute next conditioning event times per dimension

        while t < self.t_max {
            let cond_intensities = self
                .process
                .intensities_from_state(&cond_state, t, t_last_cond); // Compute intensities for conditional state at current time.
            let new_intensities = self
                .process
                .intensities_from_state(&new_state, t, t_last_new); // Compute intensities for simulated state at current time.

            // Next external event for conditioning path (updates cond_state)
            let cond_ext_event = self
                .conditioning_external_events
                .and_then(|ext| ext.events.get(cond_ext_idx));
            let t_cond_ext = cond_ext_event.map(|e| e.time).unwrap_or(f64::INFINITY);

            // Next external event for new simulation (updates new_state)
            let new_ext_event = self
                .new_external_events
                .and_then(|ext| ext.events.get(new_ext_idx));
            let t_new_ext = new_ext_event.map(|e| e.time).unwrap_or(f64::INFINITY);

            let (next_cond_internal_dim, next_cond_internal_time) = next_cond_times
                .iter()
                .enumerate()
                .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(dim, &time)| (dim, time))
                .unwrap_or((0, f64::INFINITY)); // Next internal event of conditional path.

            const EPSILON: f64 = 1e-12;
            let independent_taus: Vec<f64> = new_intensities
                .iter()
                .zip(cond_intensities.iter())
                .map(|(&l_new, &l_cond)| {
                    let c = (l_new - l_cond).max(0.0);
                    if c > EPSILON {
                        sample_exponential(&mut rng, c)
                    } else {
                        f64::INFINITY
                    }
                })
                .collect(); // Independent measure: sample from (lambda_new - lambda_cond)^+ for each dimension

            // Find the minimum independent event time and its dimension
            let (independent_dim, independent_tau) = independent_taus
                .iter()
                .enumerate()
                .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(dim, &tau)| (dim, tau))
                .unwrap_or((0, f64::INFINITY));

            let taus = [
                t_cond_ext - t,
                t_new_ext - t,
                next_cond_internal_time - t,
                independent_tau,
            ];

            let (_argmin, &tau_min) = taus
                .iter()
                .enumerate()
                .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap();

            t += tau_min;

            // Stop if we've exceeded the time horizon (matching original simulator behavior)
            if t > self.t_max {
                break;
            }

            if taus[0] == tau_min {
                if let Some(ext_event) = cond_ext_event {
                    self.process
                        .update_state(&mut cond_state, ext_event.dim, t, t_last_cond);
                    t_last_cond = t;
                    cond_ext_idx += 1;
                }
            }

            if taus[1] == tau_min {
                if let Some(ext_event) = new_ext_event {
                    self.process
                        .update_state(&mut new_state, ext_event.dim, t, t_last_new);
                    t_last_new = t;
                    new_ext_idx += 1;
                    // Record external event in result so queue path reconstruction includes it
                    result.push(MultivariateEvent {
                        time: t,
                        dim: ext_event.dim,
                    });
                }
            }

            if taus[2] == tau_min {
                // Re-compute intensities at new time t (after tau_min elapsed)
                let cond_int = self
                    .process
                    .intensities_from_state(&cond_state, t, t_last_cond)[next_cond_internal_dim];

                // If conditioning intensity is 0, this dimension's events come from externals.
                // We still update cond_state but skip the acceptance test for new_state (externals will handle recording the event).
                if cond_int > EPSILON {
                    let u = sample_uniform(&mut rng);
                    let new_int = self
                        .process
                        .intensities_from_state(&new_state, t, t_last_new)[next_cond_internal_dim];
                    if u * cond_int <= new_int {
                        self.process.update_state(
                            &mut new_state,
                            next_cond_internal_dim,
                            t,
                            t_last_new,
                        );
                        t_last_new = t;
                        result.push(MultivariateEvent {
                            time: t,
                            dim: next_cond_internal_dim,
                        });
                    }
                }

                // Always update conditioning state and advance index
                self.process
                    .update_state(&mut cond_state, next_cond_internal_dim, t, t_last_cond);
                t_last_cond = t;

                cond_indices[next_cond_internal_dim] += 1;
                next_cond_times[next_cond_internal_dim] = self.conditioning_events_by_dim
                    [next_cond_internal_dim]
                    .get(cond_indices[next_cond_internal_dim])
                    .copied()
                    .unwrap_or(f64::INFINITY);
            } else if taus[3] == tau_min {
                self.process
                    .update_state(&mut new_state, independent_dim, t, t_last_new);
                t_last_new = t;
                result.push(MultivariateEvent {
                    time: t,
                    dim: independent_dim,
                });
            }
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{MultiExponentialHawkes, MultivariateMarkovianIntensity};
    use crate::simulation::simulate_with_externals;

    struct ConstantIntensityProcess {
        lambda: f64,
    }

    impl MultivariateMarkovianIntensity for ConstantIntensityProcess {
        type State = ();

        fn dim(&self) -> usize {
            1
        }

        fn initial_state(&self) -> Self::State {
            ()
        }

        fn intensities_from_state(&self, _state: &Self::State, _t: f64, _t_last: f64) -> Vec<f64> {
            vec![self.lambda]
        }

        fn update_state(&self, _state: &mut Self::State, _dim: usize, _t: f64, _t_prev: f64) {}
    }

    #[test]
    fn test_constant_intensity_identical_path() {
        let process = ConstantIntensityProcess { lambda: 2.0 };
        let t_max = 50.0;

        // Simulate conditioning path
        let conditioning_result = crate::simulation::simulate(&process, t_max, Some(42));

        // Run conditional simulation
        let ctx = ConditionalSimulationContext::new_without_externals(
            &process,
            &conditioning_result.events_by_dim,
            t_max,
        );

        let simulated_result = ctx.simulate(None, Some(999));

        assert_eq!(
            simulated_result.events.len(),
            conditioning_result.events.len(),
            "Poisson: event count mismatch: {} vs {}",
            simulated_result.events.len(),
            conditioning_result.events.len()
        );

        for (i, (sim, cond)) in simulated_result
            .events
            .iter()
            .zip(conditioning_result.events.iter())
            .enumerate()
        {
            assert_eq!(sim.time, cond.time, "Poisson: event {} time mismatch", i);
            assert_eq!(sim.dim, cond.dim, "Poisson: event {} dim mismatch", i);
        }
    }

    #[test]
    fn test_identical_externals_produces_identical_path() {
        let mu = 1.0;
        let alpha = vec![0.3, 0.2];
        let beta = vec![1.0, 5.0];
        let t_max = 50.0;

        let hawkes = MultiExponentialHawkes::new(mu, alpha, beta);

        // Create external events (another Hawkes process)
        let external_hawkes = MultiExponentialHawkes::new(0.5, vec![0.1], vec![2.0]);
        let external_events = crate::simulation::simulate(&external_hawkes, t_max, Some(123));

        // Simulate the conditioning path with these external events
        let conditioning_result =
            simulate_with_externals(&hawkes, t_max, &external_events, Some(42));

        let mut expected_events: Vec<MultivariateEvent> = conditioning_result.events.clone();
        expected_events.extend(external_events.events.iter().cloned());
        expected_events.sort_by(|a, b| a.time.partial_cmp(&b.time).unwrap());

        // Run conditional simulation with identical external events
        let ctx = ConditionalSimulationContext::new(
            &hawkes,
            &conditioning_result.events_by_dim,
            Some(&external_events),
            Some(&external_events),
            t_max,
        );

        let simulated_result = ctx.simulate(None, Some(999)); // Different seed shouldn't matter

        // The simulated path must be identical to the conditioning path + externals
        assert_eq!(
            simulated_result.events.len(),
            expected_events.len(),
            "Event count mismatch: simulated {} vs expected {}",
            simulated_result.events.len(),
            expected_events.len()
        );

        for (i, (sim_event, exp_event)) in simulated_result
            .events
            .iter()
            .zip(expected_events.iter())
            .enumerate()
        {
            assert_eq!(
                sim_event.time, exp_event.time,
                "Event {} time mismatch: simulated {} vs expected {}",
                i, sim_event.time, exp_event.time
            );
            assert_eq!(
                sim_event.dim, exp_event.dim,
                "Event {} dimension mismatch: simulated {} vs expected {}",
                i, sim_event.dim, exp_event.dim
            );
        }
    }

    // Test without any external events - same principle applies.
    #[test]
    fn test_no_externals_identical_path() {
        let mu = 1.5;
        let alpha = vec![0.4];
        let beta = vec![2.0];
        let t_max = 30.0;

        let hawkes = MultiExponentialHawkes::new(mu, alpha, beta);

        // Simulate the conditioning path without external events
        let conditioning_result = crate::simulation::simulate(&hawkes, t_max, Some(42));

        // Run conditional simulation with no external events (None == None)
        let ctx = ConditionalSimulationContext::new_without_externals(
            &hawkes,
            &conditioning_result.events_by_dim,
            t_max,
        );

        let simulated_result = ctx.simulate(None, Some(777));

        assert_eq!(
            simulated_result.events.len(),
            conditioning_result.events.len(),
            "Event count mismatch without externals"
        );

        for (i, (sim_event, cond_event)) in simulated_result
            .events
            .iter()
            .zip(conditioning_result.events.iter())
            .enumerate()
        {
            assert_eq!(
                sim_event.time, cond_event.time,
                "Event {} time mismatch (no externals)",
                i
            );
            assert_eq!(
                sim_event.dim, cond_event.dim,
                "Event {} dim mismatch (no externals)",
                i
            );
        }
    }

    #[test]
    fn test_zero_intensity_dimension_skips_acceptance() {
        use crate::models::MarkovianProcess;

        // Create a process where dim 2 has 0 intensity (like queue-only process)
        let process = MarkovianProcess::new(
            3,
            vec![100.0], // Initial state: q=100
            |state: &[f64], _t: f64, _t_last: f64| {
                let q = state[0];
                vec![
                    (50.0 + 0.1 * q).max(0.0),  // dim 0: some intensity
                    (10.0 + 0.05 * q).max(0.0), // dim 1: some intensity
                    0.0,                        // dim 2: ZERO (external only)
                ]
            },
            |state: &[f64], event: &MultivariateEvent, _t: f64, _t_prev: f64| {
                let q = state[0];
                let new_q = match event.dim {
                    0 => q + 1.0,
                    1 | 2 => (q - 1.0).max(0.0),
                    _ => q,
                };
                vec![new_q]
            },
        );

        let t_max = 10.0;

        // Create external events for dim 2 (like pre-simulated Hawkes)
        let mut external_events = MultivariateSimulationResult::new(3);
        external_events.push(MultivariateEvent { time: 2.0, dim: 2 });
        external_events.push(MultivariateEvent { time: 5.0, dim: 2 });
        external_events.push(MultivariateEvent { time: 8.0, dim: 2 });

        // Simulate conditioning path with externals
        let cond_result = simulate_with_externals(&process, t_max, &external_events, Some(42));

        // Include ALL dims in conditioning events (including dim 2 which has 0 intensity)
        // This is the "naive" approach that Option B should handle correctly
        let cond_events_by_dim: Vec<Vec<f64>> = (0..3)
            .map(|dim| {
                cond_result
                    .events
                    .iter()
                    .filter(|e| e.dim == dim)
                    .map(|e| e.time)
                    .collect()
            })
            .collect();

        // Run conditional simulation with same externals
        let ctx = ConditionalSimulationContext::new(
            &process,
            &cond_events_by_dim, // Includes dim 2 events (Option B: not manually emptied)
            Some(&external_events),
            Some(&external_events),
            t_max,
        );

        let sim_result = ctx.simulate(None, Some(999));

        // Count dim 2 events - should NOT be duplicated
        let dim2_count = sim_result.events.iter().filter(|e| e.dim == 2).count();
        assert_eq!(
            dim2_count, 3,
            "Expected exactly 3 dim-2 events (from externals only), got {}. \
             Option B should skip conditioning events for 0-intensity dims.",
            dim2_count
        );

        // Verify the dim 2 events are at the external event times
        let dim2_times: Vec<f64> = sim_result
            .events
            .iter()
            .filter(|e| e.dim == 2)
            .map(|e| e.time)
            .collect();
        assert_eq!(
            dim2_times,
            vec![2.0, 5.0, 8.0],
            "Dim-2 events should match external event times"
        );
    }
}
