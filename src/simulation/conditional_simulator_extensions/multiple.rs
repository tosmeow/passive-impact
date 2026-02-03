use crate::models::{MultivariateMarkovianIntensity, MultivariateEvent, MultivariateSimulationResult};
use crate::simulation::{ConditionalSimulationContext, SimulationConfig};
use crate::simulation_helpers::{create_rng, sample_exponential, sample_uniform};

impl<'a, P: MultivariateMarkovianIntensity> ConditionalSimulationContext<'a, P> {

    /// Simulate multiple paths with shared acceptance random numbers.
    /// This ensures monotonicity: if initial_state_A > initial_state_B,
    /// then the queue difference will be monotonically ordered.
    pub fn simulate_multiple(
        &self,
        configs: &[SimulationConfig<'_, P::State>],
        base_seed: u64,
    ) -> Vec<MultivariateSimulationResult>
    where
        P::State: Clone,
    {
        // Pre-generate acceptance random numbers for all conditioning events
        // These are shared across all simulations to ensure monotonicity
        let mut rng = create_rng(Some(base_seed));

        // Count total conditioning events across all dimensions
        let total_cond_events: usize = self.conditioning_events_by_dim.iter().map(|v| v.len()).sum();

        // Pre-generate uniforms for acceptance decisions
        let acceptance_uniforms: Vec<f64> = (0..total_cond_events)
            .map(|_| sample_uniform(&mut rng))
            .collect();

        // Build index mapping: for each (dim, event_idx), get the global index
        let mut cond_event_global_idx: Vec<Vec<usize>> = Vec::with_capacity(self.conditioning_events_by_dim.len());
        let mut global_idx = 0;
        for dim_events in self.conditioning_events_by_dim.iter() {
            let mut dim_indices = Vec::with_capacity(dim_events.len());
            for _ in 0..dim_events.len() {
                dim_indices.push(global_idx);
                global_idx += 1;
            }
            cond_event_global_idx.push(dim_indices);
        }

        configs
            .iter()
            .enumerate()
            .map(|(sim_idx, config)| {
                self.simulate_with_shared_randoms(
                    config.external_events,
                    config.initial_state.clone(),
                    &acceptance_uniforms,
                    &cond_event_global_idx,
                    base_seed.wrapping_add(sim_idx as u64 * 1000000), // Different seed for independent events
                )
            })
            .collect()
    }

    fn simulate_with_shared_randoms(
        &self,
        new_external_events: Option<&MultivariateSimulationResult>,
        new_initial_state: Option<P::State>,
        acceptance_uniforms: &[f64],
        cond_event_global_idx: &[Vec<usize>],
        independent_seed: u64,
    ) -> MultivariateSimulationResult {
        let mut rng = create_rng(Some(independent_seed)); // For independent events only
        let k = self.process.dim();

        let mut result = MultivariateSimulationResult::new(k);

        let mut cond_state = self.process.initial_state();
        let mut new_state = new_initial_state.unwrap_or_else(|| self.process.initial_state());

        let mut t_last_cond = 0.0;
        let mut t_last_new = 0.0;

        let mut t = 0.0;

        let mut cond_indices: Vec<usize> = vec![0; k];
        let mut cond_ext_idx = 0;
        let mut new_ext_idx = 0;

        let mut next_cond_times: Vec<f64> = self.conditioning_events_by_dim
            .iter()
            .map(|events| events.first().copied().unwrap_or(f64::INFINITY))
            .collect();

        while t < self.t_max {
            let cond_intensities = self.process.intensities_from_state(&cond_state, t, t_last_cond);
            let new_intensities = self.process.intensities_from_state(&new_state, t, t_last_new);

            let cond_ext_event = self.conditioning_external_events.and_then(|ext| ext.events.get(cond_ext_idx));
            let t_cond_ext = cond_ext_event.map(|e| e.time).unwrap_or(f64::INFINITY);

            let new_ext_event = new_external_events.and_then(|ext| ext.events.get(new_ext_idx));
            let t_new_ext = new_ext_event.map(|e| e.time).unwrap_or(f64::INFINITY);

            let (next_cond_internal_dim, next_cond_internal_time) = next_cond_times
                .iter()
                .enumerate()
                .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(dim, &time)| (dim, time))
                .unwrap_or((0, f64::INFINITY));

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
                .collect();

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

            let (_argmin, &tau_min) = taus.iter()
                .enumerate()
                .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap();

            t += tau_min;

            if t > self.t_max {
                break;
            }

            if taus[0] == tau_min {
                if let Some(ext_event) = cond_ext_event {
                    self.process.update_state(&mut cond_state, ext_event.dim, t, t_last_cond);
                    t_last_cond = t;
                    cond_ext_idx += 1;
                }
            }

            if taus[1] == tau_min {
                if let Some(ext_event) = new_ext_event {
                    self.process.update_state(&mut new_state, ext_event.dim, t, t_last_new);
                    t_last_new = t;
                    new_ext_idx += 1;
                    result.push(MultivariateEvent { time: t, dim: ext_event.dim });
                }
            }

            if taus[2] == tau_min {
                let cond_int = self.process.intensities_from_state(&cond_state, t, t_last_cond)[next_cond_internal_dim];

                if cond_int > EPSILON {
                    // Use pre-generated uniform for this conditioning event
                    let event_idx = cond_indices[next_cond_internal_dim];
                    let u = acceptance_uniforms[cond_event_global_idx[next_cond_internal_dim][event_idx]];

                    let new_int = self.process.intensities_from_state(&new_state, t, t_last_new)[next_cond_internal_dim];
                    if u * cond_int <= new_int {
                        self.process.update_state(&mut new_state, next_cond_internal_dim, t, t_last_new);
                        t_last_new = t;
                        result.push(MultivariateEvent { time: t, dim: next_cond_internal_dim });
                    }
                }

                self.process.update_state(&mut cond_state, next_cond_internal_dim, t, t_last_cond);
                t_last_cond = t;

                cond_indices[next_cond_internal_dim] += 1;
                next_cond_times[next_cond_internal_dim] = self.conditioning_events_by_dim[next_cond_internal_dim]
                    .get(cond_indices[next_cond_internal_dim])
                    .copied()
                    .unwrap_or(f64::INFINITY);
            } else if taus[3] == tau_min {
                self.process.update_state(&mut new_state, independent_dim, t, t_last_new);
                t_last_new = t;
                result.push(MultivariateEvent { time: t, dim: independent_dim });
            }
        }
        result
    }
}
