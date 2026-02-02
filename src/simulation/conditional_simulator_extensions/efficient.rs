use crate::models::MultivariateMarkovianIntensity;
use crate::simulation::ConditionalSimulationContext;
use crate::simulation_helpers::{create_rng, sample_exponential, sample_uniform};

impl<'a, P: MultivariateMarkovianIntensity> ConditionalSimulationContext<'a, P> {
    pub fn simulate_queue_at_times(
        &self,
        sample_times: &[f64],
        initial_q: u32,
        new_initial_state: Option<P::State>,
        seed: Option<u64>,
    ) -> Vec<u32>
    where
        P::State: AsRef<[f64]>,
    {
        let mut rng = create_rng(seed);
        let k = self.process.dim();

        // Pre-allocate output
        let mut queue_samples: Vec<u32> = Vec::with_capacity(sample_times.len());
        let mut sample_idx: usize = 0;

        let mut cond_state = self.process.initial_state();
        let mut new_state = new_initial_state.unwrap_or_else(|| self.process.initial_state());

        let mut t_last_cond = 0.0;
        let mut t_last_new = 0.0;

        let mut t = 0.0;
        let mut current_queue = initial_q;

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

            let new_ext_event = self.new_external_events.and_then(|ext| ext.events.get(new_ext_idx));
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

            // Record queue for sample times strictly before current event time
            while sample_idx < sample_times.len() && sample_times[sample_idx] < t {
                queue_samples.push(current_queue);
                sample_idx += 1;
            }

            // Track if queue changed at this time step
            let mut queue_updated = false;

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
                    // Update queue from state
                    current_queue = new_state.as_ref()[0].max(0.0) as u32;
                    queue_updated = true;
                }
            }

            if taus[2] == tau_min {
                let cond_int = self.process.intensities_from_state(&cond_state, t, t_last_cond)[next_cond_internal_dim];

                if cond_int > EPSILON {
                    let u = sample_uniform(&mut rng);
                    let new_int = self.process.intensities_from_state(&new_state, t, t_last_new)[next_cond_internal_dim];
                    if u * cond_int <= new_int {
                        self.process.update_state(&mut new_state, next_cond_internal_dim, t, t_last_new);
                        t_last_new = t;
                        current_queue = new_state.as_ref()[0].max(0.0) as u32;
                        queue_updated = true;
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
                current_queue = new_state.as_ref()[0].max(0.0) as u32;
                queue_updated = true;
            }

            // Record queue for sample times at exactly t (after event processing)
            if queue_updated {
                while sample_idx < sample_times.len() && sample_times[sample_idx] == t {
                    queue_samples.push(current_queue);
                    sample_idx += 1;
                }
            }
        }

        // Record remaining sample times (after all events)
        while sample_idx < sample_times.len() {
            queue_samples.push(current_queue);
            sample_idx += 1;
        }

        queue_samples
    }

    pub fn simulate_bidask_queue_at_times(
        &self,
        sample_times: &[f64],
        initial_q_a: u32,
        initial_q_b: u32,
        new_initial_state: Option<P::State>,
        seed: Option<u64>,
    ) -> (Vec<u32>, Vec<u32>)
    where
        P::State: AsRef<[f64]>,
    {
        let mut rng = create_rng(seed);
        let k = self.process.dim();

        // Pre-allocate outputs
        let mut ask_samples: Vec<u32> = Vec::with_capacity(sample_times.len());
        let mut bid_samples: Vec<u32> = Vec::with_capacity(sample_times.len());
        let mut sample_idx: usize = 0;

        let mut cond_state = self.process.initial_state();
        let mut new_state = new_initial_state.unwrap_or_else(|| self.process.initial_state());

        let mut t_last_cond = 0.0;
        let mut t_last_new = 0.0;

        let mut t = 0.0;
        let mut current_q_a = initial_q_a;
        let mut current_q_b = initial_q_b;

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

            let new_ext_event = self.new_external_events.and_then(|ext| ext.events.get(new_ext_idx));
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

            // Record queue for sample times strictly before current event time
            while sample_idx < sample_times.len() && sample_times[sample_idx] < t {
                ask_samples.push(current_q_a);
                bid_samples.push(current_q_b);
                sample_idx += 1;
            }

            // Track if queue changed at this time step
            let mut queue_updated = false;

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
                    // Update queues from state
                    let state_ref = new_state.as_ref();
                    current_q_a = state_ref[0].max(0.0) as u32;
                    current_q_b = state_ref[1].max(0.0) as u32;
                    queue_updated = true;
                }
            }

            if taus[2] == tau_min {
                let cond_int = self.process.intensities_from_state(&cond_state, t, t_last_cond)[next_cond_internal_dim];

                if cond_int > EPSILON {
                    let u = sample_uniform(&mut rng);
                    let new_int = self.process.intensities_from_state(&new_state, t, t_last_new)[next_cond_internal_dim];
                    if u * cond_int <= new_int {
                        self.process.update_state(&mut new_state, next_cond_internal_dim, t, t_last_new);
                        t_last_new = t;
                        let state_ref = new_state.as_ref();
                        current_q_a = state_ref[0].max(0.0) as u32;
                        current_q_b = state_ref[1].max(0.0) as u32;
                        queue_updated = true;
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
                let state_ref = new_state.as_ref();
                current_q_a = state_ref[0].max(0.0) as u32;
                current_q_b = state_ref[1].max(0.0) as u32;
                queue_updated = true;
            }

            // Record queue for sample times at exactly t (after event processing)
            if queue_updated {
                while sample_idx < sample_times.len() && sample_times[sample_idx] == t {
                    ask_samples.push(current_q_a);
                    bid_samples.push(current_q_b);
                    sample_idx += 1;
                }
            }
        }

        // Record remaining sample times (after all events)
        while sample_idx < sample_times.len() {
            ask_samples.push(current_q_a);
            bid_samples.push(current_q_b);
            sample_idx += 1;
        }

        (ask_samples, bid_samples)
    }
}

#[cfg(test)]
mod tests {
    use crate::models::AffineQueueProcess;
    use crate::simulation::{ConditionalSimulationContext, simulate_with_externals};
    use crate::simulation_helpers::sample_queue_at_times;
    use crate::models::{MultivariateSimulationResult, MultivariateEvent};
    
    #[test]
    fn test_simulate_queue_at_times_matches_regular_pipeline() {
        // Create a queue process
        let q0 = 100.0;
        let a_l = 50.0;
        let b_l = 0.1;
        let a_c = 10.0;
        let b_c = 0.05;

        let process = AffineQueueProcess::new_queue(q0, a_l, b_l, a_c, b_c);
        let t_max = 10.0;
        let initial_q = q0 as u32;

        // Create external events (market orders from a Hawkes process)
        let hawkes = crate::models::MultiExponentialHawkes::new(2.0, vec![0.3], vec![1.0]);
        let hawkes_result = crate::simulation::simulate(&hawkes, t_max, Some(42));
        let market_order_times: Vec<f64> = hawkes_result.events.iter().map(|e| e.time).collect();

        // Convert Hawkes events to market orders (dim 2)
        let mut external_events = MultivariateSimulationResult::new(3);
        for t in &market_order_times {
            external_events.push(MultivariateEvent { time: *t, dim: 2 });
        }

        // Simulate conditioning path
        let cond_result = simulate_with_externals(&process, t_max, &external_events, Some(123));
        let cond_events_by_dim: Vec<Vec<f64>> = (0..3)
            .map(|dim| cond_result.events.iter().filter(|e| e.dim == dim).map(|e| e.time).collect())
            .collect();

        // Create conditional simulation context
        let ctx = ConditionalSimulationContext::new(
            &process,
            &cond_events_by_dim,
            Some(&external_events),
            Some(&external_events),
            t_max,
        );

        // Method 1: Regular pipeline
        let sim_result = ctx.simulate(None, Some(999));
        let sim_path = AffineQueueProcess::result_to_queue_path(&sim_result, initial_q);
        let regular_samples = sample_queue_at_times(&sim_path, &market_order_times);

        // Method 2: Memory-efficient
        let efficient_samples = ctx.simulate_queue_at_times(&market_order_times, initial_q, None, Some(999));

        // Both should produce identical results (same seed)
        assert_eq!(
            regular_samples.len(),
            efficient_samples.len(),
            "Sample count mismatch: regular {} vs efficient {}",
            regular_samples.len(),
            efficient_samples.len()
        );

        for (i, (&r, &e)) in regular_samples.iter().zip(efficient_samples.iter()).enumerate() {
            assert_eq!(
                r, e,
                "Queue sample mismatch at index {}: regular {} vs efficient {}",
                i, r, e
            );
        }
    }
}
