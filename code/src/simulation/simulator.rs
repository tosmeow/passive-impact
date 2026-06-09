use crate::models::{
    MultivariateEvent, MultivariateMarkovianIntensity, MultivariateSimulationResult,
};
use crate::simulation_helpers::{create_rng, sample_exponential, sample_uniform};

pub trait MarkovianProcessSimulator: MultivariateMarkovianIntensity {
    fn simulate(&self, t_max: f64, seed: Option<u64>) -> MultivariateSimulationResult;

    fn simulate_with_externals(
        &self,
        t_max: f64,
        external_trajectory: &MultivariateSimulationResult,
        seed: Option<u64>,
    ) -> MultivariateSimulationResult;
}

impl<P: MultivariateMarkovianIntensity> MarkovianProcessSimulator for P {
    fn simulate(&self, t_max: f64, seed: Option<u64>) -> MultivariateSimulationResult {
        let empty = MultivariateSimulationResult::new(self.dim());
        self.simulate_with_externals(t_max, &empty, seed)
    }

    fn simulate_with_externals(
        &self,
        t_max: f64,
        external_trajectory: &MultivariateSimulationResult,
        seed: Option<u64>,
    ) -> MultivariateSimulationResult {
        let mut rng = create_rng(seed);
        let k = self.dim();
        let mut result = MultivariateSimulationResult::new(k);
        let mut state = self.initial_state();
        let mut t = 0.0;
        let mut t_last = 0.0;
        let mut ext_idx = 0;

        while t < t_max {
            let t_ext = external_trajectory
                .events
                .get(ext_idx)
                .map(|e| e.time)
                .unwrap_or(f64::INFINITY);

            let lambda_star = self.intensity_upper_bound(&state, t, t_last);

            if lambda_star <= 0.0 && t_ext > t_max {
                break;
            }

            let dt = if lambda_star > 0.0 {
                sample_exponential(&mut rng, lambda_star)
            } else {
                f64::INFINITY
            };
            let t_prop = t + dt;

            if t_ext < t_prop && t_ext <= t_max {
                let ext_event = &external_trajectory.events[ext_idx];
                self.update_state(&mut state, ext_event.dim, ext_event.time, t_last);
                t_last = ext_event.time;
                t = ext_event.time;
                ext_idx += 1;
                continue;
            }

            if t_prop > t_max {
                break;
            }

            let intensities = self.intensities_from_state(&state, t_prop, t_last);
            let lambda_total: f64 = intensities.iter().sum();

            // Single uniform draw from [0, lambda_star) for both
            // accept/reject and dimension selection.
            let u = sample_uniform(&mut rng) * lambda_star;

            if u < lambda_total {
                let dim = if k == 1 {
                    0
                } else {
                    select_dimension(&intensities, u)
                };

                let event = MultivariateEvent { time: t_prop, dim };
                result.push(event);
                self.update_state(&mut state, dim, t_prop, t_last);
                t_last = t_prop;
            }

            t = t_prop;
        }

        result
    }
}

// Single uniform draw for accept/reject + dimension selection:
// u ~ U(0, lambda_star).
//   - u >= lambda_total  →  reject  (thinned region)
//   - u <  lambda_total  →  accept, and u is uniform on [0, lambda_total)
//     so we walk cumulative intensities with the same u to pick the dim.

#[inline]
fn select_dimension(intensities: &[f64], u: f64) -> usize {
    let mut cumsum = 0.0;
    for (i, &lambda_i) in intensities.iter().enumerate() {
        cumsum += lambda_i;
        if u < cumsum {
            return i;
        }
    }
    intensities.len() - 1
}

// Convenience free functions for backwards compatibility
pub fn simulate<P: MultivariateMarkovianIntensity>(
    model: &P,
    t_max: f64,
    seed: Option<u64>,
) -> MultivariateSimulationResult {
    model.simulate(t_max, seed)
}

pub fn simulate_with_externals<P: MultivariateMarkovianIntensity>(
    model: &P,
    t_max: f64,
    external_trajectory: &MultivariateSimulationResult,
    seed: Option<u64>,
) -> MultivariateSimulationResult {
    model.simulate_with_externals(t_max, external_trajectory, seed)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Test helper: simple independent Poisson processes
    struct IndependentPoissons {
        lambdas: Vec<f64>,
    }

    impl MultivariateMarkovianIntensity for IndependentPoissons {
        type State = ();

        fn dim(&self) -> usize {
            self.lambdas.len()
        }

        fn initial_state(&self) -> Self::State {
            ()
        }

        fn intensities_from_state(&self, _state: &Self::State, _t: f64, _t_last: f64) -> Vec<f64> {
            self.lambdas.clone()
        }

        fn update_state(&self, _state: &mut Self::State, _dim: usize, _t: f64, _t_prev: f64) {}
    }

    #[test]
    fn test_simulate_free_function() {
        let model = IndependentPoissons {
            lambdas: vec![1.0, 2.0, 0.5],
        };

        let result = simulate(&model, 100.0, Some(42));
        assert!(!result.is_empty());
    }

    #[test]
    fn test_simulate_trait_method() {
        let model = IndependentPoissons {
            lambdas: vec![1.0, 2.0, 0.5],
        };

        // Using extension trait method
        let result = model.simulate(100.0, Some(42));
        assert!(!result.is_empty());
    }

    #[test]
    fn test_both_produce_same_results() {
        let model = IndependentPoissons {
            lambdas: vec![1.0, 2.0],
        };

        let result1 = simulate(&model, 50.0, Some(123));
        let result2 = model.simulate(50.0, Some(123));

        assert_eq!(result1.events.len(), result2.events.len());
        for (e1, e2) in result1.events.iter().zip(result2.events.iter()) {
            assert_eq!(e1.time, e2.time);
            assert_eq!(e1.dim, e2.dim);
        }
    }

    #[test]
    fn test_simulate_equals_simulate_with_empty_externals() {
        // Test k=1 case (single dimension)
        let model_k1 = IndependentPoissons { lambdas: vec![2.0] };

        let empty = MultivariateSimulationResult::new(1);
        let result1 = model_k1.simulate(50.0, Some(42));
        let result2 = model_k1.simulate_with_externals(50.0, &empty, Some(42));

        assert_eq!(result1.events.len(), result2.events.len());
        for (e1, e2) in result1.events.iter().zip(result2.events.iter()) {
            assert_eq!(e1.time, e2.time);
            assert_eq!(e1.dim, e2.dim);
        }

        // Test k>1 case (multiple dimensions)
        let model_k3 = IndependentPoissons {
            lambdas: vec![1.0, 2.0, 0.5],
        };

        let empty3 = MultivariateSimulationResult::new(3);
        let result3 = model_k3.simulate(50.0, Some(42));
        let result4 = model_k3.simulate_with_externals(50.0, &empty3, Some(42));

        assert_eq!(result3.events.len(), result4.events.len());
        for (e1, e2) in result3.events.iter().zip(result4.events.iter()) {
            assert_eq!(e1.time, e2.time);
            assert_eq!(e1.dim, e2.dim);
        }
    }

    // -------------------------------------------------------------------
    // Statistical correctness tests
    // -------------------------------------------------------------------

    // Poisson: verify per-dim mean rates match theory.
    #[test]
    fn test_poisson_mean_rate() {
        let model = IndependentPoissons {
            lambdas: vec![1.0, 2.0, 0.5],
        };
        let t_max = 200.0;
        let n_runs: usize = 500;
        let k = model.lambdas.len();

        let mut counts = vec![vec![0usize; n_runs]; k];

        for i in 0..n_runs {
            let r = simulate(&model, t_max, Some(i as u64));
            for d in 0..k {
                counts[d][i] = r.events_by_dim[d].len();
            }
        }

        for d in 0..k {
            let mean: f64 = counts[d].iter().sum::<usize>() as f64 / n_runs as f64;
            let expected = model.lambdas[d] * t_max;
            let tol = 5.0 * (expected).sqrt() / (n_runs as f64).sqrt();

            assert!(
                (mean - expected).abs() < tol,
                "dim {} mean {:.2} vs expected {:.2} (tol {:.2})",
                d,
                mean,
                expected,
                tol,
            );
        }
    }

    // Hawkes: verify mean rate matches stationary theory.
    #[test]
    fn test_hawkes_mean_rate() {
        use crate::models::MultiExponentialHawkes;

        let mu = 1.0;
        let alpha = vec![0.3, 0.2];
        let beta = vec![1.0, 5.0];
        let hawkes = MultiExponentialHawkes::new(mu, alpha.clone(), beta.clone());

        let branching_ratio: f64 = alpha.iter().zip(&beta).map(|(a, b)| a / b).sum();
        let expected_rate = mu / (1.0 - branching_ratio);

        let t_max = 500.0;
        let n_runs: usize = 500;

        let mut counts = vec![0usize; n_runs];
        for i in 0..n_runs {
            counts[i] = simulate(&hawkes, t_max, Some(i as u64)).events.len();
        }

        let mean: f64 = counts.iter().sum::<usize>() as f64 / n_runs as f64;
        let expected = expected_rate * t_max;
        let tol = 5.0 * (expected).sqrt() / (n_runs as f64).sqrt();

        assert!(
            (mean - expected).abs() < tol,
            "Hawkes mean {:.2} vs expected {:.2} (tol {:.2})",
            mean,
            expected,
            tol,
        );
    }

    // Multi-dim process with thinning: verify per-dim proportions.
    #[test]
    fn test_multidim_proportions() {
        struct DecayingProcess;

        impl MultivariateMarkovianIntensity for DecayingProcess {
            type State = f64;

            fn dim(&self) -> usize {
                3
            }
            fn initial_state(&self) -> Self::State {
                0.0
            }

            fn intensities_from_state(
                &self,
                &t_last_event: &Self::State,
                t: f64,
                _t_last: f64,
            ) -> Vec<f64> {
                let decay = (-0.5 * (t - t_last_event)).exp();
                vec![2.0 * decay, 1.0 * decay, 0.5 * decay]
            }

            fn update_state(&self, state: &mut Self::State, _dim: usize, t: f64, _t_prev: f64) {
                *state = t;
            }

            fn intensity_upper_bound(&self, state: &Self::State, t: f64, t_last: f64) -> f64 {
                self.intensities_from_state(state, t, t_last).iter().sum()
            }
        }

        let model = DecayingProcess;
        let t_max = 200.0;
        let n_runs: usize = 500;

        let mut dim_counts = vec![0usize; 3];
        for i in 0..n_runs {
            for e in &simulate(&model, t_max, Some(i as u64)).events {
                dim_counts[e.dim] += 1;
            }
        }

        let total: usize = dim_counts.iter().sum();
        let expected_fracs = [4.0 / 7.0, 2.0 / 7.0, 1.0 / 7.0];

        for d in 0..3 {
            let frac = dim_counts[d] as f64 / total as f64;
            assert!(
                (frac - expected_fracs[d]).abs() < 0.05,
                "dim {} fraction {:.4} vs expected {:.4}",
                d,
                frac,
                expected_fracs[d],
            );
        }
    }

    // Simulation with externals: verify it runs and produces events.
    #[test]
    fn test_with_externals() {
        use crate::models::MultiExponentialHawkes;

        let hawkes = MultiExponentialHawkes::new(1.0, vec![0.3], vec![2.0]);
        let external_hawkes = MultiExponentialHawkes::new(0.5, vec![0.1], vec![2.0]);

        let t_max = 200.0;
        let n_runs: usize = 300;

        let mut counts = vec![0usize; n_runs];
        for i in 0..n_runs {
            let ext = simulate(&external_hawkes, t_max, Some(i as u64 + 2_000_000));
            counts[i] = simulate_with_externals(&hawkes, t_max, &ext, Some(i as u64))
                .events
                .len();
        }

        let mean: f64 = counts.iter().sum::<usize>() as f64 / n_runs as f64;
        assert!(
            mean > 0.0,
            "Expected non-zero mean event count with externals"
        );
    }
}
