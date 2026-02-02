use crate::models::{MultivariateMarkovianIntensity, MultivariateEvent, MultivariateSimulationResult};
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
            let t_ext = external_trajectory.events
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

            let accept_prob = if lambda_star > 0.0 {
                lambda_total / lambda_star
            } else {
                0.0
            };

            if sample_uniform(&mut rng) < accept_prob {
                // Skip dimension selection for k=1 (avoids extra random draw)
                let dim = if k == 1 {
                    0
                } else {
                    select_dimension(&intensities, lambda_total, &mut rng)
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

#[inline]
fn select_dimension(intensities: &[f64], lambda_total: f64, rng: &mut rand::rngs::StdRng) -> usize {
    let u = sample_uniform(rng) * lambda_total;
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
        let model_k1 = IndependentPoissons {
            lambdas: vec![2.0],
        };

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
}
