use super::model::{PointProcess, MarkovianIntensity};

#[derive(Debug, Clone)]
pub struct MultiExponentialHawkes {
    // Structure to collect the parameters for an intensity \lambda_t = \mu + \int_0^t (\sum_i \alpha[i] * exp(-(t-s) * \beta[i])) dN_s.
    pub mu: f64,
    pub alpha: Vec<f64>,
    pub beta: Vec<f64>,
}

impl MultiExponentialHawkes {
    pub fn new(mu: f64, alpha: Vec<f64>, beta: Vec<f64>) -> Self {
        assert!(mu >= 0.0, "mu must be non-negative");
        assert_eq!(alpha.len(), beta.len(), "alpha and beta must have same length");
        assert!(!alpha.is_empty(), "must have at least one component");
        assert!(alpha.iter().all(|&a| a >= 0.0), "all alpha must be non-negative");
        assert!(beta.iter().all(|&b| b > 0.0), "all beta must be positive");

        Self { mu, alpha, beta }
    }

    pub fn m(&self) -> usize {
        self.alpha.len()
    }
    #[inline]
    fn kernel(&self, m: usize, dt: f64) -> f64 {
        self.alpha[m] * (-self.beta[m] * dt).exp()
    }
}

impl PointProcess for MultiExponentialHawkes {
    fn intensity(&self, t: f64, events: &[f64]) -> f64 {
        let mut excitation = 0.0;

        for &ti in events.iter().filter(|&&ti| ti < t) {
            let dt = t - ti;
            for m in 0..self.m() {
                excitation += self.kernel(m, dt);
            }
        }

        self.mu + excitation
    }

    fn intensity_upper_bound(&self, t: f64, events: &[f64]) -> f64 {
        self.intensity(t, events)
    }

    fn baseline_intensity(&self) -> f64 {
        self.mu
    }

    fn num_components(&self) -> usize {
        self.m()
    }
}


impl MarkovianIntensity for MultiExponentialHawkes {
    type State = Vec<f64>;

    fn initial_state(&self) -> Self::State {
        vec![0.0; self.m()]
    }

    fn update_state(&self, state: &mut Self::State, t: f64, t_prev: f64) {
        let dt = t - t_prev;
        for m in 0..self.m() {
            state[m] = state[m] * (-self.beta[m] * dt).exp() + self.alpha[m];
        }
    }

    fn intensity_from_state(&self, state: &Self::State, t: f64, t_last: f64) -> f64 {
        let dt = t - t_last;
        let excitation: f64 = state
            .iter()
            .zip(self.beta.iter())
            .map(|(&s, &b)| s * (-b * dt).exp())
            .sum();

        self.mu + excitation
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_multi_exponential_creation() {
        let hawkes = MultiExponentialHawkes::new(
            0.5,
            vec![0.3, 0.2],
            vec![1.0, 5.0],
        );
        assert_relative_eq!(hawkes.mu, 0.5);
        assert_eq!(hawkes.m(), 2);
    }

    #[test]
    fn test_baseline_intensity() {
        let hawkes = MultiExponentialHawkes::new(0.5, vec![0.3, 0.2], vec![1.0, 5.0]);
        assert_relative_eq!(hawkes.intensity(0.0, &[]), 0.5);
    }

    #[test]
    fn test_intensity_after_event() {
        let hawkes = MultiExponentialHawkes::new(0.5, vec![0.3, 0.2], vec![1.0, 5.0]);
        let events = vec![0.0];

        // Right after event at t=0
        let intensity = hawkes.intensity(0.001, &events);
        let expected_jump = 0.3 + 0.2; // Sum of alphas
        assert!(intensity > 0.5);
        assert!(intensity < 0.5 + expected_jump + 0.01);
    }

    #[test]
    fn test_markovian_equivalence() {
        let hawkes = MultiExponentialHawkes::new(0.5, vec![0.3, 0.2], vec![1.0, 5.0]);
        let events = vec![0.5, 1.2, 2.1];

        // Compute intensity at t=3.0 using direct method
        let intensity_direct = hawkes.intensity(3.0, &events);

        // Compute using Markovian recursion
        let mut state = hawkes.initial_state();
        let mut t_prev = 0.0;
        for &t in &events {
            hawkes.update_state(&mut state, t, t_prev);
            t_prev = t;
        }
        let intensity_markov = hawkes.intensity_from_state(&state, 3.0, *events.last().unwrap());

        assert_relative_eq!(intensity_direct, intensity_markov, epsilon = 1e-10);
    }
}
