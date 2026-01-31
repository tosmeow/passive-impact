use super::markovian_process::{MarkovianProcess};
use super::multivariate_process::{MultivariateMarkovianIntensity, MultivariateEvent};

/// Multi-exponential Hawkes process built on MarkovianProcess.
pub struct MultiExponentialHawkes {
    pub mu: f64,
    pub alpha: Vec<f64>,
    pub beta: Vec<f64>,
    inner: MarkovianProcess,
}

impl MultivariateMarkovianIntensity for MultiExponentialHawkes {
    type State = Vec<f64>;

    fn dim(&self) -> usize {
        self.inner.dim()
    }

    fn initial_state(&self) -> Self::State {
        self.inner.initial_state()
    }

    fn intensities_from_state(&self, state: &Self::State, t: f64, t_last: f64) -> Vec<f64> {
        self.inner.intensities_from_state(state, t, t_last)
    }

    fn update_state(&self, state: &mut Self::State, dim: usize, t: f64, t_prev: f64) {
        self.inner.update_state(state, dim, t, t_prev)
    }
}

impl Clone for MultiExponentialHawkes {
    fn clone(&self) -> Self {
        Self::new(self.mu, self.alpha.clone(), self.beta.clone())
    }
}

impl MultiExponentialHawkes {
    pub fn new(mu: f64, alpha: Vec<f64>, beta: Vec<f64>) -> Self {
        assert!(mu >= 0.0, "mu must be non-negative");
        assert_eq!(alpha.len(), beta.len(), "alpha and beta must have same length");
        assert!(!alpha.is_empty(), "must have at least one component");
        assert!(alpha.iter().all(|&a| a >= 0.0), "all alpha must be non-negative");
        assert!(beta.iter().all(|&b| b > 0.0), "all beta must be positive");

        let m = alpha.len();
        let beta_lambda = beta.to_vec();
        let alpha_state = alpha.to_vec();
        let beta_state = beta.to_vec();

        let inner = MarkovianProcess::new(
            1,
            vec![0.0; m],
            move |state: &[f64], t: f64, t_last: f64| {
                let dt = t - t_last;
                let excitation: f64 = state
                    .iter()
                    .zip(beta_lambda.iter())
                    .map(|(&s, &b)| s * (-b * dt).exp())
                    .sum();
                vec![mu + excitation]
            },
            move |state: &[f64], _event: &MultivariateEvent, t: f64, t_prev: f64| {
                let dt = t - t_prev;
                state
                    .iter()
                    .zip(alpha_state.iter().zip(beta_state.iter()))
                    .map(|(&s, (&a, &b))| s * (-b * dt).exp() + a)
                    .collect()
            });

        Self { mu, alpha, beta, inner }
    }

    pub fn m(&self) -> usize {
        self.alpha.len()
    }

    /// Get the inner MarkovianProcess for use with simulate()
    pub fn as_markovian_process(&self) -> &MarkovianProcess {
        &self.inner
    }
}