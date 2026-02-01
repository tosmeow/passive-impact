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
        Self::new_with_state(
            self.inner.initial_state(),
            self.mu,
            self.alpha.clone(),
            self.beta.clone(),
        )
    }
}

impl MultiExponentialHawkes {
    /// Create a new multi-exponential Hawkes process.
    ///
    /// State starts at 0 for all components. Use `new_with_state` to specify
    /// a pre-excited initial state.
    pub fn new(mu: f64, alpha: Vec<f64>, beta: Vec<f64>) -> Self {
        let m = alpha.len();
        Self::new_with_state(vec![0.0; m], mu, alpha, beta)
    }

    /// Create a new multi-exponential Hawkes process with a specified initial state.
    ///
    /// State format: `[h_0, h_1, ..., h_{m-1}]` where h_i is the excitation for component i.
    ///
    /// This allows starting from a pre-excited state, avoiding warmup.
    /// For example, to start at the stationary mean:
    /// ```ignore
    /// // Stationary mean of component i: alpha[i] * mu / (beta[i] * (1 - branching_ratio))
    /// let branching_ratio: f64 = alpha.iter().zip(&beta).map(|(a, b)| a / b).sum();
    /// let initial_state: Vec<f64> = alpha.iter().zip(&beta)
    ///     .map(|(a, b)| a * mu / (b * (1.0 - branching_ratio)))
    ///     .collect();
    /// ```
    pub fn new_with_state(initial_state: Vec<f64>, mu: f64, alpha: Vec<f64>, beta: Vec<f64>) -> Self {
        assert!(mu >= 0.0, "mu must be non-negative");
        assert_eq!(alpha.len(), beta.len(), "alpha and beta must have same length");
        assert!(!alpha.is_empty(), "must have at least one component");
        assert!(alpha.iter().all(|&a| a >= 0.0), "all alpha must be non-negative");
        assert!(beta.iter().all(|&b| b > 0.0), "all beta must be positive");
        assert_eq!(initial_state.len(), alpha.len(), "initial_state must have length alpha.len()");

        let beta_lambda = beta.to_vec();
        let alpha_state = alpha.to_vec();
        let beta_state = beta.to_vec();

        let inner = MarkovianProcess::new(
            1,
            initial_state,
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

    /// Compute the stationary mean of the Hawkes state components.
    ///
    /// Returns `[h_0_mean, h_1_mean, ..., h_{m-1}_mean]` where:
    /// `h_i_mean = alpha[i] * mu / (beta[i] * (1 - branching_ratio))`
    ///
    /// Panics if branching_ratio >= 1 (non-stationary process).
    pub fn stationary_state(&self) -> Vec<f64> {
        let branching_ratio: f64 = self.alpha.iter()
            .zip(&self.beta)
            .map(|(a, b)| a / b)
            .sum();
        assert!(branching_ratio < 1.0, "process is not stationary (branching_ratio >= 1)");

        self.alpha.iter()
            .zip(&self.beta)
            .map(|(a, b)| a * self.mu / (b * (1.0 - branching_ratio)))
            .collect()
    }

    pub fn m(&self) -> usize {
        self.alpha.len()
    }

    /// Get the inner MarkovianProcess for use with simulate()
    pub fn as_markovian_process(&self) -> &MarkovianProcess {
        &self.inner
    }
}