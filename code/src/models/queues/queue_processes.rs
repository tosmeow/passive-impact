use crate::models::{MarkovianProcess, QueueEvent, QueuePath};
use crate::models::{MultivariateEvent, MultivariateSimulationResult};

pub struct QueueProcess;

impl QueueProcess {
    pub fn new<F, G>(
        q0: f64,
        mu: f64,
        alpha: Vec<f64>,
        beta: Vec<f64>,
        lambda_l: F,
        lambda_c: G,
    ) -> MarkovianProcess
    where
        F: Fn(f64) -> f64 + Send + Sync + 'static,
        G: Fn(f64) -> f64 + Send + Sync + 'static,
    {
        let m = alpha.len();
        let mut initial_state = vec![q0];
        initial_state.extend(vec![0.0; m]);
        Self::new_with_state(initial_state, mu, alpha, beta, lambda_l, lambda_c)
    }

    // Create a queue process with a specified initial state.
    //
    // State format: `[q, hawkes_0, hawkes_1, ..., hawkes_{m-1}]`
    //
    // This allows starting from a pre-excited Hawkes state, avoiding warmup.
    pub fn new_with_state<F, G>(
        initial_state: Vec<f64>,
        mu: f64,
        alpha: Vec<f64>,
        beta: Vec<f64>,
        lambda_l: F,
        lambda_c: G,
    ) -> MarkovianProcess
    where
        F: Fn(f64) -> f64 + Send + Sync + 'static,
        G: Fn(f64) -> f64 + Send + Sync + 'static,
    {
        let m = alpha.len();
        assert!(
            initial_state.len() == 1 + m,
            "initial_state must have length 1 + alpha.len()"
        );
        assert!(
            initial_state[0] >= 0.0,
            "initial queue must be non-negative"
        );
        assert!(mu >= 0.0, "mu must be non-negative");
        assert_eq!(
            alpha.len(),
            beta.len(),
            "alpha and beta must have same length"
        );
        assert!(!alpha.is_empty(), "must have at least one Hawkes component");

        let beta_lambda = beta.clone();
        let alpha_state = alpha.clone();
        let beta_state = beta.clone();

        MarkovianProcess::new(
            3,
            initial_state,
            move |state: &[f64], t: f64, t_last: f64| {
                let q = state[0];
                let dt = t - t_last;

                let hawkes_excitation: f64 = state[1..]
                    .iter()
                    .zip(beta_lambda.iter())
                    .map(|(&s, &b)| s * (-b * dt).exp())
                    .sum();
                let lambda_n = mu + hawkes_excitation;

                vec![
                    lambda_l(q).max(0.0), // dim 0: limit orders
                    lambda_c(q).max(0.0), // dim 1: cancel orders
                    lambda_n,             // dim 2: market orders (Hawkes)
                ]
            },
            move |state: &[f64], event: &MultivariateEvent, t: f64, t_prev: f64| {
                let dt = t - t_prev;
                let q = state[0];

                // Decay Hawkes state
                let mut new_state = vec![q];
                for (i, &s) in state[1..].iter().enumerate() {
                    new_state.push(s * (-beta_state[i] * dt).exp());
                }

                match event.dim {
                    0 => {
                        // Limit order: q += 1
                        new_state[0] += 1.0;
                    }
                    1 => {
                        // Cancel order: q -= 1 (saturating)
                        new_state[0] = (new_state[0] - 1.0).max(0.0);
                    }
                    2 => {
                        // Market order: q -= 1, Hawkes jumps
                        new_state[0] = (new_state[0] - 1.0).max(0.0);
                        for (s, &a) in new_state[1..].iter_mut().zip(alpha_state.iter()) {
                            *s += a;
                        }
                    }
                    _ => {}
                }

                new_state
            },
        )
    }

    /// Get queue value from state
    pub fn queue_from_state(state: &[f64]) -> f64 {
        state[0]
    }

    pub fn result_to_queue_path(
        result: &MultivariateSimulationResult,
        initial_q: u32,
    ) -> QueuePath {
        let mut events = vec![QueueEvent {
            queue_event: 0,
            queue_size: initial_q,
            time: 0.0,
        }];

        let mut q = initial_q as i64;
        for event in &result.events {
            match event.dim {
                0 => q += 1,             // Limit order: q += 1
                1 => q = (q - 1).max(0), // Cancel order: q -= 1
                2 => q = (q - 1).max(0), // Market order: q -= 1
                _ => {}
            }
            events.push(QueueEvent {
                queue_event: event.dim as u32,
                queue_size: q as u32,
                time: event.time,
            });
        }

        QueuePath { events }
    }
}

pub struct AffineQueueProcess;

impl AffineQueueProcess {
    // Create an affine queue process with Hawkes-driven market orders.
    //
    // This is the full coupled process where market order intensity
    // is computed from an internal Hawkes state. Hawkes components start at 0.
    // Use `new_with_state` to specify a pre-excited initial state.
    pub fn new(
        q0: f64,
        a_l: f64,
        b_l: f64,
        a_c: f64,
        b_c: f64,
        mu: f64,
        alpha: Vec<f64>,
        beta: Vec<f64>,
    ) -> MarkovianProcess {
        QueueProcess::new(
            q0,
            mu,
            alpha,
            beta,
            move |q| a_l + b_l * q,
            move |q| a_c + b_c * q,
        )
    }

    // Create an affine queue process with a specified initial state.
    //
    // State format: `[q, hawkes_0, hawkes_1, ..., hawkes_{m-1}]`
    //
    // This allows starting from a pre-excited Hawkes state, avoiding warmup.

    pub fn new_with_state(
        initial_state: Vec<f64>,
        a_l: f64,
        b_l: f64,
        a_c: f64,
        b_c: f64,
        mu: f64,
        alpha: Vec<f64>,
        beta: Vec<f64>,
    ) -> MarkovianProcess {
        QueueProcess::new_with_state(
            initial_state,
            mu,
            alpha,
            beta,
            move |q| a_l + b_l * q,
            move |q| a_c + b_c * q,
        )
    }

    // Create a queue-only process without internal Hawkes state.
    //
    // Market orders (dim 2) have intensity 0 and must be provided
    // as external events. This is more efficient when the Hawkes
    // process can be pre-simulated separately.
    //
    // State is just `[q]` instead of `[q, h0, h1, ..., h_k]`.
    pub fn new_queue(q0: f64, a_l: f64, b_l: f64, a_c: f64, b_c: f64) -> MarkovianProcess {
        MarkovianProcess::new(
            3,        // Still 3 dimensions: limit (0), cancel (1), market (2)
            vec![q0], // State is just queue size
            move |state: &[f64], _t: f64, _t_last: f64| {
                let q = state[0];
                vec![
                    (a_l + b_l * q).max(0.0), // dim 0: limit orders
                    (a_c + b_c * q).max(0.0), // dim 1: cancel orders
                    0.0,                      // dim 2: market orders (external)
                ]
            },
            move |state: &[f64], event: &MultivariateEvent, _t: f64, _t_prev: f64| {
                let q = state[0];
                let new_q = match event.dim {
                    0 => q + 1.0,            // Limit order: q += 1
                    1 => (q - 1.0).max(0.0), // Cancel order: q -= 1
                    2 => (q - 1.0).max(0.0), // Market order: q -= 1
                    _ => q,
                };
                vec![new_q]
            },
        )
    }

    /// Get queue value from state
    pub fn queue_from_state(state: &[f64]) -> f64 {
        QueueProcess::queue_from_state(state)
    }

    /// Compute λ^L(q) = a_l + b_l * q (static helper)
    pub fn lambda_l(a_l: f64, b_l: f64, q: f64) -> f64 {
        (a_l + b_l * q).max(0.0)
    }

    /// Compute λ^C(q) = a_c + b_c * q (static helper)
    pub fn lambda_c(a_c: f64, b_c: f64, q: f64) -> f64 {
        (a_c + b_c * q).max(0.0)
    }

    // Compute c_lambda from affine slopes.
    //
    // For affine intensities λ^L(q) = a_l + b_l*q and λ^C(q) = a_c + b_c*q,
    // the parameter c_lambda used in conditional impact is derived from:
    // slope_L - slope_C = b_l - b_c = -c_lambda
    //
    // Therefore: c_lambda = b_c - b_l
    pub fn c_lambda(b_l: f64, b_c: f64) -> f64 {
        b_c - b_l
    }

    // Compute the stationary initial state for the coupled queue+Hawkes process.
    //
    // Returns `[q0, h_0_mean, h_1_mean, ..., h_{m-1}_mean]` where:
    // - `q0` is the provided initial queue size
    // - `h_i_mean = alpha[i] * mu / (beta[i] * (1 - branching_ratio))`
    //
    // This allows starting the simulation at the Hawkes stationary distribution,
    // avoiding warmup time.
    pub fn stationary_state(q0: f64, mu: f64, alpha: &[f64], beta: &[f64]) -> Vec<f64> {
        let branching_ratio: f64 = alpha.iter().zip(beta).map(|(a, b)| a / b).sum();
        assert!(
            branching_ratio < 1.0,
            "Hawkes process is not stationary (branching_ratio >= 1)"
        );

        let mut state = vec![q0];
        state.extend(
            alpha
                .iter()
                .zip(beta)
                .map(|(a, b)| a * mu / (b * (1.0 - branching_ratio))),
        );
        state
    }

    /// Convert a MultivariateSimulationResult to a QueuePath.
    pub fn result_to_queue_path(
        result: &MultivariateSimulationResult,
        initial_q: u32,
    ) -> QueuePath {
        QueueProcess::result_to_queue_path(result, initial_q)
    }
}
