use super::markovian_process::{MarkovianProcess, QueuePath, QueueEvent};
use super::multivariate_process::{MultivariateEvent, MultivariateSimulationResult};

pub struct QueueProcess;

impl QueueProcess {
    /// Create a queue process as a MarkovianProcess.
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
        assert!(q0 >= 0.0, "initial queue must be non-negative");
        assert!(mu >= 0.0, "mu must be non-negative");
        assert_eq!(alpha.len(), beta.len(), "alpha and beta must have same length");
        assert!(!alpha.is_empty(), "must have at least one Hawkes component");

        let m = alpha.len();
        // State: [q, hawkes_0, hawkes_1, ..., hawkes_{m-1}]
        let mut initial_state = vec![q0];
        initial_state.extend(vec![0.0; m]);

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
                    lambda_l(q).max(0.0),  // dim 0: limit orders
                    lambda_c(q).max(0.0),  // dim 1: cancel orders
                    lambda_n,               // dim 2: market orders (Hawkes)
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

    pub fn result_to_queue_path(result: &MultivariateSimulationResult, initial_q: u32) -> QueuePath {
        let mut events = vec![QueueEvent {
            queue_event: 0,
            queue_size: initial_q,
            time: 0.0,
        }];

        let mut q = initial_q as i64;
        for event in &result.events {
            match event.dim {
                0 => q += 1,              // Limit order: q += 1
                1 => q = (q - 1).max(0),  // Cancel order: q -= 1
                2 => q = (q - 1).max(0),  // Market order: q -= 1
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
    /// Create an affine queue process as a MarkovianProcess.
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

    /// Compute c_lambda from affine slopes.
    ///
    /// For affine intensities λ^L(q) = a_l + b_l*q and λ^C(q) = a_c + b_c*q,
    /// the parameter c_lambda used in conditional impact is derived from:
    /// slope_L - slope_C = b_l - b_c = -c_lambda
    ///
    /// Therefore: c_lambda = b_c - b_l
    pub fn c_lambda(b_l: f64, b_c: f64) -> f64 {
        b_c - b_l
    }

    /// Convert a MultivariateSimulationResult to a QueuePath.
    pub fn result_to_queue_path(result: &MultivariateSimulationResult, initial_q: u32) -> QueuePath {
        QueueProcess::result_to_queue_path(result, initial_q)
    }
}