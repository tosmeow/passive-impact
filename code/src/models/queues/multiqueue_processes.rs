use crate::models::{MarkovianProcess, QueueEvent, QueuePath};
use crate::models::{MultivariateEvent, MultivariateSimulationResult};
pub struct BidAskQueueProcess;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BidAskDimension {
    LimitAsk = 0,
    CancelAsk = 1,
    MarketAsk = 2,
    LimitBid = 3,
    CancelBid = 4,
    MarketBid = 5,
}

impl BidAskDimension {
    pub fn from_usize(dim: usize) -> Option<Self> {
        match dim {
            0 => Some(Self::LimitAsk),
            1 => Some(Self::CancelAsk),
            2 => Some(Self::MarketAsk),
            3 => Some(Self::LimitBid),
            4 => Some(Self::CancelBid),
            5 => Some(Self::MarketBid),
            _ => None,
        }
    }
}

// Two-sided queue path for bid-ask model
#[derive(Clone)]
pub struct BidAskQueuePath {
    pub ask: QueuePath,
    pub bid: QueuePath,
}

impl BidAskQueueProcess {
    pub fn new<F1, F2, F3, F4>(
        q0_a: f64,
        q0_b: f64,
        mu_a: f64,
        mu_b: f64,
        alpha_a: Vec<f64>,
        beta_a: Vec<f64>,
        alpha_b: Vec<f64>,
        beta_b: Vec<f64>,
        lambda_l_a: F1,
        lambda_c_a: F2,
        lambda_l_b: F3,
        lambda_c_b: F4,
    ) -> MarkovianProcess
    where
        F1: Fn(f64, f64) -> f64 + Send + Sync + 'static,
        F2: Fn(f64, f64) -> f64 + Send + Sync + 'static,
        F3: Fn(f64, f64) -> f64 + Send + Sync + 'static,
        F4: Fn(f64, f64) -> f64 + Send + Sync + 'static,
    {
        let m_a = alpha_a.len();
        let m_b = alpha_b.len();

        let mut initial_state = vec![q0_a, q0_b];
        initial_state.extend(vec![0.0; m_a]);
        initial_state.extend(vec![0.0; m_b]);

        Self::new_with_state(
            initial_state,
            mu_a,
            mu_b,
            alpha_a,
            beta_a,
            alpha_b,
            beta_b,
            lambda_l_a,
            lambda_c_a,
            lambda_l_b,
            lambda_c_b,
        )
    }

    pub fn new_with_state<F1, F2, F3, F4>(
        initial_state: Vec<f64>,
        mu_a: f64,
        mu_b: f64,
        alpha_a: Vec<f64>,
        beta_a: Vec<f64>,
        alpha_b: Vec<f64>,
        beta_b: Vec<f64>,
        lambda_l_a: F1,
        lambda_c_a: F2,
        lambda_l_b: F3,
        lambda_c_b: F4,
    ) -> MarkovianProcess
    where
        F1: Fn(f64, f64) -> f64 + Send + Sync + 'static,
        F2: Fn(f64, f64) -> f64 + Send + Sync + 'static,
        F3: Fn(f64, f64) -> f64 + Send + Sync + 'static,
        F4: Fn(f64, f64) -> f64 + Send + Sync + 'static,
    {
        let m_a = alpha_a.len();
        let m_b = alpha_b.len();

        assert_eq!(
            initial_state.len(),
            2 + m_a + m_b,
            "initial_state must have length 2 + alpha_a.len() + alpha_b.len()"
        );
        assert!(
            initial_state[0] >= 0.0,
            "initial ask queue must be non-negative"
        );
        assert!(
            initial_state[1] >= 0.0,
            "initial bid queue must be non-negative"
        );
        assert_eq!(
            alpha_a.len(),
            beta_a.len(),
            "alpha_a and beta_a must have same length"
        );
        assert_eq!(
            alpha_b.len(),
            beta_b.len(),
            "alpha_b and beta_b must have same length"
        );

        let beta_a_lambda = beta_a.clone();
        let beta_b_lambda = beta_b.clone();
        let alpha_a_state = alpha_a.clone();
        let alpha_b_state = alpha_b.clone();
        let beta_a_state = beta_a.clone();
        let beta_b_state = beta_b.clone();

        MarkovianProcess::new(
            6,
            initial_state,
            move |state: &[f64], t: f64, t_last: f64| {
                let q_a = state[0];
                let q_b = state[1];
                let dt = t - t_last;

                let hawkes_a: f64 = state[2..2 + m_a]
                    .iter()
                    .zip(beta_a_lambda.iter())
                    .map(|(&s, &b)| s * (-b * dt).exp())
                    .sum();
                let lambda_n_a = mu_a + hawkes_a;

                let hawkes_b: f64 = state[2 + m_a..]
                    .iter()
                    .zip(beta_b_lambda.iter())
                    .map(|(&s, &b)| s * (-b * dt).exp())
                    .sum();
                let lambda_n_b = mu_b + hawkes_b;

                vec![
                    lambda_l_a(q_a, q_b).max(0.0),
                    lambda_c_a(q_a, q_b).max(0.0),
                    lambda_n_a,
                    lambda_l_b(q_a, q_b).max(0.0),
                    lambda_c_b(q_a, q_b).max(0.0),
                    lambda_n_b,
                ]
            },
            move |state: &[f64], event: &MultivariateEvent, t: f64, t_prev: f64| {
                let dt = t - t_prev;
                let q_a = state[0];
                let q_b = state[1];

                let mut new_state = vec![q_a, q_b];

                for (i, &s) in state[2..2 + m_a].iter().enumerate() {
                    new_state.push(s * (-beta_a_state[i] * dt).exp());
                }

                for (i, &s) in state[2 + m_a..].iter().enumerate() {
                    new_state.push(s * (-beta_b_state[i] * dt).exp());
                }

                match event.dim {
                    0 => {
                        new_state[0] += 1.0;
                    }
                    1 => {
                        new_state[0] = (new_state[0] - 1.0).max(0.0);
                    }
                    2 => {
                        new_state[0] = (new_state[0] - 1.0).max(0.0);
                        for (i, &a) in alpha_a_state.iter().enumerate() {
                            new_state[2 + i] += a;
                        }
                    }
                    3 => {
                        new_state[1] += 1.0;
                    }
                    4 => {
                        new_state[1] = (new_state[1] - 1.0).max(0.0);
                    }
                    5 => {
                        new_state[1] = (new_state[1] - 1.0).max(0.0);
                        for (i, &a) in alpha_b_state.iter().enumerate() {
                            new_state[2 + m_a + i] += a;
                        }
                    }
                    _ => {}
                }

                new_state
            },
        )
    }

    pub fn ask_queue_from_state(state: &[f64]) -> f64 {
        state[0]
    }

    pub fn bid_queue_from_state(state: &[f64]) -> f64 {
        state[1]
    }

    pub fn result_to_queue_paths(
        result: &MultivariateSimulationResult,
        initial_q_a: u32,
        initial_q_b: u32,
    ) -> BidAskQueuePath {
        let mut ask_events = vec![QueueEvent {
            queue_event: 0,
            queue_size: initial_q_a,
            time: 0.0,
        }];
        let mut bid_events = vec![QueueEvent {
            queue_event: 0,
            queue_size: initial_q_b,
            time: 0.0,
        }];

        let mut q_a = initial_q_a as i64;
        let mut q_b = initial_q_b as i64;

        for event in &result.events {
            match event.dim {
                0 => {
                    q_a += 1;
                    ask_events.push(QueueEvent {
                        queue_event: 0,
                        queue_size: q_a as u32,
                        time: event.time,
                    });
                }
                1 => {
                    q_a = (q_a - 1).max(0);
                    ask_events.push(QueueEvent {
                        queue_event: 1,
                        queue_size: q_a as u32,
                        time: event.time,
                    });
                }
                2 => {
                    q_a = (q_a - 1).max(0);
                    ask_events.push(QueueEvent {
                        queue_event: 2,
                        queue_size: q_a as u32,
                        time: event.time,
                    });
                }
                3 => {
                    q_b += 1;
                    bid_events.push(QueueEvent {
                        queue_event: 0,
                        queue_size: q_b as u32,
                        time: event.time,
                    });
                }
                4 => {
                    q_b = (q_b - 1).max(0);
                    bid_events.push(QueueEvent {
                        queue_event: 1,
                        queue_size: q_b as u32,
                        time: event.time,
                    });
                }
                5 => {
                    q_b = (q_b - 1).max(0);
                    bid_events.push(QueueEvent {
                        queue_event: 2,
                        queue_size: q_b as u32,
                        time: event.time,
                    });
                }
                _ => {}
            }
        }

        BidAskQueuePath {
            ask: QueuePath { events: ask_events },
            bid: QueuePath { events: bid_events },
        }
    }
}

pub struct AffineBidAskQueueProcess;

#[derive(Clone, Debug)]
pub struct AffineIntensityParams {
    pub a: f64,
    pub b_a: f64,
    pub b_b: f64,
}

impl AffineIntensityParams {
    pub fn new(a: f64, b_a: f64, b_b: f64) -> Self {
        Self { a, b_a, b_b }
    }

    pub fn compute(&self, q_a: f64, q_b: f64) -> f64 {
        (self.a + self.b_a * q_a + self.b_b * q_b).max(0.0)
    }
}

#[derive(Clone, Debug)]
pub struct BidAskAffineParams {
    pub lambda_l_a: AffineIntensityParams,
    pub lambda_c_a: AffineIntensityParams,
    pub lambda_l_b: AffineIntensityParams,
    pub lambda_c_b: AffineIntensityParams,
}

impl AffineBidAskQueueProcess {
    pub fn new(
        q0_a: f64,
        q0_b: f64,
        params: BidAskAffineParams,
        mu_a: f64,
        mu_b: f64,
        alpha_a: Vec<f64>,
        beta_a: Vec<f64>,
        alpha_b: Vec<f64>,
        beta_b: Vec<f64>,
    ) -> MarkovianProcess {
        let l_a = params.lambda_l_a.clone();
        let c_a = params.lambda_c_a.clone();
        let l_b = params.lambda_l_b.clone();
        let c_b = params.lambda_c_b.clone();

        BidAskQueueProcess::new(
            q0_a,
            q0_b,
            mu_a,
            mu_b,
            alpha_a,
            beta_a,
            alpha_b,
            beta_b,
            move |q_a, q_b| l_a.compute(q_a, q_b),
            move |q_a, q_b| c_a.compute(q_a, q_b),
            move |q_a, q_b| l_b.compute(q_a, q_b),
            move |q_a, q_b| c_b.compute(q_a, q_b),
        )
    }

    pub fn new_symmetric(
        q0_a: f64,
        q0_b: f64,
        a_l: f64,
        b_l_own: f64,
        b_l_cross: f64,

        a_c: f64,
        b_c_own: f64,
        b_c_cross: f64,

        mu: f64,
        alpha: Vec<f64>,
        beta: Vec<f64>,
    ) -> MarkovianProcess {
        let params = BidAskAffineParams {
            lambda_l_a: AffineIntensityParams::new(a_l, b_l_own, b_l_cross),
            lambda_c_a: AffineIntensityParams::new(a_c, b_c_own, b_c_cross),
            lambda_l_b: AffineIntensityParams::new(a_l, b_l_cross, b_l_own),
            lambda_c_b: AffineIntensityParams::new(a_c, b_c_cross, b_c_own),
        };

        Self::new(
            q0_a,
            q0_b,
            params,
            mu,
            mu,
            alpha.clone(),
            beta.clone(),
            alpha,
            beta,
        )
    }

    pub fn new_queue_symmetric(
        q0_a: f64,
        q0_b: f64,
        a_l: f64,
        b_l_own: f64,
        b_l_cross: f64,
        a_c: f64,
        b_c_own: f64,
        b_c_cross: f64,
    ) -> MarkovianProcess {
        let params = BidAskAffineParams {
            lambda_l_a: AffineIntensityParams::new(a_l, b_l_own, b_l_cross),
            lambda_c_a: AffineIntensityParams::new(a_c, b_c_own, b_c_cross),
            lambda_l_b: AffineIntensityParams::new(a_l, b_l_cross, b_l_own),
            lambda_c_b: AffineIntensityParams::new(a_c, b_c_cross, b_c_own),
        };
        Self::new_queue(q0_a, q0_b, params)
    }

    pub fn new_queue(q0_a: f64, q0_b: f64, params: BidAskAffineParams) -> MarkovianProcess {
        let l_a = params.lambda_l_a.clone();
        let c_a = params.lambda_c_a.clone();
        let l_b = params.lambda_l_b.clone();
        let c_b = params.lambda_c_b.clone();

        MarkovianProcess::new(
            6,
            vec![q0_a, q0_b],
            move |state: &[f64], _t: f64, _t_last: f64| {
                let q_a = state[0];
                let q_b = state[1];
                vec![
                    l_a.compute(q_a, q_b),
                    c_a.compute(q_a, q_b),
                    0.0,
                    l_b.compute(q_a, q_b),
                    c_b.compute(q_a, q_b),
                    0.0,
                ]
            },
            move |state: &[f64], event: &MultivariateEvent, _t: f64, _t_prev: f64| {
                let q_a = state[0];
                let q_b = state[1];
                let (new_q_a, new_q_b) = match event.dim {
                    0 => (q_a + 1.0, q_b),
                    1 => ((q_a - 1.0).max(0.0), q_b),
                    2 => ((q_a - 1.0).max(0.0), q_b),
                    3 => (q_a, q_b + 1.0),
                    4 => (q_a, (q_b - 1.0).max(0.0)),
                    5 => (q_a, (q_b - 1.0).max(0.0)),
                    _ => (q_a, q_b),
                };
                vec![new_q_a, new_q_b]
            },
        )
    }

    pub fn c_lambda_matrix(params: &BidAskAffineParams) -> [[f64; 2]; 2] {
        [
            [
                params.lambda_c_a.b_a - params.lambda_l_a.b_a,
                params.lambda_c_a.b_b - params.lambda_l_a.b_b,
            ],
            [
                params.lambda_c_b.b_a - params.lambda_l_b.b_a,
                params.lambda_c_b.b_b - params.lambda_l_b.b_b,
            ],
        ]
    }

    pub fn ask_queue_from_state(state: &[f64]) -> f64 {
        BidAskQueueProcess::ask_queue_from_state(state)
    }

    pub fn bid_queue_from_state(state: &[f64]) -> f64 {
        BidAskQueueProcess::bid_queue_from_state(state)
    }

    pub fn result_to_queue_paths(
        result: &MultivariateSimulationResult,
        initial_q_a: u32,
        initial_q_b: u32,
    ) -> BidAskQueuePath {
        BidAskQueueProcess::result_to_queue_paths(result, initial_q_a, initial_q_b)
    }

    pub fn stationary_state(
        q0_a: f64,
        q0_b: f64,
        mu_a: f64,
        mu_b: f64,
        alpha_a: &[f64],
        beta_a: &[f64],
        alpha_b: &[f64],
        beta_b: &[f64],
    ) -> Vec<f64> {
        let branching_a: f64 = alpha_a.iter().zip(beta_a).map(|(a, b)| a / b).sum();
        let branching_b: f64 = alpha_b.iter().zip(beta_b).map(|(a, b)| a / b).sum();

        assert!(
            branching_a < 1.0,
            "Ask Hawkes is not stationary (branching >= 1)"
        );
        assert!(
            branching_b < 1.0,
            "Bid Hawkes is not stationary (branching >= 1)"
        );

        let mut state = vec![q0_a, q0_b];

        state.extend(
            alpha_a
                .iter()
                .zip(beta_a)
                .map(|(a, b)| a * mu_a / (b * (1.0 - branching_a))),
        );

        state.extend(
            alpha_b
                .iter()
                .zip(beta_b)
                .map(|(a, b)| a * mu_b / (b * (1.0 - branching_b))),
        );

        state
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simulation::simulate;

    #[test]
    fn test_bidask_symmetric_creation() {
        let process = AffineBidAskQueueProcess::new_symmetric(
            10.0,
            10.0,
            5.0,
            -0.1,
            0.0,
            1.0,
            0.2,
            0.0,
            1.0,
            vec![0.3],
            vec![1.0],
        );

        let result = simulate(&process, 10.0, Some(42));
        assert!(!result.events.is_empty());
    }

    #[test]
    fn test_bidask_queue_paths() {
        let process = AffineBidAskQueueProcess::new_symmetric(
            10.0,
            10.0,
            5.0,
            -0.1,
            0.0,
            1.0,
            0.2,
            0.0,
            1.0,
            vec![0.3],
            vec![1.0],
        );

        let result = simulate(&process, 10.0, Some(42));
        let paths = AffineBidAskQueueProcess::result_to_queue_paths(&result, 10, 10);

        assert!(!paths.ask.events.is_empty());
        assert!(!paths.bid.events.is_empty());
        assert_eq!(paths.ask.events[0].queue_size, 10);
        assert_eq!(paths.bid.events[0].queue_size, 10);
    }

    #[test]
    fn test_c_lambda_matrix() {
        let params = BidAskAffineParams {
            lambda_l_a: AffineIntensityParams::new(5.0, -0.1, 0.05),
            lambda_c_a: AffineIntensityParams::new(1.0, 0.2, -0.02),
            lambda_l_b: AffineIntensityParams::new(5.0, 0.05, -0.1),
            lambda_c_b: AffineIntensityParams::new(1.0, -0.02, 0.2),
        };

        let c_matrix = AffineBidAskQueueProcess::c_lambda_matrix(&params);

        assert!((c_matrix[0][0] - 0.3).abs() < 1e-10);

        assert!((c_matrix[0][1] - (-0.07)).abs() < 1e-10);
    }

    #[test]
    fn test_dimension_enum() {
        assert_eq!(
            BidAskDimension::from_usize(0),
            Some(BidAskDimension::LimitAsk)
        );
        assert_eq!(
            BidAskDimension::from_usize(2),
            Some(BidAskDimension::MarketAsk)
        );
        assert_eq!(
            BidAskDimension::from_usize(5),
            Some(BidAskDimension::MarketBid)
        );
        assert_eq!(BidAskDimension::from_usize(6), None);
    }

    #[test]
    fn test_stationary_state() {
        let state = AffineBidAskQueueProcess::stationary_state(
            10.0,
            15.0,
            1.0,
            2.0,
            &[0.3, 0.2],
            &[1.0, 2.0],
            &[0.4],
            &[1.5],
        );

        assert_eq!(state.len(), 5);
        assert_eq!(state[0], 10.0);
        assert_eq!(state[1], 15.0);
    }
}
