use super::tail_intensity::TailIntensity;
use crate::models::{MultiExponentialHawkes, MultivariateMarkovianIntensity, AffineQueueProcess};

pub struct TailImpact {
    pub events: Vec<f64>,
    pub tail_impact_events: Vec<f64>,
}

impl TailImpact {
    // Create TailImpact with explicit Hawkes params and c_lambda.
    pub fn new(hawkes_params: MultiExponentialHawkes, c_lambda: f64, events: Vec<f64>) -> Self {
        let tail_intensity = TailIntensity::new(hawkes_params.clone(), c_lambda);
        let mut tail_impact_events: Vec<f64> = Vec::with_capacity(events.len());
        let mut prev_t: f64 = 0.0;
        let mut state: Vec<f64> = vec![0.0; hawkes_params.alpha.len()];
        for t in events.iter() {
            hawkes_params.update_state(&mut state, 0, *t, prev_t);
            tail_impact_events.push(tail_intensity.compute(&state));
            prev_t = *t;
        }
        Self { events, tail_impact_events }
    }

    // Create TailImpact for an affine queue process.
    //
    // Computes c_lambda automatically from the affine slopes:
    // c_lambda = b_c - b_l
    pub fn from_affine_queue(
        mu: f64,
        alpha: Vec<f64>,
        beta: Vec<f64>,
        b_l: f64,
        b_c: f64,
        events: Vec<f64>,
    ) -> Self {
        let hawkes_params = MultiExponentialHawkes::new(mu, alpha, beta);
        let c_lambda = AffineQueueProcess::c_lambda(b_l, b_c);
        Self::new(hawkes_params, c_lambda, events)
    }
}
