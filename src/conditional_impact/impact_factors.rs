use super::tail_intensity::{TailIntensity};
use crate::models::{MultiExponentialHawkes, MarkovianIntensity};

pub struct TailImpact {
    pub events: Vec<f64>,
    pub tail_impact_events: Vec<f64>,
}

impl TailImpact {
    pub fn new(hawkes_params: MultiExponentialHawkes, c_lambda: f64, events: Vec<f64>) -> Self {
        let tail_intensity = TailIntensity::new(hawkes_params.clone(), c_lambda);
        let mut tail_impact_events: Vec<f64> = Vec::with_capacity(events.len());
        let mut prev_t: f64 = 0.0;
        let mut state: Vec<f64> = vec![0.0; hawkes_params.alpha.len()];
        for t in events.iter(){
            hawkes_params.update_state(&mut state, *t, prev_t);
            tail_impact_events.push(tail_intensity.compute(&state));
            prev_t = *t;
        }
        Self {events, tail_impact_events}
    }
}