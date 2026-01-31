use super::impact_factors::{TailImpact};
use crate::models::{QueuePath};

pub struct ImpactPath {
    pub impact_path: Vec<f64>,
}

impl ImpactPath {
    // For now, we will get the impact path only at the timestamps where we have events from the market event Hawkes process.

    //Will need to add check logic so that we don't call for queue size when i or j becomes larger than their size, it shouldnt however happen except on the boundary case?
    pub fn new(q: QueuePath, q_bar: QueuePath, tail_impact: &TailImpact) -> Self {
        let mut impact_path = Vec::with_capacity(tail_impact.events.len());
        let (mut i, mut j) = (0, 0);
        let (mut curr_q, mut curr_q_bar) = (q.events[0].queue_size, q_bar.events[0].queue_size);
        let mut cumulative_term: f64 = 0.0;
        for (t_index, t) in tail_impact.events.iter().enumerate() {
            while q.events[i].time < *t {
                i += 1;
                curr_q = q.events[i].queue_size;
            }
            while q_bar.events[j].time < *t {
                j += 1;
                curr_q_bar = q_bar.events[j].queue_size;
            }
            let diff = curr_q_bar as f64 - curr_q as f64;
            cumulative_term += diff;
            let tail_term = diff * tail_impact.tail_impact_events[t_index];
            impact_path.push(cumulative_term + tail_term);
        }
        Self { impact_path }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simulation::simulate;
    use crate::models::{QueuePath, QueueEvent, MultiExponentialHawkes, AffineQueueProcess};

    #[test]
    fn zero_impact() {
        let model = MultiExponentialHawkes::new(1.0, vec![0.3, 3.5], vec![1.0, 5.0]);
        let result = simulate(&model, 10.0, Some(42));
        let events = result.events_by_dim[0].clone();
        let c_lambda = 1.0;
        let n = events.len();
        let mut q = Vec::with_capacity(2 * n);
        for i in 0..n {
            let event = QueueEvent { queue_event: 0, queue_size: (10 + (i % 3) - (i % 5)) as u32, time: events[i] };
            let next_event = QueueEvent { queue_event: 0, queue_size: (10 + (i % 3) - (i % 5)) as u32, time: events[i] };
            q.push(event);
            q.push(next_event);
        }
        let q_path = QueuePath { events: q };
        let tail_impact = TailImpact::new(model.clone(), c_lambda, events);
        let _impact_path = ImpactPath::new(q_path.clone(), q_path.clone(), &tail_impact);
        println!("Generated {} events", n);
        println!("{:?}", _impact_path.impact_path);
    }

    #[test]
    fn test_from_affine_queue() {
        // Test using the from_affine_queue convenience constructor
        let mu = 1.0;
        let alpha = vec![0.3, 3.5];
        let beta = vec![1.0, 5.0];
        let b_l = 0.5;  // λ^L slope
        let b_c = 1.5;  // λ^C slope
        // c_lambda = b_c - b_l = 1.0

        let model = MultiExponentialHawkes::new(mu, alpha.clone(), beta.clone());
        let result = simulate(&model, 10.0, Some(42));
        let events = result.events_by_dim[0].clone();

        // Both methods should produce same result
        let tail_impact_explicit = TailImpact::new(model.clone(), AffineQueueProcess::c_lambda(b_l, b_c), events.clone());
        let tail_impact_affine = TailImpact::from_affine_queue(mu, alpha, beta, b_l, b_c, events);

        assert_eq!(tail_impact_explicit.tail_impact_events.len(), tail_impact_affine.tail_impact_events.len());
        for (a, b) in tail_impact_explicit.tail_impact_events.iter().zip(&tail_impact_affine.tail_impact_events) {
            assert!((a - b).abs() < 1e-10);
        }
    }
}
