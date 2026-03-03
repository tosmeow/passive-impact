use crate::conditional_impact::TailImpact;
use crate::models::{QueuePath};

pub struct ImpactPath {
    pub impact_path: Vec<f64>,
}

impl ImpactPath {
    // Memory-efficient impact computation from pre-sampled queue values.
    //
    // This takes queue values already sampled at market order times, avoiding the need
    // to build full `QueuePath` objects and scan through them.
    //
    // # Arguments
    // - `q_samples`: Queue values at each market order time (without our orders)
    // - `q_bar_samples`: Queue values at each market order time (with our orders)
    // - `tail_impact`: The tail impact factors
    //
    // # Panics
    // Panics if `q_samples`, `q_bar_samples`, and `tail_impact.events` have different lengths.
    pub fn from_queue_samples(
        q_samples: &[u32],
        q_bar_samples: &[u32],
        tail_impact: &TailImpact,
    ) -> Self {
        assert_eq!(
            q_samples.len(),
            tail_impact.events.len(),
            "q_samples length ({}) must match tail_impact.events length ({})",
            q_samples.len(),
            tail_impact.events.len()
        );
        assert_eq!(
            q_bar_samples.len(),
            tail_impact.events.len(),
            "q_bar_samples length ({}) must match tail_impact.events length ({})",
            q_bar_samples.len(),
            tail_impact.events.len()
        );

        let mut impact_path = Vec::with_capacity(tail_impact.events.len());
        let mut cumulative_term: f64 = 0.0;

        for (t_index, (&q, &q_bar)) in q_samples.iter().zip(q_bar_samples.iter()).enumerate() {
            let diff = q_bar as f64 - q as f64;
            cumulative_term += diff;
            let tail_term = diff * tail_impact.tail_impact_events[t_index];
            impact_path.push(cumulative_term + tail_term);
        }

        Self { impact_path }
    }

    // Compute the impact path at market order times.
    //
    // For each market order time t, we find the queue sizes q(t-) and q_bar(t-)
    // (the queue size just before time t) and compute the cumulative impact.
    //
    // This version uses a scanning approach that tracks position in both queue paths,
    // handling the case where market order times may not align exactly with queue events.
    pub fn new(q: QueuePath, q_bar: QueuePath, tail_impact: &TailImpact) -> Self {
        let mut impact_path = Vec::with_capacity(tail_impact.events.len());

        let (mut i, mut j) = (0, 0);
        let q_len = q.events.len();
        let q_bar_len = q_bar.events.len();

        // Initialize with first queue values
        let (mut curr_q, mut curr_q_bar) = (
            q.events.first().map(|e| e.queue_size).unwrap_or(0),
            q_bar.events.first().map(|e| e.queue_size).unwrap_or(0),
        );

        let mut cumulative_term: f64 = 0.0;

        for (t_index, &t) in tail_impact.events.iter().enumerate() {
            // Advance i to find queue value at time t (last event with time <= t)
            while i + 1 < q_len && q.events[i + 1].time <= t {
                i += 1;
            }
            if i < q_len && q.events[i].time <= t {
                curr_q = q.events[i].queue_size;
            }

            // Advance j to find q_bar value at time t
            while j + 1 < q_bar_len && q_bar.events[j + 1].time <= t {
                j += 1;
            }
            if j < q_bar_len && q_bar.events[j].time <= t {
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
