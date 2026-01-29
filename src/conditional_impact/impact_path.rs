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
    use crate::simulation::simulate_markovian;
    use crate::models::{QueuePath,QueueEvent, MultiExponentialHawkes};
    #[test]
    fn zero_impact() {
        let model = MultiExponentialHawkes::new(1.0, vec![0.3, 3.5], vec![1.0, 5.0]);
        let events = simulate_markovian(&model, 10.0, Some(42));
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

}
