use super::multivariate_process::{MultivariateEvent, MultivariateMarkovianIntensity};

#[derive(Clone)]
pub struct QueueEvent {
    pub queue_event: u32,
    pub queue_size: u32,
    pub time: f64,
}

#[derive(Clone)]
pub struct QueuePath {
    pub events: Vec<QueueEvent>,
}

impl QueuePath {
    /// Get queue size at a given time.
    ///
    /// Returns the queue size of the last event with time <= t.
    pub fn queue_at_time(&self, t: f64) -> u32 {
        let mut val = self.events[0].queue_size;
        for event in &self.events {
            if event.time <= t {
                val = event.queue_size;
            } else {
                break;
            }
        }
        val
    }
}

pub struct MarkovianProcess {
    dim: usize,
    initial_state: Vec<f64>,
    lambda: Box<dyn Fn(&[f64], f64, f64) -> Vec<f64> + Send + Sync>,
    state_constr: Box<dyn Fn(&[f64], &MultivariateEvent, f64, f64) -> Vec<f64> + Send + Sync>,
}

impl MarkovianProcess {
    pub fn new<F, G>(
        dim: usize,
        initial_state: Vec<f64>,
        lambda: F,
        state_constr: G,
    ) -> Self
    where
        F: Fn(&[f64], f64, f64) -> Vec<f64> + Send + Sync + 'static,
        G: Fn(&[f64], &MultivariateEvent, f64, f64) -> Vec<f64> + Send + Sync + 'static,
    {
        Self {
            dim,
            initial_state,
            lambda: Box::new(lambda),
            state_constr: Box::new(state_constr),
        }
    }
}

impl MultivariateMarkovianIntensity for MarkovianProcess {
    type State = Vec<f64>;

    fn dim(&self) -> usize {
        self.dim
    }

    fn initial_state(&self) -> Self::State {
        self.initial_state.clone()
    }

    fn intensities_from_state(&self, state: &Self::State, t: f64, t_last: f64) -> Vec<f64> {
        (self.lambda)(state, t, t_last)
    }

    fn update_state(&self, state: &mut Self::State, dim: usize, t: f64, t_prev: f64) {
        let event = MultivariateEvent { time: t, dim };
        *state = (self.state_constr)(state, &event, t, t_prev);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simulation::simulate;
    use crate::models::MultiExponentialHawkes;

    #[test]
    fn test_hawkes_simulate() {
        let mu = 1.0;
        let alpha = vec![0.3, 0.2];
        let beta = vec![1.0, 5.0];
        let t_max = 50.0;
        let seed = Some(42u64);

        let hawkes = MultiExponentialHawkes::new(mu, alpha, beta);
        let result = simulate(&hawkes, t_max, seed);

        assert!(!result.is_empty());
        assert_eq!(result.events_by_dim.len(), 1);

        // Events should be sorted
        for i in 1..result.events.len() {
            assert!(result.events[i].time > result.events[i - 1].time);
        }
    }

    #[test]
    fn test_queue_event_creation() {
        let event = QueueEvent {
            queue_event: 1,
            queue_size: 10,
            time: 0.5,
        };
        assert_eq!(event.queue_event, 1);
        assert_eq!(event.queue_size, 10);
        assert_eq!(event.time, 0.5);
    }

    #[test]
    fn test_queue_event_clone() {
        let event = QueueEvent {
            queue_event: 2,
            queue_size: 5,
            time: 1.0,
        };
        let cloned = event.clone();
        assert_eq!(cloned.queue_event, 2);
        assert_eq!(cloned.queue_size, 5);
        assert_eq!(cloned.time, 1.0);
    }

    #[test]
    fn test_queue_path_creation() {
        let events = vec![
            QueueEvent { queue_event: 0, queue_size: 10, time: 0.0 },
            QueueEvent { queue_event: 1, queue_size: 11, time: 0.5 },
            QueueEvent { queue_event: 3, queue_size: 10, time: 1.2 },
        ];
        let path = QueuePath { events };
        assert_eq!(path.events.len(), 3);
        assert_eq!(path.events[0].queue_size, 10);
        assert_eq!(path.events[2].time, 1.2);
    }

    #[test]
    fn test_queue_path_clone() {
        let path = QueuePath {
            events: vec![
                QueueEvent { queue_event: 0, queue_size: 5, time: 0.0 },
            ],
        };
        let cloned = path.clone();
        assert_eq!(cloned.events.len(), 1);
        assert_eq!(cloned.events[0].queue_size, 5);
    }
}
