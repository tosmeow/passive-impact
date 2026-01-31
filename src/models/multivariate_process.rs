pub trait MultivariateMarkovianIntensity: Send + Sync {
    type State: Clone;

    fn dim(&self) -> usize;

    fn initial_state(&self) -> Self::State;

    fn intensities_from_state(&self, state: &Self::State, t: f64, t_last: f64) -> Vec<f64>;

    fn intensity_upper_bound(&self, state: &Self::State, t: f64, t_last: f64) -> f64 {
        self.intensities_from_state(state, t, t_last).iter().sum()
    }

    fn update_state(&self, state: &mut Self::State, dim: usize, t: f64, t_prev: f64);
}

#[derive(Debug, Clone, Copy)]
pub struct MultivariateEvent {
    pub time: f64,
    pub dim: usize,
}

#[derive(Debug, Clone)]
pub struct MultivariateSimulationResult {
    pub events: Vec<MultivariateEvent>,
    pub events_by_dim: Vec<Vec<f64>>,
}

impl MultivariateSimulationResult {
    pub fn new(dim: usize) -> Self {
        Self {
            events: Vec::new(),
            events_by_dim: vec![Vec::new(); dim],
        }
    }

    pub fn push(&mut self, event: MultivariateEvent) {
        self.events.push(event);
        self.events_by_dim[event.dim].push(event.time);
    }

    pub fn len(&self) -> usize {
        self.events.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    /// Create from a list of events.
    pub fn from_events(dim: usize, events: Vec<MultivariateEvent>) -> Self {
        let mut result = Self::new(dim);
        for event in events {
            result.push(event);
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multivariate_event_creation() {
        let event = MultivariateEvent { time: 1.5, dim: 2 };
        assert_eq!(event.time, 1.5);
        assert_eq!(event.dim, 2);
    }

    #[test]
    fn test_multivariate_result_push() {
        let mut result = MultivariateSimulationResult::new(3);
        result.push(MultivariateEvent { time: 0.5, dim: 0 });
        result.push(MultivariateEvent { time: 1.0, dim: 1 });
        result.push(MultivariateEvent { time: 1.5, dim: 0 });

        assert_eq!(result.len(), 3);
        assert_eq!(result.events_by_dim[0].len(), 2);
        assert_eq!(result.events_by_dim[1].len(), 1);
        assert_eq!(result.events_by_dim[2].len(), 0);
    }
}
