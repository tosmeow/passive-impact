use crate::models::MarkovianIntensity;
use crate::rng::{create_rng, sample_exponential, sample_uniform};

// Efficient simulation using Markovian recursion.
// For models implementing `MarkovianIntensity`, this avoids O(n) intensity
// recomputation at each step by maintaining a recursive state.

pub fn simulate_markovian<P: MarkovianIntensity>(
    model: &P,
    t_max: f64,
    seed: Option<u64>,
) -> Vec<f64> {
    let mut rng = create_rng(seed);
    let mut events = Vec::new();
    let mut state = model.initial_state();
    let mut t = 0.0;
    let mut t_last = 0.0;

    while t < t_max {
        // Compute intensity upper bound from state
        let lambda_t = model.intensity_from_state(&state, t, t_last);

        if lambda_t <= 0.0 {
            break;
        }

        // Propose next event
        let dt = sample_exponential(&mut rng, lambda_t);
        let t_prop = t + dt;

        if t_prop > t_max {
            break;
        }

        // Accept/reject
        let lambda_prop = model.intensity_from_state(&state, t_prop, t_last);
        let accept_prob = lambda_prop / lambda_t;

        if sample_uniform(&mut rng) < accept_prob {
            // Accept event
            model.update_state(&mut state, t_prop, t_last);
            events.push(t_prop);
            t_last = t_prop;
        }

        t = t_prop;
    }

    events
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::MultiExponentialHawkes;
    use approx::assert_relative_eq;

    #[test]
    fn test_thinning_produces_events() {
        let model = MultiExponentialHawkes::new(1.0, vec![0.5], vec![2.0]);
        let events = simulate_markovian(&model, 100.0, Some(42));

        // Should produce some events
        assert!(!events.is_empty());

        // Events should be sorted
        for i in 1..events.len() {
            assert!(events[i] > events[i - 1]);
        }

        // All events in [0, 100]
        for &t in &events {
            assert!(t >= 0.0 && t <= 100.0);
        }
    }

    #[test]
    fn test_reproducibility() {
        let model = MultiExponentialHawkes::new(1.0, vec![0.5], vec![2.0]);

        let events1 = simulate_markovian(&model, 100.0, Some(12345));
        let events2 = simulate_markovian(&model, 100.0, Some(12345));

        assert_eq!(events1.len(), events2.len());
        for (a, b) in events1.iter().zip(events2.iter()) {
            assert_relative_eq!(a, b);
        }
    }

    #[test]
    fn test_higher_intensity_more_events() {
        let low_intensity = MultiExponentialHawkes::new(0.5, vec![0.1], vec![2.0]);
        let high_intensity = MultiExponentialHawkes::new(2.0, vec![0.5], vec![2.0]);

        let events_low = simulate_markovian(&low_intensity, 100.0, Some(42));
        let events_high = simulate_markovian(&high_intensity, 100.0, Some(42));

        // Higher intensity should produce more events on average
        // (This is probabilistic, but with seed should be deterministic)
        assert!(events_high.len() > events_low.len());
    }
}
