use crate::models::{MultivariateEvent, MultivariateSimulationResult};

pub fn hawkes_to_market_orders(
    hawkes_result: &MultivariateSimulationResult,
) -> MultivariateSimulationResult {
    let mut result = MultivariateSimulationResult::new(3);
    for event in &hawkes_result.events {
        result.push(MultivariateEvent {
            time: event.time,
            dim: 2, // Market orders are dimension 2 in the queue process
        });
    }
    result
}

pub fn events_to_dim(
    events: &MultivariateSimulationResult,
    target_dim: usize,
    total_dims: usize,
) -> MultivariateSimulationResult {
    let mut result = MultivariateSimulationResult::new(total_dims);
    for event in &events.events {
        result.push(MultivariateEvent {
            time: event.time,
            dim: target_dim,
        });
    }
    result
}

pub fn merge_events(
    a: &MultivariateSimulationResult,
    b: &MultivariateSimulationResult,
) -> MultivariateSimulationResult {
    let total_dims = a.events_by_dim.len().max(b.events_by_dim.len());
    let mut result = MultivariateSimulationResult::new(total_dims);
    let mut i = 0;
    let mut j = 0;

    while i < a.events.len() || j < b.events.len() {
        let t_a = a.events.get(i).map(|e| e.time).unwrap_or(f64::INFINITY);
        let t_b = b.events.get(j).map(|e| e.time).unwrap_or(f64::INFINITY);

        if t_a <= t_b {
            result.push(a.events[i].clone());
            i += 1;
        } else {
            result.push(b.events[j].clone());
            j += 1;
        }
    }
    result
}

pub fn merge_all_events(streams: &[&MultivariateSimulationResult]) -> MultivariateSimulationResult {
    if streams.is_empty() {
        return MultivariateSimulationResult::new(0);
    }

    let mut result = streams[0].clone();
    for stream in &streams[1..] {
        result = merge_events(&result, stream);
    }
    result
}

pub fn events_for_dim(result: &MultivariateSimulationResult, dim: usize) -> Vec<f64> {
    result
        .events
        .iter()
        .filter(|e| e.dim == dim)
        .map(|e| e.time)
        .collect()
}

pub fn create_meta_orders(n: u32, t_start: f64, t_end: f64) -> MultivariateSimulationResult {
    let mut result = MultivariateSimulationResult::new(3);
    for i in 0..n {
        let time = if n > 1 {
            t_start + (i as f64 / (n - 1) as f64) * (t_end - t_start)
        } else {
            t_start
        };
        result.push(MultivariateEvent { time, dim: 0 });
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_merge_events() {
        let mut a = MultivariateSimulationResult::new(3);
        a.push(MultivariateEvent { time: 1.0, dim: 0 });
        a.push(MultivariateEvent { time: 3.0, dim: 0 });

        let mut b = MultivariateSimulationResult::new(3);
        b.push(MultivariateEvent { time: 2.0, dim: 1 });
        b.push(MultivariateEvent { time: 4.0, dim: 1 });

        let merged = merge_events(&a, &b);
        assert_eq!(merged.events.len(), 4);
        assert_eq!(merged.events[0].time, 1.0);
        assert_eq!(merged.events[1].time, 2.0);
        assert_eq!(merged.events[2].time, 3.0);
        assert_eq!(merged.events[3].time, 4.0);
    }

    #[test]
    fn test_create_meta_orders() {
        let meta = create_meta_orders(5, 10.0, 20.0);
        assert_eq!(meta.events.len(), 5);
        assert_eq!(meta.events[0].time, 10.0);
        assert_eq!(meta.events[4].time, 20.0);
        assert!(meta.events.iter().all(|e| e.dim == 0));
    }

    #[test]
    fn test_hawkes_to_market_orders() {
        let mut hawkes = MultivariateSimulationResult::new(1);
        hawkes.push(MultivariateEvent { time: 1.0, dim: 0 });
        hawkes.push(MultivariateEvent { time: 2.0, dim: 0 });

        let market = hawkes_to_market_orders(&hawkes);
        assert_eq!(market.events.len(), 2);
        assert!(market.events.iter().all(|e| e.dim == 2));
    }
}
