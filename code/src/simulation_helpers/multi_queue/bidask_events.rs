use crate::models::{MultivariateEvent, MultivariateSimulationResult};

pub fn hawkes_to_ask_market_orders(hawkes_result: &MultivariateSimulationResult) -> MultivariateSimulationResult {
    let mut result = MultivariateSimulationResult::new(6);
    for event in &hawkes_result.events {
        result.push(MultivariateEvent {
            time: event.time,
            dim: 2,  // N^a (ask market orders)
        });
    }
    result
}

pub fn hawkes_to_bid_market_orders(hawkes_result: &MultivariateSimulationResult) -> MultivariateSimulationResult {
    let mut result = MultivariateSimulationResult::new(6);
    for event in &hawkes_result.events {
        result.push(MultivariateEvent {
            time: event.time,
            dim: 5,  // N^b (bid market orders)
        });
    }
    result
}

pub fn hawkes_pair_to_market_orders(
    hawkes_ask: &MultivariateSimulationResult,
    hawkes_bid: &MultivariateSimulationResult,
) -> MultivariateSimulationResult {
    let ask = hawkes_to_ask_market_orders(hawkes_ask);
    let bid = hawkes_to_bid_market_orders(hawkes_bid);
    merge_bidask_events(&ask, &bid)
}

pub fn merge_bidask_events(
    a: &MultivariateSimulationResult,
    b: &MultivariateSimulationResult,
) -> MultivariateSimulationResult {
    let mut result = MultivariateSimulationResult::new(6);
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

pub fn merge_all_bidask_events(streams: &[&MultivariateSimulationResult]) -> MultivariateSimulationResult {
    if streams.is_empty() {
        return MultivariateSimulationResult::new(6);
    }

    let mut result = streams[0].clone();
    for stream in &streams[1..] {
        result = merge_bidask_events(&result, stream);
    }
    result
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Side {
    Ask,
    Bid,
}

pub fn create_bidask_meta_orders(
    n: u32,
    t_start: f64,
    t_end: f64,
    side: Side,
) -> MultivariateSimulationResult {
    let dim = match side {
        Side::Ask => 0,  // L^a (ask limit orders)
        Side::Bid => 3,  // L^b (bid limit orders)
    };

    let mut result = MultivariateSimulationResult::new(6);
    for i in 0..n {
        let time = if n > 1 {
            t_start + (i as f64 / (n - 1) as f64) * (t_end - t_start)
        } else {
            t_start
        };
        result.push(MultivariateEvent { time, dim });
    }
    result
}


pub fn create_symmetric_meta_orders(
    n: u32,
    t_start: f64,
    t_end: f64,
) -> MultivariateSimulationResult {
    let n_per_side = n / 2;
    let ask_orders = create_bidask_meta_orders(n_per_side, t_start, t_end, Side::Ask);
    let bid_orders = create_bidask_meta_orders(n - n_per_side, t_start, t_end, Side::Bid);
    merge_bidask_events(&ask_orders, &bid_orders)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hawkes_to_ask_market_orders() {
        let mut hawkes = MultivariateSimulationResult::new(1);
        hawkes.push(MultivariateEvent { time: 1.0, dim: 0 });
        hawkes.push(MultivariateEvent { time: 2.0, dim: 0 });

        let market = hawkes_to_ask_market_orders(&hawkes);
        assert_eq!(market.events.len(), 2);
        assert!(market.events.iter().all(|e| e.dim == 2));
    }

    #[test]
    fn test_hawkes_to_bid_market_orders() {
        let mut hawkes = MultivariateSimulationResult::new(1);
        hawkes.push(MultivariateEvent { time: 1.0, dim: 0 });
        hawkes.push(MultivariateEvent { time: 2.0, dim: 0 });

        let market = hawkes_to_bid_market_orders(&hawkes);
        assert_eq!(market.events.len(), 2);
        assert!(market.events.iter().all(|e| e.dim == 5));
    }

    #[test]
    fn test_create_bidask_meta_orders() {
        let ask_meta = create_bidask_meta_orders(5, 10.0, 20.0, Side::Ask);
        assert_eq!(ask_meta.events.len(), 5);
        assert!(ask_meta.events.iter().all(|e| e.dim == 0));

        let bid_meta = create_bidask_meta_orders(5, 10.0, 20.0, Side::Bid);
        assert_eq!(bid_meta.events.len(), 5);
        assert!(bid_meta.events.iter().all(|e| e.dim == 3));
    }

    #[test]
    fn test_hawkes_pair_to_market_orders() {
        let mut hawkes_a = MultivariateSimulationResult::new(1);
        hawkes_a.push(MultivariateEvent { time: 1.0, dim: 0 });
        hawkes_a.push(MultivariateEvent { time: 3.0, dim: 0 });

        let mut hawkes_b = MultivariateSimulationResult::new(1);
        hawkes_b.push(MultivariateEvent { time: 2.0, dim: 0 });
        hawkes_b.push(MultivariateEvent { time: 4.0, dim: 0 });

        let combined = hawkes_pair_to_market_orders(&hawkes_a, &hawkes_b);
        assert_eq!(combined.events.len(), 4);
        assert_eq!(combined.events[0].dim, 2);  // ask at t=1
        assert_eq!(combined.events[1].dim, 5);  // bid at t=2
        assert_eq!(combined.events[2].dim, 2);  // ask at t=3
        assert_eq!(combined.events[3].dim, 5);  // bid at t=4
    }
}
