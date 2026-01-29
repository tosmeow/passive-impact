use crate::rng::{create_rng, sample_exponential};
use crate::models::{QueuePath, QueueEvent};

pub struct QueueSimulator {
    lambda_l: Box<dyn Fn(f64) -> f64>,
    lambda_c: Box<dyn Fn(f64) -> f64>,
    time_horizon: f64,

    pub limit_orders: Vec<f64>,
    pub cancel_orders: Vec<f64>,
    pub market_orders: Vec<f64>,
    pub meta_orders: Vec<f64>,
}

impl QueueSimulator {
    pub fn new(
        lambda_l: impl Fn(f64) -> f64 + 'static,
        lambda_c: impl Fn(f64) -> f64 + 'static,
        time_horizon: f64,
    ) -> Self {
        Self {
            lambda_l: Box::new(lambda_l),
            lambda_c: Box::new(lambda_c),
            time_horizon,
            limit_orders: Vec::new(),
            cancel_orders: Vec::new(),
            market_orders: Vec::new(),
            meta_orders: Vec::new(),
        }
    }

    pub fn simulation_queue(
        &mut self,
        market_orders: &[f64],
        meta_orders: &[f64],
        q0: u32,
        stock_values: bool,
        seed: Option<u64>,
    ) -> QueuePath {
        let mut rng = create_rng(seed);

        if stock_values {
            self.limit_orders.clear();
            self.cancel_orders.clear();
            self.market_orders = market_orders.to_vec();
            self.meta_orders = meta_orders.to_vec();
        }

        let mut t = 0.0;
        let mut index_market = 0;
        let mut index_meta = 0;

        let mut tau_n = market_orders.get(0).copied().unwrap_or(f64::INFINITY);
        let mut tau_meta = meta_orders.get(0).copied().unwrap_or(f64::INFINITY);

        // Build queue as QueuePath
        let mut queue_events: Vec<QueueEvent> = vec![
            QueueEvent {
                queue_event: 0,  // initial state
                queue_size: q0,
                time: 0.0,
            }
        ];

        while t < self.time_horizon {
            let current_size = queue_events.last().unwrap().queue_size;
            let current_size_f64 = current_size as f64;

            // Compute rates
            let lambda_l = (self.lambda_l)(current_size_f64);
            let lambda_c = (self.lambda_c)(current_size_f64);

            // Sample times to next limit/cancel events
            let tau_l = sample_exponential(&mut rng, lambda_l);
            let tau_c = sample_exponential(&mut rng, lambda_c);

            // Find minimum time
            let taus = [t + tau_l, t + tau_c, tau_n, tau_meta];
            let (argmin, &time_of_next_event) = taus.iter()
                .enumerate()
                .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap();

            if time_of_next_event >= self.time_horizon {
                break;
            }

            t = time_of_next_event;

            match argmin {
                0 => {
                    // Limit order event
                    queue_events.push(QueueEvent {
                        queue_event: 1,
                        queue_size: current_size + 1,
                        time: t,
                    });
                    if stock_values {
                        self.limit_orders.push(t);
                    }
                }
                1 => {
                    // Cancel order event
                    queue_events.push(QueueEvent {
                        queue_event: 2,
                        queue_size: current_size.saturating_sub(1),
                        time: t,
                    });
                    if stock_values {
                        self.cancel_orders.push(t);
                    }
                }
                2 => {
                    // Market order event
                    queue_events.push(QueueEvent {
                        queue_event: 3,
                        queue_size: current_size.saturating_sub(1),
                        time: t,
                    });
                    index_market += 1;
                    tau_n = market_orders.get(index_market).copied().unwrap_or(f64::INFINITY);
                }
                3 => {
                    // Meta order event (adds to queue)
                    queue_events.push(QueueEvent {
                        queue_event: 4,  // meta order type
                        queue_size: current_size + 1,
                        time: t,
                    });
                    index_meta += 1;
                    tau_meta = meta_orders.get(index_meta).copied().unwrap_or(f64::INFINITY);
                }
                _ => unreachable!(),
            }
        }

        QueuePath { events: queue_events }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::MultiExponentialHawkes;
    use crate::simulation::simulate_markovian;

    #[test]
    fn test_queue_simulation() {
        // Linear intensity functions
        let lambda_l = |q: f64| 2.0 + 0.1 * q;
        let lambda_c = |q: f64| 1.0 + 0.2 * q;

        let mut simulator = QueueSimulator::new(lambda_l, lambda_c, 10.0);

        // Generate market orders from Hawkes process
        let hawkes = MultiExponentialHawkes::new(1.0, vec![0.3], vec![2.0]);
        let market_orders = simulate_markovian(&hawkes, 10.0, Some(42));

        let queue_path = simulator.simulation_queue(
            &market_orders,
            &[],  // no meta orders
            10,   // initial queue size
            true, // stock values
            Some(123),
        );

        println!("Generated {} queue events", queue_path.events.len());
        println!("Limit orders: {}", simulator.limit_orders.len());
        println!("Cancel orders: {}", simulator.cancel_orders.len());

        assert!(!queue_path.events.is_empty());
    }
}
