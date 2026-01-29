use crate::rng::{create_rng, sample_exponential, sample_uniform};
use crate::models::{QueuePath,QueueEvent};

pub struct ConditionalQueueSimulator {
    lambda_l: Box<dyn Fn(f64) -> f64>,
    lambda_c: Box<dyn Fn(f64) -> f64>,
    time_horizon: f64,
    
    limit_orders: Vec<f64>,
    cancel_orders: Vec<f64>,
    market_orders: Vec<f64>,
    
    pub bar_limit_orders: Vec<f64>,
    pub bar_cancel_orders: Vec<f64>,
    pub nb_independent_events_limit: usize,
    pub nb_independent_events_cancel: usize,
}

impl ConditionalQueueSimulator {
    pub fn new(
        lambda_l: impl Fn(f64) -> f64 + 'static,
        lambda_c: impl Fn(f64) -> f64 + 'static,
        time_horizon: f64,
        market_orders: Vec<f64>,
        limit_orders: Vec<f64>,
        cancel_orders: Vec<f64>,
    ) -> Self {
        Self {
            lambda_l: Box::new(lambda_l),
            lambda_c: Box::new(lambda_c),
            time_horizon,
            limit_orders,
            cancel_orders,
            market_orders,
            bar_limit_orders: Vec::new(),
            bar_cancel_orders: Vec::new(),
            nb_independent_events_limit: 0,
            nb_independent_events_cancel: 0,
        }
    }

    pub fn simulation(
        &mut self,
        queue: &QueuePath,
        seed: Option<u64>,
    ) -> QueuePath {
        let mut rng = create_rng(seed);

        let mut t = 0.0;
        let initial_size = queue.events[0].queue_size;

        // Build bar_queue as QueuePath
        let mut bar_queue_events: Vec<QueueEvent> = vec![
            QueueEvent {
                queue_event: 0,  // initial state
                queue_size: initial_size,
                time: 0.0,
            }
        ];

        self.bar_limit_orders.clear();
        self.bar_cancel_orders.clear();
        self.nb_independent_events_limit = 0;
        self.nb_independent_events_cancel = 0;

        let mut index_queue = 0;
        let mut index_market = 0;
        let mut index_cancel = 0;
        let mut index_limit = 0;

        let mut tau_n = self.market_orders.get(0).copied().unwrap_or(f64::INFINITY);
        let mut tau_c = self.cancel_orders.get(0).copied().unwrap_or(f64::INFINITY);
        let mut tau_l = self.limit_orders.get(0).copied().unwrap_or(f64::INFINITY);

        while t < self.time_horizon {
            let bar_queue_value = bar_queue_events.last().unwrap().queue_size as f64;

            // Find current queue value at time t
            while index_queue + 1 < queue.events.len() && queue.events[index_queue + 1].time <= t {
                index_queue += 1;
            }
            let queue_value = queue.events[index_queue].queue_size as f64;

            // Simulate limit order from independent queue measure
            let lambda_l_bar = ((self.lambda_l)(bar_queue_value) - (self.lambda_l)(queue_value)).max(0.0);
            let tau_l_bar = sample_exponential(&mut rng, lambda_l_bar);

            // Simulate cancel from independent queue measure
            let lambda_c_bar = ((self.lambda_c)(bar_queue_value) - (self.lambda_c)(queue_value)).max(0.0);
            let tau_c_bar = sample_exponential(&mut rng, lambda_c_bar);

            // Find minimum tau
            let taus = [tau_l_bar, tau_c_bar, tau_n - t, tau_c - t, tau_l - t];
            let (argmin, &tau_min) = taus.iter()
                .enumerate()
                .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap();

            t += tau_min;

            let current_size = bar_queue_events.last().unwrap().queue_size;

            match argmin {
                0 => {
                    // tauLbar was the smallest
                    bar_queue_events.push(QueueEvent {
                        queue_event: 1,
                        queue_size: current_size + 1,
                        time: t,
                    });
                    self.bar_limit_orders.push(t);
                    self.nb_independent_events_limit += 1;
                }
                1 => {
                    // tauCbar was the smallest
                    bar_queue_events.push(QueueEvent {
                        queue_event: 2,
                        queue_size: current_size.saturating_sub(1),
                        time: t,
                    });
                    self.bar_cancel_orders.push(t);
                    self.nb_independent_events_cancel += 1;
                }
                2 => {
                    // tauNbar was the smallest
                    bar_queue_events.push(QueueEvent {
                        queue_event: 3,
                        queue_size: current_size.saturating_sub(1),
                        time: t,
                    });
                    index_market += 1;
                    tau_n = self.market_orders.get(index_market).copied().unwrap_or(f64::INFINITY);
                }
                3 => {
                    // tauC was the smallest
                    let u = sample_uniform(&mut rng);
                    if u * (self.lambda_c)(queue_value) <= (self.lambda_c)(bar_queue_value) {
                        bar_queue_events.push(QueueEvent {
                            queue_event: 2,
                            queue_size: current_size.saturating_sub(1),
                            time: t,
                        });
                        self.bar_cancel_orders.push(t);
                    }
                    index_cancel += 1;
                    tau_c = self.cancel_orders.get(index_cancel).copied().unwrap_or(f64::INFINITY);
                }
                4 => {
                    // tauL was the smallest
                    let u = sample_uniform(&mut rng);
                    if u * (self.lambda_l)(queue_value) <= (self.lambda_l)(bar_queue_value) {
                        bar_queue_events.push(QueueEvent {
                            queue_event: 1,
                            queue_size: current_size + 1,
                            time: t,
                        });
                        self.bar_limit_orders.push(t);
                    }
                    index_limit += 1;
                    tau_l = self.limit_orders.get(index_limit).copied().unwrap_or(f64::INFINITY);
                }
                _ => unreachable!(),
            }
        }

        QueuePath { events: bar_queue_events }
    }
}