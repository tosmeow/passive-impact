use crate::models::{AffineQueueProcess, MultivariateSimulationResult, QueuePath, MultivariateMarkovianIntensity};
use crate::simulation::ConditionalSimulationContext;
use crate::conditional_impact::{TailImpact, ImpactPath};
use crate::utils::{write_npy_f64, write_npy_u32, write_npy_f64_1d};

use rayon::prelude::*;

pub struct SimulationResults {
    pub queue_samples: Vec<Vec<u32>>,
    pub impact_paths: Vec<Vec<f64>>,
    pub market_orders_per_sim: Vec<Vec<f64>>,
}

pub fn extract_event_type(result: &MultivariateSimulationResult, dim: usize) -> Vec<f64> {
    result.events.iter()
        .filter(|e| e.dim == dim)
        .map(|e| e.time)
        .collect()
}

pub fn extract_events_by_dim(
    result: &MultivariateSimulationResult,
    n_dims: usize,
    exclude_dim: Option<usize>,
) -> Vec<Vec<f64>> {
    (0..n_dims)
        .map(|dim| {
            if exclude_dim == Some(dim) {
                vec![]
            } else {
                extract_event_type(result, dim)
            }
        })
        .collect()
}

pub fn sample_queue_at_times(path: &QueuePath, times: &[f64]) -> Vec<u32> {
    times.iter().map(|&t| path.queue_at_time(t)).collect()
}

/// Extract market order times (dimension 2) from simulation results.
pub fn extract_market_orders(result: &MultivariateSimulationResult) -> Vec<f64> {
    extract_event_type(result, 2)
}

pub struct ParallelSimulator<'a, P: MultivariateMarkovianIntensity + Sync> {
    pub process: &'a P,
    pub cond_events_by_dim: &'a [Vec<f64>],
    pub cond_external_events: Option<&'a MultivariateSimulationResult>,
    pub new_external_events: Option<&'a MultivariateSimulationResult>,
    pub time_horizon: f64,
    pub initial_queue_size: u32,
    pub reference_path: &'a QueuePath,
    pub tail_impact: &'a TailImpact,
    pub market_orders: &'a [f64],
    /// If true, simulating bar_q (with_us); if false, simulating q (without_us)
    pub simulating_bar_q: bool,
}

impl<'a, P: MultivariateMarkovianIntensity + Sync> ParallelSimulator<'a, P> {
    pub fn run(&self, n_simulations: usize) -> SimulationResults {
        let results: Vec<(Vec<u32>, Vec<f64>, Vec<f64>)> = (0..n_simulations)
            .into_par_iter()
            .map(|sim_idx| {
                let ctx = ConditionalSimulationContext::new(
                    self.process,
                    self.cond_events_by_dim,
                    self.cond_external_events,
                    self.new_external_events,
                    self.time_horizon,
                );

                let sim_result = ctx.simulate(Some(sim_idx as u64));
                let sim_path = AffineQueueProcess::result_to_queue_path(&sim_result, self.initial_queue_size);

                // ImpactPath::new takes (q_path, bar_q_path)
                let impact_path = if self.simulating_bar_q {
                    // with_us: reference is q, simulated is bar_q
                    ImpactPath::new(self.reference_path.clone(), sim_path.clone(), self.tail_impact)
                } else {
                    // without_us: simulated is q, reference is bar_q
                    ImpactPath::new(sim_path.clone(), self.reference_path.clone(), self.tail_impact)
                };

                let sim_samples = sample_queue_at_times(&sim_path, self.market_orders);
                let sim_market_orders = extract_market_orders(&sim_result);

                (sim_samples, impact_path.impact_path, sim_market_orders)
            })
            .collect();

        SimulationResults {
            queue_samples: results.iter().map(|(q, _, _)| q.clone()).collect(),
            impact_paths: results.iter().map(|(_, ip, _)| ip.clone()).collect(),
            market_orders_per_sim: results.iter().map(|(_, _, mo)| mo.clone()).collect(),
        }
    }
}

pub fn write_results(
    results: &SimulationResults,
    reference_queue_samples: &[u32],
    market_orders: &[f64],
    suffix: &str,
) -> std::io::Result<()> {
    let n_times = market_orders.len();
    let n_simulations = results.queue_samples.len();

    let impact_data: Vec<f64> = (0..n_times)
        .flat_map(|t_idx| {
            results.impact_paths.iter().map(move |path| {
                path.get(t_idx).copied().unwrap_or(f64::NAN)
            })
        })
        .collect();
    write_npy_f64(&format!("impact_paths{}.npy", suffix), &impact_data, n_times, n_simulations)?;

    let queue_data: Vec<u32> = (0..n_times)
        .flat_map(|t_idx| {
            std::iter::once(reference_queue_samples[t_idx])
                .chain(results.queue_samples.iter().map(move |samples| samples[t_idx]))
        })
        .collect();
    write_npy_u32(&format!("queue_paths{}.npy", suffix), &queue_data, n_times, n_simulations + 1)?;

    write_npy_f64_1d(&format!("times{}.npy", suffix), market_orders)?;

    Ok(())
}
