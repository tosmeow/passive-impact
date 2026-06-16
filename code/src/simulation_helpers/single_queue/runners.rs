use crate::conditional_impact::{ImpactPath, TailImpact};
use crate::models::{
    AffineQueueProcess, MultivariateMarkovianIntensity, MultivariateSimulationResult, QueuePath,
};
use crate::simulation::ConditionalSimulationContext;
use crate::utils::{write_npy_f64, write_npy_f64_1d, write_npy_u32};

use rayon::prelude::*;

pub struct SimulationResults {
    pub queue_samples: Vec<Vec<u32>>,
    pub impact_paths: Vec<Vec<f64>>,
    pub market_orders_per_sim: Vec<Vec<f64>>,
}

pub struct MemoryEfficientResults {
    pub queue_samples: Vec<Vec<u32>>,
    pub impact_paths: Vec<Vec<f64>>,
}

pub fn extract_event_type(result: &MultivariateSimulationResult, dim: usize) -> Vec<f64> {
    result
        .events
        .iter()
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

                let sim_result = ctx.simulate(None, Some(sim_idx as u64));
                let sim_path =
                    AffineQueueProcess::result_to_queue_path(&sim_result, self.initial_queue_size);

                // ImpactPath::new takes (q_path, bar_q_path)
                let impact_path = if self.simulating_bar_q {
                    // with_us: reference is q, simulated is bar_q
                    ImpactPath::new(
                        self.reference_path.clone(),
                        sim_path.clone(),
                        self.tail_impact,
                    )
                } else {
                    // without_us: simulated is q, reference is bar_q
                    ImpactPath::new(
                        sim_path.clone(),
                        self.reference_path.clone(),
                        self.tail_impact,
                    )
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

    pub fn run_memory_efficient(&self, n_simulations: usize) -> MemoryEfficientResults
    where
        P::State: AsRef<[f64]> + Send,
    {
        // Compute reference samples once (shared across all simulations)
        let reference_samples: Vec<u32> =
            sample_queue_at_times(self.reference_path, self.market_orders);

        let results: Vec<(Vec<u32>, Vec<f64>)> = (0..n_simulations)
            .into_par_iter()
            .map(|sim_idx| {
                let ctx = ConditionalSimulationContext::new(
                    self.process,
                    self.cond_events_by_dim,
                    self.cond_external_events,
                    self.new_external_events,
                    self.time_horizon,
                );

                // Memory-efficient: directly get queue samples at market order times
                let sim_samples = ctx.simulate_queue_at_times(
                    self.market_orders,
                    self.initial_queue_size,
                    None, // Use default initial state
                    Some(sim_idx as u64),
                );

                // Memory-efficient: compute impact from pre-sampled queues
                let impact_path = if self.simulating_bar_q {
                    // with_us: reference is q, simulated is bar_q
                    ImpactPath::from_queue_samples(
                        &reference_samples,
                        &sim_samples,
                        self.tail_impact,
                    )
                } else {
                    // without_us: simulated is q, reference is bar_q
                    ImpactPath::from_queue_samples(
                        &sim_samples,
                        &reference_samples,
                        self.tail_impact,
                    )
                };

                (sim_samples, impact_path.impact_path)
            })
            .collect();

        MemoryEfficientResults {
            queue_samples: results.iter().map(|(q, _)| q.clone()).collect(),
            impact_paths: results.iter().map(|(_, ip)| ip.clone()).collect(),
        }
    }
}

pub fn write_results(
    results: &SimulationResults,
    reference_queue_samples: &[u32],
    market_orders: &[f64],
    output_dir: &str,
) -> std::io::Result<()> {
    std::fs::create_dir_all(output_dir)?;
    let n_times = market_orders.len();
    let n_simulations = results.queue_samples.len();

    let impact_data: Vec<f64> = (0..n_times)
        .flat_map(|t_idx| {
            results
                .impact_paths
                .iter()
                .map(move |path| path.get(t_idx).copied().unwrap_or(f64::NAN))
        })
        .collect();
    write_npy_f64(
        &format!("{}/impact_paths.npy", output_dir),
        &impact_data,
        n_times,
        n_simulations,
    )?;

    let queue_data: Vec<u32> = (0..n_times)
        .flat_map(|t_idx| {
            std::iter::once(reference_queue_samples[t_idx]).chain(
                results
                    .queue_samples
                    .iter()
                    .map(move |samples| samples[t_idx]),
            )
        })
        .collect();
    write_npy_u32(
        &format!("{}/queue_paths.npy", output_dir),
        &queue_data,
        n_times,
        n_simulations + 1,
    )?;

    write_npy_f64_1d(&format!("{}/times.npy", output_dir), market_orders)?;

    Ok(())
}

pub fn write_queue_samples(
    queue_samples: &[Vec<u32>],
    reference_queue_samples: &[u32],
    times: &[f64],
    output_dir: &str,
) -> std::io::Result<()> {
    std::fs::create_dir_all(output_dir)?;
    let n_times = times.len();
    let n_simulations = queue_samples.len();

    let queue_data: Vec<u32> = (0..n_times)
        .flat_map(|t_idx| {
            std::iter::once(reference_queue_samples[t_idx])
                .chain(queue_samples.iter().map(move |samples| samples[t_idx]))
        })
        .collect();
    write_npy_u32(
        &format!("{}/queue_paths.npy", output_dir),
        &queue_data,
        n_times,
        n_simulations + 1,
    )?;

    write_npy_f64_1d(&format!("{}/times.npy", output_dir), times)?;

    Ok(())
}

pub fn write_memory_efficient_results(
    results: &MemoryEfficientResults,
    reference_queue_samples: &[u32],
    market_orders: &[f64],
    output_dir: &str,
) -> std::io::Result<()> {
    std::fs::create_dir_all(output_dir)?;
    let n_times = market_orders.len();
    let n_simulations = results.queue_samples.len();

    let impact_data: Vec<f64> = (0..n_times)
        .flat_map(|t_idx| {
            results
                .impact_paths
                .iter()
                .map(move |path| path.get(t_idx).copied().unwrap_or(f64::NAN))
        })
        .collect();
    write_npy_f64(
        &format!("{}/impact_paths.npy", output_dir),
        &impact_data,
        n_times,
        n_simulations,
    )?;

    let queue_data: Vec<u32> = (0..n_times)
        .flat_map(|t_idx| {
            std::iter::once(reference_queue_samples[t_idx]).chain(
                results
                    .queue_samples
                    .iter()
                    .map(move |samples| samples[t_idx]),
            )
        })
        .collect();
    write_npy_u32(
        &format!("{}/queue_paths.npy", output_dir),
        &queue_data,
        n_times,
        n_simulations + 1,
    )?;

    write_npy_f64_1d(&format!("{}/times.npy", output_dir), market_orders)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::conditional_impact::TailImpact;
    use crate::models::{AffineQueueProcess, MultiExponentialHawkes};
    use crate::simulation::{simulate, simulate_with_externals};
    use crate::simulation_helpers::{
        create_meta_orders, extract_events_by_dim, hawkes_to_market_orders, merge_events,
    };

    /// Verify that `run` and `run_memory_efficient` produce identical queue samples
    /// for the same seeds.  They use different code paths (full path scan vs.
    /// point-sampling) so this catches any divergence in the efficient implementation.
    #[test]
    fn general_and_efficient_queue_samples_agree() {
        let mu = 1.0;
        let alpha = vec![0.3, 0.7];
        let beta = vec![1.0, 3.0];
        let a_l = 20.0;
        let b_l = -0.2;
        let a_c = 1.0;
        let b_c = 0.1;
        let initial_queue_size: u32 = 50;
        let time_horizon = 10.0;
        let n_simulations = 4;

        let process = AffineQueueProcess::new_queue(initial_queue_size as f64, a_l, b_l, a_c, b_c);

        let hawkes = MultiExponentialHawkes::new_with_state(
            MultiExponentialHawkes::new(mu, alpha.clone(), beta.clone()).stationary_state(),
            mu,
            alpha.clone(),
            beta.clone(),
        );
        let hawkes_result = simulate(&hawkes, time_horizon, Some(99));
        let hawkes_as_market = hawkes_to_market_orders(&hawkes_result);

        let n_meta: u32 = 10;
        let meta_orders = create_meta_orders(n_meta, 1.0, 0.8 * time_horizon);

        let q_result_internal =
            simulate_with_externals(&process, time_horizon, &hawkes_as_market, Some(99));
        let q_result = merge_events(&q_result_internal, &hawkes_as_market);
        let q_path = AffineQueueProcess::result_to_queue_path(&q_result, initial_queue_size);

        let market_orders: Vec<f64> = hawkes_as_market.events.iter().map(|e| e.time).collect();
        let q_events_by_dim = extract_events_by_dim(&q_result_internal, 3, Some(2));

        let tail_impact = TailImpact::from_affine_queue(
            mu,
            alpha.clone(),
            beta.clone(),
            b_l,
            b_c,
            market_orders.clone(),
        );

        let bar_q_external = merge_events(&meta_orders, &hawkes_as_market);
        let q_external = hawkes_as_market.clone();

        let simulator = ParallelSimulator {
            process: &process,
            cond_events_by_dim: &q_events_by_dim,
            cond_external_events: Some(&q_external),
            new_external_events: Some(&bar_q_external),
            time_horizon,
            initial_queue_size,
            reference_path: &q_path,
            tail_impact: &tail_impact,
            market_orders: &market_orders,
            simulating_bar_q: true,
        };

        let general_results = simulator.run(n_simulations);
        let efficient_results = simulator.run_memory_efficient(n_simulations);

        assert_eq!(general_results.queue_samples.len(), n_simulations);
        assert_eq!(efficient_results.queue_samples.len(), n_simulations);

        for sim_idx in 0..n_simulations {
            assert_eq!(
                general_results.queue_samples[sim_idx], efficient_results.queue_samples[sim_idx],
                "Queue samples differ at simulation {} (general vs memory-efficient)",
                sim_idx
            );
        }
    }
}
