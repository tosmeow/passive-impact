use crate::models::{
    AffineBidAskQueueProcess, MultivariateSimulationResult,
    MultivariateMarkovianIntensity, BidAskQueuePath,
};
use crate::simulation::ConditionalSimulationContext;
use crate::conditional_impact::{BidAskTailImpact, BidAskImpactPath};
use crate::utils::{write_npy_f64, write_npy_u32, write_npy_f64_1d};

use rayon::prelude::*;

pub struct BidAskSimulationResults {
    pub ask_queue_samples: Vec<Vec<u32>>,

    pub bid_queue_samples: Vec<Vec<u32>>,

    pub ask_impact_paths: Vec<Vec<f64>>,

    pub bid_impact_paths: Vec<Vec<f64>>,
}

pub struct BidAskMemoryEfficientResults {
    pub ask_queue_samples: Vec<Vec<u32>>,

    pub bid_queue_samples: Vec<Vec<u32>>,

    pub ask_impact_paths: Vec<Vec<f64>>,

    pub bid_impact_paths: Vec<Vec<f64>>,
}

pub fn extract_ask_market_orders(result: &MultivariateSimulationResult) -> Vec<f64> {
    result.events.iter()
        .filter(|e| e.dim == 2)
        .map(|e| e.time)
        .collect()
}

pub fn extract_bid_market_orders(result: &MultivariateSimulationResult) -> Vec<f64> {
    result.events.iter()
        .filter(|e| e.dim == 5)
        .map(|e| e.time)
        .collect()
}

pub fn sample_ask_queue_at_times(paths: &BidAskQueuePath, times: &[f64]) -> Vec<u32> {
    times.iter().map(|&t| paths.ask.queue_at_time(t)).collect()
}

pub fn sample_bid_queue_at_times(paths: &BidAskQueuePath, times: &[f64]) -> Vec<u32> {
    times.iter().map(|&t| paths.bid.queue_at_time(t)).collect()
}

pub fn extract_bidask_events_by_dim(
    result: &MultivariateSimulationResult,
    exclude_dims: Option<&[usize]>,
) -> Vec<Vec<f64>> {
    (0..6)
        .map(|dim| {
            if exclude_dims.map(|ex| ex.contains(&dim)).unwrap_or(false) {
                vec![]
            } else {
                result.events.iter()
                    .filter(|e| e.dim == dim)
                    .map(|e| e.time)
                    .collect()
            }
        })
        .collect()
}

pub struct BidAskParallelSimulator<'a, P: MultivariateMarkovianIntensity + Sync> {
    pub process: &'a P,
    pub cond_events_by_dim: &'a [Vec<f64>],
    pub cond_external_events: Option<&'a MultivariateSimulationResult>,
    pub new_external_events: Option<&'a MultivariateSimulationResult>,
    pub time_horizon: f64,
    pub initial_q_a: u32,
    pub initial_q_b: u32,

    pub reference_paths: &'a BidAskQueuePath,

    pub tail_impact: &'a BidAskTailImpact,

    pub ask_market_orders: &'a [f64],

    pub bid_market_orders: &'a [f64],

    pub simulating_bar_q: bool,
}

impl<'a, P: MultivariateMarkovianIntensity + Sync> BidAskParallelSimulator<'a, P> {
    pub fn run(&self, n_simulations: usize) -> BidAskSimulationResults {

        let has_conditioning = self.cond_events_by_dim.iter().any(|v| !v.is_empty());

        let results: Vec<(Vec<u32>, Vec<u32>, Vec<f64>, Vec<f64>)> = (0..n_simulations)
            .into_par_iter()
            .map(|sim_idx| {
                // If no conditioning events, use direct simulation with externals.
                // The conditional simulator only works properly when there ARE conditioning events: conditionning on nothing is not the same as
                // unconditional path, it is conditioning on q having zero event.
                let sim_result = if has_conditioning {
                    let ctx = ConditionalSimulationContext::new(
                        self.process,
                        self.cond_events_by_dim,
                        self.cond_external_events,
                        self.new_external_events,
                        self.time_horizon,
                    );
                    ctx.simulate(None, Some(sim_idx as u64))
                } else {
                    // No conditioning: simulate fresh with new external events
                    use crate::simulation::simulate_with_externals;
                    use crate::simulation_helpers::merge_bidask_events;

                    let internal_result = if let Some(ext) = self.new_external_events {
                        simulate_with_externals(self.process, self.time_horizon, ext, Some(sim_idx as u64))
                    } else {
                        crate::simulation::simulate(self.process, self.time_horizon, Some(sim_idx as u64))
                    };

                    // Merge internal events with external events for queue path reconstruction
                    if let Some(ext) = self.new_external_events {
                        merge_bidask_events(&internal_result, ext)
                    } else {
                        internal_result
                    }
                };
                let sim_paths = AffineBidAskQueueProcess::result_to_queue_paths(
                    &sim_result, self.initial_q_a, self.initial_q_b
                );

                // Compute impact using eigenvalue decomposition
                let impact = if self.simulating_bar_q {
                    // with_us: reference is q, simulated is bar_q (q')
                    BidAskImpactPath::new(
                        &self.reference_paths.ask,
                        &self.reference_paths.bid,
                        &sim_paths.ask,
                        &sim_paths.bid,
                        self.tail_impact,
                    )
                } else {
                    // without_us: simulated is q, reference is bar_q (q')
                    BidAskImpactPath::new(
                        &sim_paths.ask,
                        &sim_paths.bid,
                        &self.reference_paths.ask,
                        &self.reference_paths.bid,
                        self.tail_impact,
                    )
                };

                let ask_samples = sample_ask_queue_at_times(&sim_paths, self.ask_market_orders);
                let bid_samples = sample_bid_queue_at_times(&sim_paths, self.bid_market_orders);

                (ask_samples, bid_samples, impact.ask_impact, impact.bid_impact)
            })
            .collect();

        BidAskSimulationResults {
            ask_queue_samples: results.iter().map(|(a, _, _, _)| a.clone()).collect(),
            bid_queue_samples: results.iter().map(|(_, b, _, _)| b.clone()).collect(),
            ask_impact_paths: results.iter().map(|(_, _, ai, _)| ai.clone()).collect(),
            bid_impact_paths: results.iter().map(|(_, _, _, bi)| bi.clone()).collect(),
        }
    }

    pub fn run_memory_efficient(&self, n_simulations: usize) -> BidAskMemoryEfficientResults
    where
        P::State: AsRef<[f64]> + Send,
    {
        let ref_ask_at_ask = sample_ask_queue_at_times(self.reference_paths, self.ask_market_orders);
        let ref_bid_at_ask = sample_bid_queue_at_times(self.reference_paths, self.ask_market_orders);
        let ref_ask_at_bid = sample_ask_queue_at_times(self.reference_paths, self.bid_market_orders);
        let ref_bid_at_bid = sample_bid_queue_at_times(self.reference_paths, self.bid_market_orders);

        let results: Vec<(Vec<u32>, Vec<u32>, Vec<f64>, Vec<f64>)> = (0..n_simulations)
            .into_par_iter()
            .map(|sim_idx| {
                let ctx = ConditionalSimulationContext::new(
                    self.process,
                    self.cond_events_by_dim,
                    self.cond_external_events,
                    self.new_external_events,
                    self.time_horizon,
                );

                // Sample at ask market order times
                let (sim_ask_at_ask, sim_bid_at_ask) = ctx.simulate_bidask_queue_at_times(
                    self.ask_market_orders,
                    self.initial_q_a,
                    self.initial_q_b,
                    None,  // Use default initial state
                    Some(sim_idx as u64),
                );

                // Sample at bid market order times (need separate simulation with same seed for same path)
                let (sim_ask_at_bid, sim_bid_at_bid) = ctx.simulate_bidask_queue_at_times(
                    self.bid_market_orders,
                    self.initial_q_a,
                    self.initial_q_b,
                    None,  // Use default initial state
                    Some(sim_idx as u64),
                );

                // Compute impact using pre-sampled values
                let impact = if self.simulating_bar_q {
                    // with_us: reference is q, simulated is bar_q (q')
                    BidAskImpactPath::from_queue_samples(
                        &ref_ask_at_ask, &ref_bid_at_ask, &sim_ask_at_ask, &sim_bid_at_ask,
                        &ref_ask_at_bid, &ref_bid_at_bid, &sim_ask_at_bid, &sim_bid_at_bid,
                        self.tail_impact,
                    )
                } else {
                    // without_us: simulated is q, reference is bar_q (q')
                    BidAskImpactPath::from_queue_samples(
                        &sim_ask_at_ask, &sim_bid_at_ask, &ref_ask_at_ask, &ref_bid_at_ask,
                        &sim_ask_at_bid, &sim_bid_at_bid, &ref_ask_at_bid, &ref_bid_at_bid,
                        self.tail_impact,
                    )
                };

                (sim_ask_at_ask, sim_bid_at_bid, impact.ask_impact, impact.bid_impact)
            })
            .collect();

        BidAskMemoryEfficientResults {
            ask_queue_samples: results.iter().map(|(a, _, _, _)| a.clone()).collect(),
            bid_queue_samples: results.iter().map(|(_, b, _, _)| b.clone()).collect(),
            ask_impact_paths: results.iter().map(|(_, _, ai, _)| ai.clone()).collect(),
            bid_impact_paths: results.iter().map(|(_, _, _, bi)| bi.clone()).collect(),
        }
    }
}

pub fn write_bidask_results(
    results: &BidAskSimulationResults,
    reference_ask_samples: &[u32],
    reference_bid_samples: &[u32],
    ask_market_orders: &[f64],
    bid_market_orders: &[f64],
    output_dir: &str,
) -> std::io::Result<()> {
    let n_ask_times = ask_market_orders.len();
    let n_bid_times = bid_market_orders.len();
    let n_simulations = results.ask_queue_samples.len();

    // Ask impact paths
    let ask_impact_data: Vec<f64> = (0..n_ask_times)
        .flat_map(|t_idx| {
            results.ask_impact_paths.iter().map(move |path| {
                path.get(t_idx).copied().unwrap_or(f64::NAN)
            })
        })
        .collect();
    write_npy_f64(&format!("{}/ask_impact_paths.npy", output_dir), &ask_impact_data, n_ask_times, n_simulations)?;

    // Bid impact paths
    let bid_impact_data: Vec<f64> = (0..n_bid_times)
        .flat_map(|t_idx| {
            results.bid_impact_paths.iter().map(move |path| {
                path.get(t_idx).copied().unwrap_or(f64::NAN)
            })
        })
        .collect();
    write_npy_f64(&format!("{}/bid_impact_paths.npy", output_dir), &bid_impact_data, n_bid_times, n_simulations)?;

    // Ask queue paths
    let ask_queue_data: Vec<u32> = (0..n_ask_times)
        .flat_map(|t_idx| {
            std::iter::once(reference_ask_samples[t_idx])
                .chain(results.ask_queue_samples.iter().map(move |samples| samples[t_idx]))
        })
        .collect();
    write_npy_u32(&format!("{}/ask_queue_paths.npy", output_dir), &ask_queue_data, n_ask_times, n_simulations + 1)?;

    // Bid queue paths
    let bid_queue_data: Vec<u32> = (0..n_bid_times)
        .flat_map(|t_idx| {
            std::iter::once(reference_bid_samples[t_idx])
                .chain(results.bid_queue_samples.iter().map(move |samples| samples[t_idx]))
        })
        .collect();
    write_npy_u32(&format!("{}/bid_queue_paths.npy", output_dir), &bid_queue_data, n_bid_times, n_simulations + 1)?;

    // Times
    write_npy_f64_1d(&format!("{}/ask_times.npy", output_dir), ask_market_orders)?;
    write_npy_f64_1d(&format!("{}/bid_times.npy", output_dir), bid_market_orders)?;

    Ok(())
}

pub fn write_bidask_memory_efficient_results(
    results: &BidAskMemoryEfficientResults,
    reference_ask_samples: &[u32],
    reference_bid_samples: &[u32],
    ask_market_orders: &[f64],
    bid_market_orders: &[f64],
    output_dir: &str,
) -> std::io::Result<()> {
    let n_ask_times = ask_market_orders.len();
    let n_bid_times = bid_market_orders.len();
    let n_simulations = results.ask_queue_samples.len();

    // Ask impact paths
    let ask_impact_data: Vec<f64> = (0..n_ask_times)
        .flat_map(|t_idx| {
            results.ask_impact_paths.iter().map(move |path| {
                path.get(t_idx).copied().unwrap_or(f64::NAN)
            })
        })
        .collect();
    write_npy_f64(&format!("{}/ask_impact_paths.npy", output_dir), &ask_impact_data, n_ask_times, n_simulations)?;

    // Bid impact paths
    let bid_impact_data: Vec<f64> = (0..n_bid_times)
        .flat_map(|t_idx| {
            results.bid_impact_paths.iter().map(move |path| {
                path.get(t_idx).copied().unwrap_or(f64::NAN)
            })
        })
        .collect();
    write_npy_f64(&format!("{}/bid_impact_paths.npy", output_dir), &bid_impact_data, n_bid_times, n_simulations)?;

    // Ask queue paths
    let ask_queue_data: Vec<u32> = (0..n_ask_times)
        .flat_map(|t_idx| {
            std::iter::once(reference_ask_samples[t_idx])
                .chain(results.ask_queue_samples.iter().map(move |samples| samples[t_idx]))
        })
        .collect();
    write_npy_u32(&format!("{}/ask_queue_paths.npy", output_dir), &ask_queue_data, n_ask_times, n_simulations + 1)?;

    // Bid queue paths
    let bid_queue_data: Vec<u32> = (0..n_bid_times)
        .flat_map(|t_idx| {
            std::iter::once(reference_bid_samples[t_idx])
                .chain(results.bid_queue_samples.iter().map(move |samples| samples[t_idx]))
        })
        .collect();
    write_npy_u32(&format!("{}/bid_queue_paths.npy", output_dir), &bid_queue_data, n_bid_times, n_simulations + 1)?;

    // Times
    write_npy_f64_1d(&format!("{}/ask_times.npy", output_dir), ask_market_orders)?;
    write_npy_f64_1d(&format!("{}/bid_times.npy", output_dir), bid_market_orders)?;

    Ok(())
}
