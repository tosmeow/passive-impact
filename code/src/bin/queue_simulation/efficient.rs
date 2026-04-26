//! queue_simulation_efficient — counterfactual queue paths in presence of a metaorder.
//!
//! Same underlying simulation as single_queue_efficient_with_us but without
//! computing the impact curve.

use simulation_project::models::{AffineQueueProcess, MultiExponentialHawkes};
use simulation_project::simulation::{simulate, simulate_with_externals, ConditionalSimulationContext};
use simulation_project::simulation_helpers::{
    hawkes_to_market_orders, merge_events, create_meta_orders,
    extract_events_by_dim, sample_queue_at_times,
};
use simulation_project::utils::{write_npy_f64_1d, write_npy_u32};

use rayon::prelude::*;
use std::time::Instant;

fn main() {
    let t_total = Instant::now();

    // ===== Configuration =====
    let time_horizon = 100.0;
    let n_simulations = 500;
    let initial_queue_size: u32 = 200;

    let a_l = 100.0;
    let b_l = -0.275;
    let a_c = 2.0;
    let b_c = 0.125;

    let mu = 1.0;
    let alpha = vec![0.065, 0.2, 0.325, 0.65];
    let beta = vec![0.15, 0.60, 2.5, 10.0];

    let n_meta: u32 = 375;
    let meta_start = 1.0;
    let meta_end = 4.0 * time_horizon / 5.0;

    println!("=== Queue Simulation (single, efficient) ===");

    // ===== Pre-simulate Hawkes =====
    let hawkes = MultiExponentialHawkes::new_with_state(
        MultiExponentialHawkes::new(mu, alpha.clone(), beta.clone()).stationary_state(),
        mu, alpha.clone(), beta.clone(),
    );
    let hawkes_result = simulate(&hawkes, time_horizon, Some(42));
    let hawkes_as_market = hawkes_to_market_orders(&hawkes_result);

    // ===== Simulate baseline q =====
    let process = AffineQueueProcess::new_queue(initial_queue_size as f64, a_l, b_l, a_c, b_c);
    let q_result_internal = simulate_with_externals(&process, time_horizon, &hawkes_as_market, Some(42));
    let q_result = merge_events(&q_result_internal, &hawkes_as_market);
    let q_path = AffineQueueProcess::result_to_queue_path(&q_result, initial_queue_size);

    // ===== Build evaluation grid (uniform in time) =====
    let n_times = 1000usize;
    let times: Vec<f64> = (0..n_times)
        .map(|i| i as f64 * time_horizon / (n_times as f64 - 1.0))
        .collect();

    // ===== Conditioning + externals =====
    let q_events_by_dim = extract_events_by_dim(&q_result_internal, 3, Some(2));
    let meta_orders = create_meta_orders(n_meta, meta_start, meta_end);
    let bar_q_external = merge_events(&meta_orders, &hawkes_as_market);

    // ===== Run parallel conditional simulations =====
    let q_at_times = sample_queue_at_times(&q_path, &times);

    let t0 = Instant::now();
    let bar_q_paths: Vec<Vec<u32>> = (0..n_simulations).into_par_iter().map(|sim_idx| {
        let ctx = ConditionalSimulationContext::new(
            &process,
            &q_events_by_dim,
            Some(&hawkes_as_market),
            Some(&bar_q_external),
            time_horizon,
        );
        ctx.simulate_queue_at_times(&times, initial_queue_size, None, Some(sim_idx as u64))
    }).collect();
    println!("[TIMING] {} parallel simulations: {:?}", n_simulations, t0.elapsed());

    // ===== Output =====
    let output_dir = "experiments/queue_simulation/load_experiments/data/single/efficient";
    std::fs::create_dir_all(output_dir).unwrap();

    // Queue paths: (n_times, n_simulations + 1) — first column = q, rest = bar_q_sim_i
    let queue_data: Vec<u32> = (0..n_times).flat_map(|t_idx| {
        std::iter::once(q_at_times[t_idx])
            .chain(bar_q_paths.iter().map(move |bar_q| bar_q[t_idx]))
    }).collect();
    write_npy_u32(&format!("{}/queue_paths.npy", output_dir), &queue_data, n_times, n_simulations + 1).unwrap();
    write_npy_f64_1d(&format!("{}/times.npy", output_dir), &times).unwrap();

    println!("[TIMING] TOTAL: {:?}", t_total.elapsed());
}
