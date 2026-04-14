use simulation_project::models::{AffineQueueProcess, MultiExponentialHawkes};
use simulation_project::simulation::{simulate, simulate_with_externals, ConditionalSimulationContext};
use simulation_project::simulation_helpers::{
    hawkes_to_market_orders, merge_events, create_meta_orders, events_to_dim,
    extract_events_by_dim, sample_queue_at_times,
};
use simulation_project::conditional_impact::AggressiveImpactPath;
use simulation_project::utils::{write_npy_f64, write_npy_u32, write_npy_f64_1d};

use rayon::prelude::*;
use std::time::Instant;

fn main() {
    let t_total = Instant::now();

    // ==========================================================================
    // Configuration (same queue/Hawkes parameters as aggressive_impact)
    // ==========================================================================
    let time_horizon = 100.0;
    let n_simulations = 500;
    let initial_queue_size: u32 = 200;

    // Affine queue parameters: lambda^L(q) = a_l + b_l * q, lambda^C(q) = a_c + b_c * q
    let a_l = 100.0;
    let b_l = -0.275;
    let a_c = 2.0;
    let b_c = 0.125;

    // Hawkes parameters for market orders
    let mu = 1.0;
    let alpha = vec![0.065, 0.2, 0.325, 0.65];
    let beta = vec![0.15, 0.60, 2.5, 10.0];

    // Meta orders configuration (aggressive: dim=2, reduce queue)
    let n_meta: u32 = 1000;
    let meta_start = 1.0;
    let meta_end = 3.0 * time_horizon / 4.0;

    // Kappa function: kappa(q) = -c_kappa * q  (purely linear, no constant)
    let c_kappa = 0.1_f64;
    let kappa = |q: f64| -c_kappa * q;

    // bar_kappa: constant weight for the propagator term (fixed small value)
    let bar_kappa = 0.1_f64;

    println!("=== Aggressive Impact Hybrid Experiment ===");
    println!("Time horizon: {}, Simulations: {}, Initial queue: {}",
             time_horizon, n_simulations, initial_queue_size);
    println!("kappa(q) = -{} * q, bar_kappa = {}", c_kappa, bar_kappa);

    let c_lambda = AffineQueueProcess::c_lambda(b_l, b_c);
    println!("c_lambda = {}", c_lambda);

    // ==========================================================================
    // Pre-simulate Hawkes -> market orders (decoupled mode)
    // ==========================================================================
    let t0 = Instant::now();

    let hawkes = MultiExponentialHawkes::new_with_state(
        MultiExponentialHawkes::new(mu, alpha.clone(), beta.clone()).stationary_state(),
        mu, alpha.clone(), beta.clone(),
    );
    let hawkes_result = simulate(&hawkes, time_horizon, Some(42));
    let hawkes_as_market = hawkes_to_market_orders(&hawkes_result);
    println!("[TIMING] Hawkes pre-simulation: {:?} ({} events)", t0.elapsed(), hawkes_result.events.len());

    // ==========================================================================
    // Simulate q path (base queue, no meta orders)
    // ==========================================================================
    let t0 = Instant::now();

    let process = AffineQueueProcess::new_queue(initial_queue_size as f64, a_l, b_l, a_c, b_c);
    let q_result_internal = simulate_with_externals(&process, time_horizon, &hawkes_as_market, Some(42));
    let q_result = merge_events(&q_result_internal, &hawkes_as_market);
    let q_path = AffineQueueProcess::result_to_queue_path(&q_result, initial_queue_size);
    println!("[TIMING] q simulation: {:?} ({} events)", t0.elapsed(), q_path.events.len());

    // ==========================================================================
    // Create aggressive meta orders
    // ==========================================================================
    let meta_orders_raw = create_meta_orders(n_meta, meta_start, meta_end);
    let meta_orders = events_to_dim(&meta_orders_raw, 2, 3);
    let meta_order_times: Vec<f64> = meta_orders.events.iter().map(|e| e.time).collect();
    let market_order_times: Vec<f64> = hawkes_as_market.events.iter().map(|e| e.time).collect();
    println!("Market orders: {}, Meta orders: {}", market_order_times.len(), meta_order_times.len());

    let hawkes_model = MultiExponentialHawkes::new(mu, alpha.clone(), beta.clone());
    let norm: f64 = alpha.iter().zip(&beta).map(|(a, b)| a / b).sum::<f64>();
    println!("Propagator: G(0) = {:.4} (mean cluster size), G(∞) = 1 (permanent)", 1.0 / (1.0 - norm));

    // ==========================================================================
    // Merge market + meta order times into sorted evaluation times
    // ==========================================================================
    let mut eval_entries: Vec<(f64, bool)> = Vec::new();
    for &t in &market_order_times {
        eval_entries.push((t, true));   // is_market_order = true
    }
    for &t in &meta_order_times {
        eval_entries.push((t, false));  // is_market_order = false (meta order)
    }
    eval_entries.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    let eval_times: Vec<f64> = eval_entries.iter().map(|e| e.0).collect();
    let is_market_order: Vec<bool> = eval_entries.iter().map(|e| e.1).collect();
    println!("Evaluation times (merged): {}", eval_times.len());

    // ==========================================================================
    // Sample q at evaluation times (shared across all simulations)
    // ==========================================================================
    let q_at_eval_times = sample_queue_at_times(&q_path, &eval_times);

    // ==========================================================================
    // Build conditioning events and externals
    // ==========================================================================
    let q_events_by_dim = extract_events_by_dim(&q_result_internal, 3, Some(2));
    let bar_q_external = merge_events(&meta_orders, &hawkes_as_market);

    // ==========================================================================
    // Run parallel conditional simulations
    // ==========================================================================
    let t0 = Instant::now();

    let results: Vec<(Vec<u32>, Vec<f64>)> = (0..n_simulations)
        .into_par_iter()
        .map(|sim_idx| {
            let ctx = ConditionalSimulationContext::new(
                &process,
                &q_events_by_dim,
                Some(&hawkes_as_market),
                Some(&bar_q_external),
                time_horizon,
            );

            let q_bar_samples = ctx.simulate_queue_at_times(
                &eval_times,
                initial_queue_size,
                None,
                Some(sim_idx as u64),
            );

            // Hybrid impact: metaorders through propagator with bar_kappa,
            // market orders contribute instantaneous queue-dependent correction.
            let impact_path = AggressiveImpactPath::from_queue_samples_hybrid(
                &q_at_eval_times,
                &q_bar_samples,
                &eval_times,
                &is_market_order,
                &hawkes_model,
                &kappa,
                bar_kappa,
            );

            (q_bar_samples, impact_path.impact_path)
        })
        .collect();

    println!("[TIMING] Parallel simulations ({}x): {:?}", n_simulations, t0.elapsed());

    // ==========================================================================
    // Output
    // ==========================================================================
    let t0 = Instant::now();
    let output_dir = "data/agressive_impact_hybrid";
    std::fs::create_dir_all(output_dir).expect("Failed to create output directory");

    let n_times = eval_times.len();

    // Impact paths: (n_times, n_simulations)
    let impact_data: Vec<f64> = (0..n_times)
        .flat_map(|t_idx| {
            results.iter().map(move |(_, impact)| {
                impact.get(t_idx).copied().unwrap_or(f64::NAN)
            })
        })
        .collect();
    write_npy_f64(&format!("{}/impact_paths.npy", output_dir), &impact_data, n_times, n_simulations).unwrap();

    // Queue paths: (n_times, n_simulations + 1) [first col = q, rest = bar_q_sim_i]
    let queue_data: Vec<u32> = (0..n_times)
        .flat_map(|t_idx| {
            std::iter::once(q_at_eval_times[t_idx])
                .chain(results.iter().map(move |(q_bar, _)| q_bar[t_idx]))
        })
        .collect();
    write_npy_u32(&format!("{}/queue_paths.npy", output_dir), &queue_data, n_times, n_simulations + 1).unwrap();

    // Times
    write_npy_f64_1d(&format!("{}/times.npy", output_dir), &eval_times).unwrap();

    // Event types: 1.0 for market order, 0.0 for meta order
    let event_types: Vec<f64> = is_market_order.iter().map(|&b| if b { 1.0 } else { 0.0 }).collect();
    write_npy_f64_1d(&format!("{}/event_types.npy", output_dir), &event_types).unwrap();

    // bar_kappa value used (scalar, stored as 1-element array for easy Python loading)
    write_npy_f64_1d(&format!("{}/bar_kappa.npy", output_dir), &[bar_kappa]).unwrap();

    println!("[TIMING] Data write: {:?}", t0.elapsed());
    println!("[TIMING] TOTAL: {:?}", t_total.elapsed());
}
