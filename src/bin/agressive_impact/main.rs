use simulation_project::models::{AffineQueueProcess, MultiExponentialHawkes};
use simulation_project::simulation::{simulate, simulate_with_externals, ConditionalSimulationContext};
use simulation_project::simulation_helpers::{
    hawkes_to_market_orders, merge_events, create_meta_orders, events_to_dim,
    extract_events_by_dim, sample_queue_at_times,
};
use simulation_project::conditional_impact::propagator::Propagator;
use simulation_project::utils::{write_npy_f64, write_npy_u32, write_npy_f64_1d};

use rayon::prelude::*;
use std::time::Instant;

/// Compute the aggressive market impact MI(t) using the propagator.
///
/// MI(t) = int_0^t [kappa(q_bar_s) - kappa(q_s)] xi(t-s) dN_s
///       + int_0^t kappa(q_bar_s) xi(t-s) dN^o_s
///
/// where xi(u) = delta_0(u) + sum_j c_j e^{-lambda_j u} is the propagator,
/// and kappa(q) = c_kappa * q + d_kappa.
fn compute_aggressive_impact(
    q_samples: &[u32],
    q_bar_samples: &[u32],
    eval_times: &[f64],
    is_market_order: &[bool],
    propagator_lambda: &[f64],
    propagator_c: &[f64],
    c_kappa: f64,
    d_kappa: f64,
) -> Vec<f64> {
    let n = eval_times.len();
    let n_components = propagator_lambda.len();
    let mut state = vec![0.0f64; n_components];
    let mut instantaneous = 0.0f64;
    let mut impact_path = Vec::with_capacity(n);
    let mut prev_t = 0.0f64;

    for idx in 0..n {
        let t = eval_times[idx];
        let dt = t - prev_t;

        // Decay propagator state
        for j in 0..n_components {
            state[j] *= (-propagator_lambda[j] * dt).exp();
        }

        let q = q_samples[idx] as f64;
        let q_bar = q_bar_samples[idx] as f64;
        let kappa_q = c_kappa * q + d_kappa;
        let kappa_q_bar = c_kappa * q_bar + d_kappa;

        if is_market_order[idx] {
            // Market order event (dN): contribute kappa(q_bar) - kappa(q)
            let contribution = kappa_q_bar - kappa_q;
            instantaneous += contribution;
            for j in 0..n_components {
                state[j] += propagator_c[j] * contribution;
            }
        } else {
            // Meta order event (dN^o): contribute kappa(q_bar)
            instantaneous += kappa_q_bar;
            for j in 0..n_components {
                state[j] += propagator_c[j] * kappa_q_bar;
            }
        }

        let propagator_term: f64 = state.iter().sum();
        impact_path.push(instantaneous + propagator_term);
        prev_t = t;
    }

    impact_path
}

fn main() {
    let t_total = Instant::now();

    // ==========================================================================
    // Configuration (same queue/Hawkes parameters as single_queue)
    // ==========================================================================
    let time_horizon = 100.0;
    let n_simulations = 500;
    let initial_queue_size: u32 = 200;

    // Affine queue parameters: lambda^L(q) = a_l + b_l * q, lambda^C(q) = a_c + b_c * q
    let a_l = 100.0;
    let b_l = -0.275;
    let a_c = 2.0;
    let b_c = 0.125;

    // Hawkes parameters for market orders (close to t^{-1.5} power-law kernel)
    let mu = 1.0;
    let alpha = vec![0.065, 0.2, 0.325, 0.65];
    let beta = vec![0.15, 0.60, 2.5, 10.0];

    // Meta orders configuration (aggressive: dim=2, reduce queue)
    let n_meta: u32 = 100;
    let meta_start = 1.0;
    let meta_end = 3.0 * time_horizon / 4.0;

    // Kappa function: kappa(q) = c_kappa * q + d_kappa
    // Negative c_kappa: smaller queue means larger impact per order
    let c_kappa = -0.005;
    let d_kappa = 1.0;

    println!("=== Aggressive Impact Experiment ===");
    println!("Time horizon: {}, Simulations: {}, Initial queue: {}",
             time_horizon, n_simulations, initial_queue_size);
    println!("kappa(q) = {} * q + {}", c_kappa, d_kappa);

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
    // Create aggressive meta orders and compute propagator
    // ==========================================================================
    // Meta orders are market orders (dim=2) that reduce the queue
    let meta_orders_raw = create_meta_orders(n_meta, meta_start, meta_end);
    let meta_orders = events_to_dim(&meta_orders_raw, 2, 3);
    let meta_order_times: Vec<f64> = meta_orders.events.iter().map(|e| e.time).collect();
    let market_order_times: Vec<f64> = hawkes_as_market.events.iter().map(|e| e.time).collect();
    println!("Market orders: {}, Meta orders: {}", market_order_times.len(), meta_order_times.len());

    let t0 = Instant::now();
    let hawkes_model = MultiExponentialHawkes::new(mu, alpha.clone(), beta.clone());
    let propagator = Propagator::new(hawkes_model);
    println!("[TIMING] Propagator computation: {:?}", t0.elapsed());
    println!("Propagator: lambda = {:?}", propagator.lambda);
    println!("Propagator: c = {:?}", propagator.c);

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
    // Condition on dims 0,1 (limit, cancel) - dim 2 (market) is external
    let q_events_by_dim = extract_events_by_dim(&q_result_internal, 3, Some(2));

    // bar_q external events = meta orders + Hawkes market orders
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
                Some(&hawkes_as_market),     // q conditioning externals
                Some(&bar_q_external),        // bar_q externals (meta + hawkes)
                time_horizon,
            );

            // Simulate bar_q at evaluation times (memory-efficient)
            let q_bar_samples = ctx.simulate_queue_at_times(
                &eval_times,
                initial_queue_size,
                None,
                Some(sim_idx as u64),
            );

            // Compute aggressive impact using propagator
            let impact = compute_aggressive_impact(
                &q_at_eval_times,
                &q_bar_samples,
                &eval_times,
                &is_market_order,
                &propagator.lambda,
                &propagator.c,
                c_kappa,
                d_kappa,
            );

            (q_bar_samples, impact)
        })
        .collect();

    println!("[TIMING] Parallel simulations ({}x): {:?}", n_simulations, t0.elapsed());

    // ==========================================================================
    // Output
    // ==========================================================================
    let t0 = Instant::now();
    let output_dir = "data/agressive_impact";
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

    println!("[TIMING] Data write: {:?}", t0.elapsed());
    println!("[TIMING] TOTAL: {:?}", t_total.elapsed());
}
