use simulation_project::models::{AffineQueueProcess, MultiExponentialHawkes};
use simulation_project::simulation::{simulate, simulate_with_externals, ConditionalSimulationContext};
use simulation_project::simulation_helpers::{
    hawkes_to_market_orders, merge_events, create_meta_orders,
    extract_events_by_dim, sample_queue_at_times,
};
use simulation_project::conditional_impact::{TailImpact, ImpactPath};
use simulation_project::conditional_impact::propagator::Propagator;
use simulation_project::utils::{write_npy_f64, write_npy_u32, write_npy_f64_1d};

use rayon::prelude::*;
use std::time::Instant;

/// Model B: Propagator-based passive impact with kappa(q) = q.
///
/// I(t_i) = sum_{j<=i} (q_bar_j - q_j) * [1 + sum_k c_k e^{-lambda_k (t_i - t_j)}]
///
/// Tracked via Markovian propagator state.
fn compute_propagator_impact(
    q_samples: &[u32],
    q_bar_samples: &[u32],
    market_times: &[f64],
    propagator_lambda: &[f64],
    propagator_c: &[f64],
) -> Vec<f64> {
    let n = market_times.len();
    let n_comp = propagator_lambda.len();
    let mut state = vec![0.0f64; n_comp];
    let mut instantaneous = 0.0f64;
    let mut impact_path = Vec::with_capacity(n);
    let mut prev_t = 0.0f64;

    for i in 0..n {
        let t = market_times[i];
        let dt = t - prev_t;

        // Decay propagator state
        for j in 0..n_comp {
            state[j] *= (-propagator_lambda[j] * dt).exp();
        }

        let diff = q_bar_samples[i] as f64 - q_samples[i] as f64;
        instantaneous += diff;
        for j in 0..n_comp {
            state[j] += propagator_c[j] * diff;
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
    // Configuration (identical to single_queue)
    // ==========================================================================
    let time_horizon = 100.0;
    let n_simulations = 500;
    let initial_queue_size: u32 = 200;

    // Affine queue parameters
    let a_l = 100.0;   // lambda^L(q) = a_l + b_l * q
    let b_l = -0.275;
    let a_c = 2.0;     // lambda^C(q) = a_c + b_c * q
    let b_c = 0.125;

    // Hawkes parameters for market orders
    let mu = 1.0;
    let alpha = vec![0.065, 0.2, 0.325, 0.65];
    let beta = vec![0.15, 0.60, 2.5, 10.0];

    // Meta orders configuration (passive: dim=0, limit orders)
    let n_meta: u32 = 375;
    let meta_start = 1.0;
    let meta_end = 3.0 * time_horizon / 4.0;

    println!("=== Comparative Pricing Models ===");
    println!("Time horizon: {}, Simulations: {}, Initial queue: {}",
             time_horizon, n_simulations, initial_queue_size);
    println!("kappa(q) = q for both models");

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

    // Extract market orders
    let market_orders: Vec<f64> = hawkes_as_market.events.iter().map(|e| e.time).collect();
    println!("Market orders: {}", market_orders.len());

    // Build conditioning events (dims 0,1 only)
    let q_events_by_dim = extract_events_by_dim(&q_result_internal, 3, Some(2));

    // ==========================================================================
    // Setup passive meta orders
    // ==========================================================================
    let meta_orders = create_meta_orders(n_meta, meta_start, meta_end);

    // Build external events for q and bar_q
    let bar_q_external = merge_events(&meta_orders, &hawkes_as_market);

    // ==========================================================================
    // Setup Model A (TailImpact) and Model B (Propagator)
    // ==========================================================================
    let t0 = Instant::now();
    let tail_impact = TailImpact::from_affine_queue(
        mu, alpha.clone(), beta.clone(), b_l, b_c, market_orders.clone()
    );
    println!("[TIMING] TailImpact (Model A) setup: {:?}", t0.elapsed());

    let t0 = Instant::now();
    let hawkes_model = MultiExponentialHawkes::new(mu, alpha.clone(), beta.clone());
    let propagator = Propagator::new(hawkes_model);
    println!("[TIMING] Propagator (Model B) setup: {:?}", t0.elapsed());
    println!("Propagator: lambda = {:?}", propagator.lambda);
    println!("Propagator: c = {:?}", propagator.c);

    // ==========================================================================
    // Sample q at market order times (shared)
    // ==========================================================================
    let q_at_market_orders = sample_queue_at_times(&q_path, &market_orders);

    // ==========================================================================
    // Run parallel conditional simulations, compute BOTH impact models
    // ==========================================================================
    let t0 = Instant::now();

    let results: Vec<(Vec<u32>, Vec<f64>, Vec<f64>)> = (0..n_simulations)
        .into_par_iter()
        .map(|sim_idx| {
            let ctx = ConditionalSimulationContext::new(
                &process,
                &q_events_by_dim,
                Some(&hawkes_as_market),
                Some(&bar_q_external),
                time_horizon,
            );

            // Simulate bar_q at market order times
            let q_bar_samples = ctx.simulate_queue_at_times(
                &market_orders,
                initial_queue_size,
                None,
                Some(sim_idx as u64),
            );

            // Model A: TailImpact decomposition
            let impact_a = ImpactPath::from_queue_samples(
                &q_at_market_orders,
                &q_bar_samples,
                &tail_impact,
            ).impact_path;

            // Model B: Propagator-based
            let impact_b = compute_propagator_impact(
                &q_at_market_orders,
                &q_bar_samples,
                &market_orders,
                &propagator.lambda,
                &propagator.c,
            );

            (q_bar_samples, impact_a, impact_b)
        })
        .collect();

    println!("[TIMING] Parallel simulations ({}x): {:?}", n_simulations, t0.elapsed());

    // ==========================================================================
    // Output
    // ==========================================================================
    let t0 = Instant::now();
    let output_dir = "data/comparative_pricing_models";
    std::fs::create_dir_all(output_dir).expect("Failed to create output directory");

    let n_times = market_orders.len();

    // Model A impact paths
    let impact_a_data: Vec<f64> = (0..n_times)
        .flat_map(|t_idx| {
            results.iter().map(move |(_, impact_a, _)| {
                impact_a.get(t_idx).copied().unwrap_or(f64::NAN)
            })
        })
        .collect();
    write_npy_f64(&format!("{}/impact_model_a.npy", output_dir), &impact_a_data, n_times, n_simulations).unwrap();

    // Model B impact paths
    let impact_b_data: Vec<f64> = (0..n_times)
        .flat_map(|t_idx| {
            results.iter().map(move |(_, _, impact_b)| {
                impact_b.get(t_idx).copied().unwrap_or(f64::NAN)
            })
        })
        .collect();
    write_npy_f64(&format!("{}/impact_model_b.npy", output_dir), &impact_b_data, n_times, n_simulations).unwrap();

    // Queue paths (shared)
    let queue_data: Vec<u32> = (0..n_times)
        .flat_map(|t_idx| {
            std::iter::once(q_at_market_orders[t_idx])
                .chain(results.iter().map(move |(q_bar, _, _)| q_bar[t_idx]))
        })
        .collect();
    write_npy_u32(&format!("{}/queue_paths.npy", output_dir), &queue_data, n_times, n_simulations + 1).unwrap();

    // Times
    write_npy_f64_1d(&format!("{}/times.npy", output_dir), &market_orders).unwrap();

    println!("[TIMING] Data write: {:?}", t0.elapsed());
    println!("[TIMING] TOTAL: {:?}", t_total.elapsed());
}
