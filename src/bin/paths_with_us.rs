use simulation_project::models::{
    AffineQueueProcess,
    MultivariateEvent, MultivariateSimulationResult,
};
use simulation_project::simulation::{simulate, ConditionalSimulationContext};
use simulation_project::conditional_impact::{TailImpact, ImpactPath};
use simulation_project::utils::{write_npy_f64, write_npy_u32, write_npy_f64_1d};

use rayon::prelude::*;
use std::time::Instant;

fn main() {
    let t_total = Instant::now();
    // Configuration
    let time_horizon = 250.0;
    let n_simulations = 500;
    let initial_queue_size: u32 = 250;

    // Affine queue parameters
    let a_l = 100.0;   // λ^L(q) = a_l + b_l * q
    let b_l = -0.275;
    let a_c = 2.0;    // λ^C(q) = a_c + b_c * q
    let b_c = 0.125;

    // Hawkes parameters for market orders
    let mu = 1.0;
    let alpha = vec![0.065, 0.2, 0.325, 0.65];//vec![0.6, 0.6, 1.35];
    let beta = vec![0.15, 0.60, 2.5, 10.0];//vec![1.0, 3.0, 9.0];


    // Compute c_lambda from affine slopes
    let _c_lambda = AffineQueueProcess::c_lambda(b_l, b_c);
    println!("c_lambda = {}", _c_lambda);

    // Create the queue process
    let process = AffineQueueProcess::new(
        initial_queue_size as f64,
        a_l, b_l, a_c, b_c,
        mu, alpha.clone(), beta.clone(),
    );

    // 1) Simulate q path (without meta orders)
    let t0 = Instant::now();
    let q_result = simulate(&process, time_horizon, None);
    let q_path = AffineQueueProcess::result_to_queue_path(&q_result, initial_queue_size);
    println!("[TIMING] Initial simulation: {:?}", t0.elapsed());
    println!("Generated q path with {} events", q_path.events.len());

    // Extract market orders (dimension 2) for impact calculation
    let market_orders: Vec<f64> = q_result.events.iter()
        .filter(|e| e.dim == 2)
        .map(|e| e.time)
        .collect();
    println!("Generated {} market order events", market_orders.len());

    // Meta orders - trader's limit orders that ADD to the queue (dim 0)
    // These are external events ONLY for bar_q, not for the original q
    let n: u32 = 250;
    let mut meta_orders = MultivariateSimulationResult::new(3);
    for i in 0..n {
        meta_orders.push(MultivariateEvent {
            time: time_horizon / 4.0 + (i as f64 / (n - 1) as f64) * (time_horizon / 2.0),
            dim: 0
        });
    }

    // 2) Prepare TailImpact for impact path calculation
    let t0 = Instant::now();
    let tail_impact = TailImpact::from_affine_queue(
        mu, alpha.clone(), beta.clone(), b_l, b_c, market_orders.clone()
    );
    println!("[TIMING] TailImpact setup: {:?}", t0.elapsed());

    // 3) Run conditional simulations in parallel
    // Simulate bar_q (with meta orders) conditioned on q (without meta orders)
    //
    // KEY INSIGHT: The meta orders (dim 0) only affect the queue size, NOT the Hawkes
    // intensity (which is driven by dim 2 market orders). Therefore:
    // - Original q: No external events
    // - Conditional bar_q: Meta orders as external events on dim 0
    // - The Hawkes process (dim 2) will naturally produce identical events in both paths
    //   because the dim 0 events don't influence the Hawkes state
    // - Only dims 0 and 1 (limit and cancel) will differ due to the meta orders

    // Events from q to condition on (all 3 dimensions)
    let q_events_by_dim: Vec<Vec<f64>> = (0..3)
        .map(|dim| q_result.events.iter().filter(|e| e.dim == dim).map(|e| e.time).collect())
        .collect();

    // For detailed analysis, run one simulation with diagnostics
    println!("\n=== Detailed Divergence Analysis (sim_0) ===");
    {
        let ctx = ConditionalSimulationContext::new(
            &process,
            &q_events_by_dim,
            None,
            Some(&meta_orders),
            time_horizon,
        );
        let bar_q_result = ctx.simulate(Some(0));

        // Count events by dimension and by time window
        let meta_start = time_horizon / 4.0;
        let meta_end = 3.0 * time_horizon / 4.0;

        let q_events_before: Vec<_> = q_result.events.iter().filter(|e| e.time < meta_start).collect();
        let q_events_during: Vec<_> = q_result.events.iter().filter(|e| e.time >= meta_start && e.time <= meta_end).collect();
        let q_events_after: Vec<_> = q_result.events.iter().filter(|e| e.time > meta_end).collect();

        let bar_q_events_before: Vec<_> = bar_q_result.events.iter().filter(|e| e.time < meta_start).collect();
        let bar_q_events_during: Vec<_> = bar_q_result.events.iter().filter(|e| e.time >= meta_start && e.time <= meta_end).collect();
        let bar_q_events_after: Vec<_> = bar_q_result.events.iter().filter(|e| e.time > meta_end).collect();

        println!("Time window: [0, {:.1}) - before meta orders", meta_start);
        for dim in 0..3 {
            let q_count = q_events_before.iter().filter(|e| e.dim == dim).count();
            let bar_q_count = bar_q_events_before.iter().filter(|e| e.dim == dim).count();
            println!("  dim {}: q={}, bar_q={}, diff={}", dim, q_count, bar_q_count, bar_q_count as i64 - q_count as i64);
        }

        println!("Time window: [{:.1}, {:.1}] - during meta orders (+{} external)", meta_start, meta_end, n);
        for dim in 0..3 {
            let q_count = q_events_during.iter().filter(|e| e.dim == dim).count();
            let bar_q_count = bar_q_events_during.iter().filter(|e| e.dim == dim).count();
            println!("  dim {}: q={}, bar_q={}, diff={}", dim, q_count, bar_q_count, bar_q_count as i64 - q_count as i64);
        }

        println!("Time window: ({:.1}, {:.1}] - after meta orders", meta_end, time_horizon);
        for dim in 0..3 {
            let q_count = q_events_after.iter().filter(|e| e.dim == dim).count();
            let bar_q_count = bar_q_events_after.iter().filter(|e| e.dim == dim).count();
            println!("  dim {}: q={}, bar_q={}, diff={}", dim, q_count, bar_q_count, bar_q_count as i64 - q_count as i64);
        }

        // Check queue values at key times
        let bar_q_path_diag = AffineQueueProcess::result_to_queue_path(&bar_q_result, initial_queue_size);
        println!("\nQueue values at key times:");
        println!("  t=0: q={}, bar_q={}", q_path.queue_at_time(0.0), bar_q_path_diag.queue_at_time(0.0));
        println!("  t={:.1} (meta start): q={}, bar_q={}", meta_start, q_path.queue_at_time(meta_start), bar_q_path_diag.queue_at_time(meta_start));
        println!("  t={:.1} (meta end): q={}, bar_q={}", meta_end, q_path.queue_at_time(meta_end), bar_q_path_diag.queue_at_time(meta_end));
        println!("  t={:.1} (end): q={}, bar_q={}", time_horizon, q_path.queue_at_time(time_horizon), bar_q_path_diag.queue_at_time(time_horizon));
    }
    println!("=== End Detailed Analysis ===\n");

    // Pre-compute q values at each market order time (once, not per simulation)
    let q_at_market_orders: Vec<u32> = market_orders.iter()
        .map(|&t| q_path.queue_at_time(t))
        .collect();

    let t0 = Instant::now();
    let results: Vec<(Vec<u32>, Vec<f64>, Vec<f64>)> = (0..n_simulations)
        .into_par_iter()
        .map(|sim_idx| {
            let ctx = ConditionalSimulationContext::new(
                &process,
                &q_events_by_dim,
                None,                   // q had no external events
                Some(&meta_orders),     // bar_q has meta orders as external events
                time_horizon,
            );

            let bar_q_result = ctx.simulate(Some(sim_idx as u64));
            let bar_q_path = AffineQueueProcess::result_to_queue_path(&bar_q_result, initial_queue_size);
            let impact_path = ImpactPath::new(q_path.clone(), bar_q_path.clone(), &tail_impact);

            // Sample bar_q at market order times ONCE during this iteration
            let bar_q_at_market_orders: Vec<u32> = market_orders.iter()
                .map(|&t| bar_q_path.queue_at_time(t))
                .collect();

            // Extract market orders (dim 2) from bar_q for comparison
            let bar_q_market_orders: Vec<f64> = bar_q_result.events.iter()
                .filter(|e| e.dim == 2)
                .map(|e| e.time)
                .collect();

            (bar_q_at_market_orders, impact_path.impact_path, bar_q_market_orders)
        })
        .collect();
    println!("[TIMING] Parallel simulations ({}x): {:?}", n_simulations, t0.elapsed());

    println!("Completed all {} simulations in parallel", n_simulations);

    // Check if market orders (dim 2) are identical between q and bar_q
    println!("\n=== Market Order (dim 2) Comparison ===");
    println!("q has {} market orders", market_orders.len());

    for (sim_idx, (_, _, bar_q_mo)) in results.iter().enumerate().take(5) {
        let matches = bar_q_mo.len() == market_orders.len()
            && bar_q_mo.iter().zip(market_orders.iter()).all(|(a, b)| (a - b).abs() < 1e-10);
        println!("sim_{}: bar_q has {} market orders, identical={}",
            sim_idx, bar_q_mo.len(), matches);

        if !matches && bar_q_mo.len() > 0 {
            // Show first few differences
            let mut diff_count = 0;
            for (i, (bq, q)) in bar_q_mo.iter().zip(market_orders.iter()).enumerate() {
                if (bq - q).abs() >= 1e-10 {
                    println!("  Event {}: bar_q={:.6}, q={:.6}", i, bq, q);
                    diff_count += 1;
                    if diff_count >= 3 { break; }
                }
            }
            if bar_q_mo.len() != market_orders.len() {
                println!("  Length mismatch: bar_q={}, q={}", bar_q_mo.len(), market_orders.len());
            }
        }
    }
    println!("=== End Comparison ===\n");

    // Separate the results - bar_q values are already sampled at market order times
    let all_bar_q_samples: Vec<_> = results.iter().map(|(bq, _, _)| bq).collect();
    let all_impact_paths: Vec<_> = results.iter().map(|(_, ip, _)| ip).collect();

    let n_times = market_orders.len();

    // 4) Store impact paths as .npy (n_times x n_simulations)
    let t0 = Instant::now();
    let impact_data: Vec<f64> = (0..n_times)
        .flat_map(|t_idx| {
            all_impact_paths.iter().map(move |path| {
                path.get(t_idx).copied().unwrap_or(f64::NAN)
            })
        })
        .collect();
    write_npy_f64("impact_paths.npy", &impact_data, n_times, n_simulations).unwrap();
    println!("Saved impact paths to impact_paths.npy");

    // 5) Store queue paths as .npy
    // First column is q (reference), rest are bar_q simulations
    // Data is already sampled - just interleave
    let queue_data: Vec<u32> = (0..n_times)
        .flat_map(|t_idx| {
            std::iter::once(q_at_market_orders[t_idx])
                .chain(all_bar_q_samples.iter().map(move |bq_samples| bq_samples[t_idx]))
        })
        .collect();
    write_npy_u32("queue_paths.npy", &queue_data, n_times, n_simulations + 1).unwrap();
    println!("Saved queue paths to queue_paths.npy");

    // 6) Store time index separately
    write_npy_f64_1d("times.npy", &market_orders).unwrap();
    println!("[TIMING] Data prep + write: {:?}", t0.elapsed());
    println!("[TIMING] TOTAL: {:?}", t_total.elapsed());
}
