use simulation_project::conditional_impact::TailImpact;
use simulation_project::models::{AffineQueueProcess, MultiExponentialHawkes};
use simulation_project::simulation::{simulate, simulate_with_externals};
use simulation_project::simulation_helpers::{
    create_meta_orders, extract_events_by_dim, hawkes_to_market_orders, merge_events,
    sample_queue_at_times, write_memory_efficient_results, write_queue_samples, ParallelSimulator,
};

use std::time::Instant;

fn main() {
    let t_total = Instant::now();

    // ==========================================================================
    // Configuration
    // ==========================================================================
    let time_horizon = 90.0;
    let n_simulations = 500;
    let initial_queue_size: u32 = 200;

    // Memory-efficient version only supports decoupled mode
    let decoupled = true;

    // Affine queue parameters
    let a_l = 100.0; // λ^L(q) = a_l + b_l * q
    let b_l = -0.275;
    let a_c = 2.0; // λ^C(q) = a_c + b_c * q
    let b_c = 0.125;

    // Hawkes parameters for market orders: these parameters are set to be close to a t ** -1.5 power-law for the kernel.
    let mu = 1.0;
    let alpha = vec![0.065, 0.2, 0.325, 0.65];
    let beta = vec![0.15, 0.60, 2.5, 10.0];

    // Meta orders configuration
    let n_meta: u32 = 270;
    let meta_start = 0.0;
    let meta_end = 2.0 * time_horizon / 3.0;

    // Output file suffix

    println!("=== Paths WITHOUT Us (Memory-Efficient) ===");
    println!(
        "Time horizon: {}, Simulations: {}, Initial queue: {}",
        time_horizon, n_simulations, initial_queue_size
    );

    let _c_lambda = AffineQueueProcess::c_lambda(b_l, b_c);
    println!("c_lambda = {}", _c_lambda);

    assert!(
        decoupled,
        "Memory-efficient version requires decoupled mode"
    );

    // ==========================================================================
    // Create process and simulate bar_q path (with meta orders)
    // ==========================================================================
    let t0 = Instant::now();

    // Create decoupled queue process (state is just [q])
    let process = AffineQueueProcess::new_queue(initial_queue_size as f64, a_l, b_l, a_c, b_c);

    // Pre-simulate Hawkes path (same seed as original: 42)
    let hawkes = MultiExponentialHawkes::new_with_state(
        MultiExponentialHawkes::new(mu, alpha.clone(), beta.clone()).stationary_state(),
        mu,
        alpha.clone(),
        beta.clone(),
    );
    let hawkes_result = simulate(&hawkes, time_horizon, Some(42));
    println!(
        "[TIMING] Hawkes pre-simulation: {:?} ({} events)",
        t0.elapsed(),
        hawkes_result.events.len()
    );
    let hawkes_as_market = hawkes_to_market_orders(&hawkes_result);

    // Create metaorder
    let meta_orders = create_meta_orders(n_meta, meta_start, meta_end);

    // Simulate bar_q path with the metaorder (same seed as original: 42)
    let t0 = Instant::now();
    let bar_q_externals = merge_events(&meta_orders, &hawkes_as_market);
    let bar_q_result_internal =
        simulate_with_externals(&process, time_horizon, &bar_q_externals, Some(42));
    let bar_q_result = merge_events(&bar_q_result_internal, &bar_q_externals);
    let bar_q_path = AffineQueueProcess::result_to_queue_path(&bar_q_result, initial_queue_size);
    println!(
        "[TIMING] bar_q simulation: {:?} ({} events)",
        t0.elapsed(),
        bar_q_path.events.len()
    );

    // Extract market orders (from pre-simulated Hawkes)
    let market_orders: Vec<f64> = hawkes_as_market.events.iter().map(|e| e.time).collect();
    println!("Generated {} market order events", market_orders.len());

    // Build conditioning events (dims 0,1 only - market orders are external)
    let bar_q_events_by_dim = extract_events_by_dim(&bar_q_result_internal, 3, Some(2));

    // ==========================================================================
    // Setup TailImpact
    // ==========================================================================
    let t0 = Instant::now();
    let tail_impact = TailImpact::from_affine_queue(
        mu,
        alpha.clone(),
        beta.clone(),
        b_l,
        b_c,
        market_orders.clone(),
    );
    println!("[TIMING] TailImpact setup: {:?}", t0.elapsed());

    // Build external events for bar_q and q
    let bar_q_external = merge_events(&meta_orders, &hawkes_as_market);
    let q_external = hawkes_as_market.clone();

    // ==========================================================================
    // Run parallel conditional simulations (MEMORY-EFFICIENT)
    // ==========================================================================
    let bar_q_at_market_orders = sample_queue_at_times(&bar_q_path, &market_orders);

    let t0 = Instant::now();
    let simulator = ParallelSimulator {
        process: &process,
        cond_events_by_dim: &bar_q_events_by_dim,
        cond_external_events: Some(&bar_q_external),
        new_external_events: Some(&q_external),
        time_horizon,
        initial_queue_size,
        reference_path: &bar_q_path,
        tail_impact: &tail_impact,
        market_orders: &market_orders,
        simulating_bar_q: false,
    };
    let results = simulator.run_memory_efficient(n_simulations);
    println!(
        "[TIMING] Parallel simulations ({}x, memory-efficient): {:?}",
        n_simulations,
        t0.elapsed()
    );

    // ==========================================================================
    // Output
    // ==========================================================================
    let t0 = Instant::now();
    write_memory_efficient_results(
        &results,
        &bar_q_at_market_orders,
        &market_orders,
        "experiments/passive_impact/load_experiments/data/single/efficient/without",
    )
    .unwrap();
    write_queue_samples(
        &results.queue_samples,
        &bar_q_at_market_orders,
        &market_orders,
        "experiments/queue_simulation/load_experiments/data/single/efficient/without",
    )
    .unwrap();
    println!("[TIMING] Data write: {:?}", t0.elapsed());
    println!("[TIMING] TOTAL: {:?}", t_total.elapsed());
}
