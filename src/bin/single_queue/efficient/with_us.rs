use simulation_project::models::{AffineQueueProcess, MultiExponentialHawkes};
use simulation_project::simulation::{simulate, simulate_with_externals};
use simulation_project::simulation_helpers::{
    hawkes_to_market_orders, merge_events, create_meta_orders,
    extract_events_by_dim, sample_queue_at_times,
    ParallelSimulator, write_memory_efficient_results,
};
use simulation_project::conditional_impact::TailImpact;

use std::time::Instant;

fn main() {
    let t_total = Instant::now();

    // ==========================================================================
    // Configuration
    // ==========================================================================
    let time_horizon = 100.0;
    let n_simulations = 500;
    let initial_queue_size: u32 = 200;

    // Memory-efficient version only supports decoupled mode
    let decoupled = true;

    // Affine queue parameters
    let a_l = 100.0;   // λ^L(q) = a_l + b_l * q
    let b_l = -0.275;
    let a_c = 2.0;     // λ^C(q) = a_c + b_c * q
    let b_c = 0.125;

    // Hawkes parameters for market orders: these parameters are set to be close to a t ** -1.5 power-law for the kernel.
    let mu = 6.0; //1.0;
    //let alpha = vec![0.065, 0.2, 0.325, 0.65];
    //let beta = vec![0.15, 0.60, 2.5, 10.0];
    let alpha = vec![0.000939493885, 0.00709728833, 0.0147626864, 0.0345610486, 0.110927373, 0.498447091, 3.52737838, 29.7662263, 523.666117, 1188582440.0];
    let beta  = vec![0.0083852488, 0.153562388, 0.580891232, 1.93130619, 6.79859487, 24.9704201, 108.954711, 552.110376, 6059.92156, 2592609630.0];

    // Meta orders configuration
    let n_meta: u32 = 375;
    let meta_start = 1.0;
    let meta_end = 3.0 * time_horizon / 4.0;

    println!("=== Paths WITH Us (Memory-Efficient) ===");
    println!("Time horizon: {}, Simulations: {}, Initial queue: {}",
             time_horizon, n_simulations, initial_queue_size);

    let _c_lambda = AffineQueueProcess::c_lambda(b_l, b_c);
    println!("c_lambda = {}", _c_lambda);

    assert!(decoupled, "Memory-efficient version requires decoupled mode");

    // ==========================================================================
    // Create process and simulate q path (without meta orders)
    // ==========================================================================
    let t0 = Instant::now();

    // Create decoupled queue process (state is just [q])
    let process = AffineQueueProcess::new_queue(initial_queue_size as f64, a_l, b_l, a_c, b_c);

    // Pre-simulate Hawkes path
    let hawkes = MultiExponentialHawkes::new_with_state(
        MultiExponentialHawkes::new(mu, alpha.clone(), beta.clone()).stationary_state(),
        mu, alpha.clone(), beta.clone(),
    );
    let hawkes_result = simulate(&hawkes, time_horizon, Some(42));
    println!("[TIMING] Hawkes pre-simulation: {:?} ({} events)", t0.elapsed(), hawkes_result.events.len());
    let hawkes_as_market = hawkes_to_market_orders(&hawkes_result);

    // Simulate q path without meta orders (same seed as original for comparison)
    let t0 = Instant::now();
    let q_result_internal = simulate_with_externals(&process, time_horizon, &hawkes_as_market, Some(42));
    let q_result = merge_events(&q_result_internal, &hawkes_as_market);
    let q_path = AffineQueueProcess::result_to_queue_path(&q_result, initial_queue_size);
    println!("[TIMING] q simulation: {:?} ({} events)", t0.elapsed(), q_path.events.len());

    // Extract market orders (from pre-simulated Hawkes)
    let market_orders: Vec<f64> = hawkes_as_market.events.iter().map(|e| e.time).collect();
    println!("Generated {} market order events", market_orders.len());

    // Build conditioning events (dims 0,1 only - market orders are external)
    let q_events_by_dim = extract_events_by_dim(&q_result_internal, 3, Some(2));

    // ==========================================================================
    // Setup meta orders and TailImpact
    // ==========================================================================
    let meta_orders = create_meta_orders(n_meta, meta_start, meta_end);

    let t0 = Instant::now();
    let tail_impact = TailImpact::from_affine_queue(
        mu, alpha.clone(), beta.clone(), b_l, b_c, market_orders.clone()
    );
    println!("[TIMING] TailImpact setup: {:?}", t0.elapsed());

    // Build external events for q and bar_q
    let bar_q_external = merge_events(&meta_orders, &hawkes_as_market);
    let q_external = hawkes_as_market.clone();

    // ==========================================================================
    // Run parallel conditional simulations (MEMORY-EFFICIENT)
    // ==========================================================================
    let q_at_market_orders = sample_queue_at_times(&q_path, &market_orders);

    let t0 = Instant::now();
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
    let results = simulator.run_memory_efficient(n_simulations);
    println!("[TIMING] Parallel simulations ({}x, memory-efficient): {:?}", n_simulations, t0.elapsed());

    // ==========================================================================
    // Output
    // ==========================================================================
    let t0 = Instant::now();
    write_memory_efficient_results(&results, &q_at_market_orders, &market_orders, "data/single_queue/efficient/with").unwrap();
    println!("[TIMING] Data write: {:?}", t0.elapsed());
    println!("[TIMING] TOTAL: {:?}", t_total.elapsed());
}
