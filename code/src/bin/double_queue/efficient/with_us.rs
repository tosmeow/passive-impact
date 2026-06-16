use simulation_project::conditional_impact::{BidAskTailImpact, SymmetricCMatrix};
use simulation_project::models::{
    AffineBidAskQueueProcess, AffineIntensityParams, BidAskAffineParams, MultiExponentialHawkes,
};
use simulation_project::simulation::{simulate, simulate_with_externals};
use simulation_project::simulation_helpers::{
    create_bidask_meta_orders, extract_bidask_events_by_dim, hawkes_pair_to_market_orders,
    merge_bidask_events, sample_ask_queue_at_times, sample_bid_queue_at_times,
    write_bidask_memory_efficient_results, BidAskParallelSimulator, Side,
};

use std::time::Instant;

fn main() {
    let t_total = Instant::now();

    // ==========================================================================
    // Configuration
    // ==========================================================================
    let time_horizon = 90.0;
    let n_simulations = 500;
    let initial_q_a: u32 = 350;
    let initial_q_b: u32 = 350;

    // Memory-efficient version only supports decoupled mode
    let decoupled = true;

    // Affine queue parameters (symmetric)
    let a_l = 100.0;
    let b_l_own = -0.15;
    let b_l_cross = 0.05;

    let a_c = 2.0;
    let b_c_own = 0.10;
    let b_c_cross = 0.02;

    // Hawkes parameters (same for both sides)
    let mu = 1.0;
    let alpha = vec![0.065, 0.2, 0.325, 0.65];
    let beta = vec![0.15, 0.60, 2.5, 10.0];

    // Meta orders configuration (on ask side only)
    let n_meta: u32 = 270;
    let meta_start = 1.0;
    let meta_end = 2.0 * time_horizon / 3.0;
    let meta_side = Side::Ask;

    println!("=== Bid-Ask Paths WITH Us (Memory-Efficient) ===");
    println!(
        "Time horizon: {}, Simulations: {}",
        time_horizon, n_simulations
    );
    println!("Initial queues: ask={}, bid={}", initial_q_a, initial_q_b);

    assert!(
        decoupled,
        "Memory-efficient version requires decoupled mode"
    );

    // Build C matrix for impact computation
    let c_matrix = SymmetricCMatrix::from_affine_symmetric(b_l_own, b_l_cross, b_c_own, b_c_cross);
    println!("C matrix: c={}, a={}", c_matrix.c, c_matrix.a);

    // ==========================================================================
    // Create process and simulate q paths (without meta orders)
    // ==========================================================================
    let t0 = Instant::now();

    let params = BidAskAffineParams {
        lambda_l_a: AffineIntensityParams::new(a_l, b_l_own, b_l_cross),
        lambda_c_a: AffineIntensityParams::new(a_c, b_c_own, b_c_cross),
        lambda_l_b: AffineIntensityParams::new(a_l, b_l_cross, b_l_own),
        lambda_c_b: AffineIntensityParams::new(a_c, b_c_cross, b_c_own),
    };

    let process =
        AffineBidAskQueueProcess::new_queue(initial_q_a as f64, initial_q_b as f64, params.clone());

    // Pre-simulate both Hawkes paths (same seeds as original for comparison)
    let hawkes_a = MultiExponentialHawkes::new_with_state(
        MultiExponentialHawkes::new(mu, alpha.clone(), beta.clone()).stationary_state(),
        mu,
        alpha.clone(),
        beta.clone(),
    );
    let hawkes_b = MultiExponentialHawkes::new_with_state(
        MultiExponentialHawkes::new(mu, alpha.clone(), beta.clone()).stationary_state(),
        mu,
        alpha.clone(),
        beta.clone(),
    );

    let hawkes_a_result = simulate(&hawkes_a, time_horizon, Some(42));
    let hawkes_b_result = simulate(&hawkes_b, time_horizon, Some(43));

    println!(
        "[TIMING] Hawkes pre-simulation: {:?} (ask: {} events, bid: {} events)",
        t0.elapsed(),
        hawkes_a_result.events.len(),
        hawkes_b_result.events.len()
    );

    let hawkes_as_market = hawkes_pair_to_market_orders(&hawkes_a_result, &hawkes_b_result);

    // Simulate q paths without meta orders
    let t0 = Instant::now();
    let q_result_internal =
        simulate_with_externals(&process, time_horizon, &hawkes_as_market, Some(42));
    let q_result = merge_bidask_events(&q_result_internal, &hawkes_as_market);
    let q_paths =
        AffineBidAskQueueProcess::result_to_queue_paths(&q_result, initial_q_a, initial_q_b);
    println!(
        "[TIMING] q simulation: {:?} (ask: {} events, bid: {} events)",
        t0.elapsed(),
        q_paths.ask.events.len(),
        q_paths.bid.events.len()
    );

    // Extract market orders
    let ask_market_orders: Vec<f64> = hawkes_a_result.events.iter().map(|e| e.time).collect();
    let bid_market_orders: Vec<f64> = hawkes_b_result.events.iter().map(|e| e.time).collect();
    println!(
        "Market orders: ask={}, bid={}",
        ask_market_orders.len(),
        bid_market_orders.len()
    );

    // Build conditioning events (exclude dims 2 and 5 - market orders are external)
    let q_events_by_dim = extract_bidask_events_by_dim(&q_result_internal, Some(&[2, 5]));

    // ==========================================================================
    // Setup meta orders and TailImpact
    // ==========================================================================
    let meta_orders = create_bidask_meta_orders(n_meta, meta_start, meta_end, meta_side);
    println!("Created {} meta orders on {:?} side", n_meta, meta_side);

    let t0 = Instant::now();
    let tail_impact = BidAskTailImpact::new_symmetric_hawkes(
        mu,
        alpha.clone(),
        beta.clone(),
        c_matrix.clone(),
        ask_market_orders.clone(),
        bid_market_orders.clone(),
    );
    println!("[TIMING] BidAskTailImpact setup: {:?}", t0.elapsed());

    // Build external events for q and bar_q
    let bar_q_external = merge_bidask_events(&meta_orders, &hawkes_as_market);
    let q_external = hawkes_as_market.clone();

    // ==========================================================================
    // Run parallel conditional simulations (MEMORY-EFFICIENT)
    // ==========================================================================
    let ask_at_market = sample_ask_queue_at_times(&q_paths, &ask_market_orders);
    let bid_at_market = sample_bid_queue_at_times(&q_paths, &bid_market_orders);

    let t0 = Instant::now();
    let simulator = BidAskParallelSimulator {
        process: &process,
        cond_events_by_dim: &q_events_by_dim,
        cond_external_events: Some(&q_external),
        new_external_events: Some(&bar_q_external),
        time_horizon,
        initial_q_a,
        initial_q_b,
        reference_paths: &q_paths,
        tail_impact: &tail_impact,
        ask_market_orders: &ask_market_orders,
        bid_market_orders: &bid_market_orders,
        simulating_bar_q: true,
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
    write_bidask_memory_efficient_results(
        &results,
        &ask_at_market,
        &bid_at_market,
        &ask_market_orders,
        &bid_market_orders,
        "experiments/passive_impact/load_experiments/data/double/efficient/with",
    )
    .unwrap();
    println!("[TIMING] Data write: {:?}", t0.elapsed());
    println!("[TIMING] TOTAL: {:?}", t_total.elapsed());
}
