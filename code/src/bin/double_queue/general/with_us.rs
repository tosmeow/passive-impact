use simulation_project::models::{
    AffineBidAskQueueProcess, MultiExponentialHawkes, BidAskAffineParams, AffineIntensityParams,
};
use simulation_project::simulation::{simulate, simulate_with_externals};
use simulation_project::simulation_helpers::{
    hawkes_pair_to_market_orders, merge_bidask_events, create_bidask_meta_orders, Side,
    extract_bidask_events_by_dim, extract_ask_market_orders, extract_bid_market_orders,
    sample_ask_queue_at_times, sample_bid_queue_at_times,
    BidAskParallelSimulator, write_bidask_results,
};
use simulation_project::conditional_impact::{BidAskTailImpact, SymmetricCMatrix};

use std::time::Instant;

fn main() {
    let t_total = Instant::now();

    // ==========================================================================
    // Configuration
    // ==========================================================================
    let time_horizon = 100.0;
    let n_simulations = 500;
    let initial_q_a: u32 = 350;
    let initial_q_b: u32 = 350;

    // Set to true for efficient decoupled simulation
    let decoupled = true;

    // Affine queue parameters (symmetric)
    // λ^{L,a}(q^a, q^b) = a_l + b_l_own * q^a + b_l_cross * q^b
    let a_l = 100.0;
    let b_l_own = -0.15;   // own queue large → less incentive to add
    let b_l_cross = 0.05;  // other queue large → more incentive to add (rebalancing)

    let a_c = 2.0;
    let b_c_own = 0.10;    // own queue large → more cancellations
    let b_c_cross = 0.02;  // other queue large → slightly more cancellations

    // Hawkes parameters (same for both sides)
    let mu = 1.0;
    let alpha = vec![0.065, 0.2, 0.325, 0.65];
    let beta = vec![0.15, 0.60, 2.5, 10.0];

    // Meta orders configuration (on ask side only)
    let n_meta: u32 = 375;
    let meta_start = 1.0;
    let meta_end = 3.0 * time_horizon / 4.0;
    let meta_side = Side::Ask;

    println!("=== Bid-Ask Paths WITH Us ({}) ===", if decoupled { "decoupled" } else { "coupled" });
    println!("Time horizon: {}, Simulations: {}", time_horizon, n_simulations);
    println!("Initial queues: ask={}, bid={}", initial_q_a, initial_q_b);

    // Build C matrix for impact computation
    let c_matrix = SymmetricCMatrix::from_affine_symmetric(b_l_own, b_l_cross, b_c_own, b_c_cross);
    println!("C matrix: c={}, a={}", c_matrix.c, c_matrix.a);
    println!("Eigenvalues: λ_+ = {}, λ_- = {}", c_matrix.lambda_plus(), c_matrix.lambda_minus());

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

    let process = if decoupled {
        AffineBidAskQueueProcess::new_queue(
            initial_q_a as f64, initial_q_b as f64, params.clone()
        )
    } else {
        AffineBidAskQueueProcess::new(
            initial_q_a as f64, initial_q_b as f64,
            params.clone(),
            mu, mu,
            alpha.clone(), beta.clone(),
            alpha.clone(), beta.clone(),
        )
    };

    // For decoupled mode, pre-simulate both Hawkes paths
    let (hawkes_a_result, hawkes_b_result, hawkes_as_market) = if decoupled {
        let hawkes_a = MultiExponentialHawkes::new_with_state(
            MultiExponentialHawkes::new(mu, alpha.clone(), beta.clone()).stationary_state(),
            mu, alpha.clone(), beta.clone(),
        );
        let hawkes_b = MultiExponentialHawkes::new_with_state(
            MultiExponentialHawkes::new(mu, alpha.clone(), beta.clone()).stationary_state(),
            mu, alpha.clone(), beta.clone(),
        );

        let hawkes_a_result = simulate(&hawkes_a, time_horizon, Some(42));
        let hawkes_b_result = simulate(&hawkes_b, time_horizon, Some(43));

        println!("[TIMING] Hawkes pre-simulation: {:?} (ask: {} events, bid: {} events)",
            t0.elapsed(), hawkes_a_result.events.len(), hawkes_b_result.events.len());

        let combined = hawkes_pair_to_market_orders(&hawkes_a_result, &hawkes_b_result);
        (Some(hawkes_a_result), Some(hawkes_b_result), Some(combined))
    } else {
        (None, None, None)
    };

    // Simulate q paths without meta orders
    let t0 = Instant::now();
    let (q_result, q_result_internal) = if decoupled {
        let q_result_internal = simulate_with_externals(
            &process, time_horizon, hawkes_as_market.as_ref().unwrap(), Some(42)
        );
        let q_result = merge_bidask_events(&q_result_internal, hawkes_as_market.as_ref().unwrap());
        (q_result, Some(q_result_internal))
    } else {
        let q_result = simulate(&process, time_horizon, None);
        (q_result, None)
    };

    let q_paths = AffineBidAskQueueProcess::result_to_queue_paths(
        &q_result, initial_q_a, initial_q_b
    );
    println!("[TIMING] q simulation: {:?} (ask: {} events, bid: {} events)",
        t0.elapsed(), q_paths.ask.events.len(), q_paths.bid.events.len());

    // Extract market orders
    let (ask_market_orders, bid_market_orders) = if decoupled {
        (
            hawkes_a_result.as_ref().unwrap().events.iter().map(|e| e.time).collect::<Vec<_>>(),
            hawkes_b_result.as_ref().unwrap().events.iter().map(|e| e.time).collect::<Vec<_>>(),
        )
    } else {
        (
            extract_ask_market_orders(&q_result),
            extract_bid_market_orders(&q_result),
        )
    };
    println!("Market orders: ask={}, bid={}", ask_market_orders.len(), bid_market_orders.len());

    // Build conditioning events: condition on all L/C events from q path
    // The conditional simulation will accept/reject based on intensity ratios
    let q_events_by_dim = if decoupled {
        // Exclude dims 2 and 5 (market orders are external)
        extract_bidask_events_by_dim(q_result_internal.as_ref().unwrap(), Some(&[2, 5]))
    } else {
        extract_bidask_events_by_dim(&q_result, None)
    };

    // ==========================================================================
    // Setup meta orders and TailImpact
    // ==========================================================================
    let meta_orders = create_bidask_meta_orders(n_meta, meta_start, meta_end, meta_side);
    println!("Created {} meta orders on {:?} side", n_meta, meta_side);

    let t0 = Instant::now();
    let tail_impact = BidAskTailImpact::new_symmetric_hawkes(
        mu, alpha.clone(), beta.clone(),
        c_matrix.clone(),
        ask_market_orders.clone(),
        bid_market_orders.clone(),
    );
    println!("[TIMING] BidAskTailImpact setup: {:?}", t0.elapsed());

    // Build external events for q and bar_q
    let bar_q_external = if decoupled {
        Some(merge_bidask_events(&meta_orders, hawkes_as_market.as_ref().unwrap()))
    } else {
        Some(meta_orders.clone())
    };

    let q_external = hawkes_as_market.clone();

    // ==========================================================================
    // Run parallel conditional simulations
    // ==========================================================================
    let ask_at_market = sample_ask_queue_at_times(&q_paths, &ask_market_orders);
    let bid_at_market = sample_bid_queue_at_times(&q_paths, &bid_market_orders);

    let t0 = Instant::now();
    let simulator = BidAskParallelSimulator {
        process: &process,
        cond_events_by_dim: &q_events_by_dim,
        cond_external_events: q_external.as_ref(),
        new_external_events: bar_q_external.as_ref(),
        time_horizon,
        initial_q_a,
        initial_q_b,
        reference_paths: &q_paths,
        tail_impact: &tail_impact,
        ask_market_orders: &ask_market_orders,
        bid_market_orders: &bid_market_orders,
        simulating_bar_q: true,
    };
    let results = simulator.run(n_simulations);
    println!("[TIMING] Parallel simulations ({}x): {:?}", n_simulations, t0.elapsed());

    // ==========================================================================
    // Output
    // ==========================================================================
    let t0 = Instant::now();
    write_bidask_results(
        &results,
        &ask_at_market,
        &bid_at_market,
        &ask_market_orders,
        &bid_market_orders,
        "data/double_queue/general/with",
    ).unwrap();
    println!("[TIMING] Data write: {:?}", t0.elapsed());
    println!("[TIMING] TOTAL: {:?}", t_total.elapsed());
}
