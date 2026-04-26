use simulation_project::models::{AffineQueueProcess, MultiExponentialHawkes};
use simulation_project::simulation::{simulate, simulate_with_externals};
use simulation_project::simulation_helpers::{
    hawkes_to_market_orders, merge_events, create_meta_orders,
    extract_events_by_dim, sample_queue_at_times, extract_market_orders,
    ParallelSimulator, write_results,
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

    // Set to true for efficient decoupled simulation
    let decoupled = true;

    // Affine queue parameters
    let a_l = 100.0;   // λ^L(q) = a_l + b_l * q
    let b_l = -0.275;
    let a_c = 2.0;     // λ^C(q) = a_c + b_c * q
    let b_c = 0.125;

    // Hawkes parameters for market orders: these parameters are set to be close to a t ** -1.5 power-law for the kernel.
    let mu = 1.0;
    let alpha = vec![0.065, 0.2, 0.325, 0.65];
    let beta = vec![0.15, 0.60, 2.5, 10.0];

    // Meta orders configuration
    let n_meta: u32 = 375;
    let meta_start = 1.0;
    let meta_end = 4.0 * time_horizon / 5.0;

    // Output file suffix

    println!("=== Paths WITHOUT Us ===");
    println!("Time horizon: {}, Simulations: {}, Initial queue: {}",
             time_horizon, n_simulations, initial_queue_size);

    let _c_lambda = AffineQueueProcess::c_lambda(b_l, b_c);
    println!("c_lambda = {}", _c_lambda);

    // ==========================================================================
    // Create process and simulate bar_q path (with meta orders)
    // ==========================================================================
    let t0 = Instant::now();

    // Create process based on mode: decoupled only keeps queue states and otherwise keeps queue and Hawkes markovian states.
    let process = if decoupled {
        AffineQueueProcess::new_queue(initial_queue_size as f64, a_l, b_l, a_c, b_c)
    } else {
        let initial_state = AffineQueueProcess::stationary_state(
            initial_queue_size as f64, mu, &alpha, &beta
        );
        AffineQueueProcess::new_with_state(
            initial_state, a_l, b_l, a_c, b_c, mu, alpha.clone(), beta.clone(),
        )
    };

    // For decoupled mode, pre-simulate Hawkes path:
    let hawkes_as_market = if decoupled {
        let hawkes = MultiExponentialHawkes::new_with_state(
            MultiExponentialHawkes::new(mu, alpha.clone(), beta.clone()).stationary_state(),
            mu, alpha.clone(), beta.clone(),
        );
        let hawkes_result = simulate(&hawkes, time_horizon, Some(42));
        println!("[TIMING] Hawkes pre-simulation: {:?} ({} events)", t0.elapsed(), hawkes_result.events.len());
        Some(hawkes_to_market_orders(&hawkes_result))
    } else {
        None
    };

    // Create metaorder.
    let meta_orders = create_meta_orders(n_meta, meta_start, meta_end);

    // Simulate q path with the metaorder.
    let t0 = Instant::now();
    let (bar_q_result, bar_q_result_internal) = if decoupled {
        // Decoupled: bar_q has the metaorder and the Hawkes process as external.
        let bar_q_externals = merge_events(&meta_orders, hawkes_as_market.as_ref().unwrap());
        let bar_q_result_internal = simulate_with_externals(&process, time_horizon, &bar_q_externals, Some(42));
        let bar_q_result = merge_events(&bar_q_result_internal, &bar_q_externals);
        (bar_q_result, Some(bar_q_result_internal))
    } else {
        // Coupled: bar_q has only meta orders as external
        let bar_q_result_internal = simulate_with_externals(&process, time_horizon, &meta_orders, None);
        let bar_q_result = merge_events(&bar_q_result_internal, &meta_orders);
        (bar_q_result, Some(bar_q_result_internal))
    };
    let bar_q_path = AffineQueueProcess::result_to_queue_path(&bar_q_result, initial_queue_size);
    println!("[TIMING] bar_q simulation: {:?} ({} events)", t0.elapsed(), bar_q_path.events.len());

    // Extract market orders
    let market_orders: Vec<f64> = if decoupled {
        hawkes_as_market.as_ref().unwrap().events.iter().map(|e| e.time).collect()
    } else {
        extract_market_orders(bar_q_result_internal.as_ref().unwrap())
    };
    println!("Generated {} market order events", market_orders.len());

    // Build conditioning events
    let bar_q_events_by_dim = if decoupled {
        // Condition only on dims 0,1 (internal) corresponding to limits and cancels: dim 2, corresponding the market orders, is external.
        extract_events_by_dim(bar_q_result_internal.as_ref().unwrap(), 3, Some(2))
    } else {
        // Condition on all dims
        extract_events_by_dim(bar_q_result_internal.as_ref().unwrap(), 3, None)
    };

    // ==========================================================================
    // Setup TailImpact
    // ==========================================================================
    let t0 = Instant::now();
    let tail_impact = TailImpact::from_affine_queue(
        mu, alpha.clone(), beta.clone(), b_l, b_c, market_orders.clone()
    );
    println!("[TIMING] TailImpact setup: {:?}", t0.elapsed());

    // Build external events for bar_q and q
    let bar_q_external = if decoupled {
        Some(merge_events(&meta_orders, hawkes_as_market.as_ref().unwrap()))
    } else {
        Some(meta_orders.clone())
    };
    
    let q_external = hawkes_as_market.clone();  // q has either only the Hawkes path or None.

    // ==========================================================================
    // Run parallel conditional simulations
    // ==========================================================================
    let bar_q_at_market_orders = sample_queue_at_times(&bar_q_path, &market_orders);

    let t0 = Instant::now();
    let simulator = ParallelSimulator {
        process: &process,
        cond_events_by_dim: &bar_q_events_by_dim,
        cond_external_events: bar_q_external.as_ref(),
        new_external_events: q_external.as_ref(),
        time_horizon,
        initial_queue_size,
        reference_path: &bar_q_path,
        tail_impact: &tail_impact,
        market_orders: &market_orders,
        simulating_bar_q: false,
    };
    let results = simulator.run(n_simulations);
    println!("[TIMING] Parallel simulations ({}x): {:?}", n_simulations, t0.elapsed());

    // ==========================================================================
    // Output
    // ==========================================================================
    let t0 = Instant::now();
    write_results(&results, &bar_q_at_market_orders, &market_orders, "experiments/passive_impact/load_experiments/data/single/general/without").unwrap();
    println!("[TIMING] Data write: {:?}", t0.elapsed());
    println!("[TIMING] TOTAL: {:?}", t_total.elapsed());
}
