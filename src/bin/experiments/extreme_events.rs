use simulation_project::models::{AffineQueueProcess, MultiExponentialHawkes, MultivariateEvent, MultivariateSimulationResult};
use simulation_project::simulation::{simulate, simulate_with_externals, ConditionalSimulationContext, SimulationConfig};
use simulation_project::simulation_helpers::{
    hawkes_to_market_orders, merge_events,
    extract_events_by_dim, sample_queue_at_times,
};
use simulation_project::conditional_impact::{TailImpact, ImpactPath};
use simulation_project::utils::{write_npy_f64, write_npy_u32, write_npy_f64_1d};

use std::time::Instant;

fn create_alternating_lc_events(n: u32, t_start: f64, t_end: f64) -> MultivariateSimulationResult {
    let mut result = MultivariateSimulationResult::new(3);

    for i in 0..n {
        let time = if n > 1 {
            t_start + (i as f64 / (n - 1) as f64) * (t_end - t_start)
        } else {
            t_start
        };

        // Alternating L, C, L, C... (net zero effect on queue)
        let dim = (i % 2) as usize;
        result.push(MultivariateEvent { time, dim });
    }
    result
}

fn main() {
    let t_total = Instant::now();

    // ==========================================================================
    // Configuration
    // ==========================================================================
    let time_horizon = 5.0;
    let initial_queue_size: u32 = 200;

    // Affine queue parameters (same as single_queue)
    let a_l = 100.0;   // λ^L(q) = a_l + b_l * q
    let b_l = -0.275;
    let a_c = 2.0;     // λ^C(q) = a_c + b_c * q
    let b_c = 0.125;

    // Hawkes parameters for market orders
    let mu = 1.0;
    let alpha = vec![0.065, 0.2, 0.325, 0.65];
    let beta = vec![0.15, 0.60, 2.5, 10.0];

    // Extreme events configuration
    let n_lc_events: u32 = 1000;  // Number of L/C events in [0, 1]
    let lc_start = 0.0;
    let lc_end = 1.0;

    // Range of initial queue state perturbations: q+1 to q+p
    let p: u32 = 50;  // Maximum perturbation

    // Sampling grid for queue difference evolution
    let m_samples: u32 = 1000;  // Number of sample points in [0, 1]

    // Create sampling grid: linspace on [0, 1]
    let sample_times: Vec<f64> = (0..m_samples)
        .map(|i| i as f64 / (m_samples - 1) as f64)
        .collect();

    println!("=== Extreme Events Experiment ===");
    println!("Time horizon: {}, Initial queue: {}", time_horizon, initial_queue_size);
    println!("L/C events: {} events in [{}, {}]", n_lc_events, lc_start, lc_end);
    println!("Initial state perturbations: q+1 to q+{}", p);
    println!("Sample grid: {} points in [0, 1]", m_samples);

    let c_lambda = AffineQueueProcess::c_lambda(b_l, b_c);
    println!("c_lambda = {}", c_lambda);

    // ==========================================================================
    // Create process and simulate reference paths
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

    // Create alternating L/C events (net zero effect on queue level)
    let lc_events = create_alternating_lc_events(n_lc_events, lc_start, lc_end);
    println!("Generated {} L/C events", lc_events.events.len());

    // Simulate reference q path with market orders only as externals
    let t0 = Instant::now();
    let q_result_internal = simulate_with_externals(&process, time_horizon, &hawkes_as_market, Some(42));
    let q_result = merge_events(&q_result_internal, &hawkes_as_market);

    // Build q_path INCLUDING the synthetic L/C events (as if q generated them)
    let q_result_with_lc = merge_events(&q_result, &lc_events);
    let q_path = AffineQueueProcess::result_to_queue_path(&q_result_with_lc, initial_queue_size);
    println!("[TIMING] q simulation (reference): {:?} ({} total events, {} synthetic L/C)",
             t0.elapsed(), q_path.events.len(), lc_events.events.len());

    // Get reference queue value at t=0 (this is the q we'll perturb)
    let q_at_zero = q_path.queue_at_time(0.0);
    println!("Reference queue at t=0: {}", q_at_zero);

    // Extract market orders
    let market_orders: Vec<f64> = hawkes_as_market.events.iter().map(|e| e.time).collect();
    println!("Generated {} market order events", market_orders.len());

    // Build conditioning events: q's internal events + synthetic L/C events
    // This makes q̄ thin against the L/C events
    let mut q_events_by_dim = extract_events_by_dim(&q_result_internal, 3, Some(2));

    // Add synthetic L/C events to conditioning (dim 0 = L, dim 1 = C)
    let lc_times_by_dim = extract_events_by_dim(&lc_events, 3, Some(2));
    for dim in 0..2 {
        q_events_by_dim[dim].extend(&lc_times_by_dim[dim]);
        q_events_by_dim[dim].sort_by(|a, b| a.partial_cmp(b).unwrap());
    }
    println!("Conditioning events: {} L, {} C", q_events_by_dim[0].len(), q_events_by_dim[1].len());

    // ==========================================================================
    // Setup TailImpact
    // ==========================================================================
    let t0 = Instant::now();
    let tail_impact = TailImpact::from_affine_queue(
        mu, alpha.clone(), beta.clone(), b_l, b_c, market_orders.clone()
    );
    println!("[TIMING] TailImpact setup: {:?}", t0.elapsed());

    // ==========================================================================
    // Run conditional simulations with different initial states
    // ==========================================================================
    // Both q and q̄ see the same external events (market orders + L/C events)
    // The only difference is the initial state

    // Sample q at both market order times (for impact) and sample grid (for queue diff)
    let q_at_market_orders = sample_queue_at_times(&q_path, &market_orders);
    let q_at_sample_times = sample_queue_at_times(&q_path, &sample_times);

    // Create simulation configs for different initial states
    // Initial states range from q+1 to q+p
    let initial_states: Vec<Vec<f64>> = (1..=p)
        .map(|delta| vec![(initial_queue_size + delta) as f64])
        .collect();

    let t0 = Instant::now();

    // Create conditional simulation context
    // L/C events are in conditioning, only market orders are external
    let ctx = ConditionalSimulationContext::new(
        &process,
        &q_events_by_dim,
        Some(&hawkes_as_market),  // q's externals (market only)
        Some(&hawkes_as_market),  // q̄'s externals (same: market only)
        time_horizon,
    );

    // Build configs for each initial state (same externals, different initial states)
    let configs: Vec<SimulationConfig<Vec<f64>>> = initial_states
        .iter()
        .map(|state| SimulationConfig::new(Some(&hawkes_as_market), Some(state.clone())))
        .collect();

    // Run all simulations
    let sim_results = ctx.simulate_multiple(&configs, 42);
    println!("[TIMING] Conditional simulations ({}x): {:?}", p, t0.elapsed());

    // ==========================================================================
    // Compute results for each initial state
    // ==========================================================================
    let t0 = Instant::now();

    // For each simulation, compute queue samples and impact
    let mut all_queue_samples_market: Vec<Vec<u32>> = Vec::with_capacity(p as usize);
    let mut all_queue_samples_grid: Vec<Vec<u32>> = Vec::with_capacity(p as usize);
    let mut all_impact_paths: Vec<Vec<f64>> = Vec::with_capacity(p as usize);
    let mut initial_deltas: Vec<u32> = Vec::with_capacity(p as usize);

    for (i, sim_result) in sim_results.iter().enumerate() {
        let delta = (i + 1) as u32;
        initial_deltas.push(delta);

        let initial_q_bar = initial_queue_size + delta;
        let bar_q_path = AffineQueueProcess::result_to_queue_path(sim_result, initial_q_bar);

        // Sample queue at market order times (for impact) and sample grid (for queue diff)
        let bar_q_samples_market = sample_queue_at_times(&bar_q_path, &market_orders);
        let bar_q_samples_grid = sample_queue_at_times(&bar_q_path, &sample_times);

        // Compute impact: bar_q - q
        let impact_path = ImpactPath::new(q_path.clone(), bar_q_path, &tail_impact);

        all_queue_samples_market.push(bar_q_samples_market);
        all_queue_samples_grid.push(bar_q_samples_grid);
        all_impact_paths.push(impact_path.impact_path);
    }
    println!("[TIMING] Result computation: {:?}", t0.elapsed());

    // ==========================================================================
    // Output
    // ==========================================================================
    let t0 = Instant::now();
    let output_dir = "data/experiments/extreme_events";

    // Create output directory
    std::fs::create_dir_all(output_dir).expect("Failed to create output directory");

    let n_market_times = market_orders.len();
    let n_sample_times = sample_times.len();
    let n_sims = p as usize;

    // Write impact paths at market order times: [n_market_times x n_sims]
    let impact_data: Vec<f64> = (0..n_market_times)
        .flat_map(|t_idx| {
            all_impact_paths.iter().map(move |path| {
                path.get(t_idx).copied().unwrap_or(f64::NAN)
            })
        })
        .collect();
    write_npy_f64(&format!("{}/impact_paths.npy", output_dir), &impact_data, n_market_times, n_sims).unwrap();

    // Write queue paths at market order times: [n_market_times x (1 + n_sims)]
    let queue_data_market: Vec<u32> = (0..n_market_times)
        .flat_map(|t_idx| {
            std::iter::once(q_at_market_orders[t_idx])
                .chain(all_queue_samples_market.iter().map(move |samples| samples[t_idx]))
        })
        .collect();
    write_npy_u32(&format!("{}/queue_paths.npy", output_dir), &queue_data_market, n_market_times, n_sims + 1).unwrap();

    // Write queue paths at sample grid: [n_sample_times x (1 + n_sims)]
    let queue_data_grid: Vec<u32> = (0..n_sample_times)
        .flat_map(|t_idx| {
            std::iter::once(q_at_sample_times[t_idx])
                .chain(all_queue_samples_grid.iter().map(move |samples| samples[t_idx]))
        })
        .collect();
    write_npy_u32(&format!("{}/queue_paths_grid.npy", output_dir), &queue_data_grid, n_sample_times, n_sims + 1).unwrap();

    // Write market order times
    write_npy_f64_1d(&format!("{}/times.npy", output_dir), &market_orders).unwrap();

    // Write sample grid times
    write_npy_f64_1d(&format!("{}/sample_times.npy", output_dir), &sample_times).unwrap();

    // Write initial deltas
    let initial_deltas_f64: Vec<f64> = initial_deltas.iter().map(|&d| d as f64).collect();
    write_npy_f64_1d(&format!("{}/initial_deltas.npy", output_dir), &initial_deltas_f64).unwrap();

    // Write L/C event times for reference
    let lc_times: Vec<f64> = lc_events.events.iter().map(|e| e.time).collect();
    write_npy_f64_1d(&format!("{}/lc_event_times.npy", output_dir), &lc_times).unwrap();

    // Write metadata
    let metadata = format!(
        "time_horizon: {}\ninitial_queue_size: {}\nn_lc_events: {}\nlc_start: {}\nlc_end: {}\np: {}\nm_samples: {}\na_l: {}\nb_l: {}\na_c: {}\nb_c: {}\nmu: {}\nalpha: {:?}\nbeta: {:?}\n",
        time_horizon, initial_queue_size, n_lc_events, lc_start, lc_end, p, m_samples, a_l, b_l, a_c, b_c, mu, alpha, beta
    );
    std::fs::write(&format!("{}/metadata.txt", output_dir), metadata).unwrap();

    println!("[TIMING] Data write: {:?}", t0.elapsed());
    println!("[TIMING] TOTAL: {:?}", t_total.elapsed());
    println!("\nOutput written to: {}", output_dir);
    println!("Files:");
    println!("  - impact_paths.npy: [{} x {}] impact at market order times", n_market_times, n_sims);
    println!("  - queue_paths.npy: [{} x {}] queues at market order times", n_market_times, n_sims + 1);
    println!("  - queue_paths_grid.npy: [{} x {}] queues at sample grid (for diff evolution)", n_sample_times, n_sims + 1);
    println!("  - times.npy: [{}] market order times", n_market_times);
    println!("  - sample_times.npy: [{}] sample grid times", n_sample_times);
    println!("  - initial_deltas.npy: [{}] initial queue perturbations", n_sims);
    println!("  - lc_event_times.npy: [{}] L/C event times", n_lc_events);
}
