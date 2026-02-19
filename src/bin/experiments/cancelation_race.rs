use simulation_project::models::{AffineQueueProcess, MultiExponentialHawkes, MultivariateEvent, MultivariateSimulationResult};
use simulation_project::simulation::{simulate, simulate_with_externals, ConditionalSimulationContext, SimulationConfig};
use simulation_project::simulation_helpers::{
    hawkes_to_market_orders, merge_events,
    extract_events_by_dim, sample_queue_at_times,
};
use simulation_project::conditional_impact::{TailImpact, ImpactPath};
use simulation_project::utils::{write_npy_f64, write_npy_u32, write_npy_f64_1d};

use std::time::Instant;

/// Create n cancelation events evenly spaced in [t_start, t_end]
fn create_cancel_events(n: u32, t_start: f64, t_end: f64) -> MultivariateSimulationResult {
    let mut result = MultivariateSimulationResult::new(3);

    for i in 0..n {
        let time = if n > 1 {
            t_start + (i as f64 / (n - 1) as f64) * (t_end - t_start)
        } else {
            t_start
        };

        // Dimension 1 = Cancel
        result.push(MultivariateEvent { time, dim: 1 });
    }
    result
}

/// Create n limit events evenly spaced in [t_start, t_end]
fn create_limit_events(n: u32, t_start: f64, t_end: f64) -> MultivariateSimulationResult {
    let mut result = MultivariateSimulationResult::new(3);

    for i in 0..n {
        let time = if n > 1 {
            t_start + (i as f64 / (n - 1) as f64) * (t_end - t_start)
        } else {
            t_start
        };

        // Dimension 0 = Limit
        result.push(MultivariateEvent { time, dim: 0 });
    }
    result
}

/// Create p cancel events as a burst starting at time x, spaced by epsilon
fn create_cancel_burst(p: u32, x: f64, epsilon: f64) -> MultivariateSimulationResult {
    let mut result = MultivariateSimulationResult::new(3);

    for i in 0..p {
        let time = x + (i as f64) * epsilon;
        // Dimension 1 = Cancel
        result.push(MultivariateEvent { time, dim: 1 });
    }
    result
}

fn main() {
    let t_total = Instant::now();

    // ==========================================================================
    // Configuration
    // ==========================================================================
    let time_horizon = 50.0;
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

    // Cancelation race configuration
    let n_cancels: u32 = 100;     // Number of cancel events in conditioning path [0, 1]
    let n_limits: u32 = 200;      // Number of limit events in external path (to keep queue stable)
    let p_burst: u32 = 20;        // Number of cancels in the burst
    let epsilon: f64 = 1e-8;      // Time spacing between burst cancels
    let initial_delta: u32 = 0;  // Initial queue perturbation: q̄₀ = q₀ + initial_delta

    // Time window for bursts and sampling
    let burst_window_end: f64 = 10.0;  // Bursts happen in [0, burst_window_end]
    let sample_window_end: f64 = 15.0; // Sample grid extends past burst window

    // Different x values to test (when the burst happens)
    let n_x_values: u32 = 50;
    let x_values: Vec<f64> = (0..n_x_values)
        .map(|i| (i as f64 + 0.5) * burst_window_end / n_x_values as f64)
        .collect();

    // Sampling grid for queue/impact evolution
    let m_samples: u32 = 1000;
    let sample_times: Vec<f64> = (0..m_samples)
        .map(|i| i as f64 * sample_window_end / (m_samples - 1) as f64)
        .collect();

    println!("=== Cancelation Race Experiment ===");
    println!("Time horizon: {}, Initial queue: {}", time_horizon, initial_queue_size);
    println!("Initial perturbation: q̄₀ = q₀ + {}", initial_delta);
    println!("Conditioning cancels: {} events in [0, {}]", n_cancels, burst_window_end);
    println!("External limits: {} events in [0, {}] (for queue stability)", n_limits, burst_window_end);
    println!("Burst: {} cancels at time x with spacing {}", p_burst, epsilon);
    println!("Testing {} different x values in (0, {})", n_x_values, burst_window_end);

    let c_lambda = AffineQueueProcess::c_lambda(b_l, b_c);
    println!("c_lambda = {}", c_lambda);

    // ==========================================================================
    // Create process and simulate reference paths
    // ==========================================================================
    let t0 = Instant::now();

    // Create queue process
    let process = AffineQueueProcess::new_queue(initial_queue_size as f64, a_l, b_l, a_c, b_c);

    // Pre-simulate Hawkes path
    let hawkes = MultiExponentialHawkes::new_with_state(
        MultiExponentialHawkes::new(mu, alpha.clone(), beta.clone()).stationary_state(),
        mu, alpha.clone(), beta.clone(),
    );
    let hawkes_result = simulate(&hawkes, time_horizon, Some(42));
    println!("[TIMING] Hawkes pre-simulation: {:?} ({} events)", t0.elapsed(), hawkes_result.events.len());
    let hawkes_as_market = hawkes_to_market_orders(&hawkes_result);

    // Create conditioning events: n cancels evenly spaced in [0, burst_window_end]
    let conditioning_cancels = create_cancel_events(n_cancels, 0.0, burst_window_end);
    println!("Generated {} conditioning cancel events", conditioning_cancels.events.len());

    // Create external limit events (to balance the cancels and keep queue stable)
    let external_limits = create_limit_events(n_limits, 0.0, burst_window_end);
    println!("Generated {} external limit events", external_limits.events.len());

    // Build q externals: market orders + limit events
    let q_externals = merge_events(&hawkes_as_market, &external_limits);

    // Simulate reference q path with market orders + limits as externals
    let t0 = Instant::now();
    let q_result_internal = simulate_with_externals(&process, time_horizon, &q_externals, Some(42));
    let q_result = merge_events(&q_result_internal, &q_externals);

    // Build q_path WITHOUT conditioning cancels - they are purely for thinning
    let q_path = AffineQueueProcess::result_to_queue_path(&q_result, initial_queue_size);
    println!("[TIMING] q simulation (reference): {:?} ({} total events)",
             t0.elapsed(), q_path.events.len());

    // Extract market orders
    let market_orders: Vec<f64> = hawkes_as_market.events.iter().map(|e| e.time).collect();
    println!("Generated {} market order events", market_orders.len());

    // Build conditioning events by dim
    // Cancels are in conditioning (dim 1), q's internal L/C events are also in conditioning
    let mut q_events_by_dim = extract_events_by_dim(&q_result_internal, 3, Some(2));

    // Add conditioning cancels to dim 1
    let conditioning_times: Vec<f64> = conditioning_cancels.events.iter().map(|e| e.time).collect();
    q_events_by_dim[1].extend(&conditioning_times);
    q_events_by_dim[1].sort_by(|a, b| a.partial_cmp(b).unwrap());

    println!("Conditioning events: {} L, {} C", q_events_by_dim[0].len(), q_events_by_dim[1].len());

    // ==========================================================================
    // Setup TailImpact
    // ==========================================================================
    let t0 = Instant::now();
    let tail_impact = TailImpact::from_affine_queue(
        mu, alpha.clone(), beta.clone(), b_l, b_c, market_orders.clone()
    );
    println!("[TIMING] TailImpact setup: {:?}", t0.elapsed());

    // Sample q at times for reference
    let q_at_market_orders = sample_queue_at_times(&q_path, &market_orders);
    let q_at_sample_times = sample_queue_at_times(&q_path, &sample_times);

    // ==========================================================================
    // Run conditional simulations with different burst times x
    // ==========================================================================
    // For each x, q_bar sees the same conditioning events, but gets p extra cancels
    // injected at time x in its external path.
    // Using simulate_multiple ensures shared randomness across all x values.

    let t0 = Instant::now();

    // Create conditional simulation context
    // Conditioning events are the same for all x values
    let ctx = ConditionalSimulationContext::new(
        &process,
        &q_events_by_dim,
        Some(&q_externals),      // q's externals (market + limits)
        Some(&q_externals),      // q_bar's base externals (will be augmented per config)
        time_horizon,
    );

    // Build configs for each x value
    // Each config has the perturbed initial state and different externals (with burst at x)
    let bar_initial_state = vec![(initial_queue_size + initial_delta) as f64];

    let configs: Vec<(f64, MultivariateSimulationResult)> = x_values
        .iter()
        .map(|&x| {
            let burst = create_cancel_burst(p_burst, x, epsilon);
            let bar_externals = merge_events(&q_externals, &burst);
            (x, bar_externals)
        })
        .collect();

    let sim_configs: Vec<SimulationConfig<Vec<f64>>> = configs
        .iter()
        .map(|(_x, externals)| SimulationConfig::new(Some(externals), Some(bar_initial_state.clone())))
        .collect();

    // Run all simulations with shared randomness
    let sim_results = ctx.simulate_multiple(&sim_configs, 42);
    println!("[TIMING] Conditional simulations ({}x): {:?}", n_x_values, t0.elapsed());

    // ==========================================================================
    // Compute results for each x value
    // ==========================================================================
    let t0 = Instant::now();

    let mut all_queue_samples_market: Vec<Vec<u32>> = Vec::with_capacity(n_x_values as usize);
    let mut all_queue_samples_grid: Vec<Vec<u32>> = Vec::with_capacity(n_x_values as usize);
    let mut all_impact_paths: Vec<Vec<f64>> = Vec::with_capacity(n_x_values as usize);
    let mut all_queue_diff_grid: Vec<Vec<i32>> = Vec::with_capacity(n_x_values as usize);

    let bar_initial_queue = initial_queue_size + initial_delta;

    for sim_result in sim_results.iter() {
        let bar_q_path = AffineQueueProcess::result_to_queue_path(sim_result, bar_initial_queue);

        // Sample queue at market order times and sample grid
        let bar_q_samples_market = sample_queue_at_times(&bar_q_path, &market_orders);
        let bar_q_samples_grid = sample_queue_at_times(&bar_q_path, &sample_times);

        // Compute queue difference: bar_q - q
        let queue_diff: Vec<i32> = bar_q_samples_grid.iter()
            .zip(q_at_sample_times.iter())
            .map(|(&bq, &q)| bq as i32 - q as i32)
            .collect();

        // Compute impact
        let impact_path = ImpactPath::new(q_path.clone(), bar_q_path, &tail_impact);

        all_queue_samples_market.push(bar_q_samples_market);
        all_queue_samples_grid.push(bar_q_samples_grid);
        all_impact_paths.push(impact_path.impact_path);
        all_queue_diff_grid.push(queue_diff);
    }
    println!("[TIMING] Result computation: {:?}", t0.elapsed());

    // ==========================================================================
    // Output
    // ==========================================================================
    let t0 = Instant::now();
    let output_dir = "data/experiments/cancelation_race";

    std::fs::create_dir_all(output_dir).expect("Failed to create output directory");

    let n_market_times = market_orders.len();
    let n_sample_times = sample_times.len();
    let n_sims = n_x_values as usize;

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

    // Write queue difference at sample grid: [n_sample_times x n_sims]
    let queue_diff_data: Vec<i32> = (0..n_sample_times)
        .flat_map(|t_idx| {
            all_queue_diff_grid.iter().map(move |diff| diff[t_idx])
        })
        .collect();
    // Convert i32 to f64 for npy writing
    let queue_diff_f64: Vec<f64> = queue_diff_data.iter().map(|&x| x as f64).collect();
    write_npy_f64(&format!("{}/queue_diff_grid.npy", output_dir), &queue_diff_f64, n_sample_times, n_sims).unwrap();

    // Write time arrays
    write_npy_f64_1d(&format!("{}/times.npy", output_dir), &market_orders).unwrap();
    write_npy_f64_1d(&format!("{}/sample_times.npy", output_dir), &sample_times).unwrap();

    // Write x values (burst times)
    write_npy_f64_1d(&format!("{}/x_values.npy", output_dir), &x_values).unwrap();

    // Write conditioning cancel times
    write_npy_f64_1d(&format!("{}/conditioning_cancel_times.npy", output_dir), &conditioning_times).unwrap();

    // Write initial_delta
    write_npy_f64_1d(&format!("{}/initial_delta.npy", output_dir), &[initial_delta as f64]).unwrap();

    // Write metadata
    let metadata = format!(
        "time_horizon: {}\ninitial_queue_size: {}\ninitial_delta: {}\nn_cancels: {}\nn_limits: {}\np_burst: {}\nepsilon: {}\nn_x_values: {}\nm_samples: {}\na_l: {}\nb_l: {}\na_c: {}\nb_c: {}\nmu: {}\nalpha: {:?}\nbeta: {:?}\n",
        time_horizon, initial_queue_size, initial_delta, n_cancels, n_limits, p_burst, epsilon, n_x_values, m_samples, a_l, b_l, a_c, b_c, mu, alpha, beta
    );
    std::fs::write(&format!("{}/metadata.txt", output_dir), metadata).unwrap();

    println!("[TIMING] Data write: {:?}", t0.elapsed());
    println!("[TIMING] TOTAL: {:?}", t_total.elapsed());
    println!("\nOutput written to: {}", output_dir);
    println!("Files:");
    println!("  - impact_paths.npy: [{} x {}] impact at market order times", n_market_times, n_sims);
    println!("  - queue_paths.npy: [{} x {}] queues at market order times", n_market_times, n_sims + 1);
    println!("  - queue_paths_grid.npy: [{} x {}] queues at sample grid", n_sample_times, n_sims + 1);
    println!("  - queue_diff_grid.npy: [{} x {}] queue differences (bar_q - q) at sample grid", n_sample_times, n_sims);
    println!("  - times.npy: [{}] market order times", n_market_times);
    println!("  - sample_times.npy: [{}] sample grid times", n_sample_times);
    println!("  - x_values.npy: [{}] burst times tested", n_sims);
    println!("  - conditioning_cancel_times.npy: [{}] conditioning cancel times", n_cancels);
}
