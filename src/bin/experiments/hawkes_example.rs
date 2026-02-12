use simulation_project::models::{MultiExponentialHawkes, MultivariateMarkovianIntensity, MultivariateEvent, MultivariateSimulationResult};
use simulation_project::simulation::{simulate, ConditionalSimulationContext};
use simulation_project::utils::{write_npy_f64, write_npy_f64_1d};

use std::time::Instant;

fn compute_intensity_on_grid(
    hawkes: &MultiExponentialHawkes,
    result: &MultivariateSimulationResult,
    grid: &[f64],
) -> Vec<f64> {
    let mut intensities = Vec::with_capacity(grid.len());
    let mut state = hawkes.initial_state();
    let mut t_last = 0.0;
    let mut event_idx = 0;

    for &t in grid {
        while event_idx < result.events.len() && result.events[event_idx].time <= t {
            let event = &result.events[event_idx];
            hawkes.update_state(&mut state, event.dim, event.time, t_last);
            t_last = event.time;
            event_idx += 1;
        }

        let lambda = hawkes.intensities_from_state(&state, t, t_last);
        intensities.push(lambda[0]);
    }

    intensities
}

/// Compute N(t) on the grid via searchsorted (count of events <= t)
fn cumulative_count_on_grid(event_times: &[f64], grid: &[f64]) -> Vec<f64> {
    let mut counts = Vec::with_capacity(grid.len());
    let mut idx = 0;
    for &t in grid {
        while idx < event_times.len() && event_times[idx] <= t {
            idx += 1;
        }
        counts.push(idx as f64);
    }
    counts
}

fn main() {
    let t_total = Instant::now();

    // ==========================================================================
    // Configuration
    // ==========================================================================
    let time_horizon = 10.0;

    // Hawkes parameters (4-component exponential kernel)
    let mu = 100.0;
    let alpha = vec![0.065, 0.2, 0.325, 0.65];
    let beta = vec![0.15, 0.60, 2.5, 10.0];

    // Shock configuration: n extra events packed into [t_shock, t_shock + dt]
    let n_extra: usize = 100;
    let t_shock = 2.0;
    let dt_shock = 0.01;

    // Number of counterfactual paths
    let n_paths: usize = 500;

    println!("=== Hawkes Conditional Simulation Example ===");
    println!("Time horizon: {}", time_horizon);
    println!("Hawkes: mu={}, alpha={:?}, beta={:?}", mu, alpha, beta);
    println!("Shock: {} events in [{}, {}]", n_extra, t_shock, t_shock + dt_shock);
    println!("Counterfactual paths: {}", n_paths);

    // ==========================================================================
    // Step 1: Simulate original Hawkes path
    // ==========================================================================
    let hawkes = MultiExponentialHawkes::new_with_state(
        MultiExponentialHawkes::new(mu, alpha.clone(), beta.clone()).stationary_state(),
        mu, alpha.clone(), beta.clone(),
    );

    let t0 = Instant::now();
    let original_result = simulate(&hawkes, time_horizon, Some(42));
    println!("\n[Step 1] Original simulation: {:?} ({} events)",
        t0.elapsed(), original_result.events.len());

    // ==========================================================================
    // Step 2: Create n extra events in [t_shock, t_shock + dt]
    // ==========================================================================
    let extra_times: Vec<f64> = (0..n_extra)
        .map(|i| {
            if n_extra > 1 {
                t_shock + (i as f64 / (n_extra - 1) as f64) * dt_shock
            } else {
                t_shock
            }
        })
        .collect();

    let mut extra_as_result = MultivariateSimulationResult::new(1);
    for &t in &extra_times {
        extra_as_result.push(MultivariateEvent { time: t, dim: 0 });
    }

    println!("[Step 2] Shock: {} events in [{:.4}, {:.4}]",
        n_extra, extra_times.first().unwrap_or(&0.0), extra_times.last().unwrap_or(&0.0));

    // ==========================================================================
    // Step 3: Run n_paths conditional simulations
    // ==========================================================================
    let t0 = Instant::now();
    let ctx = ConditionalSimulationContext::new(
        &hawkes,
        &original_result.events_by_dim,
        None,                    // original path had no externals
        Some(&extra_as_result),  // new path receives the shock
        time_horizon,
    );

    let conditional_results: Vec<MultivariateSimulationResult> = (0..n_paths)
        .map(|i| ctx.simulate(None, Some(i as u64)))
        .collect();

    let event_counts: Vec<usize> = conditional_results.iter().map(|r| r.events.len()).collect();
    let mean_events = event_counts.iter().sum::<usize>() as f64 / n_paths as f64;
    let mean_extra = mean_events - original_result.events.len() as f64;
    println!("[Step 3] {} conditional simulations: {:?}", n_paths, t0.elapsed());
    println!("  Events: original={}, mean conditional={:.1} (mean {:+.1} from shock)",
        original_result.events.len(), mean_events, mean_extra);

    // ==========================================================================
    // Step 4: Compute intensities and cumulative counts on grid
    // ==========================================================================
    let n_grid: usize = 2000;
    let grid_times: Vec<f64> = (0..n_grid)
        .map(|i| i as f64 / (n_grid - 1) as f64 * time_horizon)
        .collect();

    let original_intensities = compute_intensity_on_grid(&hawkes, &original_result, &grid_times);
    let original_times: Vec<f64> = original_result.events.iter().map(|e| e.time).collect();
    let original_counts = cumulative_count_on_grid(&original_times, &grid_times);

    // Compute intensities and counts for all conditional paths → 2D arrays [n_grid x n_paths]
    let mut all_intensities: Vec<f64> = Vec::with_capacity(n_grid * n_paths);
    let mut all_counts: Vec<f64> = Vec::with_capacity(n_grid * n_paths);

    for result in &conditional_results {
        let intensities = compute_intensity_on_grid(&hawkes, result, &grid_times);
        let times: Vec<f64> = result.events.iter().map(|e| e.time).collect();
        let counts = cumulative_count_on_grid(&times, &grid_times);

        all_intensities.extend_from_slice(&intensities);
        all_counts.extend_from_slice(&counts);
    }

    // Reshape to row-major [n_grid x n_paths]: row i = grid point i, col j = path j
    // Currently stored as [path0_grid, path1_grid, ...] i.e. [n_paths x n_grid]
    // Transpose to [n_grid x n_paths]
    let mut intensities_grid_major: Vec<f64> = Vec::with_capacity(n_grid * n_paths);
    let mut counts_grid_major: Vec<f64> = Vec::with_capacity(n_grid * n_paths);
    for t_idx in 0..n_grid {
        for p_idx in 0..n_paths {
            intensities_grid_major.push(all_intensities[p_idx * n_grid + t_idx]);
            counts_grid_major.push(all_counts[p_idx * n_grid + t_idx]);
        }
    }

    // ==========================================================================
    // Output
    // ==========================================================================
    let output_dir = "data/experiments/hawkes_example";
    std::fs::create_dir_all(output_dir).expect("Failed to create output directory");

    write_npy_f64_1d(&format!("{}/original_events.npy", output_dir), &original_times).unwrap();
    write_npy_f64_1d(&format!("{}/extra_events.npy", output_dir), &extra_times).unwrap();
    write_npy_f64_1d(&format!("{}/grid_times.npy", output_dir), &grid_times).unwrap();
    write_npy_f64_1d(&format!("{}/original_intensities.npy", output_dir), &original_intensities).unwrap();
    write_npy_f64_1d(&format!("{}/original_counts.npy", output_dir), &original_counts).unwrap();

    // 2D arrays: [n_grid x n_paths]
    write_npy_f64(&format!("{}/conditional_intensities.npy", output_dir),
        &intensities_grid_major, n_grid, n_paths).unwrap();
    write_npy_f64(&format!("{}/conditional_counts.npy", output_dir),
        &counts_grid_major, n_grid, n_paths).unwrap();

    // Print summary
    println!("\n=== Summary ===");
    println!("Original: {} events", original_times.len());
    println!("Conditional: {} paths, mean {:.1} events ({:+.1})", n_paths, mean_events, mean_extra);

    let shock_idx = grid_times.iter().position(|&t| t >= t_shock).unwrap_or(0);
    let post_shock_idx = grid_times.iter().position(|&t| t >= t_shock + 0.5).unwrap_or(n_grid - 1);
    println!("\nIntensity at t={}:  original={:.2}", t_shock, original_intensities[shock_idx]);
    println!("Intensity at t={:.1}: original={:.2}", t_shock + 0.5, original_intensities[post_shock_idx]);

    println!("\n[TIMING] TOTAL: {:?}", t_total.elapsed());
    println!("Output: {}/", output_dir);
    println!("  original_intensities.npy:    [{}]", n_grid);
    println!("  original_counts.npy:         [{}]", n_grid);
    println!("  conditional_intensities.npy: [{} x {}]", n_grid, n_paths);
    println!("  conditional_counts.npy:      [{} x {}]", n_grid, n_paths);
}
