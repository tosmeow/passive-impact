//! Markovian Queue Illustration for Presentation
//!
//! Creates data for two figures:
//!
//! 1. Original trajectory figure (4 panels):
//!    - Limit order intensity λ^L(q) over time
//!    - Cancel intensity λ^C(q) over time
//!    - Hawkes (market order) intensity over time
//!    - Queue size over time
//!
//! 2. Counterfactual comparison figure (3 panels):
//!    - Limit order intensity λ^L(q) (original + counterfactuals)
//!    - Cancel intensity λ^C(q) (original + counterfactuals)
//!    - Queue sizes (original + counterfactuals)
//!
//! Uses ConditionalSimulationContext with meta orders injected at the start
//! to create initial gap, then shows natural convergence.
//!
//! Output: NPY files for efficient storage.

use simulation_project::models::{AffineQueueProcess, MultiExponentialHawkes, MultivariateSimulationResult, MultivariateEvent, MultivariateMarkovianIntensity};
use simulation_project::simulation::{simulate, simulate_with_externals, ConditionalSimulationContext};
use simulation_project::simulation_helpers::{
    hawkes_to_market_orders, merge_events, extract_events_by_dim, sample_queue_at_times,
};
use simulation_project::utils::{write_npy_f64, write_npy_f64_1d};

/// Sample Hawkes intensity at given time points
fn sample_hawkes_intensity_at_times(
    hawkes: &MultiExponentialHawkes,
    events: &MultivariateSimulationResult,
    sample_times: &[f64],
) -> Vec<f64> {
    let mut intensities = Vec::with_capacity(sample_times.len());
    let mut state = hawkes.initial_state();
    let mut t_last = 0.0;
    let mut event_idx = 0;

    for &t in sample_times {
        // Process all events up to time t
        while event_idx < events.events.len() && events.events[event_idx].time <= t {
            let event = &events.events[event_idx];
            hawkes.update_state(&mut state, event.dim, event.time, t_last);
            t_last = event.time;
            event_idx += 1;
        }

        // Compute intensity at time t
        let lambda = hawkes.intensities_from_state(&state, t, t_last);
        intensities.push(lambda[0]);  // Univariate Hawkes has 1 dimension
    }

    intensities
}

/// Create meta orders at the very start to create initial gap
fn create_early_meta_orders(n_meta: u32, t_start: f64, t_end: f64) -> MultivariateSimulationResult {
    let mut result = MultivariateSimulationResult::new(3);
    let dt = (t_end - t_start) / n_meta as f64;
    for i in 0..n_meta {
        result.push(MultivariateEvent {
            time: t_start + (i as f64 + 0.5) * dt,
            dim: 0,  // Limit order dimension
        });
    }
    result
}

fn main() {
    // ==========================================================================
    // Configuration
    // ==========================================================================
    let initial_queue_size: u32 = 200;

    // Affine queue parameters
    let a_l = 100.0;   // λ^L(q) = a_l + b_l * q
    let b_l = -0.275;
    let a_c = 2.0;     // λ^C(q) = a_c + b_c * q
    let b_c = 0.125;

    // Hawkes parameters
    let mu = 1.0;
    let alpha = vec![0.065, 0.2, 0.325, 0.65];
    let beta = vec![0.15, 0.60, 2.5, 10.0];

    // Early meta orders to create initial gap (injected at very start)
    let n_meta_orders = 10;  // Creates gap of ~10 in queue
    let meta_start = 1e-5;
    let meta_end = 1e-4;

    // Number of counterfactual paths
    let n_counterfactual: usize = 50;

    // Time horizon
    let time_horizon = 5.0;

    // Number of sample points for output
    let n_sample_points: usize = 500;

    println!("=== Markovian Queue Illustration ===");
    println!("Initial queue: {}, Time horizon: {}", initial_queue_size, time_horizon);
    println!("Meta orders: {} (t={:.0e} to {:.0e})", n_meta_orders, meta_start, meta_end);
    println!("Counterfactual paths: {}, Sample points: {}", n_counterfactual, n_sample_points);

    // ==========================================================================
    // Pre-simulate Hawkes path (market orders)
    // ==========================================================================
    let hawkes = MultiExponentialHawkes::new_with_state(
        MultiExponentialHawkes::new(mu, alpha.clone(), beta.clone()).stationary_state(),
        mu, alpha.clone(), beta.clone(),
    );
    let hawkes_result = simulate(&hawkes, time_horizon, Some(42));
    let hawkes_as_market = hawkes_to_market_orders(&hawkes_result);
    println!("Simulated {} Hawkes events", hawkes_result.events.len());

    // ==========================================================================
    // Simulate original q path (without meta orders)
    // ==========================================================================
    let process = AffineQueueProcess::new_queue(initial_queue_size as f64, a_l, b_l, a_c, b_c);

    let q_result_internal = simulate_with_externals(&process, time_horizon, &hawkes_as_market, Some(42));
    let q_result = merge_events(&q_result_internal, &hawkes_as_market);
    let q_path = AffineQueueProcess::result_to_queue_path(&q_result, initial_queue_size);
    println!("Original q path: {} events", q_path.events.len());

    // Extract conditioning events (dims 0,1 - L/C events)
    let q_events_by_dim = extract_events_by_dim(&q_result_internal, 3, Some(2));

    // ==========================================================================
    // Create meta orders and external events for bar_q
    // ==========================================================================
    let meta_orders = create_early_meta_orders(n_meta_orders, meta_start, meta_end);
    let bar_q_external = merge_events(&meta_orders, &hawkes_as_market);

    // ==========================================================================
    // Simulate counterfactual bar_q paths using ConditionalSimulationContext
    // ==========================================================================
    let mut counterfactual_paths = Vec::new();

    for i in 0..n_counterfactual {
        let ctx = ConditionalSimulationContext::new(
            &process,
            &q_events_by_dim,
            Some(&hawkes_as_market),   // q was simulated with these externals
            Some(&bar_q_external),      // bar_q gets meta orders + hawkes
            time_horizon,
        );

        let cf_result = ctx.simulate(None, Some((i + 100) as u64));
        let cf_path = AffineQueueProcess::result_to_queue_path(&cf_result, initial_queue_size);

        if i < 5 || i == n_counterfactual - 1 {
            println!("Counterfactual {} path: {} events", i + 1, cf_path.events.len());
        } else if i == 5 {
            println!("...");
        }
        counterfactual_paths.push(cf_path);
    }

    // ==========================================================================
    // Sample all paths at common time points
    // ==========================================================================
    let sample_times: Vec<f64> = (0..n_sample_points)
        .map(|i| time_horizon * (i as f64) / (n_sample_points as f64 - 1.0))
        .collect();

    // Sample original path
    let q_samples = sample_queue_at_times(&q_path, &sample_times);

    // Sample Hawkes intensity at each time point
    let hawkes_intensities = sample_hawkes_intensity_at_times(&hawkes, &hawkes_result, &sample_times);

    // Sample counterfactual paths
    let cf_samples: Vec<Vec<u32>> = counterfactual_paths
        .iter()
        .map(|cf_path| sample_queue_at_times(cf_path, &sample_times))
        .collect();

    // ==========================================================================
    // Write NPY files
    // ==========================================================================
    let output_dir = "python/experiments/presentation";
    std::fs::create_dir_all(output_dir).unwrap();

    // Write times
    write_npy_f64_1d(&format!("{}/times.npy", output_dir), &sample_times).unwrap();
    println!("Wrote: {}/times.npy ({} points)", output_dir, n_sample_points);

    // Write queues: shape (n_times, 1 + n_counterfactual)
    // Column 0 = original q, columns 1..=n_counterfactual = counterfactuals
    let n_cols = 1 + n_counterfactual;
    let mut queue_data: Vec<f64> = Vec::with_capacity(n_sample_points * n_cols);
    for t_idx in 0..n_sample_points {
        queue_data.push(q_samples[t_idx] as f64);
        for cf_idx in 0..n_counterfactual {
            queue_data.push(cf_samples[cf_idx][t_idx] as f64);
        }
    }
    write_npy_f64(&format!("{}/queues.npy", output_dir), &queue_data, n_sample_points, n_cols).unwrap();
    println!("Wrote: {}/queues.npy (shape: {} x {})", output_dir, n_sample_points, n_cols);

    // Write parameters for Python to compute intensities
    let params = vec![a_l, b_l, a_c, b_c];
    write_npy_f64_1d(&format!("{}/params.npy", output_dir), &params).unwrap();
    println!("Wrote: {}/params.npy [a_l, b_l, a_c, b_c]", output_dir);

    // Write Hawkes intensity for original trajectory figure
    write_npy_f64_1d(&format!("{}/hawkes_intensity.npy", output_dir), &hawkes_intensities).unwrap();
    println!("Wrote: {}/hawkes_intensity.npy ({} points)", output_dir, n_sample_points);

    println!("\nDone! Run:");
    println!("  python python/experiments/presentation/plot_original_queue_illustration.py");
    println!("  python python/experiments/presentation/plot_queue_illustration.py");
}
