use simulation_project::conditional_impact::AggressiveImpactPath;
use simulation_project::models::{AffineQueueProcess, MultiExponentialHawkes};
use simulation_project::simulation::{
    simulate, simulate_with_externals, ConditionalSimulationContext,
};
use simulation_project::simulation_helpers::{
    create_meta_orders, events_to_dim, extract_events_by_dim, hawkes_to_market_orders,
    merge_events, sample_queue_at_times,
};
use simulation_project::utils::{write_npy_f64, write_npy_f64_1d, write_npy_u32};

use rayon::prelude::*;
use std::time::Instant;
use std::{env, process};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Direction {
    WithUs,
    WithoutUs,
}

impl Direction {
    fn is_counterfactual(self) -> bool {
        matches!(self, Self::WithoutUs)
    }

    fn label(self) -> &'static str {
        match self {
            Self::WithUs => "with us",
            Self::WithoutUs => "without us",
        }
    }
}

fn main() {
    for direction in parse_directions() {
        run_direction(direction);
    }
}

fn run_direction(direction: Direction) {
    let t_total = Instant::now();

    // ==========================================================================
    // Configuration (hybrid aggressive-impact baseline)
    // ==========================================================================
    let time_horizon = 90.0;
    let n_simulations = 500;
    let initial_queue_size: u32 = 200;

    // Affine queue parameters: lambda^L(q) = a_l + b_l * q, lambda^C(q) = a_c + b_c * q
    let a_l = 100.0;
    let b_l = -0.275;
    let a_c = 2.0;
    let b_c = 0.125;

    // Hawkes parameters for market orders
    let mu = 1.0;
    let alpha = vec![0.065, 0.2, 0.325, 0.65];
    let beta = vec![0.15, 0.60, 2.5, 10.0];

    // Meta orders configuration (aggressive: dim=2, reduce queue)
    let n_meta: u32 = 156;
    let meta_start = 0.0;
    let meta_end = 2.0 * time_horizon / 3.0;

    // Kappa function: kappa(q) = -c_kappa * q  (purely linear, no constant)
    let c_kappa = 0.001_f64;
    let kappa = |q: f64| -c_kappa * q;

    // bar_kappa: constant weight for the propagator term (fixed small value)
    let bar_kappa = 0.01_f64;

    println!(
        "=== Aggressive Impact Experiment ({}) ===",
        direction.label()
    );
    println!(
        "Time horizon: {}, Simulations: {}, Initial queue: {}",
        time_horizon, n_simulations, initial_queue_size
    );
    println!("kappa(q) = -{} * q, bar_kappa = {}", c_kappa, bar_kappa);

    let c_lambda = AffineQueueProcess::c_lambda(b_l, b_c);
    println!("c_lambda = {}", c_lambda);

    // ==========================================================================
    // Pre-simulate Hawkes -> market orders (decoupled mode)
    // ==========================================================================
    let t0 = Instant::now();

    let hawkes = MultiExponentialHawkes::new_with_state(
        MultiExponentialHawkes::new(mu, alpha.clone(), beta.clone()).stationary_state(),
        mu,
        alpha.clone(),
        beta.clone(),
    );
    let hawkes_result = simulate(&hawkes, time_horizon, Some(42));
    let hawkes_as_market = hawkes_to_market_orders(&hawkes_result);
    println!(
        "[TIMING] Hawkes pre-simulation: {:?} ({} events)",
        t0.elapsed(),
        hawkes_result.events.len()
    );

    // ==========================================================================
    // Create aggressive meta orders
    // ==========================================================================
    // Meta orders are market orders (dim=2) that reduce the queue
    let meta_orders_raw = create_meta_orders(n_meta, meta_start, meta_end);
    let meta_orders = events_to_dim(&meta_orders_raw, 2, 3);
    let meta_order_times: Vec<f64> = meta_orders.events.iter().map(|e| e.time).collect();
    let market_order_times: Vec<f64> = hawkes_as_market.events.iter().map(|e| e.time).collect();
    println!(
        "Market orders: {}, Meta orders: {}",
        market_order_times.len(),
        meta_order_times.len()
    );

    let hawkes_model = MultiExponentialHawkes::new(mu, alpha.clone(), beta.clone());
    let norm: f64 = alpha.iter().zip(&beta).map(|(a, b)| a / b).sum::<f64>();
    println!(
        "Propagator: G(0) = {:.4} (mean cluster size), G(∞) = 1 (permanent)",
        1.0 / (1.0 - norm)
    );

    // ==========================================================================
    // Merge market + meta order times into sorted evaluation times
    // ==========================================================================
    let mut eval_entries: Vec<(f64, bool)> = Vec::new();
    for &t in &market_order_times {
        eval_entries.push((t, true)); // is_market_order = true
    }
    for &t in &meta_order_times {
        eval_entries.push((t, false)); // is_market_order = false (meta order)
    }
    eval_entries.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    let eval_times: Vec<f64> = eval_entries.iter().map(|e| e.0).collect();
    let is_market_order: Vec<bool> = eval_entries.iter().map(|e| e.1).collect();
    println!("Evaluation times (merged): {}", eval_times.len());

    // ==========================================================================
    // Build conditioning path and externals
    // ==========================================================================
    let process = AffineQueueProcess::new_queue(initial_queue_size as f64, a_l, b_l, a_c, b_c);

    // bar_q external events = meta orders + Hawkes market orders
    let bar_q_external = merge_events(&meta_orders, &hawkes_as_market);

    let (
        conditioning_events_by_dim,
        cond_external_events,
        new_external_events,
        reference_samples,
        simulating_bar_q,
    ) = if direction.is_counterfactual() {
        // without_us: condition on bar_q (with meta orders), simulate q (without meta orders).
        let t0 = Instant::now();
        let bar_q_result_internal =
            simulate_with_externals(&process, time_horizon, &bar_q_external, Some(42));
        let bar_q_result = merge_events(&bar_q_result_internal, &bar_q_external);
        let bar_q_path =
            AffineQueueProcess::result_to_queue_path(&bar_q_result, initial_queue_size);
        println!(
            "[TIMING] bar_q conditioning simulation: {:?} ({} events)",
            t0.elapsed(),
            bar_q_path.events.len()
        );
        (
            extract_events_by_dim(&bar_q_result_internal, 3, Some(2)),
            &bar_q_external,
            &hawkes_as_market,
            sample_queue_at_times(&bar_q_path, &eval_times),
            false,
        )
    } else {
        // with_us: condition on q (without meta orders), simulate bar_q (with meta orders).
        let t0 = Instant::now();
        let q_result_internal =
            simulate_with_externals(&process, time_horizon, &hawkes_as_market, Some(42));
        let q_result = merge_events(&q_result_internal, &hawkes_as_market);
        let q_path = AffineQueueProcess::result_to_queue_path(&q_result, initial_queue_size);
        println!(
            "[TIMING] q conditioning simulation: {:?} ({} events)",
            t0.elapsed(),
            q_path.events.len()
        );
        (
            extract_events_by_dim(&q_result_internal, 3, Some(2)),
            &hawkes_as_market,
            &bar_q_external,
            sample_queue_at_times(&q_path, &eval_times),
            true,
        )
    };

    println!(
        "Conditioning on {}; simulating {}",
        if simulating_bar_q { "q" } else { "bar_q" },
        if simulating_bar_q { "bar_q" } else { "q" }
    );

    // ==========================================================================
    // Run parallel conditional simulations
    // ==========================================================================
    let t0 = Instant::now();

    let results: Vec<(Vec<u32>, Vec<f64>)> = (0..n_simulations)
        .into_par_iter()
        .map(|sim_idx| {
            let ctx = ConditionalSimulationContext::new(
                &process,
                &conditioning_events_by_dim,
                Some(cond_external_events),
                Some(new_external_events),
                time_horizon,
            );

            let sim_samples = ctx.simulate_queue_at_times(
                &eval_times,
                initial_queue_size,
                None,
                Some(sim_idx as u64),
            );

            // Hybrid impact: metaorders propagate with bar_kappa; market orders
            // contribute instantaneous queue-dependent corrections.
            let impact_path = if simulating_bar_q {
                AggressiveImpactPath::from_queue_samples(
                    &reference_samples,
                    &sim_samples,
                    &eval_times,
                    &is_market_order,
                    &hawkes_model,
                    &kappa,
                    bar_kappa,
                )
            } else {
                AggressiveImpactPath::from_queue_samples(
                    &sim_samples,
                    &reference_samples,
                    &eval_times,
                    &is_market_order,
                    &hawkes_model,
                    &kappa,
                    bar_kappa,
                )
            };

            (sim_samples, impact_path.impact_path)
        })
        .collect();

    println!(
        "[TIMING] Parallel simulations ({}x): {:?}",
        n_simulations,
        t0.elapsed()
    );

    // ==========================================================================
    // Output
    // ==========================================================================
    let t0 = Instant::now();
    let output_dirs = output_dirs_for(
        "experiments/agressive_impact/load_experiments/data",
        direction,
    );
    write_aggressive_outputs(
        &output_dirs,
        &eval_times,
        &is_market_order,
        &reference_samples,
        &results,
        n_simulations,
        bar_kappa,
    );

    println!("[TIMING] Data write: {:?}", t0.elapsed());
    println!("[TIMING] TOTAL: {:?}", t_total.elapsed());
}

fn parse_directions() -> Vec<Direction> {
    let mut directions = vec![Direction::WithUs, Direction::WithoutUs];
    for arg in env::args().skip(1) {
        match arg.as_str() {
            "--both" => directions = vec![Direction::WithUs, Direction::WithoutUs],
            "--counterfactual" | "--without-us" => directions = vec![Direction::WithoutUs],
            "--with-us" => directions = vec![Direction::WithUs],
            "-h" | "--help" => {
                print_help();
                process::exit(0);
            }
            other => {
                eprintln!("Unknown argument: {other}");
                print_help();
                process::exit(2);
            }
        }
    }
    directions
}

fn print_help() {
    println!(
        "\
Usage: agressive_impact [--both | --with-us | --without-us | --counterfactual]

Options:
  --both             Run both with-us and without-us scenarios. This is the default.
  --with-us          Condition on q and simulate bar_q.
  --without-us      Condition on bar_q and simulate q.
  --counterfactual  Alias for --without-us.
  -h, --help        Show this help message.
"
    );
}

fn output_dirs_for(base_output_dir: &str, direction: Direction) -> Vec<String> {
    match direction {
        Direction::WithUs => vec![format!("{base_output_dir}/with")],
        Direction::WithoutUs => vec![format!("{base_output_dir}/without")],
    }
}

fn write_aggressive_outputs(
    output_dirs: &[String],
    eval_times: &[f64],
    is_market_order: &[bool],
    reference_samples: &[u32],
    results: &[(Vec<u32>, Vec<f64>)],
    n_simulations: usize,
    bar_kappa: f64,
) {
    let n_times = eval_times.len();

    // Impact paths: (n_times, n_simulations)
    let impact_data: Vec<f64> = (0..n_times)
        .flat_map(|t_idx| {
            results
                .iter()
                .map(move |(_, impact)| impact.get(t_idx).copied().unwrap_or(f64::NAN))
        })
        .collect();

    // Queue paths: (n_times, n_simulations + 1)
    // with_us: first col = q, rest = bar_q_sim_i
    // without_us: first col = bar_q, rest = q_sim_i
    let queue_data: Vec<u32> = (0..n_times)
        .flat_map(|t_idx| {
            std::iter::once(reference_samples[t_idx]).chain(
                results
                    .iter()
                    .map(move |(sim_samples, _)| sim_samples[t_idx]),
            )
        })
        .collect();

    // Event types: 1.0 for market order, 0.0 for meta order
    let event_types: Vec<f64> = is_market_order
        .iter()
        .map(|&b| if b { 1.0 } else { 0.0 })
        .collect();

    for output_dir in output_dirs {
        std::fs::create_dir_all(output_dir).expect("Failed to create output directory");
        write_npy_f64(
            &format!("{}/impact_paths.npy", output_dir),
            &impact_data,
            n_times,
            n_simulations,
        )
        .unwrap();
        write_npy_u32(
            &format!("{}/queue_paths.npy", output_dir),
            &queue_data,
            n_times,
            n_simulations + 1,
        )
        .unwrap();
        write_npy_f64_1d(&format!("{}/times.npy", output_dir), eval_times).unwrap();
        write_npy_f64_1d(&format!("{}/event_types.npy", output_dir), &event_types).unwrap();
        write_npy_f64_1d(&format!("{}/bar_kappa.npy", output_dir), &[bar_kappa]).unwrap();
    }
}
