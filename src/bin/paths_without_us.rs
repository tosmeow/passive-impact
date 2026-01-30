use simulation_project::models::{MultiExponentialHawkes, QueuePath};
use simulation_project::simulation::{simulate_markovian, QueueSimulator, ConditionalQueueRemoval};
use simulation_project::conditional_impact::{TailImpact, ImpactPath};

use std::fs::File;
use std::io::Write;
use rayon::prelude::*;

// Helper to get queue value at a given time
fn queue_at_time(path: &QueuePath, t: f64) -> u32 {
    let mut val = path.events[0].queue_size;
    for event in &path.events {
        if event.time <= t {
            val = event.queue_size;
        } else {
            break;
        }
    }
    val
}

fn main() {
    // Configuration
    let time_horizon = 100.0;
    let n_simulations = 500;
    let initial_queue_size = 500;
    let c_lambda = 0.1;

    // 1) Simulate a Hawkes path (market orders)
    let hawkes_model = MultiExponentialHawkes::new(1.0, vec![0.6, 0.6, 1.15], vec![1.0, 3.0, 9.0]);
    let market_orders = simulate_markovian(&hawkes_model, time_horizon, Some(42));
    println!("Generated {} market order events", market_orders.len());

    // Meta orders - trader's limit orders that ADD to the queue (distinguishes q from q_bar)
    let n: u32 = 50;
    let meta_orders: Vec<f64> = (0..n).map(|i| time_horizon / 2.0 + (i as f64 / (n - 1) as f64) * (2.0 * time_horizon / 3.0)).collect();
    println!("Generated {} meta order events", meta_orders.len());

    // 2) Use QueueSimulator to generate the reference queue path (with meta orders)
    let lambda_l = |q: f64| 50.0 - 0.075 * q;
    let lambda_c = |q: f64| 0.025 * q;

    let mut bar_queue_sim = QueueSimulator::new(lambda_l, lambda_c, time_horizon);
    let bar_q_path = bar_queue_sim.simulation_queue(
        &market_orders,
        &meta_orders,
        initial_queue_size,
        true,
        Some(123),
    );
    println!("Generated bar_q path with {} events", bar_q_path.events.len());

    // 3) Prepare TailImpact for the impact path calculation
    let _tail_impact = TailImpact::new(hawkes_model.clone(), c_lambda, market_orders.clone());

    // 4) Run simulations in parallel - return both queue path and impact path
    let bar_limit_orders = &bar_queue_sim.limit_orders;
    let bar_cancel_orders = &bar_queue_sim.cancel_orders;

    let results: Vec<(QueuePath, Vec<f64>)> = (0..n_simulations)
        .into_par_iter()
        .map(|sim_idx| {
            let mut cond_sim = ConditionalQueueRemoval::new(
                lambda_l,
                lambda_c,
                time_horizon,
                &market_orders,
                bar_limit_orders,
                bar_cancel_orders,
                &meta_orders,
            );

            // Simulate q (without meta) given bar_q (with meta)
            let q_path = cond_sim.simulation(&bar_q_path, Some(sim_idx as u64));
            let impact_path = ImpactPath::new(q_path.clone(), bar_q_path.clone(), &_tail_impact);

            (q_path, impact_path.impact_path)
        })
        .collect();

    println!("Completed all {} simulations in parallel", n_simulations);

    // Separate the results
    let all_q_paths: Vec<_> = results.iter().map(|(q, _)| q).collect();
    let all_impact_paths: Vec<_> = results.iter().map(|(_, ip)| ip).collect();

    // 5) Store impact paths as CSV
    let mut file = File::create("impact_paths_without.csv").unwrap();

    let header: Vec<String> = (0..n_simulations).map(|i| format!("sim_{}", i)).collect();
    writeln!(file, "time,{}", header.join(",")).unwrap();

    let max_len = all_impact_paths.iter().map(|p| p.len()).max().unwrap_or(0);

    for t_idx in 0..max_len {
        let values: Vec<String> = all_impact_paths
            .iter()
            .map(|path| {
                path.get(t_idx)
                    .map(|v| format!("{:.6}", v))
                    .unwrap_or_default()
            })
            .collect();
        let time_value = market_orders.get(t_idx).map(|t| format!("{:.6}", t)).unwrap_or_default();
        writeln!(file, "{},{}", time_value, values.join(",")).unwrap();
    }

    println!("Saved impact paths to impact_paths_without.csv");

    // 6) Store queue paths as CSV
    let mut queue_file = File::create("queue_paths_without.csv").unwrap();

    let q_header: Vec<String> = (0..n_simulations).map(|i| format!("q_sim_{}", i)).collect();
    writeln!(queue_file, "time,bar_q,{}", q_header.join(",")).unwrap();

    for &t in &market_orders {
        let bar_q_val = queue_at_time(&bar_q_path, t);
        let q_vals: Vec<String> = all_q_paths
            .iter()
            .map(|qp| format!("{}", queue_at_time(qp, t)))
            .collect();
        writeln!(queue_file, "{:.6},{},{}", t, bar_q_val, q_vals.join(",")).unwrap();
    }

    println!("Saved queue paths to queue_paths_without.csv");
}
