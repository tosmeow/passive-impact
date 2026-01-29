use simulation_project::models::MultiExponentialHawkes;
use simulation_project::simulation::{simulate_markovian, QueueSimulator, ConditionalQueueSimulator};
use simulation_project::conditional_impact::{TailImpact, ImpactPath};

use std::fs::File;
use std::io::Write;
use rayon::prelude::*;

fn main() {
    // Configuration
    let time_horizon = 500.0;
    let n_simulations = 500;
    let initial_queue_size = 500;
    let c_lambda = 0.1;

    // 1) Simulate a Hawkes path (market orders)
    let hawkes_model = MultiExponentialHawkes::new(1.0, vec![0.6, 0.6, 1.15], vec![1.0, 3.0, 9.0]);
    let market_orders = simulate_markovian(&hawkes_model, time_horizon, Some(42));
    println!("Generated {} market order events", market_orders.len());

    // Meta orders - trader's limit orders that ADD to the queue (distinguishes q from q_bar)
    // Generate from another Hawkes process
    let n: u32 = 250;
    let meta_orders : Vec<f64> = (0..n).map(|i| time_horizon / 4.0 + (i as f64 / (n - 1) as f64) * (2.0 * time_horizon / 3.0)).collect();
    println!("Generated {} meta order events", meta_orders.len());

    // 2) Use QueueSimulator to generate the reference queue path
    let lambda_l = |q: f64| 50.0 - 0.075 * q;
    let lambda_c = |q: f64| 0.025 * q;

    let mut queue_sim = QueueSimulator::new(lambda_l, lambda_c, time_horizon);
    let q_path = queue_sim.simulation_queue(
        &market_orders,
        &meta_orders,
        initial_queue_size,
        true,  // stock values
        Some(123),
    );
    println!("Generated queue path with {} events", q_path.events.len());

    // 3) Prepare TailImpact for the impact path calculation
    let _tail_impact = TailImpact::new(hawkes_model.clone(), c_lambda, market_orders.clone());

    // 4) Run simulations in parallel using Rayon
    let limit_orders = queue_sim.limit_orders.clone();
    let cancel_orders = queue_sim.cancel_orders.clone();

    let all_impact_paths: Vec<Vec<f64>> = (0..n_simulations)
        .into_par_iter()
        .map(|sim_idx| {
            // Each thread creates its own simulator.
            let mut cond_sim = ConditionalQueueSimulator::new(
                lambda_l,
                lambda_c,
                time_horizon,
                market_orders.clone(),
                limit_orders.clone(),
                cancel_orders.clone(),
            );

            // Simulate q_bar given q
            let q_bar_path = cond_sim.simulation(&q_path, Some(sim_idx as u64));

            // Generate impact path for this simulation
            let impact_path = ImpactPath::new(q_path.clone(), q_bar_path, &_tail_impact);

            impact_path.impact_path
        })
        .collect();

    println!("Completed all {} simulations in parallel", n_simulations);

    // 5) Store all impact paths as CSV
    let mut file = File::create("impact_paths.csv").unwrap();

    // Write header (simulation indices)
    let header: Vec<String> = (0..n_simulations).map(|i| format!("sim_{}", i)).collect();
    writeln!(file, "time,{}", header.join(",")).unwrap();

    // Find max length across all paths
    let max_len = all_impact_paths.iter().map(|p| p.len()).max().unwrap_or(0);

    // Write rows (each row is one time point across all simulations)
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

    println!("Saved impact paths to impact_paths.csv");
}