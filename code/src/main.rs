use std::process::Command;
use std::time::Instant;

fn main() {
    let t_total = Instant::now();

    println!("{}", "=".repeat(70));
    println!("Running all simulation binaries");
    println!("{}", "=".repeat(70));

    let binaries = [
        // Single queue
        (
            "single_queue_efficient_with_us",
            "Single Queue - Efficient - With Us",
        ),
        (
            "single_queue_efficient_without_us",
            "Single Queue - Efficient - Without Us",
        ),
        // Double queue (bid-ask)
        (
            "double_queue_efficient_with_us",
            "Double Queue - Efficient - With Us",
        ),
        (
            "double_queue_efficient_without_us",
            "Double Queue - Efficient - Without Us",
        ),
    ];

    let mut successes = 0;
    let mut failures = 0;

    for (binary, description) in binaries {
        println!("\n{}", "-".repeat(70));
        println!("[RUNNING] {}", description);
        println!("{}", "-".repeat(70));

        let t0 = Instant::now();

        // Get the path to the binary in the same directory as this executable
        let exe_path = std::env::current_exe().expect("Failed to get current executable path");
        let bin_dir = exe_path.parent().expect("Failed to get binary directory");
        let binary_path = bin_dir.join(binary);

        let result = Command::new(&binary_path).status();

        match result {
            Ok(status) if status.success() => {
                println!("[SUCCESS] {} completed in {:?}", description, t0.elapsed());
                successes += 1;
            }
            Ok(status) => {
                println!("[FAILED] {} exited with status: {}", description, status);
                failures += 1;
            }
            Err(e) => {
                println!("[ERROR] Failed to run {}: {}", binary, e);
                println!("  Binary path: {:?}", binary_path);
                failures += 1;
            }
        }
    }

    println!("\n{}", "=".repeat(70));
    println!("SUMMARY");
    println!("{}", "=".repeat(70));
    println!("Successes: {}", successes);
    println!("Failures:  {}", failures);
    println!("Total time: {:?}", t_total.elapsed());
}
