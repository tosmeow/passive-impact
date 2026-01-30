use std::process::Command;

fn main() {
    println!("=== Running paths_with_us ===");
    let status = Command::new("cargo")
        .args(["run", "--bin", "paths_with_us", "--release"])
        .status()
        .expect("Failed to run paths_with_us");

    if !status.success() {
        eprintln!("paths_with_us failed with status: {}", status);
    }

    println!("\n=== Running paths_without_us ===");
    let status = Command::new("cargo")
        .args(["run", "--bin", "paths_without_us", "--release"])
        .status()
        .expect("Failed to run paths_without_us");

    if !status.success() {
        eprintln!("paths_without_us failed with status: {}", status);
    }

    println!("\n=== Done ===");
}
