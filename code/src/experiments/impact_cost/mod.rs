//! Native support for the empirical passive impact-cost workflow.
//!
//! This module is the Rust side of `experiments/impact_cost`. It adapts
//! row-sized empirical order-flow data to fast anchored queue simulation and
//! passive fill tracking:
//!
//! - anchored queue simulation treats observed post-event snapshots as the
//!   exogenous factual path `bar_q` and simulates only `dq = q - bar_q`;
//! - passive selection helpers flag limit rows that represent our displayed
//!   orders;
//! - fill tracking applies first-level queue-priority conventions to those
//!   passive orders;
//! - latency grids repeat the same fill tracker across many minute windows.
//!
//! The Python package re-exports these functions through `simproj` and keeps
//! file I/O, dataframe adaptation, and plotting in `experiments/impact_cost`.

pub mod anchored;
pub mod events;
pub mod execution;
pub mod execution_grid;
pub mod policies;

pub use anchored::{
    simulate_anchored_affine_queue, AffineQueueIntensity, AnchoredConditionalSimulationContext,
    AnchoredConditioningPath, AnchoredOffsetPath, AnchoredQueueInput, AnchoredQueueSimulation,
    AnchoredSimulatedEvent, CANCEL_DIM, LIMIT_DIM, MARKET_DIM,
};
pub use execution::{
    track_passive_fills, CancellationPolicy, PassiveFillTrackerInput, PassiveFillTrackerResult,
};
pub use execution_grid::{
    build_execution_latency_grid, ExecutionLatencyGridInput, ExecutionLatencyGridResult,
    LatencyFillRow, LatencyOrderRow,
};
pub use policies::{
    limit_positions, select_first_limit_every, select_limit_indices, select_random_limit_fraction,
};
