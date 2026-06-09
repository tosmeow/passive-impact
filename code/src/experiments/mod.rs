//! Experiment-scoped native helpers.
//!
//! Modules under `experiments` support reproducible research workflows that
//! need Rust speed but are not part of the core point-process abstractions.
//! Public items here are stable enough for the repository's Python pipelines,
//! but they should be treated as workflow APIs rather than general-purpose
//! model primitives.

pub mod impact_cost;
