//! Passive market impact under the flow imbalance model.
//!
//! This module contains impact computations based on the conditional expectation
//! of queue differences, leading to closed-form formulas via tail intensity.
//!
//! The model assumes the underlying price satisfies:
//! ```text
//! P_t = E_t[∫₀^∞ κ(q̄_s - q_s) dN_s]
//! ```
//! where the queue difference decays mean-revertingly due to affine queue dynamics.

pub mod multi_queue;
pub mod single_queue;

pub use multi_queue::{BidAskImpactPath, BidAskTailImpact, ModeTailImpact, SymmetricCMatrix};
pub use single_queue::ImpactPath;
