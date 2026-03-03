//! Conditional impact computation for bid-ask (two-queue) processes.
//!
//! # Mathematical Framework
//!
//! For the bid-ask model with symmetric coupling matrix:
//!
//! ```text
//! C = [[c, a],
//!      [a, c]]
//! ```
//!
//! The vector `f_s = E_t[(q' - q)_s]` satisfies the ODE:
//! ```text
//! f_s = f_t + ∫_t^s C f_u du
//! ```
//!
//! With solution `f_s = f_t × exp(C(s-t))`.
//!
//! ## Eigendecomposition
//!
//! The matrix C has eigenvectors and eigenvalues:
//! - `(1, 1)` with eigenvalue `λ_+ = c + a`
//! - `(1, -1)` with eigenvalue `λ_- = c - a`
//!
//! ## Decomposition into Modes
//!
//! Define:
//! - `f_+ = (q'^a - q^a) + (q'^b - q^b)` — symmetric mode
//! - `f_- = (q'^a - q^a) - (q'^b - q^b)` — antisymmetric mode
//!
//! Each mode evolves independently with its own decay rate:
//! - `f_+(s) = f_+(t) × exp((c+a)(s-t))`
//! - `f_-(s) = f_-(t) × exp((c-a)(s-t))`
//!
//! ## Reconstructing Per-Side Impact
//!
//! The impact on each side is reconstructed from the two modes:
//! - `Impact^a = ½ × [I(f_+, c+a, Hawkes^a) + I(f_-, c-a, Hawkes^a)]`
//! - `Impact^b = ½ × [I(f_+, c+a, Hawkes^b) - I(f_-, c-a, Hawkes^b)]`
//!
//! Where `I(init, c_lambda, Hawkes)` is the single-queue impact path computation.

use crate::conditional_impact::TailImpact;
use crate::models::{MultiExponentialHawkes, QueuePath, BidAskQueuePath};

/// Parameters for the symmetric C matrix.
#[derive(Clone, Debug)]
pub struct SymmetricCMatrix {
    /// Diagonal term (c)
    pub c: f64,
    /// Off-diagonal term (a)
    pub a: f64,
}

impl SymmetricCMatrix {
    pub fn new(c: f64, a: f64) -> Self {
        Self { c, a }
    }

    /// Eigenvalue for symmetric mode (1, 1)
    pub fn lambda_plus(&self) -> f64 {
        self.c + self.a
    }

    /// Eigenvalue for antisymmetric mode (1, -1)
    pub fn lambda_minus(&self) -> f64 {
        self.c - self.a
    }

    /// Create from affine queue parameters.
    ///
    /// Given symmetric bid-ask with:
    /// - λ^{L,a}(q^a, q^b) = a_l + b_l_own * q^a + b_l_cross * q^b
    /// - λ^{C,a}(q^a, q^b) = a_c + b_c_own * q^a + b_c_cross * q^b
    ///
    /// Then:
    /// - c = b_c_own - b_l_own (diagonal term)
    /// - a = b_c_cross - b_l_cross (off-diagonal term)
    pub fn from_affine_symmetric(
        b_l_own: f64,
        b_l_cross: f64,
        b_c_own: f64,
        b_c_cross: f64,
    ) -> Self {
        Self {
            c: b_c_own - b_l_own,
            a: b_c_cross - b_l_cross,
        }
    }
}

/// Tail impact factors for a single mode (symmetric or antisymmetric).
///
/// This wraps the single-queue TailImpact, parameterized by the mode's
/// c_lambda (either c+a or c-a) and the Hawkes process for one side.
pub struct ModeTailImpact {
    /// The underlying single-queue tail impact computation
    pub tail_impact: TailImpact,
    /// c_lambda for this mode (c+a or c-a)
    pub c_lambda: f64,
}

impl ModeTailImpact {
    /// Create a mode tail impact from Hawkes parameters and events.
    pub fn new(
        hawkes_params: MultiExponentialHawkes,
        c_lambda: f64,
        events: Vec<f64>,
    ) -> Self {
        let tail_impact = TailImpact::new(hawkes_params, c_lambda, events);
        Self { tail_impact, c_lambda }
    }
}

/// Tail impact computation for bid-ask model.
///
/// Holds the TailImpact for both modes (symmetric and antisymmetric)
/// for both sides (ask and bid Hawkes processes).
pub struct BidAskTailImpact {
    /// Symmetric mode (c+a) with ask Hawkes
    pub plus_ask: ModeTailImpact,
    /// Antisymmetric mode (c-a) with ask Hawkes
    pub minus_ask: ModeTailImpact,
    /// Symmetric mode (c+a) with bid Hawkes
    pub plus_bid: ModeTailImpact,
    /// Antisymmetric mode (c-a) with bid Hawkes
    pub minus_bid: ModeTailImpact,
    /// The C matrix parameters
    pub c_matrix: SymmetricCMatrix,
}

impl BidAskTailImpact {
    /// Create bid-ask tail impact from model parameters.
    ///
    /// # Arguments
    ///
    /// * `hawkes_a` - Hawkes parameters for ask side
    /// * `hawkes_b` - Hawkes parameters for bid side
    /// * `c_matrix` - The symmetric C matrix
    /// * `events_a` - Market order times on ask side (N^a events)
    /// * `events_b` - Market order times on bid side (N^b events)
    pub fn new(
        hawkes_a: MultiExponentialHawkes,
        hawkes_b: MultiExponentialHawkes,
        c_matrix: SymmetricCMatrix,
        events_a: Vec<f64>,
        events_b: Vec<f64>,
    ) -> Self {
        let lambda_plus = c_matrix.lambda_plus();
        let lambda_minus = c_matrix.lambda_minus();

        Self {
            plus_ask: ModeTailImpact::new(hawkes_a.clone(), lambda_plus, events_a.clone()),
            minus_ask: ModeTailImpact::new(hawkes_a, lambda_minus, events_a),
            plus_bid: ModeTailImpact::new(hawkes_b.clone(), lambda_plus, events_b.clone()),
            minus_bid: ModeTailImpact::new(hawkes_b, lambda_minus, events_b),
            c_matrix,
        }
    }

    /// Convenience constructor for symmetric Hawkes (same params both sides).
    pub fn new_symmetric_hawkes(
        mu: f64,
        alpha: Vec<f64>,
        beta: Vec<f64>,
        c_matrix: SymmetricCMatrix,
        events_a: Vec<f64>,
        events_b: Vec<f64>,
    ) -> Self {
        let hawkes_a = MultiExponentialHawkes::new(mu, alpha.clone(), beta.clone());
        let hawkes_b = MultiExponentialHawkes::new(mu, alpha, beta);
        Self::new(hawkes_a, hawkes_b, c_matrix, events_a, events_b)
    }
}

/// Impact path for the bid-ask model.
///
/// Computes impact on both ask and bid sides using the eigenvalue decomposition.
pub struct BidAskImpactPath {
    /// Impact values at ask market order times
    pub ask_impact: Vec<f64>,
    /// Impact values at bid market order times
    pub bid_impact: Vec<f64>,
}

impl BidAskImpactPath {
    /// Compute bid-ask impact paths.
    ///
    /// # Arguments
    ///
    /// * `q_a, q_b` - Actual queue paths (under scenario with metaorder)
    /// * `q_prime_a, q_prime_b` - Counterfactual queue paths (without metaorder)
    /// * `tail_impact` - Pre-computed tail impact factors
    ///
    /// # Returns
    ///
    /// Impact paths evaluated at market order times for each side.
    pub fn new(
        q_a: &QueuePath,
        q_b: &QueuePath,
        q_prime_a: &QueuePath,
        q_prime_b: &QueuePath,
        tail_impact: &BidAskTailImpact,
    ) -> Self {
        // Compute impact on ask side (at N^a event times)
        let ask_impact = Self::compute_side_impact(
            q_a, q_b, q_prime_a, q_prime_b,
            &tail_impact.plus_ask.tail_impact,
            &tail_impact.minus_ask.tail_impact,
            true, // ask side: add the minus term
        );

        // Compute impact on bid side (at N^b event times)
        let bid_impact = Self::compute_side_impact(
            q_a, q_b, q_prime_a, q_prime_b,
            &tail_impact.plus_bid.tail_impact,
            &tail_impact.minus_bid.tail_impact,
            false, // bid side: subtract the minus term
        );

        Self { ask_impact, bid_impact }
    }

    /// Compute impact for one side.
    ///
    /// For ask: Impact = ½ × [I_+ + I_-]
    /// For bid: Impact = ½ × [I_+ - I_-]
    fn compute_side_impact(
        q_a: &QueuePath,
        q_b: &QueuePath,
        q_prime_a: &QueuePath,
        q_prime_b: &QueuePath,
        tail_plus: &TailImpact,
        tail_minus: &TailImpact,
        is_ask: bool,
    ) -> Vec<f64> {
        let events = &tail_plus.events; // Events are the same for both modes on same side
        let n = events.len();
        let mut impact = Vec::with_capacity(n);

        // Track positions in queue paths
        let (mut i_a, mut i_b) = (0, 0);
        let (mut i_prime_a, mut i_prime_b) = (0, 0);

        // Current queue values
        let (mut curr_q_a, mut curr_q_b) = (
            q_a.events.first().map(|e| e.queue_size).unwrap_or(0) as f64,
            q_b.events.first().map(|e| e.queue_size).unwrap_or(0) as f64,
        );
        let (mut curr_q_prime_a, mut curr_q_prime_b) = (
            q_prime_a.events.first().map(|e| e.queue_size).unwrap_or(0) as f64,
            q_prime_b.events.first().map(|e| e.queue_size).unwrap_or(0) as f64,
        );

        // Cumulative terms for each mode
        let mut cumulative_plus: f64 = 0.0;
        let mut cumulative_minus: f64 = 0.0;

        for (t_idx, &t) in events.iter().enumerate() {
            // Update queue values to time t
            Self::advance_queue(&mut i_a, &mut curr_q_a, q_a, t);
            Self::advance_queue(&mut i_b, &mut curr_q_b, q_b, t);
            Self::advance_queue(&mut i_prime_a, &mut curr_q_prime_a, q_prime_a, t);
            Self::advance_queue(&mut i_prime_b, &mut curr_q_prime_b, q_prime_b, t);

            // Compute differences
            let diff_a = curr_q_prime_a - curr_q_a;
            let diff_b = curr_q_prime_b - curr_q_b;

            // Mode decomposition
            let f_plus = diff_a + diff_b;  // symmetric mode
            let f_minus = diff_a - diff_b; // antisymmetric mode

            // Update cumulative terms
            cumulative_plus += f_plus;
            cumulative_minus += f_minus;

            // Tail terms
            let tail_term_plus = f_plus * tail_plus.tail_impact_events[t_idx];
            let tail_term_minus = f_minus * tail_minus.tail_impact_events[t_idx];

            // Mode impacts
            let impact_plus = cumulative_plus + tail_term_plus;
            let impact_minus = cumulative_minus + tail_term_minus;

            // Reconstruct side impact
            let side_impact = if is_ask {
                0.5 * (impact_plus + impact_minus)
            } else {
                0.5 * (impact_plus - impact_minus)
            };

            impact.push(side_impact);
        }

        impact
    }

    /// Advance queue pointer to time t and update current value.
    fn advance_queue(idx: &mut usize, curr: &mut f64, path: &QueuePath, t: f64) {
        let len = path.events.len();
        while *idx + 1 < len && path.events[*idx + 1].time <= t {
            *idx += 1;
        }
        if *idx < len && path.events[*idx].time <= t {
            *curr = path.events[*idx].queue_size as f64;
        }
    }

    /// Create from BidAskQueuePath structs for convenience.
    pub fn from_bidask_paths(
        q: &BidAskQueuePath,
        q_prime: &BidAskQueuePath,
        tail_impact: &BidAskTailImpact,
    ) -> Self {
        Self::new(&q.ask, &q.bid, &q_prime.ask, &q_prime.bid, tail_impact)
    }

    /// Memory-efficient impact computation from pre-sampled queue values.
    ///
    /// This takes queue values already sampled at market order times, avoiding the need
    /// to build full `QueuePath` objects and scan through them.
    ///
    /// # Arguments
    ///
    /// For ask impact (evaluated at ask market order times):
    /// - `q_a_at_ask`, `q_b_at_ask` - queue values at ask market times
    /// - `q_prime_a_at_ask`, `q_prime_b_at_ask` - counterfactual queue values at ask market times
    ///
    /// For bid impact (evaluated at bid market order times):
    /// - `q_a_at_bid`, `q_b_at_bid` - queue values at bid market times
    /// - `q_prime_a_at_bid`, `q_prime_b_at_bid` - counterfactual queue values at bid market times
    ///
    /// # Panics
    /// Panics if sample lengths don't match the corresponding tail impact event counts.
    #[allow(clippy::too_many_arguments)]
    pub fn from_queue_samples(
        // At ask market order times
        q_a_at_ask: &[u32],
        q_b_at_ask: &[u32],
        q_prime_a_at_ask: &[u32],
        q_prime_b_at_ask: &[u32],
        // At bid market order times
        q_a_at_bid: &[u32],
        q_b_at_bid: &[u32],
        q_prime_a_at_bid: &[u32],
        q_prime_b_at_bid: &[u32],
        tail_impact: &BidAskTailImpact,
    ) -> Self {
        let n_ask = tail_impact.plus_ask.tail_impact.events.len();
        let n_bid = tail_impact.plus_bid.tail_impact.events.len();

        // Validate lengths
        assert_eq!(q_a_at_ask.len(), n_ask, "q_a_at_ask length mismatch");
        assert_eq!(q_b_at_ask.len(), n_ask, "q_b_at_ask length mismatch");
        assert_eq!(q_prime_a_at_ask.len(), n_ask, "q_prime_a_at_ask length mismatch");
        assert_eq!(q_prime_b_at_ask.len(), n_ask, "q_prime_b_at_ask length mismatch");
        assert_eq!(q_a_at_bid.len(), n_bid, "q_a_at_bid length mismatch");
        assert_eq!(q_b_at_bid.len(), n_bid, "q_b_at_bid length mismatch");
        assert_eq!(q_prime_a_at_bid.len(), n_bid, "q_prime_a_at_bid length mismatch");
        assert_eq!(q_prime_b_at_bid.len(), n_bid, "q_prime_b_at_bid length mismatch");

        // Compute ask impact
        let ask_impact = Self::compute_side_impact_from_samples(
            q_a_at_ask,
            q_b_at_ask,
            q_prime_a_at_ask,
            q_prime_b_at_ask,
            &tail_impact.plus_ask.tail_impact.tail_impact_events,
            &tail_impact.minus_ask.tail_impact.tail_impact_events,
            true, // ask side: add the minus term
        );

        // Compute bid impact
        let bid_impact = Self::compute_side_impact_from_samples(
            q_a_at_bid,
            q_b_at_bid,
            q_prime_a_at_bid,
            q_prime_b_at_bid,
            &tail_impact.plus_bid.tail_impact.tail_impact_events,
            &tail_impact.minus_bid.tail_impact.tail_impact_events,
            false, // bid side: subtract the minus term
        );

        Self { ask_impact, bid_impact }
    }

    /// Compute impact for one side from pre-sampled queue values.
    fn compute_side_impact_from_samples(
        q_a: &[u32],
        q_b: &[u32],
        q_prime_a: &[u32],
        q_prime_b: &[u32],
        tail_plus_events: &[f64],
        tail_minus_events: &[f64],
        is_ask: bool,
    ) -> Vec<f64> {
        let n = q_a.len();
        let mut impact = Vec::with_capacity(n);

        let mut cumulative_plus: f64 = 0.0;
        let mut cumulative_minus: f64 = 0.0;

        for t_idx in 0..n {
            // Compute differences
            let diff_a = q_prime_a[t_idx] as f64 - q_a[t_idx] as f64;
            let diff_b = q_prime_b[t_idx] as f64 - q_b[t_idx] as f64;

            // Mode decomposition
            let f_plus = diff_a + diff_b;  // symmetric mode
            let f_minus = diff_a - diff_b; // antisymmetric mode

            // Update cumulative terms
            cumulative_plus += f_plus;
            cumulative_minus += f_minus;

            // Tail terms
            let tail_term_plus = f_plus * tail_plus_events[t_idx];
            let tail_term_minus = f_minus * tail_minus_events[t_idx];

            // Mode impacts
            let impact_plus = cumulative_plus + tail_term_plus;
            let impact_minus = cumulative_minus + tail_term_minus;

            // Reconstruct side impact
            let side_impact = if is_ask {
                0.5 * (impact_plus + impact_minus)
            } else {
                0.5 * (impact_plus - impact_minus)
            };

            impact.push(side_impact);
        }

        impact
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_c_matrix_eigenvalues() {
        let c_matrix = SymmetricCMatrix::new(1.0, 0.3);
        assert!((c_matrix.lambda_plus() - 1.3).abs() < 1e-10);
        assert!((c_matrix.lambda_minus() - 0.7).abs() < 1e-10);
    }

    #[test]
    fn test_c_matrix_from_affine() {
        // b_l_own = -0.1, b_l_cross = 0.05
        // b_c_own = 0.2, b_c_cross = -0.02
        // c = 0.2 - (-0.1) = 0.3
        // a = -0.02 - 0.05 = -0.07
        let c_matrix = SymmetricCMatrix::from_affine_symmetric(-0.1, 0.05, 0.2, -0.02);
        assert!((c_matrix.c - 0.3).abs() < 1e-10);
        assert!((c_matrix.a - (-0.07)).abs() < 1e-10);
    }

    #[test]
    fn test_zero_impact_identical_paths() {
        use crate::models::QueueEvent;

        // If q = q', impact should be zero
        let events = vec![
            QueueEvent { queue_event: 2, queue_size: 10, time: 0.0 },
            QueueEvent { queue_event: 2, queue_size: 9, time: 1.0 },
            QueueEvent { queue_event: 2, queue_size: 8, time: 2.0 },
        ];
        let q_a = QueuePath { events: events.clone() };
        let q_b = QueuePath { events: events.clone() };

        let c_matrix = SymmetricCMatrix::new(1.0, 0.2);
        let tail_impact = BidAskTailImpact::new_symmetric_hawkes(
            1.0,
            vec![0.3],
            vec![1.0],
            c_matrix,
            vec![1.0, 2.0],
            vec![1.0, 2.0],
        );

        let impact = BidAskImpactPath::new(&q_a, &q_b, &q_a, &q_b, &tail_impact);

        for val in &impact.ask_impact {
            assert!(val.abs() < 1e-10, "Expected zero ask impact, got {}", val);
        }
        for val in &impact.bid_impact {
            assert!(val.abs() < 1e-10, "Expected zero bid impact, got {}", val);
        }
    }

    #[test]
    fn test_bidask_tail_impact_creation() {
        let c_matrix = SymmetricCMatrix::new(1.0, 0.2);
        let tail_impact = BidAskTailImpact::new_symmetric_hawkes(
            1.0,
            vec![0.3, 0.2],
            vec![1.0, 2.0],
            c_matrix.clone(),
            vec![0.5, 1.5, 2.5],
            vec![0.3, 1.2],
        );

        // Check that events are stored correctly
        assert_eq!(tail_impact.plus_ask.tail_impact.events.len(), 3);
        assert_eq!(tail_impact.minus_ask.tail_impact.events.len(), 3);
        assert_eq!(tail_impact.plus_bid.tail_impact.events.len(), 2);
        assert_eq!(tail_impact.minus_bid.tail_impact.events.len(), 2);

        // Check c_lambda values
        assert!((tail_impact.plus_ask.c_lambda - 1.2).abs() < 1e-10);
        assert!((tail_impact.minus_ask.c_lambda - 0.8).abs() < 1e-10);
    }
}
