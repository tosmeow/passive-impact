//! Aggressive market impact computation under the propagator price model.
//!
//! In this model, the underlying price evolves as:
//! ```text
//! P_t = P_0 + ∫₀ᵗ κ(q^a_s) G(t-s) dN^a_s - ∫₀ᵗ κ(q^b_s) G(t-s) dN^b_s
//! ```
//! where G is the martingale-consistent propagator kernel derived from the
//! Hawkes kernel φ via the condition G'(t) = -G(0)φ(t).
//!
//! Matching the expectation-based price P_t = lim_{T→∞} E_t[N^a_T - N^b_T]
//! (constant-κ case) fixes G(0) = 1/(1-‖φ‖₁), the mean cluster size.
//!
//! For a sum-of-exponentials Hawkes kernel φ(t) = Σᵢ αᵢ e^{-βᵢt}:
//! ```text
//! G(t) = 1/(1-‖φ‖₁) · [(1 - ‖φ‖₁) + Σᵢ (αᵢ/βᵢ) e^{-βᵢt}]
//!      = 1 + Σᵢ (αᵢ/βᵢ)/(1-‖φ‖₁) · e^{-βᵢt}
//! ```
//! where ‖φ‖₁ = Σᵢ αᵢ/βᵢ < 1 (stationarity). Note G(0) = 1/(1-‖φ‖₁)
//! (instantaneous overshoot) and G(∞) = 1 (permanent impact per event).
//!
//! The aggressive market impact (from a metaorder that depletes the queue) is:
//! ```text
//! MI(t) = ∫₀ᵗ [κ(q̄_s) - κ(q_s)] G(t-s) dN_s + ∫₀ᵗ κ(q̄_s) G(t-s) dN^o_s
//! ```
//! where dN is the market order stream, dN^o is the metaorder stream,
//! and κ(q) = c_κ·q + d_κ is an affine impact function.

use crate::models::MultiExponentialHawkes;

/// Impact path computed under the propagator price model.
///
/// Tracks both ordinary market orders and metaorders, with the propagator
/// kernel G(t) controlling how historical price impacts decay over time.
///
/// The propagator kernel is derived from the Hawkes kernel via the martingale
/// condition. Its decay rates are the Hawkes kernel rates βᵢ themselves
/// (not the resolvent roots λⱼ used in the passive model).
pub struct AggressiveImpactPath {
    /// Impact values at each evaluation time
    pub impact_path: Vec<f64>,
}

impl AggressiveImpactPath {
    /// Compute aggressive impact from pre-sampled queue values.
    ///
    /// # Arguments
    /// * `q_samples` — baseline queue sizes at each evaluation time
    /// * `q_bar_samples` — counterfactual queue sizes (with metaorder) at each time
    /// * `eval_times` — time points at which to evaluate impact
    /// * `is_market_order` — boolean flag for each time: true = ordinary market order, false = metaorder
    /// * `hawkes` — the Hawkes model whose kernel defines the propagator
    /// * `c_kappa` — linear coefficient in κ(q) = c_κ·q + d_κ
    /// * `d_kappa` — constant term in κ(q)
    pub fn from_queue_samples(
        q_samples: &[u32],
        q_bar_samples: &[u32],
        eval_times: &[f64],
        is_market_order: &[bool],
        hawkes: &MultiExponentialHawkes,
        c_kappa: f64,
        d_kappa: f64,
    ) -> Self {
        let n = eval_times.len();
        let n_components = hawkes.alpha.len();

        // Hawkes kernel norm: ‖φ‖₁ = Σᵢ αᵢ/βᵢ
        let norm: f64 = hawkes.alpha.iter().zip(&hawkes.beta)
            .map(|(a, b)| a / b).sum();
        // G(0) = 1/(1-‖φ‖₁) = mean cluster size
        let g0 = 1.0 / (1.0 - norm);
        // Propagator weights: G(0) · αᵢ/βᵢ for each exponential component
        let weights: Vec<f64> = hawkes.alpha.iter().zip(&hawkes.beta)
            .map(|(a, b)| g0 * a / b).collect();
        let mut state = vec![0.0f64; n_components];
        let mut permanent_acc = 0.0f64; // G(∞) = 1, so permanent weight is 1
        let mut impact_path = Vec::with_capacity(n);
        let mut prev_t = 0.0f64;

        for idx in 0..n {
            let t = eval_times[idx];
            let dt = t - prev_t;

            // Decay exponential states: e^{-βⱼ · Δt}
            for j in 0..n_components {
                state[j] *= (-hawkes.beta[j] * dt).exp();
            }

            let q = q_samples[idx] as f64;
            let q_bar = q_bar_samples[idx] as f64;
            let kappa_q = c_kappa * q + d_kappa;
            let kappa_q_bar = c_kappa * q_bar + d_kappa;

            let contribution = if is_market_order[idx] {
                // Market order event (dN): contribute κ(q̄) - κ(q)
                kappa_q_bar - kappa_q
            } else {
                // Metaorder event (dN^o): contribute κ(q̄)
                kappa_q_bar
            };

            // Permanent component: G(∞) · contribution = 1 · contribution
            permanent_acc += contribution;

            // Decaying components: G(0)·(αᵢ/βᵢ) · contribution
            for j in 0..n_components {
                state[j] += weights[j] * contribution;
            }

            // Total impact = permanent + Σⱼ state[j]
            let decay_term: f64 = state.iter().sum();
            impact_path.push(permanent_acc + decay_term);
            prev_t = t;
        }

        AggressiveImpactPath { impact_path }
    }
}
