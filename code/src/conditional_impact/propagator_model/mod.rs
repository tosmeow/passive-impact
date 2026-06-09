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
//! and κ is any decreasing impact function supplied by the caller.

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
    /// * `kappa` — impact function κ(q); must be decreasing in q
    pub fn from_queue_samples(
        q_samples: &[u32],
        q_bar_samples: &[u32],
        eval_times: &[f64],
        is_market_order: &[bool],
        hawkes: &MultiExponentialHawkes,
        kappa: impl Fn(f64) -> f64,
    ) -> Self {
        let n = eval_times.len();
        let n_components = hawkes.alpha.len();

        // Hawkes kernel norm: ‖φ‖₁ = Σᵢ αᵢ/βᵢ
        let norm: f64 = hawkes
            .alpha
            .iter()
            .zip(&hawkes.beta)
            .map(|(a, b)| a / b)
            .sum();
        // G(0) = 1/(1-‖φ‖₁) = mean cluster size
        let g0 = 1.0 / (1.0 - norm);
        // Propagator weights: G(0) · αᵢ/βᵢ for each exponential component
        let weights: Vec<f64> = hawkes
            .alpha
            .iter()
            .zip(&hawkes.beta)
            .map(|(a, b)| g0 * a / b)
            .collect();
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
            let kappa_q = kappa(q);
            let kappa_q_bar = kappa(q_bar);

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

    /// Compute hybrid aggressive impact from pre-sampled queue values.
    ///
    /// Uses a decomposed price model:
    /// ```text
    /// P_t = κ̄ ∫₀ᵗ G(t-s) d(N^a - N^b)_s + ∫₀ᵗ (κ(q^a_s) - κ̄) dN^a_s - ∫₀ᵗ (κ(q^b_s) - κ̄) dN^b_s
    /// ```
    /// where κ̄ is a constant (e.g. the average value of κ(q) at stationarity).
    ///
    /// Only metaorder events propagate through the kernel G with weight κ̄.
    /// Market orders contribute an instantaneous non-decaying cumulative sum
    /// with weight κ(q̄) - κ(q) (the queue-dependent correction to κ̄).
    ///
    /// The resulting impact formula (single-sided sell metaorder n) is:
    /// ```text
    /// MI_t = κ̄ ∫₀ᵗ G(t-s) dN^o_s + ∫₀ᵗ (κ(q̄_s) - κ(q_s)) dN_s
    /// ```
    ///
    /// # Arguments
    /// * `q_samples` — baseline queue sizes at each evaluation time
    /// * `q_bar_samples` — counterfactual queue sizes (with metaorder) at each time
    /// * `eval_times` — time points at which to evaluate impact
    /// * `is_market_order` — true = ordinary market order (dN), false = metaorder (dn)
    /// * `hawkes` — the Hawkes model whose kernel defines the propagator G
    /// * `kappa` — impact function κ(q); must be decreasing in q
    /// * `bar_kappa` — constant weight κ̄ for the propagator term (e.g. κ(E\[q\]))
    pub fn from_queue_samples_hybrid(
        q_samples: &[u32],
        q_bar_samples: &[u32],
        eval_times: &[f64],
        is_market_order: &[bool],
        hawkes: &MultiExponentialHawkes,
        kappa: impl Fn(f64) -> f64,
        bar_kappa: f64,
    ) -> Self {
        let n = eval_times.len();
        let n_components = hawkes.alpha.len();

        // Hawkes kernel norm: ‖φ‖₁ = Σᵢ αᵢ/βᵢ
        let norm: f64 = hawkes
            .alpha
            .iter()
            .zip(&hawkes.beta)
            .map(|(a, b)| a / b)
            .sum();
        // G(0) = 1/(1-‖φ‖₁)
        let g0 = 1.0 / (1.0 - norm);
        // Propagator weights for the decaying components
        let weights: Vec<f64> = hawkes
            .alpha
            .iter()
            .zip(&hawkes.beta)
            .map(|(a, b)| g0 * a / b)
            .collect();

        // Propagator state: decaying part of κ̄·G(t-s) summed over metaorder events
        let mut prop_state = vec![0.0f64; n_components];
        // Permanent part of propagator: G(∞)=1, so accumulates bar_kappa per metaorder
        let mut prop_permanent = 0.0f64;
        // Non-decaying cumulative sum from market orders: Σ (κ(q̄) - κ(q)) at dN times
        let mut instant_acc = 0.0f64;
        let mut impact_path = Vec::with_capacity(n);
        let mut prev_t = 0.0f64;

        for idx in 0..n {
            let t = eval_times[idx];
            let dt = t - prev_t;

            // Decay propagator exponential states
            for j in 0..n_components {
                prop_state[j] *= (-hawkes.beta[j] * dt).exp();
            }

            let q = q_samples[idx] as f64;
            let q_bar = q_bar_samples[idx] as f64;

            if is_market_order[idx] {
                // Market order (dN): instantaneous contribution, no propagator decay
                instant_acc += kappa(q_bar) - kappa(q);
            } else {
                // Metaorder (dn): propagator contribution with constant weight κ̄
                prop_permanent += bar_kappa;
                for j in 0..n_components {
                    prop_state[j] += weights[j] * bar_kappa;
                }
            }

            // Total impact = propagator(t) + instant cumulative sum
            let prop_decay: f64 = prop_state.iter().sum();
            impact_path.push(prop_permanent + prop_decay + instant_acc);
            prev_t = t;
        }

        AggressiveImpactPath { impact_path }
    }
}
