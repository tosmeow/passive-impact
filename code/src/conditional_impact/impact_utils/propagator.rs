// Resolvent computation for a sum-of-exponentials Hawkes kernel.
//
// For φ(s) = Σᵢ αᵢ e^{-βᵢ s}, we compute the resolvent ψ such that
// (δ₀ - φ)⁻¹ = δ₀ + ψ, where ψ = Σⱼ cⱼ e^{-λⱼ s}.
//
// The roots λⱼ and residues cⱼ are found by solving the characteristic
// equation 1 - Σᵢ αᵢ/(x + βᵢ) = 0.
//
// NOTE: This resolvent is used by the passive (flow imbalance) model to
// solve the Volterra equation for E_t[λ_s]. It is NOT the propagator kernel
// G used in the aggressive hybrid price model. The propagator kernel
// G(t) = (1 - ‖φ‖₁) + Σᵢ (αᵢ/βᵢ) e^{-βᵢt} is derived from the
// martingale condition and uses the Hawkes rates βᵢ directly — see
// propagator_model/mod.rs.

use crate::models::MultiExponentialHawkes;
use crate::utils::FDSolver;
use crate::utils::IVTSolver;

/// Resolvent of the Hawkes kernel: (δ₀ - φ)⁻¹ = δ₀ + Σⱼ cⱼ e^{-λⱼ ·}
///
/// Used by `TailIntensity` (passive model) to compute E_t\[λ_s\] in closed form.
pub struct Propagator {
    pub hawkes_params: MultiExponentialHawkes,
    //Decay rate of each exponential component
    pub lambda: Vec<f64>,
    //Excitation amplitude for each component
    pub c: Vec<f64>,
}

impl Propagator {
    pub fn new(hawkes_params: MultiExponentialHawkes) -> Self {
        let n = hawkes_params.alpha.len();
        let mut lambda = Vec::with_capacity(n);
        let mut c = Vec::with_capacity(n);
        let norm = hawkes_params
            .alpha
            .iter()
            .zip(&hawkes_params.beta)
            .map(|(a, b)| a * b)
            .sum::<f64>();
        let len = hawkes_params.alpha.len();
        let f = |x: f64| {
            1.0 - hawkes_params
                .alpha
                .iter()
                .zip(&hawkes_params.beta)
                .map(|(a, b)| a / (x + b))
                .sum::<f64>()
        };

        // Smallest root
        let solver = IVTSolver::new(&f, norm, -hawkes_params.beta[0]);
        let root = solver.find_zero(1e-10, 100).expect("root not found");
        let (lo, hi) = ((root + hawkes_params.beta[0]), norm - root); //1.0 - norm - root);
        let bounds = if lo.min(hi) != 0.0 {
            lo.min(hi) / 2.0
        } else {
            1e-3
        };
        let fd_solver = FDSolver::new(&f, root, bounds);
        let derivative = fd_solver
            .solve(100, bounds * 1e-6)
            .expect("derivative failed");
        lambda.push(-root);
        c.push(1.0 / derivative);
        for i in 0..len - 1 {
            let solver = IVTSolver::new(&f, -hawkes_params.beta[i + 1], -hawkes_params.beta[i]);
            let root = solver.find_zero(1e-10, 100).expect("root not found");
            let bounds =
                (root + hawkes_params.beta[i + 1]).min(-hawkes_params.beta[i] - root) / 2.0;
            let fd_solver = FDSolver::new(&f, root, bounds);
            let derivative = fd_solver
                .solve(100, bounds * 1e-6)
                .expect("derivative failed");
            lambda.push(-root);
            c.push(1.0 / derivative);
        }

        Self {
            hawkes_params,
            lambda,
            c,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_solver_1() {
        let params = MultiExponentialHawkes::new(1.0, vec![1.0], vec![1.0]);
        let propag = Propagator::new(params);
        assert!((propag.lambda[0] - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_solver_2() {
        let params = MultiExponentialHawkes::new(1.0, vec![1.0], vec![2.0]);
        let propag = Propagator::new(params);
        assert!((propag.lambda[0] - 1.0).abs() < 1e-9);
    }
}
