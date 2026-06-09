use super::propagator::Propagator;
use crate::models::MultiExponentialHawkes;

// TailIntensity computes ∫_t^∞ e^{-c_lambda·(s-t)} E_t[λ_s] ds for a Hawkes intensity λ.
//
// Sign convention note (differs from the paper's notation):
//   In the paper (Assumption A), c_λ = slope(λ^L) - slope(λ^C) = b_l - b_c < 0.
//   The paper writes the kernel as e^{c_λ·(s-t)} with c_λ < 0 (decaying).
//   In this codebase, `c_lambda` = b_c - b_l > 0 (see AffineQueueProcess::c_lambda), so:
//       c_lambda (code) = -(c_λ in paper) > 0.
//   The kernel written as e^{-c_lambda·(s-t)} here is identical to e^{c_λ·(s-t)} in the paper.
//   The integral converges in both cases because c_lambda > 0 (equivalently c_λ < 0).

pub struct TailIntensity {
    pub hawkes_params: MultiExponentialHawkes,
    //Decay rate of each exponential component
    pub c_lambda: f64,
    //Excitation amplitude for each component
    pub factors: Vec<f64>,

    pub lambda: Vec<f64>,

    pub c: Vec<f64>,
}

impl TailIntensity {
    pub fn new(hawkes_params: MultiExponentialHawkes, c_lambda: f64) -> Self {
        let propagator_params = Propagator::new(hawkes_params.clone());
        let n = hawkes_params.alpha.len();
        let mut factors = Vec::with_capacity(n);
        for i in 0..n {
            let beta = hawkes_params.beta[i];
            let factor = propagator_params
                .lambda
                .iter()
                .zip(&propagator_params.c)
                .map(|(lambda, c)| {
                    c / (lambda - beta) * ((1.0 / (c_lambda + beta)) - (1.0 / (c_lambda + lambda)))
                })
                .sum::<f64>()
                + 1.0 / (c_lambda + beta);
            factors.push(factor);
        }
        Self {
            hawkes_params,
            c_lambda,
            factors,
            lambda: propagator_params.lambda.clone(),
            c: propagator_params.c.clone(),
        }
    }

    pub fn compute(&self, state: &[f64]) -> f64 {
        self.factors
            .iter()
            .zip(state)
            .map(|(f, r)| r * f)
            .sum::<f64>()
            + self.hawkes_params.mu
                * (1.0 / self.c_lambda
                    + self
                        .lambda
                        .iter()
                        .zip(&self.c)
                        .map(|(lbd, c)| {
                            ((1.0 / self.c_lambda) - (1.0 / (self.c_lambda + lbd))) * c / lbd
                        })
                        .sum::<f64>())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_solver_1() {
        // Control in the 1-Dimensional case that the factor coefficient we implemented is exactly the intended one.
        let params = MultiExponentialHawkes::new(0.0, vec![1.0], vec![2.0]);
        let tail_intensity = TailIntensity::new(params.clone(), 0.0);
        assert!((tail_intensity.factors[0] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_solver_2() {
        // Now, when factor is well-controlled, does compute produce the right value for the impact term?
        let params = MultiExponentialHawkes::new(0.0, vec![1.0], vec![2.0]);
        let tail_intensity = TailIntensity::new(params.clone(), 1e-8);
        assert!((tail_intensity.compute(&vec![1.0]) - 1.0).abs() < 1e-5);
    }
}
