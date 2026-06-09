use super::tail_intensity::TailIntensity;
use crate::models::{AffineQueueProcess, MultiExponentialHawkes, MultivariateMarkovianIntensity};

pub struct TailImpact {
    pub events: Vec<f64>,
    pub tail_impact_events: Vec<f64>,
}

impl TailImpact {
    // Create TailImpact with explicit Hawkes params and c_lambda.
    pub fn new(hawkes_params: MultiExponentialHawkes, c_lambda: f64, events: Vec<f64>) -> Self {
        let tail_intensity = TailIntensity::new(hawkes_params.clone(), c_lambda);
        let mut tail_impact_events: Vec<f64> = Vec::with_capacity(events.len());
        let mut prev_t: f64 = 0.0;
        let mut state: Vec<f64> = vec![0.0; hawkes_params.alpha.len()];
        for t in events.iter() {
            hawkes_params.update_state(&mut state, 0, *t, prev_t);
            tail_impact_events.push(tail_intensity.compute(&state));
            prev_t = *t;
        }
        Self {
            events,
            tail_impact_events,
        }
    }

    // Create TailImpact for an affine queue process.
    //
    // Computes c_lambda automatically from the affine slopes:
    // c_lambda = b_c - b_l
    pub fn from_affine_queue(
        mu: f64,
        alpha: Vec<f64>,
        beta: Vec<f64>,
        b_l: f64,
        b_c: f64,
        events: Vec<f64>,
    ) -> Self {
        let hawkes_params = MultiExponentialHawkes::new(mu, alpha, beta);
        let c_lambda = AffineQueueProcess::c_lambda(b_l, b_c);
        Self::new(hawkes_params, c_lambda, events)
    }

    // Create TailImpact directly from a fitted price-propagator exponential sum.
    //
    // If g(t) = kappa + sum_i weights_i exp(-beta_i t), the dimensionless
    // effective passive kernel is
    //
    //   K_C(t) = sum_i eta_i exp(-beta_i t),
    //   eta_i = beta_i weights_i / (kappa * (beta_i + c_lambda)).
    //
    // This avoids interpreting signed propagator weights as Hawkes amplitudes.
    pub fn from_tail_propagator(
        propagator_kappa: f64,
        propagator_weights: Vec<f64>,
        propagator_beta: Vec<f64>,
        c_lambda: f64,
        zeta: f64,
        events: Vec<f64>,
    ) -> Result<Self, String> {
        let eta = Self::tail_propagator_effective_coefficients(
            propagator_kappa,
            &propagator_weights,
            &propagator_beta,
            c_lambda,
        )?;
        let mut tail_impact_events: Vec<f64> = Vec::with_capacity(events.len());
        let mut state: Vec<f64> = vec![0.0; propagator_beta.len()];
        let mut prev_t: f64 = 0.0;

        for &t in &events {
            if !t.is_finite() {
                return Err("events must be finite".to_string());
            }
            let dt = t - prev_t;
            if dt < 0.0 {
                return Err("events must be sorted".to_string());
            }
            for i in 0..state.len() {
                state[i] = state[i] * (-propagator_beta[i] * dt).exp() + eta[i];
            }
            tail_impact_events.push(zeta + state.iter().sum::<f64>());
            prev_t = t;
        }

        Ok(Self {
            events,
            tail_impact_events,
        })
    }

    pub fn tail_propagator_effective_coefficients(
        propagator_kappa: f64,
        propagator_weights: &[f64],
        propagator_beta: &[f64],
        c_lambda: f64,
    ) -> Result<Vec<f64>, String> {
        if propagator_weights.len() != propagator_beta.len() {
            return Err(
                "propagator_weights and propagator_beta must have matching lengths".to_string(),
            );
        }
        if propagator_kappa == 0.0 || !propagator_kappa.is_finite() {
            return Err("propagator_kappa must be finite and nonzero".to_string());
        }
        if c_lambda <= 0.0 || !c_lambda.is_finite() {
            return Err("c_lambda must be finite and positive".to_string());
        }
        if propagator_beta
            .iter()
            .any(|&beta| beta <= 0.0 || !beta.is_finite())
        {
            return Err("propagator_beta values must be finite and positive".to_string());
        }
        if propagator_weights.iter().any(|&weight| !weight.is_finite()) {
            return Err("propagator_weights values must be finite".to_string());
        }

        Ok(propagator_weights
            .iter()
            .zip(propagator_beta)
            .map(|(&weight, &beta)| beta * weight / (propagator_kappa * (beta + c_lambda)))
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tail_propagator_coefficients_scale_by_c_lambda() {
        let eta =
            TailImpact::tail_propagator_effective_coefficients(2.0, &[0.2, -0.4], &[1.0, 3.0], 1.0)
                .unwrap();

        assert!((eta[0] - 0.05).abs() < 1e-12);
        assert!((eta[1] + 0.15).abs() < 1e-12);
    }

    #[test]
    fn tail_propagator_updates_event_states() {
        let tail =
            TailImpact::from_tail_propagator(1.0, vec![2.0], vec![3.0], 1.0, 0.5, vec![0.0, 1.0])
                .unwrap();
        let eta = 3.0 * 2.0 / (1.0 * (3.0 + 1.0));

        assert!((tail.tail_impact_events[0] - (0.5 + eta)).abs() < 1e-12);
        assert!((tail.tail_impact_events[1] - (0.5 + eta * (-3.0_f64).exp() + eta)).abs() < 1e-12);
    }
}
