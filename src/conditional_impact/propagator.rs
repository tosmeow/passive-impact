// For a sum of exponentials kernel $\phi(s) = \sum_{i=1}^N \alpha_i e^{-\beta_i * s}, we compute numerically the coefficients such that (\delta_0 - \phi) = 1 + \psi
// where \psi is another sum of exponentials whose parameters we specify.

use crate::utils::IVTSolver;
use crate::utils::FDSolver;
use crate::models::MultiExponentialHawkes;

// This structure is meant to store the kernel parameters of the propagator (delta_0 - f)^-1 - delta_0 associated to a kernel of type MultiExponentialHawkes.

pub struct Propagator {
    pub hawkes_params: MultiExponentialHawkes,
    //Decay rate of each exponential component
    pub lambda: Vec<f64>,
    //Excitation amplitude for each component
    pub c: Vec<f64>
}

impl Propagator {
    pub fn new(hawkes_params: MultiExponentialHawkes) -> Self {
        let n = hawkes_params.alpha.len();
        let mut lambda = Vec::with_capacity(n);
        let mut c = Vec::with_capacity(n);
        let norm = hawkes_params.alpha.iter().zip(&hawkes_params.beta).map(|(a, b)| a * b).sum::<f64>();
        let len = hawkes_params.alpha.len();
        let f = |x:f64| 1.0 - hawkes_params.alpha.iter().zip(&hawkes_params.beta).map(|(a, b)| a / (x + b)).sum::<f64>();
        
        // Smallest root
        let solver = IVTSolver::new(&f, norm, -hawkes_params.beta[0]);
        let root = solver.find_zero(1e-10, 100).expect("root not found");
        let (lo, hi) = ((root + hawkes_params.beta[0]), norm - root);//1.0 - norm - root);
        let bounds = if lo.min(hi) != 0.0 {lo.min(hi) / 2.0} else {1e-3};
        let fd_solver = FDSolver::new(&f, root, bounds);
        lambda.push(-root);
        c.push(fd_solver.solve(100, bounds * 1e-6).expect("derivative failed"));
        for i in 0..len-1 {
            let solver = IVTSolver::new(&f, -hawkes_params.beta[i+1], -hawkes_params.beta[i]);
            let root = solver.find_zero(1e-10, 100).expect("root not found");
            let bounds = (root + hawkes_params.beta[i+1]).min(-hawkes_params.beta[i] - root) / 2.0;
            let fd_solver = FDSolver::new(&f, root, bounds);
            lambda.push(-root);
            c.push(fd_solver.solve(100, bounds * 1e-6).expect("derivative failed"));
        }
        
        Self { hawkes_params, lambda, c }
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
