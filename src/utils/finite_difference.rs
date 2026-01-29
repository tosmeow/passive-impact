/// Finite Difference derivative finder.
/// F incorporates a function f, where we want its derivative at a point a and know that its defined properly on [f-eps, f+eps] (callable interval).
pub struct FDSolver<F>
where
    F: Fn(f64) -> f64,
{
    f: F,
    a: f64,
    eps: f64,
}

impl<F> FDSolver<F>
where
    F: Fn(f64) -> f64,
{
    pub fn new(f: F, a: f64, eps: f64) -> Self {
        Self { f, a, eps}
    }
    // We want tolerance of < tol on the computed derivative at a: that is, for 2 different choices of h, if f' - f'_bis is < eps, then we stop and output the average of these two derivatives FD.
    pub fn solve(&self, max_iter: usize, tol: f64) -> Option<f64> {
        let mut curr_y = ((self.f)(self.a + self.eps) - (self.f)(self.a - self.eps)) / (2.0 * self.eps);
        for i in 1..=max_iter {
            // Exponentially close to the point a, until the precision of derivatives is good enough for our goal.
            let fraction = 0.1_f64.powi(i as i32) * self.eps;
            let y = ((self.f)(self.a + fraction) - (self.f)(self.a - fraction)) / (2.0 * fraction);
            if (y - curr_y).abs() <= tol {
                return Some((y + curr_y) / 2.0)
            }
            else {
                curr_y = y;
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_squared() {
        // f(x) = x ** 2
        // Has derivative 2 at 1.
        let solver = FDSolver::new(
            |x| x.powi(2),
            1.0,
            1.0,
        );
        let derivative = solver.solve(100, 1e-9).unwrap();
        assert!((derivative - 2.0).abs() < 1e-9);
    }
}
