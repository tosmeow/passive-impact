/// Intermediate Value Theorem root finder for functions undefined at endpoints.
/// Assumes f has limits of opposite signs at a and b.

pub struct IVTSolver<F>
where
    F: Fn(f64) -> f64,
{
    f: F,
    a: f64,
    b: f64,
}

impl<F> IVTSolver<F>
where
    F: Fn(f64) -> f64,
{
    pub fn new(f: F, a: f64, b: f64) -> Self {
        Self { f, a, b }
    }

    fn find_finite_point(&self, max_probes: usize) -> Option<(f64, f64)> {
        let len = self.b - self.a;

        for i in 1..=max_probes {
            // Exponentially approach the interior: start at 1/2, then 1/4, 1/8, etc. till we find opposite signs endpoints.
            let fraction = 0.1_f64.powi(i as i32);
            let x = (self.a + fraction * len, self.b - fraction * len);

            let fx = (self.f)(x.0) * (self.f)(x.1);
            if fx <= 0.0 {
                return Some(x);
            }
        }
        None
    }

    /// Finds a zero of f in the open interval (a, b), when we only assume lim f at a and lim f at b to have opposite signs without necessarly being defined at these endpoints.
    pub fn find_zero(&self, tol: f64, max_iter: usize) -> Option<f64> {
        const MAX_PROBES: usize = 50;

        // Find finite starting points from both ends
        let (mut lo, mut hi) = self.find_finite_point(MAX_PROBES)?;
        // Ensure lo < hi
        if lo > hi {
            std::mem::swap(&mut lo, &mut hi);
        }
        let mut f_lo = (self.f)(lo);

        for _ in 0..max_iter {
            let mid = (lo + hi) / 2.0;
            let f_mid = (self.f)(mid);

            if f_mid.abs() < tol || (hi - lo) / 2.0 < tol {
                return Some(mid);
            }

            if f_lo * f_mid < 0.0 {
                hi = mid;
            } else {
                lo = mid;
                f_lo = f_mid;
            }
        }

        Some((lo + hi) / 2.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_singular_endpoints() {
        // f(x) = 1/(x-1) - 1/(2-x), undefined at x=1 and x=2
        // Has a root at x = 1.5
        let solver = IVTSolver::new(
            |x| 1.0 / (x - 1.0) - 1.0 / (2.0 - x),
            1.0,
            2.0,
        );
        let root = solver.find_zero(1e-10, 100).unwrap();
        assert!((root - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_opposite_sign_infinities() {
        // f(x) = 1/x - 1/(1-x), undefined at 0 and 1
        // At x→0: f→+∞, at x→1: f→-∞ (opposite signs!)
        // Root at x = 0.5
        let solver = IVTSolver::new(
            |x| 1.0 / x - 1.0 / (1.0 - x),
            0.0,
            1.0,
        );
        let root = solver.find_zero(1e-10, 100).unwrap();
        assert!((root - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_laplace_inverse() {
        // f(x) = 1 - 1/(x+2), undefined at -2 with positive value at 0
        // Root at x = -1.
        let solver = IVTSolver::new(
            |x| 1.0 - 1.0 / (x + 2.0),
            -2.0,
            0.0,
        );
        let root = solver.find_zero(1e-10, 100).unwrap();
        assert!((root + 1.0).abs() < 1e-9);
    }
    #[test]
    fn test_same_sign_limits() {
        // f(x) = 1/x² + 1/(1-x)², both endpoints → +∞
        // No sign change possible, so find_zero should return None
        let solver = IVTSolver::new(
            |x| 1.0 / (x * x) + 1.0 / ((1.0 - x) * (1.0 - x)),
            0.0,
            1.0,
        );
        assert!(solver.find_zero(1e-10, 100).is_none());
    }
}
