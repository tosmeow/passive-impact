# Utils

Numerical solvers for propagator computation.

## Structure

```
utils/
├── ivt.rs              # Root-finding via bisection
├── finite_difference.rs # Numerical derivatives
└── npy_io.rs           # NumPy binary output
```

## IVT Solver

Find roots on open intervals $(a, b)$ where $f$ may be undefined at endpoints.

**Algorithm**:
1. Search towards endpoints until $f$ has opposite signs.
2. Bisection on the valid interior interval.

```rust
use simulation_project::utils::IVTSolver;

// Find root of 1/(x-1) - 1/(2-x) on (1, 2)
let f = |x: f64| 1.0 / (x - 1.0) - 1.0 / (2.0 - x);
let solver = IVTSolver::new(f, 1.0, 2.0);
let root = solver.find_zero(1e-10, 100);  // tol, max_iter
```

**Application**: Propagator roots in intervals $(-\beta_{i+1}, -\beta_i)$.

## Finite Difference Solver

Central differences with adaptive step halving:

$$f'(a) \approx \frac{f(a+h) - f(a-h)}{2h}$$

Converges when consecutive estimates agree within tolerance: $|\frac{f(a+h) - f(a-h)}{2h} - \frac{f(a+h') - f(a-h')}{2h'}| < tol$ for two successive choices $h,h'$.

```rust
use simulation_project::utils::FDSolver;

let f = |x: f64| x.sin();
let solver = FDSolver::new(f, 0.0, 0.1);  // point, initial step
let deriv = solver.solve(100, 1e-10);     // max_iter, tol
```

**Application**: Propagator coefficients $c_j = 1/g'(\lambda_j)$.

## NumPy I/O

```rust
use simulation_project::utils::{write_npy_f64, write_npy_u32, write_npy_f64_1d};

write_npy_f64("data.npy", &data, n_rows, n_cols)?;
write_npy_u32("counts.npy", &counts, n_rows, n_cols)?;
write_npy_f64_1d("times.npy", &times)?;
```
