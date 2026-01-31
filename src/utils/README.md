# Utils Module

Numerical solvers for root-finding and derivative estimation.

## Overview

This module provides specialized numerical methods designed for the specific challenges encountered in Hawkes propagator computation:

- **Singular endpoints:** Functions undefined at interval boundaries
- **Infinite limits:** Functions approaching $\pm\infty$ at endpoints
- **High precision:** Derivatives needed for coefficient computation

## Architecture

```
utils/
├── ivt.rs              # Root-finding via Intermediate Value Theorem
└── finite_difference.rs # Numerical derivative estimation
```

## IVT Solver

### Problem Setting

Find a root of $f(x) = 0$ on an open interval $(a, b)$ where:

- $f$ may be **undefined** at $a$ and $b$
- $\lim_{x \to a^+} f(x)$ and $\lim_{x \to b^-} f(x)$ have **opposite signs**
- The limits may be $\pm\infty$

This situation arises when finding roots of the characteristic equation:

$$g(x) = 1 - \sum_{i=1}^{k} \frac{\alpha_i}{x + \beta_i}$$

which has poles at $x = -\beta_i$ and roots in each interval $(-\beta_{i+1}, -\beta_i)$.

### Algorithm

The `IVTSolver` implements a two-phase approach:

**Phase 1: Endpoint Search**
```
Starting from both endpoints, move inward exponentially until f is defined:
    a' = a + ε,  2ε,  4ε, ...  until f(a') is finite
    b' = b - ε,  2ε,  4ε, ...  until f(b') is finite
```

**Phase 2: Bisection**
```
Standard bisection on [a', b']:
    mid = (a' + b') / 2
    if sign(f(mid)) == sign(f(a')): a' = mid
    else: b' = mid
    repeat until |b' - a'| < tol
```

### Usage

```rust
use utils::IVTSolver;

// Find root of 1/(x-1) - 1/(2-x) on (1, 2)
// Note: f is undefined at x=1 and x=2
let f = |x: f64| 1.0 / (x - 1.0) - 1.0 / (2.0 - x);
let solver = IVTSolver::new(f, 1.0, 2.0);

let root = solver.find_zero(1e-10, 100);
assert!((root.unwrap() - 1.5).abs() < 1e-9);  // Root at x = 1.5
```

### API

```rust
pub struct IVTSolver<F: Fn(f64) -> f64> {
    f: F,
    a: f64,  // Left endpoint (exclusive)
    b: f64,  // Right endpoint (exclusive)
}

impl<F: Fn(f64) -> f64> IVTSolver<F> {
    pub fn new(f: F, a: f64, b: f64) -> Self;

    /// Find root with given tolerance and max iterations
    /// Returns None if no valid interior interval found
    pub fn find_zero(&self, tol: f64, max_iter: usize) -> Option<f64>;
}
```

### Application: Propagator Roots

For the Hawkes characteristic equation, roots lie in intervals $(-\beta_{i+1}, -\beta_i)$:

```rust
let alphas = vec![0.1, 0.2, 0.3];
let betas = vec![1.0, 2.0, 5.0];

// Characteristic function
let g = |x: f64| {
    1.0 - alphas.iter().zip(&betas)
        .map(|(a, b)| a / (x + b))
        .sum::<f64>()
};

// Find root in (-2, -1) — between -β₂ and -β₁
let solver = IVTSolver::new(g, -2.0, -1.0);
let lambda = solver.find_zero(1e-12, 100).unwrap();
```

## Finite Difference Solver

### Problem Setting

Estimate $f'(a)$ when:
- Only function evaluations $f(x)$ are available
- High precision is required
- The function may have numerical noise

### Algorithm

The `FDSolver` uses **central differences** with adaptive step size:

$$f'(a) \approx \frac{f(a + h) - f(a - h)}{2h}$$

**Convergence criterion:** The derivative estimate is accepted when two consecutive estimates (with halved step size) agree within tolerance:

```
h = eps
d1 = (f(a+h) - f(a-h)) / 2h
repeat:
    h = h / 2
    d2 = (f(a+h) - f(a-h)) / 2h
    if |d1 - d2| < tol:
        return (d1 + d2) / 2
    d1 = d2
```

### Usage

```rust
use utils::FDSolver;

// Estimate derivative of sin(x) at x = 0
let f = |x: f64| x.sin();
let solver = FDSolver::new(f, 0.0, 0.1);  // a=0, eps=0.1

let derivative = solver.solve(100, 1e-10);
assert!((derivative.unwrap() - 1.0).abs() < 1e-8);  // sin'(0) = cos(0) = 1
```

### API

```rust
pub struct FDSolver<F: Fn(f64) -> f64> {
    f: F,
    a: f64,    // Point at which to estimate derivative
    eps: f64,  // Initial step size (callable half-width)
}

impl<F: Fn(f64) -> f64> FDSolver<F> {
    pub fn new(f: F, a: f64, eps: f64) -> Self;

    /// Estimate f'(a) with given max iterations and tolerance
    /// Returns None if convergence not achieved
    pub fn solve(&self, max_iter: usize, tol: f64) -> Option<f64>;
}
```

### Application: Propagator Coefficients

After finding the propagator roots $\lambda_j$, the coefficients $c_j$ are computed via:

$$c_j = \frac{1}{g'(\lambda_j)}$$

where $g$ is the characteristic function. Since $g$ involves sums of rational functions, finite differences provide a robust way to estimate $g'$:

```rust
let g = |x: f64| { /* characteristic function */ };

for &lambda_j in &lambdas {
    let solver = FDSolver::new(&g, lambda_j, 0.01);
    let g_prime = solver.solve(50, 1e-12).unwrap();
    let c_j = 1.0 / g_prime;
}
```

## Numerical Considerations

### Tolerance Selection

| Application | Recommended Tolerance |
|-------------|----------------------|
| Root-finding (IVT) | `1e-10` to `1e-12` |
| Derivative (FD) | `1e-8` to `1e-10` |

### Potential Issues

1. **Nearly coincident roots:** If $\lambda_i \approx \lambda_j$, numerical instability may occur
2. **Very small $\alpha_i$:** Can cause ill-conditioning in the characteristic equation
3. **Large $k$:** Many exponential components increase root-finding complexity

### Debugging Tips

```rust
// Enable verbose output for debugging
let solver = IVTSolver::new(f, a, b);
if let Some(root) = solver.find_zero(1e-10, 100) {
    println!("Root found: {}", root);
    println!("f(root) = {}", f(root));  // Should be ~0
} else {
    println!("No root found — check interval bounds");
}
```
