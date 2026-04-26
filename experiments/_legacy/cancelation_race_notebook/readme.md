# Cancellation Race

**Paper section**: Additional experiment — not in paper

## What This Shows

When a cancellation cascade occurs (e.g., after a mid-price move), there is a race to cancel orders. This experiment investigates the **advantage of being early** to cancel: how does the timing of a cancellation burst affect the resulting queue dynamics and market impact?

## Motivation

When a cancelation cascade occurs (e.g., after a mid-price move), there's a race to cancel orders. This experiment investigates the **advantage of being early** to cancel when such a cascade happens.

## Setup

We simulate a queue process over a time horizon T = 5, with:
- **Reference path** q: A queue starting at q₀ = 200, driven by Hawkes market orders
- **Counterfactual paths** q̄(x): Same setup, but with p additional cancels injected at time x

### Conditioning Path Structure

In the interval [0, 1], we inject:
1. **n = 100 cancelation events** evenly spaced (conditioning path) - these represent the "cascade"
2. **n = 100 limit events** evenly spaced (external path) - to keep the queue stable near stationary

The counterfactual paths q̄(x) see all the same events as q, but additionally receive a **burst of p = 20 cancels** injected almost instantaneously at time x (spaced by ε = 10⁻⁸).

### Key Design: Shared Randomness

Using `simulate_multiple`, all simulations across different x values share the same random numbers for the thinning algorithm. This ensures:
- The only difference between paths is **when** the burst happens
- No spurious variance from different random seeds
- Clean comparison of the timing effect

### Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| n_cancels | 100 | Conditioning cancel events in [0, 1] |
| n_limits | 100 | External limit events (queue stability) |
| p_burst | 20 | Cancels in the injected burst |
| epsilon | 10⁻⁸ | Time spacing between burst cancels |
| n_x_values | 50 | Different burst times tested |
| a_L | 100 | Limit order baseline intensity |
| b_L | -0.275 | Limit order queue sensitivity |
| a_C | 2 | Cancel order baseline intensity |
| b_C | 0.125 | Cancel order queue sensitivity |
| μ | 1 | Hawkes baseline intensity |
| α | [0.065, 0.2, 0.325, 0.65] | Hawkes excitation coefficients |
| β | [0.15, 0.6, 2.5, 10.0] | Hawkes decay rates |

---

## Question: What is the advantage of early cancelation?

When a cascade of cancels happens, being early to cancel means:
1. Your cancels are more likely to be **accepted** (higher intensity ratio)
2. The queue difference (q̄ - q) grows more negative (you removed more from queue)
3. Your **impact** accumulates differently

### Mechanism

Due to the affine intensity structure:
- When q̄ < q (we've canceled more), q̄ has:
  - **Higher** limit intensity (b_L < 0)
  - **Lower** cancel intensity (b_C > 0)
- This creates a mean-reverting effect: q̄ tends to catch up to q over time
- Early cancels have more time to benefit from this coupling before it reverts

---

## How to Run

```bash
# Generate data for all 50 burst times
cargo run --release --bin cancelation_race

# Analyze and visualize (plots are saved to images/ folder)
cd python/experiments/cancelation_race
python plot_utils.py
```

## Output Files

| File | Shape | Description |
|------|-------|-------------|
| impact_paths.npy | [n_market_times × n_x] | Impact at market order times |
| queue_paths.npy | [n_market_times × (1 + n_x)] | Queue values (q, then q̄ for each x) |
| queue_paths_grid.npy | [n_sample_times × (1 + n_x)] | Queue on fine grid |
| queue_diff_grid.npy | [n_sample_times × n_x] | q̄ - q on fine grid |
| times.npy | [n_market_times] | Market order times |
| sample_times.npy | [n_sample_times] | Fine sample grid |
| x_values.npy | [n_x] | Burst times tested |

---

## Expected Results

1. **Earlier bursts → more negative q̄ - q**: The earlier you cancel, the more queue reduction you achieve
2. **Impact depends on timing**: Early cancels accumulate more favorable impact before mean reversion kicks in
3. **Monotonic ordering**: Due to shared randomness, if x₁ < x₂, then q̄(x₁) ≤ q̄(x₂) at all times after x₂
