//! Event dimensions and queue update helpers for anchored queue experiments.

/// Limit-order row dimension.
pub const LIMIT_DIM: usize = 0;
/// Cancellation row dimension.
pub const CANCEL_DIM: usize = 1;
/// Market-order row dimension.
pub const MARKET_DIM: usize = 2;

/// Affine single-queue limit and cancellation intensity parameters.
///
/// The experiment only simulates independent limit/cancel deviations from the
/// empirical queue path. Market rows in the input path are common background
/// rows and therefore have no intensity here.
#[derive(Clone, Copy, Debug)]
pub struct AffineQueueIntensity {
    /// Limit-order intercept.
    pub a_l: f64,
    /// Limit-order queue slope.
    pub b_l: f64,
    /// Cancellation intercept.
    pub a_c: f64,
    /// Cancellation queue slope.
    pub b_c: f64,
}

impl AffineQueueIntensity {
    /// Return the non-negative intensity for `dim` at queue size `q`.
    #[inline]
    pub fn intensity(&self, q: f64, dim: usize) -> f64 {
        let q = q.max(0.0);
        match dim {
            LIMIT_DIM => (self.a_l + self.b_l * q).max(0.0),
            CANCEL_DIM => (self.a_c + self.b_c * q).max(0.0),
            _ => 0.0,
        }
    }
}

/// Convert an external integer dimension into an anchored queue dimension.
#[inline]
pub fn valid_dim(dim: i32) -> Option<usize> {
    match dim {
        0 => Some(LIMIT_DIM),
        1 => Some(CANCEL_DIM),
        2 => Some(MARKET_DIM),
        _ => None,
    }
}

/// Apply one unit event to a queue size.
#[inline]
pub fn apply_event_to_queue(q: f64, dim: usize) -> f64 {
    match dim {
        LIMIT_DIM => q + 1.0,
        CANCEL_DIM | MARKET_DIM => (q - 1.0).max(0.0),
        _ => q,
    }
}

/// Apply one sized row to a queue size.
#[inline]
pub fn apply_sized_event_to_queue(q: f64, dim: usize, qty: u32) -> f64 {
    let qty = qty as f64;
    match dim {
        LIMIT_DIM => q + qty,
        CANCEL_DIM | MARKET_DIM => (q - qty).max(0.0),
        _ => q,
    }
}

/// Apply an independent counterfactual unit event to the displaced queue.
#[inline]
pub fn apply_q_only_event(dq: f64, bar_q: f64, dim: usize) -> f64 {
    let q = (bar_q + dq).max(0.0);
    apply_event_to_queue(q, dim) - bar_q.max(0.0)
}

/// Keep `q = bar_q + dq` non-negative by clipping the offset from below.
#[inline]
pub fn clamp_offset(dq: f64, bar_q: f64) -> f64 {
    dq.max(-bar_q.max(0.0))
}

/// Sample the empirical post-event queue as a right-continuous step path.
pub fn sample_bar_queue(
    event_times: &[f64],
    bar_q_post: &[f64],
    sample_times: &[f64],
    initial_q: f64,
) -> Vec<f64> {
    sample_times
        .iter()
        .map(|&t| {
            let idx = upper_bound(event_times, t);
            if idx == 0 {
                initial_q
            } else {
                bar_q_post[idx - 1].max(0.0)
            }
        })
        .collect()
}

/// Sample an arbitrary right-continuous step path.
pub fn sample_step_values(
    step_times: &[f64],
    step_values: &[f64],
    sample_times: &[f64],
) -> Vec<f64> {
    sample_times
        .iter()
        .map(|&t| {
            let idx = upper_bound(step_times, t).saturating_sub(1);
            step_values[idx]
        })
        .collect()
}

/// Return the first index whose value is strictly greater than `x`.
pub fn upper_bound(values: &[f64], x: f64) -> usize {
    let mut lo = 0;
    let mut hi = values.len();
    while lo < hi {
        let mid = (lo + hi) / 2;
        if values[mid] <= x {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    lo
}
