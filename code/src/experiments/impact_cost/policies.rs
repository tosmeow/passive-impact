use rand::Rng;

use crate::simulation_helpers::create_rng;

use super::events::LIMIT_DIM;

/// Return positions of rows whose dimension is `LIMIT_DIM`.
pub fn limit_positions(event_dims: &[i32]) -> Vec<usize> {
    event_dims
        .iter()
        .enumerate()
        .filter_map(|(idx, &dim)| (dim == LIMIT_DIM as i32).then_some(idx))
        .collect()
}

/// Select the first limit row in each `every_seconds` time bucket.
///
/// Buckets are anchored at the first limit row. Non-limit rows are ignored but
/// kept in the returned flag vector.
pub fn select_first_limit_every(
    event_times: &[f64],
    event_dims: &[i32],
    every_seconds: f64,
) -> Result<Vec<bool>, String> {
    if event_times.len() != event_dims.len() {
        return Err("event_times and event_dims must have matching lengths".to_string());
    }
    if every_seconds <= 0.0 {
        return Err("every_seconds must be positive".to_string());
    }

    let l_pos = limit_positions(event_dims);
    let mut flags = vec![false; event_dims.len()];
    if l_pos.is_empty() {
        return Ok(flags);
    }

    let origin = event_times[l_pos[0]];
    let mut last_bucket: Option<i64> = None;
    for row_pos in l_pos {
        let bucket = ((event_times[row_pos] - origin) / every_seconds).floor() as i64;
        if last_bucket != Some(bucket) {
            flags[row_pos] = true;
            last_bucket = Some(bucket);
        }
    }
    Ok(flags)
}

/// Select explicit indices within the sequence of limit rows.
///
/// `index_base` controls whether caller-facing indices are one-based or
/// zero-based.
pub fn select_limit_indices(
    event_dims: &[i32],
    indices: &[usize],
    index_base: usize,
) -> Result<Vec<bool>, String> {
    let l_pos = limit_positions(event_dims);
    let mut flags = vec![false; event_dims.len()];
    for &raw_idx in indices {
        let Some(idx) = raw_idx.checked_sub(index_base) else {
            return Err(format!("limit index {} is outside valid range", raw_idx));
        };
        if idx >= l_pos.len() {
            return Err(format!(
                "limit index {} is outside valid range with {} limit events",
                raw_idx,
                l_pos.len()
            ));
        }
        flags[l_pos[idx]] = true;
    }
    Ok(flags)
}

/// Select exactly `round(fraction * n_limit_rows)` limit rows at random.
pub fn select_random_limit_fraction(
    event_dims: &[i32],
    fraction: f64,
    seed: Option<u64>,
) -> Result<Vec<bool>, String> {
    if !(0.0..=1.0).contains(&fraction) {
        return Err("fraction must be in [0, 1]".to_string());
    }

    let mut l_pos = limit_positions(event_dims);
    let mut flags = vec![false; event_dims.len()];
    let n_selected = (fraction * l_pos.len() as f64).round() as usize;
    if n_selected == 0 {
        return Ok(flags);
    }

    let mut rng = create_rng(seed);
    for idx in (1..l_pos.len()).rev() {
        let swap_idx = rng.gen_range(0..=idx);
        l_pos.swap(idx, swap_idx);
    }
    for row_pos in l_pos.into_iter().take(n_selected) {
        flags[row_pos] = true;
    }
    Ok(flags)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_every_selects_first_limit_per_bucket() {
        let times = vec![0.0, 0.5, 2.1, 2.5, 4.2];
        let dims = vec![0, 0, 0, 0, 0];
        let flags = select_first_limit_every(&times, &dims, 2.0).unwrap();
        assert_eq!(flags, vec![true, false, true, false, true]);
    }

    #[test]
    fn indices_are_within_limit_sequence() {
        let dims = vec![2, 0, 1, 0, 0];
        let flags = select_limit_indices(&dims, &[2], 1).unwrap();
        assert_eq!(flags, vec![false, false, false, true, false]);
    }

    #[test]
    fn random_fraction_selects_exact_count() {
        let dims = vec![0, 1, 0, 2, 0, 0];
        let flags = select_random_limit_fraction(&dims, 0.5, Some(7)).unwrap();
        assert_eq!(flags.iter().filter(|&&x| x).count(), 2);
    }
}
