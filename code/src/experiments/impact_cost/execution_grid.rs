use super::events::LIMIT_DIM;
use super::execution::{track_passive_fills, CancellationPolicy, PassiveFillTrackerInput};

/// Input for minute-by-minute passive execution latency measurement.
///
/// For each minute start, the grid selects up to `n_orders` own passive limit
/// rows by taking the first limit row in each `order_spacing_seconds` slot,
/// then tracks fills over `tracking_horizon_seconds`.
#[derive(Clone, Debug)]
pub struct ExecutionLatencyGridInput {
    /// Sorted absolute event times in seconds.
    pub event_times: Vec<f64>,
    /// Event dimensions. Only limit, cancel, and market dimensions are used.
    pub event_dims: Vec<i32>,
    /// Event row sizes.
    pub event_qtys: Vec<u32>,
    /// Post-event queue snapshots for priority tracking.
    pub queue_post: Vec<f64>,
    /// Original source row positions preserved in output rows.
    pub source_row_positions: Vec<usize>,
    /// Absolute window start times in seconds.
    pub minute_starts: Vec<f64>,
    /// Tracking horizon for each window.
    pub tracking_horizon_seconds: f64,
    /// Maximum number of passive orders to post in each window.
    pub n_orders: usize,
    /// Width of each posting slot.
    pub order_spacing_seconds: f64,
    /// Cancellation priority convention used by the fill tracker.
    pub cancellation_policy: CancellationPolicy,
    /// Whether to cap active order positions by post-event queue snapshots.
    pub cap_position_by_queue_post: bool,
    /// Optional base seed; each window uses a deterministic offset.
    pub seed: Option<u64>,
}

/// One passive order row emitted by the latency grid.
#[derive(Clone, Debug, PartialEq)]
pub struct LatencyOrderRow {
    /// Window index in `minute_starts`.
    pub minute_index: usize,
    /// Absolute window start time in seconds.
    pub minute_start_time: f64,
    /// Order id local to this window.
    pub order_id: usize,
    /// Posting slot within the window.
    pub order_slot: usize,
    /// Event row position in the filtered arrays.
    pub event_row_pos: usize,
    /// Original source row position.
    pub source_row_pos: usize,
    /// Absolute posting time.
    pub post_time: f64,
    /// Posting time relative to the window start.
    pub post_time_relative: f64,
    /// Initial passive order quantity.
    pub initial_qty: u32,
    /// Executed quantity by the end of the tracking window.
    pub executed_qty: u32,
    /// Remaining quantity by the end of the tracking window.
    pub remaining_qty: u32,
    /// Final scalar position.
    pub final_position_qty: f64,
    /// Final volume above the passive order.
    pub final_top_qty: u32,
    /// Absolute completion time, if filled.
    pub completed_time: Option<f64>,
    /// Completion time relative to the window start, if filled.
    pub completed_time_relative: Option<f64>,
    /// Latency from post time to completion, if filled.
    pub latency_seconds: Option<f64>,
}

/// One fill row emitted by the latency grid.
#[derive(Clone, Debug, PartialEq)]
pub struct LatencyFillRow {
    /// Window index in `minute_starts`.
    pub minute_index: usize,
    /// Absolute window start time in seconds.
    pub minute_start_time: f64,
    /// Order id local to this window.
    pub order_id: usize,
    /// Posting slot within the window.
    pub order_slot: usize,
    /// Passive order event row position in the filtered arrays.
    pub order_event_row_pos: usize,
    /// Passive order original source row position.
    pub order_source_row_pos: usize,
    /// Fill event row position in the filtered arrays.
    pub fill_event_row_pos: usize,
    /// Fill event original source row position.
    pub fill_source_row_pos: usize,
    /// Absolute fill time.
    pub fill_time: f64,
    /// Fill time relative to the window start.
    pub fill_time_relative: f64,
    /// Executed fill quantity.
    pub fill_qty: u32,
    /// Latency from order post time to this fill.
    pub latency_seconds: f64,
}

/// Output of passive execution latency grid construction.
#[derive(Clone, Debug, PartialEq)]
pub struct ExecutionLatencyGridResult {
    /// One row per selected passive order.
    pub order_rows: Vec<LatencyOrderRow>,
    /// One row per market-event fill contribution.
    pub fill_rows: Vec<LatencyFillRow>,
    /// Number of passive orders selected in each window.
    pub posted_counts: Vec<usize>,
}

/// Build passive order/fill rows for all configured windows.
pub fn build_execution_latency_grid(
    input: ExecutionLatencyGridInput,
) -> Result<ExecutionLatencyGridResult, String> {
    validate_input(&input)?;

    let mut order_rows = Vec::new();
    let mut fill_rows = Vec::new();
    let mut posted_counts = Vec::with_capacity(input.minute_starts.len());

    for (minute_index, &minute_start_time) in input.minute_starts.iter().enumerate() {
        let start_idx = lower_bound(&input.event_times, minute_start_time);
        let end_time = minute_start_time + input.tracking_horizon_seconds;
        let end_idx = lower_bound(&input.event_times, end_time);
        let window_len = end_idx.saturating_sub(start_idx);

        if window_len == 0 {
            posted_counts.push(0);
            continue;
        }

        let window_times_abs = &input.event_times[start_idx..end_idx];
        let window_dims = input.event_dims[start_idx..end_idx].to_vec();
        let window_qtys = input.event_qtys[start_idx..end_idx].to_vec();
        let window_queue_post = input.queue_post[start_idx..end_idx].to_vec();
        let window_source_rows = &input.source_row_positions[start_idx..end_idx];
        let window_times_rel: Vec<f64> = window_times_abs
            .iter()
            .map(|&time| time - minute_start_time)
            .collect();
        let (passive_flags, slot_by_relative_row) = select_window_passive_orders(
            &window_times_rel,
            &window_dims,
            &window_qtys,
            input.n_orders,
            input.order_spacing_seconds,
        );
        let posted_count = passive_flags.iter().filter(|&&flag| flag).count();
        posted_counts.push(posted_count);
        if posted_count == 0 {
            continue;
        }

        let tracker_seed = input
            .seed
            .map(|seed| seed.wrapping_add(minute_index as u64));
        let tracker_result = track_passive_fills(PassiveFillTrackerInput {
            event_times: window_times_rel,
            event_dims: window_dims,
            event_qtys: window_qtys,
            queue_post: window_queue_post,
            passive_flags,
            cancellation_policy: input.cancellation_policy,
            cap_position_by_queue_post: input.cap_position_by_queue_post,
            seed: tracker_seed,
        })?;

        let mut order_slot_by_id = vec![0_usize; tracker_result.order_ids.len()];
        let mut order_rel_row_by_id = vec![0_usize; tracker_result.order_ids.len()];
        let mut order_rel_time_by_id = vec![0.0_f64; tracker_result.order_ids.len()];

        for out_idx in 0..tracker_result.order_ids.len() {
            let order_id = tracker_result.order_ids[out_idx];
            let rel_row_pos = tracker_result.order_row_pos[out_idx];
            let order_slot = slot_by_relative_row[rel_row_pos]
                .ok_or_else(|| "tracker returned an unflagged order row".to_string())?;
            let post_time_relative = tracker_result.order_times[out_idx];
            let completed_time_relative = finite_option(tracker_result.completed_times[out_idx]);
            let completed_time = completed_time_relative.map(|t| minute_start_time + t);
            let latency_seconds = completed_time_relative.map(|t| t - post_time_relative);

            if order_id >= order_slot_by_id.len() {
                return Err("tracker returned an order id outside the order table".to_string());
            }
            order_slot_by_id[order_id] = order_slot;
            order_rel_row_by_id[order_id] = rel_row_pos;
            order_rel_time_by_id[order_id] = post_time_relative;

            order_rows.push(LatencyOrderRow {
                minute_index,
                minute_start_time,
                order_id,
                order_slot,
                event_row_pos: start_idx + rel_row_pos,
                source_row_pos: window_source_rows[rel_row_pos],
                post_time: minute_start_time + post_time_relative,
                post_time_relative,
                initial_qty: tracker_result.initial_qtys[out_idx],
                executed_qty: tracker_result.executed_qtys[out_idx],
                remaining_qty: tracker_result.remaining_qtys[out_idx],
                final_position_qty: tracker_result.final_position_qtys[out_idx],
                final_top_qty: tracker_result.final_top_qtys[out_idx],
                completed_time,
                completed_time_relative,
                latency_seconds,
            });
        }

        for fill_idx in 0..tracker_result.fill_order_ids.len() {
            let order_id = tracker_result.fill_order_ids[fill_idx];
            if order_id >= order_slot_by_id.len() {
                return Err("tracker returned a fill order id outside the order table".to_string());
            }
            let order_rel_row = order_rel_row_by_id[order_id];
            let fill_rel_row = tracker_result.fill_event_row_pos[fill_idx];
            let fill_time_relative = tracker_result.fill_times[fill_idx];
            fill_rows.push(LatencyFillRow {
                minute_index,
                minute_start_time,
                order_id,
                order_slot: order_slot_by_id[order_id],
                order_event_row_pos: start_idx + order_rel_row,
                order_source_row_pos: window_source_rows[order_rel_row],
                fill_event_row_pos: start_idx + fill_rel_row,
                fill_source_row_pos: window_source_rows[fill_rel_row],
                fill_time: minute_start_time + fill_time_relative,
                fill_time_relative,
                fill_qty: tracker_result.fill_qtys[fill_idx],
                latency_seconds: fill_time_relative - order_rel_time_by_id[order_id],
            });
        }
    }

    Ok(ExecutionLatencyGridResult {
        order_rows,
        fill_rows,
        posted_counts,
    })
}

fn select_window_passive_orders(
    event_times_relative: &[f64],
    event_dims: &[i32],
    event_qtys: &[u32],
    n_orders: usize,
    order_spacing_seconds: f64,
) -> (Vec<bool>, Vec<Option<usize>>) {
    let mut flags = vec![false; event_times_relative.len()];
    let mut slot_by_row = vec![None; event_times_relative.len()];
    let mut slot_taken = vec![false; n_orders];
    let selection_horizon = n_orders as f64 * order_spacing_seconds;

    for (row_pos, &time) in event_times_relative.iter().enumerate() {
        if !(0.0..selection_horizon).contains(&time) {
            continue;
        }
        if event_dims[row_pos] != LIMIT_DIM as i32 || event_qtys[row_pos] == 0 {
            continue;
        }

        let slot = (time / order_spacing_seconds).floor() as usize;
        if slot < n_orders && !slot_taken[slot] {
            flags[row_pos] = true;
            slot_by_row[row_pos] = Some(slot);
            slot_taken[slot] = true;
        }
    }

    (flags, slot_by_row)
}

fn validate_input(input: &ExecutionLatencyGridInput) -> Result<(), String> {
    let n = input.event_times.len();
    let same_len = input.event_dims.len() == n
        && input.event_qtys.len() == n
        && input.queue_post.len() == n
        && input.source_row_positions.len() == n;
    if !same_len {
        return Err("execution latency grid event arrays must have matching lengths".to_string());
    }
    if input.tracking_horizon_seconds <= 0.0 {
        return Err("tracking_horizon_seconds must be positive".to_string());
    }
    if input.n_orders == 0 {
        return Err("n_orders must be positive".to_string());
    }
    if input.order_spacing_seconds <= 0.0 {
        return Err("order_spacing_seconds must be positive".to_string());
    }
    for &time in input.event_times.iter().chain(input.minute_starts.iter()) {
        if !time.is_finite() {
            return Err("event times and minute starts must be finite".to_string());
        }
    }
    for pair in input.event_times.windows(2) {
        if pair[0] > pair[1] {
            return Err("event_times must be sorted in nondecreasing order".to_string());
        }
    }
    Ok(())
}

fn finite_option(value: f64) -> Option<f64> {
    value.is_finite().then_some(value)
}

fn lower_bound(values: &[f64], x: f64) -> usize {
    let mut lo = 0;
    let mut hi = values.len();
    while lo < hi {
        let mid = (lo + hi) / 2;
        if values[mid] < x {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    lo
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base_grid_input() -> ExecutionLatencyGridInput {
        ExecutionLatencyGridInput {
            event_times: Vec::new(),
            event_dims: Vec::new(),
            event_qtys: Vec::new(),
            queue_post: Vec::new(),
            source_row_positions: Vec::new(),
            minute_starts: vec![0.0],
            tracking_horizon_seconds: 10.0,
            n_orders: 3,
            order_spacing_seconds: 1.0,
            cancellation_policy: CancellationPolicy::Top,
            cap_position_by_queue_post: false,
            seed: Some(7),
        }
    }

    #[test]
    fn selects_first_limit_in_each_one_second_slot() {
        let mut input = base_grid_input();
        input.event_times = vec![10.10, 10.20, 10.90, 11.00, 11.40, 12.40, 12.80, 13.00];
        input.event_dims = vec![0, 0, 1, 0, 0, 2, 0, 0];
        input.event_qtys = vec![1; input.event_times.len()];
        input.queue_post = vec![5.0; input.event_times.len()];
        input.source_row_positions = vec![100, 101, 102, 103, 104, 105, 106, 107];
        input.minute_starts = vec![10.0];

        let out = build_execution_latency_grid(input).unwrap();

        assert_eq!(out.posted_counts, vec![3]);
        assert_eq!(
            out.order_rows
                .iter()
                .map(|row| row.source_row_pos)
                .collect::<Vec<_>>(),
            vec![100, 103, 106]
        );
        assert_eq!(
            out.order_rows
                .iter()
                .map(|row| row.order_slot)
                .collect::<Vec<_>>(),
            vec![0, 1, 2]
        );
    }

    #[test]
    fn preserves_order_priority_through_tracker() {
        let mut input = base_grid_input();
        input.event_times = vec![0.10, 1.10, 2.10, 2.50, 3.00];
        input.event_dims = vec![0, 0, 0, 2, 2];
        input.event_qtys = vec![1, 1, 1, 2, 10];
        input.queue_post = vec![5.0, 1.0, 1.0, 0.0, 0.0];
        input.source_row_positions = vec![0, 1, 2, 3, 4];

        let out = build_execution_latency_grid(input).unwrap();

        assert_eq!(
            out.order_rows
                .iter()
                .map(|row| row.completed_time)
                .collect::<Vec<_>>(),
            vec![Some(3.0), Some(3.0), Some(3.0)]
        );
        assert_eq!(
            out.fill_rows
                .iter()
                .map(|row| row.order_id)
                .collect::<Vec<_>>(),
            vec![0, 1, 2]
        );
    }

    #[test]
    fn handles_windows_with_fewer_than_three_candidate_orders() {
        let mut input = base_grid_input();
        input.event_times = vec![0.20, 0.40, 60.20, 61.10];
        input.event_dims = vec![1, 0, 2, 1];
        input.event_qtys = vec![1; input.event_times.len()];
        input.queue_post = vec![5.0; input.event_times.len()];
        input.source_row_positions = vec![10, 11, 12, 13];
        input.minute_starts = vec![0.0, 60.0];

        let out = build_execution_latency_grid(input).unwrap();

        assert_eq!(out.posted_counts, vec![1, 0]);
        assert_eq!(out.order_rows.len(), 1);
        assert_eq!(out.order_rows[0].source_row_pos, 11);
        assert_eq!(out.fill_rows.len(), 0);
    }
}
