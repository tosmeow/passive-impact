use crate::simulation_helpers::{create_rng, sample_uniform};

use super::events::{valid_dim, CANCEL_DIM, LIMIT_DIM, MARKET_DIM};

/// How cancellations should affect passive order priority.
#[derive(Clone, Copy, Debug)]
pub enum CancellationPolicy {
    /// Treat cancellation quantity as removing displayed volume above us first.
    Top,
    /// Treat cancellation quantity as reducing our position directly.
    Below,
    /// Each cancellation unit removes top-side displayed volume with probability
    /// `theta`; residual quantity reduces our position.
    ProbabilisticTop { theta: f64 },
}

impl CancellationPolicy {
    /// Parse a Python/CLI policy name.
    pub fn from_name(name: &str, theta: f64) -> Result<Self, String> {
        match name.to_ascii_lowercase().as_str() {
            "top" => Ok(Self::Top),
            "below" | "position" => Ok(Self::Below),
            "probabilistic_top" | "top_probability" => {
                if !(0.0..=1.0).contains(&theta) {
                    return Err("theta must be in [0, 1]".to_string());
                }
                Ok(Self::ProbabilisticTop { theta })
            }
            _ => Err(
                "cancellation_policy must be 'top', 'below', 'position', or 'probabilistic_top'"
                    .to_string(),
            ),
        }
    }
}

/// Row-aligned input for passive fill tracking.
///
/// `passive_flags` marks own limit rows. Market rows consume queue volume and
/// can fill active passive orders; cancellation rows update priority according
/// to `cancellation_policy`.
#[derive(Clone, Debug)]
pub struct PassiveFillTrackerInput {
    /// Event row times in seconds.
    pub event_times: Vec<f64>,
    /// Event dimensions. Invalid dimensions are ignored.
    pub event_dims: Vec<i32>,
    /// Event row sizes.
    pub event_qtys: Vec<u32>,
    /// Post-event queue snapshots used to initialize/cap own-order positions.
    pub queue_post: Vec<f64>,
    /// Flags selecting own passive limit rows.
    pub passive_flags: Vec<bool>,
    /// Cancellation priority convention.
    pub cancellation_policy: CancellationPolicy,
    /// Whether to cap active order positions by each row's post-event queue.
    pub cap_position_by_queue_post: bool,
    /// Optional RNG seed for probabilistic cancellation handling.
    pub seed: Option<u64>,
}

/// Passive fill tracker output.
#[derive(Clone, Debug)]
pub struct PassiveFillTrackerResult {
    /// Dense order ids for flagged passive limit rows.
    pub order_ids: Vec<usize>,
    /// Source row position for each passive order.
    pub order_row_pos: Vec<usize>,
    /// Posting time for each passive order.
    pub order_times: Vec<f64>,
    /// Initial quantity for each passive order.
    pub initial_qtys: Vec<u32>,
    /// Total executed quantity for each passive order.
    pub executed_qtys: Vec<u32>,
    /// Remaining quantity for each passive order.
    pub remaining_qtys: Vec<u32>,
    /// Final scalar position for each passive order.
    pub final_position_qtys: Vec<f64>,
    /// Final volume above each passive order.
    pub final_top_qtys: Vec<u32>,
    /// Completion time for each passive order, or `NaN` if incomplete.
    pub completed_times: Vec<f64>,
    /// Order id for each fill row.
    pub fill_order_ids: Vec<usize>,
    /// Passive order source row for each fill row.
    pub fill_order_row_pos: Vec<usize>,
    /// Market event source row for each fill row.
    pub fill_event_row_pos: Vec<usize>,
    /// Fill time for each fill row.
    pub fill_times: Vec<f64>,
    /// Executed quantity for each fill row.
    pub fill_qtys: Vec<u32>,
}

#[derive(Clone, Debug)]
struct ActiveOrder {
    order_id: usize,
    row_pos: usize,
    time: f64,
    initial_qty: u32,
    remaining_qty: u32,
    position_qty: f64,
    top_qty: u32,
    completed_time: Option<f64>,
}

/// Track execution of flagged passive limit rows through a row event stream.
pub fn track_passive_fills(
    input: PassiveFillTrackerInput,
) -> Result<PassiveFillTrackerResult, String> {
    validate_input(&input)?;

    let mut rng = create_rng(input.seed);
    let mut orders: Vec<ActiveOrder> = Vec::new();
    let mut fill_order_ids = Vec::new();
    let mut fill_order_row_pos = Vec::new();
    let mut fill_event_row_pos = Vec::new();
    let mut fill_times = Vec::new();
    let mut fill_qtys = Vec::new();
    let mut global_top_qty = 0_u32;

    for row_idx in 0..input.event_times.len() {
        let dim = valid_dim(input.event_dims[row_idx]);
        let event_qty = input.event_qtys[row_idx];
        let time = input.event_times[row_idx];

        if event_qty > 0 {
            match dim {
                Some(LIMIT_DIM) => {
                    let is_own_limit = input.passive_flags[row_idx];
                    if !is_own_limit {
                        let has_active = orders.iter().any(|o| o.remaining_qty > 0);
                        if has_active {
                            global_top_qty = global_top_qty.saturating_add(event_qty);
                            for order in orders.iter_mut().filter(|o| o.remaining_qty > 0) {
                                order.top_qty = global_top_qty;
                            }
                        }
                    }

                    if is_own_limit {
                        let initial_position = input.queue_post[row_idx]
                            .max(minimum_new_order_position(&orders, event_qty));
                        orders.push(ActiveOrder {
                            order_id: orders.len(),
                            row_pos: row_idx,
                            time,
                            initial_qty: event_qty,
                            remaining_qty: event_qty,
                            position_qty: initial_position,
                            top_qty: global_top_qty,
                            completed_time: None,
                        });
                        enforce_order_positions(&mut orders);
                    }
                }
                Some(MARKET_DIM) => {
                    for order in orders.iter_mut().filter(|o| o.remaining_qty > 0) {
                        let fill_qty =
                            market_fill_qty(order.position_qty, order.remaining_qty, event_qty);
                        if fill_qty > 0 {
                            order.remaining_qty -= fill_qty;
                            fill_order_ids.push(order.order_id);
                            fill_order_row_pos.push(order.row_pos);
                            fill_event_row_pos.push(row_idx);
                            fill_times.push(time);
                            fill_qtys.push(fill_qty);
                            if order.remaining_qty == 0 && order.completed_time.is_none() {
                                order.completed_time = Some(time);
                            }
                        }
                        order.position_qty = if order.remaining_qty == 0 {
                            0.0
                        } else {
                            (order.position_qty - event_qty as f64).max(order.remaining_qty as f64)
                        };
                        order.top_qty = global_top_qty;
                    }
                    enforce_order_positions(&mut orders);
                }
                Some(CANCEL_DIM) => {
                    let desired_top_qty =
                        desired_top_cancel_qty(&mut rng, event_qty, input.cancellation_policy);
                    let cancel_top_qty = global_top_qty.min(desired_top_qty);
                    global_top_qty -= cancel_top_qty;
                    let cancel_position_qty = event_qty - cancel_top_qty;
                    for order in orders.iter_mut().filter(|o| o.remaining_qty > 0) {
                        order.position_qty = (order.position_qty - cancel_position_qty as f64)
                            .max(order.remaining_qty as f64);
                        order.top_qty = global_top_qty;
                    }
                    enforce_order_positions(&mut orders);
                }
                _ => {}
            }
        }

        if input.cap_position_by_queue_post {
            cap_positions_by_queue_post(&mut orders, input.queue_post[row_idx]);
        }
    }

    let mut order_ids = Vec::with_capacity(orders.len());
    let mut order_row_pos = Vec::with_capacity(orders.len());
    let mut order_times = Vec::with_capacity(orders.len());
    let mut initial_qtys = Vec::with_capacity(orders.len());
    let mut executed_qtys = Vec::with_capacity(orders.len());
    let mut remaining_qtys = Vec::with_capacity(orders.len());
    let mut final_position_qtys = Vec::with_capacity(orders.len());
    let mut final_top_qtys = Vec::with_capacity(orders.len());
    let mut completed_times = Vec::with_capacity(orders.len());

    for order in orders.iter_mut() {
        order_ids.push(order.order_id);
        order_row_pos.push(order.row_pos);
        order_times.push(order.time);
        initial_qtys.push(order.initial_qty);
        executed_qtys.push(order.initial_qty - order.remaining_qty);
        remaining_qtys.push(order.remaining_qty);
        final_position_qtys.push(order.position_qty);
        final_top_qtys.push(order.top_qty);
        completed_times.push(order.completed_time.unwrap_or(f64::NAN));
    }

    Ok(PassiveFillTrackerResult {
        order_ids,
        order_row_pos,
        order_times,
        initial_qtys,
        executed_qtys,
        remaining_qtys,
        final_position_qtys,
        final_top_qtys,
        completed_times,
        fill_order_ids,
        fill_order_row_pos,
        fill_event_row_pos,
        fill_times,
        fill_qtys,
    })
}

fn market_fill_qty(position: f64, remaining: u32, event_qty: u32) -> u32 {
    if remaining == 0 || event_qty == 0 {
        return 0;
    }
    let ahead_before = (position - remaining as f64).max(0.0);
    let consumed_until = position.min(event_qty as f64);
    (consumed_until - ahead_before)
        .max(0.0)
        .floor()
        .min(remaining as f64) as u32
}

fn desired_top_cancel_qty<R: rand::Rng>(
    rng: &mut R,
    event_qty: u32,
    policy: CancellationPolicy,
) -> u32 {
    match policy {
        CancellationPolicy::Top => event_qty,
        CancellationPolicy::Below => 0,
        CancellationPolicy::ProbabilisticTop { theta } => {
            let mut count = 0;
            for _ in 0..event_qty {
                if sample_uniform(rng) <= theta {
                    count += 1;
                }
            }
            count
        }
    }
}

fn minimum_new_order_position(orders: &[ActiveOrder], qty: u32) -> f64 {
    orders
        .iter()
        .filter(|order| order.remaining_qty > 0)
        .map(|order| order.position_qty)
        .fold(qty as f64, |acc, pos| acc.max(pos + qty as f64))
}

fn cap_positions_by_queue_post(orders: &mut [ActiveOrder], queue_post: f64) {
    let cap = queue_post.max(0.0);
    let mut min_position = 0.0;
    for order in orders.iter_mut() {
        if order.remaining_qty == 0 {
            order.position_qty = 0.0;
            continue;
        }
        min_position += order.remaining_qty as f64;
        order.position_qty = order
            .position_qty
            .min(cap)
            .max(min_position)
            .max(order.remaining_qty as f64);
        min_position = order.position_qty;
    }
    debug_assert_priority_invariants(orders);
}

fn enforce_order_positions(orders: &mut [ActiveOrder]) {
    let mut min_position = 0.0;
    for order in orders.iter_mut() {
        if order.remaining_qty == 0 {
            order.position_qty = 0.0;
            continue;
        }
        min_position += order.remaining_qty as f64;
        order.position_qty = order.position_qty.max(min_position);
        min_position = order.position_qty;
    }
    debug_assert_priority_invariants(orders);
}

fn debug_assert_priority_invariants(orders: &[ActiveOrder]) {
    let mut last_active_position = 0.0;
    for order in orders {
        if order.remaining_qty == 0 {
            debug_assert_eq!(order.position_qty, 0.0);
            continue;
        }
        let min_position = last_active_position + order.remaining_qty as f64;
        debug_assert!(
            order.position_qty + f64::EPSILON >= min_position,
            "own-order priority violation: order_id={} position={} minimum={}",
            order.order_id,
            order.position_qty,
            min_position
        );
        last_active_position = order.position_qty;
    }
}

fn validate_input(input: &PassiveFillTrackerInput) -> Result<(), String> {
    let n = input.event_times.len();
    let same_len = input.event_dims.len() == n
        && input.event_qtys.len() == n
        && input.queue_post.len() == n
        && input.passive_flags.len() == n;
    if !same_len {
        return Err("passive fill tracker arrays must have matching lengths".to_string());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tracks_market_fills_with_top_cancellation_buffer() {
        let input = PassiveFillTrackerInput {
            event_times: vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            event_dims: vec![0, 0, 2, 2, 1, 2],
            event_qtys: vec![5, 3, 7, 2, 3, 4],
            queue_post: vec![10.0, 13.0, 6.0, 4.0, 4.0, 0.0],
            passive_flags: vec![true, false, false, false, false, false],
            cancellation_policy: CancellationPolicy::Top,
            cap_position_by_queue_post: false,
            seed: Some(7),
        };

        let out = track_passive_fills(input).unwrap();
        assert_eq!(out.initial_qtys, vec![5]);
        assert_eq!(out.executed_qtys, vec![5]);
        assert_eq!(out.remaining_qtys, vec![0]);
        assert_eq!(out.fill_qtys, vec![2, 2, 1]);
        assert_eq!(out.fill_event_row_pos, vec![2, 3, 5]);
    }

    #[test]
    fn top_cancellations_fall_through_when_no_top_buffer_exists() {
        let input = PassiveFillTrackerInput {
            event_times: vec![0.0, 1.0, 2.0],
            event_dims: vec![0, 1, 2],
            event_qtys: vec![5, 5, 2],
            queue_post: vec![10.0, 5.0, 3.0],
            passive_flags: vec![true, false, false],
            cancellation_policy: CancellationPolicy::Top,
            cap_position_by_queue_post: false,
            seed: Some(7),
        };

        let out = track_passive_fills(input).unwrap();
        assert_eq!(out.fill_qtys, vec![2]);
        assert_eq!(out.remaining_qtys, vec![3]);
    }

    #[test]
    fn queue_post_cap_can_improve_position_on_unmodeled_first_queue_drop() {
        let input = PassiveFillTrackerInput {
            event_times: vec![0.0, 1.0, 2.0],
            event_dims: vec![0, -1, 2],
            event_qtys: vec![1, 0, 1],
            queue_post: vec![10.0, 1.0, 0.0],
            passive_flags: vec![true, false, false],
            cancellation_policy: CancellationPolicy::Top,
            cap_position_by_queue_post: true,
            seed: Some(7),
        };

        let out = track_passive_fills(input).unwrap();
        assert_eq!(out.fill_event_row_pos, vec![2]);
        assert_eq!(out.completed_times, vec![2.0]);
    }

    #[test]
    fn preserves_priority_between_own_orders() {
        let input = PassiveFillTrackerInput {
            event_times: vec![0.0, 1.0, 2.0, 3.0],
            event_dims: vec![0, 0, 2, 2],
            event_qtys: vec![1, 1, 2, 5],
            queue_post: vec![5.0, 2.0, 0.0, 0.0],
            passive_flags: vec![true, true, false, false],
            cancellation_policy: CancellationPolicy::Top,
            cap_position_by_queue_post: false,
            seed: Some(7),
        };

        let out = track_passive_fills(input).unwrap();
        assert_eq!(out.fill_order_ids, vec![0, 1]);
        assert_eq!(out.fill_event_row_pos, vec![3, 3]);
        assert_eq!(out.completed_times, vec![3.0, 3.0]);
    }

    #[test]
    fn below_cancellations_cannot_move_later_order_ahead_of_older_order() {
        let input = PassiveFillTrackerInput {
            event_times: vec![0.0, 1.0, 2.0, 3.0, 4.0],
            event_dims: vec![0, 0, 1, 2, 2],
            event_qtys: vec![5, 5, 10, 5, 5],
            queue_post: vec![10.0, 15.0, 5.0, 0.0, 0.0],
            passive_flags: vec![true, true, false, false, false],
            cancellation_policy: CancellationPolicy::Below,
            cap_position_by_queue_post: false,
            seed: Some(7),
        };

        let out = track_passive_fills(input).unwrap();
        assert_eq!(out.fill_order_ids, vec![0, 1]);
        assert_eq!(out.fill_event_row_pos, vec![3, 4]);
        assert_eq!(out.fill_qtys, vec![5, 5]);
        assert_eq!(out.completed_times, vec![3.0, 4.0]);
    }

    #[test]
    fn low_post_snapshot_cannot_place_new_own_order_ahead_of_active_orders() {
        let input = PassiveFillTrackerInput {
            event_times: vec![0.0, 1.0, 2.0, 3.0, 4.0],
            event_dims: vec![0, 0, 2, 2, 2],
            event_qtys: vec![5, 5, 5, 5, 5],
            queue_post: vec![10.0, 5.0, 0.0, 0.0, 0.0],
            passive_flags: vec![true, true, false, false, false],
            cancellation_policy: CancellationPolicy::Top,
            cap_position_by_queue_post: false,
            seed: Some(7),
        };

        let out = track_passive_fills(input).unwrap();
        assert_eq!(out.fill_order_ids, vec![0, 1]);
        assert_eq!(out.fill_event_row_pos, vec![3, 4]);
        assert_eq!(out.fill_qtys, vec![5, 5]);
        assert_eq!(out.completed_times, vec![3.0, 4.0]);
    }
}
