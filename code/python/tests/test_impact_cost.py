import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
CODE_PYTHON = REPO_ROOT / "code" / "python"
if str(CODE_PYTHON) not in sys.path:
    sys.path.insert(0, str(CODE_PYTHON))

from experiments.impact_cost.core.cost_utils import (
    cost_from_fills,
    event_seconds,
    expand_event_times_by_dim,
    flag_passive_limits,
    regroup_event_times_by_dim,
    track_passive_fills,
)
from experiments.impact_cost.core.experiment_utils import (
    sample_previous_value,
    select_limit_sequences,
)
from experiments.impact_cost.core.empirical_lifecycle import (
    resolve_lifecycle_to_observed_rows,
)
from experiments.impact_cost.core.level_execution import (
    first_level_execution_events_from_snapshots,
    market_side_for_queue,
    price_sign_for_queue,
    q1_column_for_side,
)
from experiments.impact_cost.core.anchored_simulator import (
    select_passive_limit_flags,
    simulate_anchored_queue_paths,
)
from experiments.impact_cost.core.latency_filters import LatencyFilterConfig, select_latency_orders
from experiments.impact_cost.core.passive_impact import (
    PassiveImpactModelConfig,
    execution_cost_jump_series,
    passive_impact_path_from_queue_samples,
)
from experiments.impact_cost.core.reduced_form_impact import (
    DEFAULT_PROPAGATOR_GAMMA,
    DEFAULT_PROPAGATOR_KAPPA,
    DEFAULT_PROPAGATOR_WEIGHTS,
    ReducedFormPropagator,
    effective_passive_kernel_coefficients,
    passive_propagator_tail_impact_from_queue_samples,
    passive_reduced_form_impact_from_queue_samples,
    propagator_impact_from_events,
)
from experiments.impact_cost.core.queue_replay import replay_consistency_report


def _toy_df():
    return pd.DataFrame(
        {
            "ts": pd.to_datetime(
                [
                    "2025-01-01 09:30:00",
                    "2025-01-01 09:30:01",
                    "2025-01-01 09:30:02",
                    "2025-01-01 09:30:03",
                    "2025-01-01 09:30:04",
                    "2025-01-01 09:30:05",
                ]
            ),
            "order_type": ["limit", "limit", "market", "market", "cancel", "market"],
            "side": ["B", "B", "B", "B", "B", "B"],
            "qty": [5, 3, 7, 2, 3, 4],
            "q_a": [10, 13, 6, 4, 4, 0],
            "q_b": [20, 20, 20, 20, 20, 20],
        }
    )


def test_flag_passive_limits_by_indices_within_l_events():
    df = _toy_df()
    flags = flag_passive_limits(df, "indices", side="B", indices=[2])
    assert np.flatnonzero(flags).tolist() == [1]


def test_flag_passive_limits_first_every_seconds():
    df = pd.DataFrame(
        {
            "ts": pd.to_datetime(
                [
                    "2025-01-01 09:30:00.0",
                    "2025-01-01 09:30:00.5",
                    "2025-01-01 09:30:02.1",
                    "2025-01-01 09:30:02.5",
                    "2025-01-01 09:30:04.2",
                ]
            ),
            "order_type": ["limit"] * 5,
            "side": ["B"] * 5,
            "qty": [1] * 5,
        }
    )
    flags = flag_passive_limits(df, "first_every", side="B", every_seconds=2.0)
    assert np.flatnonzero(flags).tolist() == [0, 2, 4]


def test_impact_series_selector_samples_limit_sequences_with_span_constraint():
    df = pd.DataFrame(
        {
            "ts": pd.to_datetime(
                [
                    "2025-01-01 09:30:00.0",
                    "2025-01-01 09:30:00.1",
                    "2025-01-01 09:30:00.2",
                    "2025-01-01 09:30:01.0",
                    "2025-01-01 09:30:01.1",
                    "2025-01-01 09:30:01.2",
                    "2025-01-01 09:30:04.0",
                ]
            ),
            "order_type": ["limit", "limit", "limit", "limit", "limit", "limit", "market"],
            "side": ["B", "B", "B", "B", "B", "B", "B"],
            "qty": [1, 1, 1, 2, 2, 2, 1],
            "q_a": [10, 11, 12, 13, 14, 15, 14],
            "q_b": [20, 20, 20, 20, 20, 20, 20],
            "source_row_pos": np.arange(7, dtype=np.int64),
        }
    )
    cfg = SimpleNamespace(
        n_orders_per_episode=3,
        n_episodes=10,
        post_span_seconds=0.25,
        horizon_seconds=1.0,
        raw_side="B",
        start_time=None,
        end_time=None,
        seed=3,
    )
    episodes, orders = select_limit_sequences(df, cfg)

    assert len(episodes) == 2
    assert set(episodes["selected_source_rows"]) == {"0,1,2", "3,4,5"}
    assert len(orders) == 6


def test_sample_previous_value_aligns_impact_path_to_output_grid():
    out = sample_previous_value(
        event_times=np.asarray([0.2, 0.5, 1.0]),
        event_values=np.asarray([1.0, -2.0, 4.0]),
        output_grid=np.asarray([0.0, 0.2, 0.49, 0.5, 0.9, 1.1]),
        initial_value=0.0,
    )

    np.testing.assert_allclose(out, [0.0, 1.0, 1.0, -2.0, -2.0, 4.0])


def test_native_passive_limit_policies_match_python_examples():
    simproj = pytest.importorskip("simproj")
    df = pd.DataFrame(
        {
            "ts": pd.to_datetime(
                [
                    "2025-01-01 09:30:00.0",
                    "2025-01-01 09:30:00.5",
                    "2025-01-01 09:30:02.1",
                    "2025-01-01 09:30:02.5",
                    "2025-01-01 09:30:04.2",
                ]
            ),
            "order_type": ["limit"] * 5,
            "side": ["B"] * 5,
            "qty": [1] * 5,
        }
    )
    times = event_seconds(df)
    dims = np.zeros(len(df), dtype=np.int32)

    native_first = simproj.select_limit_flags_first_every(times, dims, 2.0)
    assert np.flatnonzero(native_first).tolist() == [0, 2, 4]

    native_indices = simproj.select_limit_flags_indices(dims, [2, 4], 1)
    assert np.flatnonzero(native_indices).tolist() == [1, 3]

    native_random = simproj.select_limit_flags_random_fraction(dims, 0.4, 7)
    assert int(np.asarray(native_random).sum()) == 2


def test_expand_event_times_by_dim_repeats_quantities():
    df = _toy_df().iloc[:3].copy()
    expanded = expand_event_times_by_dim(
        df,
        {
            ("limit", "B"): 0,
            ("market", "B"): 2,
        },
    )
    assert np.allclose(expanded[0], [0.0] * 5 + [1.0] * 3)
    assert np.allclose(expanded[2], [2.0] * 7)


def test_regroup_event_times_by_dim_counts_repeated_unit_events():
    regrouped = regroup_event_times_by_dim(
        {
            0: np.array([0.0, 0.0, 1.0, 1.0, 1.0]),
            2: np.array([1.0, 2.0, 2.0]),
        }
    )
    assert regrouped.to_dict("records") == [
        {"time": 0.0, "dim": 0, "qty": 2},
        {"time": 1.0, "dim": 0, "qty": 3},
        {"time": 1.0, "dim": 2, "qty": 1},
        {"time": 2.0, "dim": 2, "qty": 2},
    ]


def test_regroup_event_times_by_dim_accepts_simulation_times_and_dims():
    regrouped = regroup_event_times_by_dim(
        times=np.array([3.0, 3.0, 3.0, 4.0]),
        dims=np.array([1, 1, 2, 1]),
    )
    assert regrouped.to_dict("records") == [
        {"time": 3.0, "dim": 1, "qty": 2},
        {"time": 3.0, "dim": 2, "qty": 1},
        {"time": 4.0, "dim": 1, "qty": 1},
    ]


def test_resolve_lifecycle_uses_first_observed_limit_units_and_keeps_random_fill():
    window = pd.DataFrame(
        {
            "ts": pd.to_datetime(
                [
                    "2025-01-01 09:30:00.05",
                    "2025-01-01 09:30:00.10",
                    "2025-01-01 09:30:00.20",
                ]
            ),
            "order_type": ["limit", "limit", "limit"],
            "side": ["A", "A", "A"],
            "qty": [2, 4, 8],
            "q_b": [12, 16, 24],
            "q_a": [0, 0, 0],
            "source_row_pos": [10, 11, 12],
        }
    )
    lifecycle = {
        "orders": pd.DataFrame(
            [
                {
                    "episode_id": 0,
                    "policy_path_id": 0,
                    "cycle_id": 0,
                    "order_id": 7,
                    "order_slot": 0,
                    "post_time_s": 0.08,
                    "qty": 5,
                    "filled": True,
                    "fill_time_s": 0.25,
                    "cancel_time_s": np.nan,
                    "terminal_time_s": 0.25,
                    "terminal_action": "fill",
                }
            ]
        ),
        "fills": pd.DataFrame(
            [
                {
                    "episode_id": 0,
                    "policy_path_id": 0,
                    "cycle_id": 0,
                    "order_id": 7,
                    "order_slot": 0,
                    "post_time_s": 0.08,
                    "qty": 5,
                    "fill_time_s": 0.25,
                }
            ]
        ),
        "cancels": pd.DataFrame(),
        "events": pd.DataFrame(),
        "cycle_summary": pd.DataFrame(),
    }

    resolved = resolve_lifecycle_to_observed_rows(
        window,
        lifecycle,
        raw_side="A",
        origin=pd.Timestamp("2025-01-01 09:30:00"),
        horizon_seconds=1.0,
    )

    assert resolved["own_qtys"].tolist() == [0, 4, 1]
    events = resolved["events"].sort_values(["time_s", "event_kind"]).reset_index(drop=True)
    assert events[["time_s", "event_kind", "qty", "displacement_delta"]].to_dict("records") == [
        {"time_s": 0.1, "event_kind": "post", "qty": 4, "displacement_delta": 4},
        {"time_s": 0.2, "event_kind": "post", "qty": 1, "displacement_delta": 1},
        {"time_s": 0.25, "event_kind": "fill", "qty": 5, "displacement_delta": -5},
    ]
    assert resolved["fills"]["fill_time_s"].tolist() == [0.25]


def test_resolve_lifecycle_snaps_cancel_intentions_to_observed_cancel_units():
    window = pd.DataFrame(
        {
            "ts": pd.to_datetime(
                [
                    "2025-01-01 09:30:00.10",
                    "2025-01-01 09:30:00.12",
                    "2025-01-01 09:30:00.30",
                ]
            ),
            "order_type": ["limit", "cancel", "cancel"],
            "side": ["A", "A", "A"],
            "qty": [5, 2, 5],
            "q_b": [15, 13, 8],
            "q_a": [0, 0, 0],
            "source_row_pos": [20, 21, 22],
        }
    )
    lifecycle = {
        "orders": pd.DataFrame(
            [
                {
                    "episode_id": 0,
                    "policy_path_id": 0,
                    "cycle_id": 0,
                    "order_id": 3,
                    "order_slot": 0,
                    "post_time_s": 0.05,
                    "qty": 5,
                    "filled": False,
                    "fill_time_s": np.nan,
                    "cancel_time_s": 0.20,
                    "terminal_time_s": 0.20,
                    "terminal_action": "cancel",
                }
            ]
        ),
        "fills": pd.DataFrame(),
        "cancels": pd.DataFrame(
            [
                {
                    "episode_id": 0,
                    "policy_path_id": 0,
                    "cycle_id": 0,
                    "order_id": 3,
                    "order_slot": 0,
                    "post_time_s": 0.05,
                    "qty": 5,
                    "cancel_time_s": 0.20,
                }
            ]
        ),
        "events": pd.DataFrame(),
        "cycle_summary": pd.DataFrame(),
    }

    resolved = resolve_lifecycle_to_observed_rows(
        window,
        lifecycle,
        raw_side="A",
        origin=pd.Timestamp("2025-01-01 09:30:00"),
        horizon_seconds=1.0,
    )

    assert resolved["own_qtys"].tolist() == [5, 0, 5]
    events = resolved["events"].sort_values(["time_s", "event_kind"]).reset_index(drop=True)
    assert events[["time_s", "event_kind", "qty", "displacement_delta"]].to_dict("records") == [
        {"time_s": 0.1, "event_kind": "post", "qty": 5, "displacement_delta": 5},
        {"time_s": 0.3, "event_kind": "cancel", "qty": 5, "displacement_delta": -5},
    ]


def test_replay_consistency_reconstruction_saturates_at_zero():
    df = pd.DataFrame(
        {
            "ts": pd.to_datetime(
                ["2025-01-01 09:30:00", "2025-01-01 09:30:01"]
            ),
            "order_type": ["market", "limit"],
            "side": ["B", "B"],
            "qty": [3, 2],
            "q_a": [0, 2],
        }
    )
    report, summary = replay_consistency_report(
        df, raw_side="B", queue_col="q_a", market_side="B", initial_q=1
    )

    assert report["expected_delta"].tolist() == [-1.0, 2.0]
    assert report["reconstructed_queue"].tolist() == [0.0, 2.0]
    assert report["level_diff"].tolist() == [0.0, 0.0]
    assert summary["max_abs_level_diff"] == 0.0


def test_anchored_no_passive_matches_raw_snapshots_when_replay_diverges():
    pytest.importorskip("simproj")
    df = pd.DataFrame(
        {
            "ts": pd.to_datetime(
                ["2025-01-01 09:30:00", "2025-01-01 09:30:01"]
            ),
            "order_type": ["limit", "cancel"],
            "side": ["B", "B"],
            "qty": [1, 1],
            # Deliberately not reproducible by +L-C replay from initial_q=10.
            "q_a": [20, 25],
            "q_b": [0, 0],
        }
    )
    flags = np.zeros(len(df), dtype=bool)
    grid = np.array([0.0, 0.5, 1.0])
    _, replay_summary = replay_consistency_report(
        df, raw_side="B", queue_col="q_a", market_side="B", initial_q=10
    )

    result = simulate_anchored_queue_paths(
        df,
        flags,
        raw_side="B",
        queue_col="q_a",
        market_side="B",
        initial_q=10,
        horizon_seconds=1.0,
        grid=grid,
        n_simulations=3,
        seed=123,
        a_l=10.0,
        b_l=0.0,
        a_c=10.0,
        b_c=0.0,
    )

    assert replay_summary["max_abs_level_diff"] > 0
    assert result.factual_queue.tolist() == [20.0, 20.0, 25.0]
    assert np.all(result.simulated_queues == result.factual_queue[:, None])
    assert np.all(result.simulated_offsets == 0.0)


def test_anchored_no_us_removes_passive_limit_but_keeps_market_common():
    pytest.importorskip("simproj")
    df = pd.DataFrame(
        {
            "ts": pd.to_datetime(
                ["2025-01-01 09:30:00", "2025-01-01 09:30:01"]
            ),
            "order_type": ["limit", "market"],
            "side": ["B", "B"],
            "qty": [1, 1],
            "q_a": [11, 10],
            "q_b": [0, 0],
        }
    )
    flags = np.array([True, False])

    result = simulate_anchored_queue_paths(
        df,
        flags,
        raw_side="B",
        queue_col="q_a",
        market_side="B",
        initial_q=10,
        horizon_seconds=1.0,
        grid=np.array([0.0, 1.0]),
        n_simulations=2,
        seed=123,
        a_l=10.0,
        b_l=0.0,
        a_c=10.0,
        b_c=0.0,
    )

    assert result.factual_queue.tolist() == [11.0, 10.0]
    assert result.mechanical_no_us_queue.tolist() == [10.0, 9.0]
    assert np.all(result.simulated_queues == result.mechanical_no_us_queue[:, None])


def test_anchored_sized_events_are_handled_without_expanded_output():
    pytest.importorskip("simproj")
    df = pd.DataFrame(
        {
            "ts": pd.to_datetime(["2025-01-01 09:30:00"]),
            "order_type": ["limit"],
            "side": ["B"],
            "qty": [3],
            "q_a": [13],
            "q_b": [0],
        }
    )

    result = simulate_anchored_queue_paths(
        df,
        np.array([False]),
        raw_side="B",
        queue_col="q_a",
        market_side="B",
        initial_q=10,
        horizon_seconds=0.0,
        grid=np.array([0.0]),
        n_simulations=1,
        seed=123,
        a_l=10.0,
        b_l=0.0,
        a_c=10.0,
        b_c=0.0,
    )

    assert result.factual_queue.tolist() == [13.0]
    assert result.simulated_queues[:, 0].tolist() == [13.0]
    assert result.simulated_events.to_dict("records") == [
        {"simulation": 0, "time": 0.0, "dim": 0, "qty": 3}
    ]


def test_native_passive_fill_tracker_matches_python_top_policy():
    simproj = pytest.importorskip("simproj")
    df = _toy_df()
    flags = np.array([True, False, False, False, False, False])
    dims = np.array([0, 0, 2, 2, 1, 2], dtype=np.int32)

    native = simproj.track_passive_fills(
        event_seconds(df).astype(np.float64),
        dims,
        df["qty"].to_numpy(dtype=np.uint32),
        df["q_a"].to_numpy(dtype=np.float64),
        flags,
        "top",
        1.0,
        7,
    )

    assert np.asarray(native["initial_qtys"]).tolist() == [5]
    assert np.asarray(native["executed_qtys"]).tolist() == [5]
    assert np.asarray(native["remaining_qtys"]).tolist() == [0]
    assert np.asarray(native["fill_qtys"]).tolist() == [2, 2, 1]
    assert np.asarray(native["fill_event_row_pos"]).tolist() == [2, 3, 5]


def test_unit_size_expansion_matches_direct_conditional_simulation():
    simproj = pytest.importorskip("simproj")
    df = pd.DataFrame(
        {
            "ts": pd.to_datetime(
                [
                    "2025-01-01 09:30:00.10",
                    "2025-01-01 09:30:00.25",
                    "2025-01-01 09:30:00.50",
                    "2025-01-01 09:30:00.75",
                    "2025-01-01 09:30:01.00",
                    "2025-01-01 09:30:01.25",
                ]
            ),
            "order_type": ["limit", "cancel", "limit", "limit", "cancel", "limit"],
            "side": ["B"] * 6,
            "qty": [1] * 6,
            "q_a": [11, 10, 11, 12, 11, 12],
            "q_b": [0] * 6,
        }
    )
    t_max = 2.0
    expanded = expand_event_times_by_dim(
        df,
        {
            ("limit", "B"): 0,
            ("cancel", "B"): 1,
        },
    )
    cond_from_expanded = [
        expanded.get(0, np.array([], dtype=np.float64)).tolist(),
        expanded.get(1, np.array([], dtype=np.float64)).tolist(),
        [],
    ]

    seconds = event_seconds(df)
    direct_limit_times = seconds[df["order_type"].to_numpy() == "limit"].tolist()
    direct_cancel_times = seconds[df["order_type"].to_numpy() == "cancel"].tolist()
    cond_direct = [direct_limit_times, direct_cancel_times, []]
    assert cond_from_expanded == cond_direct

    process = simproj.AffineQueueProcess.new_queue(
        q0=10.0, a_l=4.0, b_l=-0.05, a_c=1.0, b_c=0.05
    )
    ctx_expanded = simproj.ConditionalSimulationContext(
        process, cond_from_expanded, t_max
    )
    ctx_direct = simproj.ConditionalSimulationContext(process, cond_direct, t_max)

    seed = 2027
    sim_expanded = ctx_expanded.simulate(seed=seed)
    sim_direct = ctx_direct.simulate(seed=seed)

    assert np.allclose(sim_expanded.times(), sim_direct.times())
    assert np.array_equal(sim_expanded.dims(), sim_direct.dims())

    grouped_expanded = regroup_event_times_by_dim(
        times=sim_expanded.times(), dims=sim_expanded.dims()
    )
    grouped_direct = regroup_event_times_by_dim(
        times=sim_direct.times(), dims=sim_direct.dims()
    )
    pd.testing.assert_frame_equal(grouped_expanded, grouped_direct)


def test_anchored_matches_legacy_conditional_on_clean_library_path():
    simproj = pytest.importorskip("simproj")
    initial_q = 20
    t_max = 1.0
    process = simproj.AffineQueueProcess.new_queue(
        q0=float(initial_q),
        a_l=30.0,
        b_l=0.0,
        a_c=20.0,
        b_c=0.0,
    )
    market_times = np.array([0.12, 0.47, 0.81], dtype=np.float64)
    market_events = simproj.create_meta_orders_from_times(
        market_times, target_dim=2, total_dims=3
    )
    internal_events = simproj.simulate_with_externals(
        process, t_max, market_events, seed=123
    )
    full_events = simproj.merge_events(internal_events, market_events)

    event_times = np.asarray(full_events.times(), dtype=np.float64)
    event_dims = np.asarray(full_events.dims(), dtype=np.int64)
    assert event_times.size > 0
    assert np.any(event_dims == 0)

    q_snapshots = simproj.sample_queue_at_times(
        full_events, initial_q=initial_q, times=event_times
    )
    type_by_dim = {0: "limit", 1: "cancel", 2: "market"}
    clean_df = pd.DataFrame(
        {
            "ts": event_times,
            "order_type": [type_by_dim[int(dim)] for dim in event_dims],
            "side": ["B"] * len(event_times),
            "qty": [1] * len(event_times),
            "q_a": np.asarray(q_snapshots, dtype=np.int64),
            "q_b": [0] * len(event_times),
        }
    )
    passive_flags = np.zeros(len(clean_df), dtype=bool)
    passive_flags[np.flatnonzero(event_dims == 0)[0]] = True

    background = expand_event_times_by_dim(
        clean_df[~passive_flags],
        {
            ("limit", "B"): 0,
            ("cancel", "B"): 1,
        },
        origin=0.0,
    )
    own_limits = expand_event_times_by_dim(
        clean_df[passive_flags],
        {("limit", "B"): 0},
        origin=0.0,
    )
    cond_by_dim = [
        background.get(0, np.array([], dtype=np.float64)).tolist(),
        background.get(1, np.array([], dtype=np.float64)).tolist(),
        [],
    ]
    own_limit_events = simproj.create_meta_orders_from_times(
        own_limits.get(0, np.array([], dtype=np.float64)),
        target_dim=0,
        total_dims=3,
    )
    cond_externals = simproj.merge_events(own_limit_events, market_events)
    legacy_ctx = simproj.ConditionalSimulationContext(
        process,
        cond_by_dim,
        t_max,
        cond_externals=cond_externals,
        new_externals=market_events,
    )

    grid = np.linspace(0.0, t_max, 50)
    anchored = simulate_anchored_queue_paths(
        clean_df,
        passive_flags,
        raw_side="B",
        queue_col="q_a",
        market_side="B",
        initial_q=initial_q,
        horizon_seconds=t_max,
        grid=grid,
        n_simulations=3,
        seed=900,
        a_l=30.0,
        b_l=0.0,
        a_c=20.0,
        b_c=0.0,
        origin=0.0,
    )

    np.testing.assert_array_equal(
        anchored.factual_queue,
        simproj.sample_queue_at_times(full_events, initial_q=initial_q, times=grid),
    )
    for sim_idx in range(3):
        legacy_result = legacy_ctx.simulate(seed=900 + sim_idx)
        legacy_samples = simproj.sample_queue_at_times(
            legacy_result, initial_q=initial_q, times=grid
        )
        np.testing.assert_array_equal(
            anchored.simulated_queues[:, sim_idx],
            np.asarray(legacy_samples, dtype=np.float64),
        )


def test_track_passive_fills_with_top_cancellations():
    df = _toy_df()
    flags = np.array([True, False, False, False, False, False])
    result = track_passive_fills(
        df,
        flags,
        side="B",
        queue_col="q_a",
        cancellation_policy="top",
        include_ledger=True,
    )

    assert result.orders.loc[0, "initial_qty"] == 5
    assert result.orders.loc[0, "executed_qty"] == 5
    assert result.orders.loc[0, "remaining_qty"] == 0
    assert result.fills["qty"].tolist() == [2, 2, 1]
    assert result.fills["event_row_pos"].tolist() == [2, 3, 5]
    assert result.ledger is not None
    cancel_row = result.ledger[result.ledger["event_row_pos"] == 4].iloc[0]
    assert cancel_row["cancel_top_qty"] == 3
    assert cancel_row["cancel_position_qty"] == 0


def test_cancellation_policy_below_decreases_position_without_fill():
    df = pd.DataFrame(
        {
            "ts": pd.to_datetime(
                [
                    "2025-01-01 09:30:00",
                    "2025-01-01 09:30:01",
                    "2025-01-01 09:30:02",
                ]
            ),
            "order_type": ["limit", "cancel", "market"],
            "side": ["B", "B", "B"],
            "qty": [5, 5, 2],
            "q_a": [10, 5, 3],
            "q_b": [0, 0, 0],
        }
    )
    flags = np.array([True, False, False])

    top = track_passive_fills(
        df, flags, side="B", queue_col="q_a", cancellation_policy="top"
    )
    below = track_passive_fills(
        df, flags, side="B", queue_col="q_a", cancellation_policy="below"
    )

    # There is no later L above our order, so "top" has no buffer and falls
    # through to position-decreasing behavior.
    assert top.fills["qty"].tolist() == [2]
    assert below.fills["qty"].tolist() == [2]


def test_top_cancellations_are_capped_by_later_limit_buffer():
    df = pd.DataFrame(
        {
            "ts": pd.to_datetime(
                [
                    "2025-01-01 09:30:00",
                    "2025-01-01 09:30:01",
                    "2025-01-01 09:30:02",
                    "2025-01-01 09:30:03",
                    "2025-01-01 09:30:04",
                    "2025-01-01 09:30:05",
                    "2025-01-01 09:30:06",
                    "2025-01-01 09:30:07",
                ]
            ),
            "order_type": [
                "limit",
                "cancel",
                "limit",
                "limit",
                "market",
                "cancel",
                "cancel",
                "cancel",
            ],
            "side": ["B"] * 8,
            "qty": [1] * 8,
            # q_a was 10 before the first L; these are post-event snapshots.
            "q_a": [11, 10, 11, 12, 11, 10, 9, 8],
            "q_b": [0] * 8,
        }
    )
    flags = np.array([True, False, False, False, False, False, False, False])

    top = track_passive_fills(
        df,
        flags,
        side="B",
        queue_col="q_a",
        cancellation_policy="top",
        include_ledger=True,
    )
    below = track_passive_fills(
        df,
        flags,
        side="B",
        queue_col="q_a",
        cancellation_policy="below",
        include_ledger=True,
    )

    assert top.fills.empty
    assert top.ledger["position_after"].tolist() == [11.0, 10.0, 10.0, 10.0, 9.0, 9.0, 9.0, 8.0]
    assert top.ledger["top_after"].tolist() == [0, 0, 1, 2, 2, 1, 0, 0]
    assert top.ledger["cancel_top_qty"].tolist() == [0, 0, 0, 0, 0, 1, 1, 0]
    assert top.ledger["cancel_position_qty"].tolist() == [0, 1, 0, 0, 0, 0, 0, 1]

    assert below.fills.empty
    assert below.ledger["position_after"].tolist() == [11.0, 10.0, 10.0, 10.0, 9.0, 8.0, 7.0, 6.0]


def test_track_passive_fills_uses_only_target_queue_level():
    df = pd.DataFrame(
        {
            "ts": pd.to_datetime(
                [
                    "2025-01-01 09:30:00",
                    "2025-01-01 09:30:01",
                    "2025-01-01 09:30:02",
                    "2025-01-01 09:30:03",
                    "2025-01-01 09:30:04",
                ]
            ),
            "order_type": ["limit", "market", "limit", "cancel", "market"],
            "side": ["B"] * 5,
            "level": [1, 2, 2, 1, 1],
            "qty": [2, 1, 10, 10, 2],
            "q1": [5, 5, 5, 2, 0],
        }
    )
    flags = np.array([True, False, False, False, False])

    level1 = track_passive_fills(
        df,
        flags,
        side="B",
        queue_col="q1",
        level_col="level",
        target_level=1,
        cancellation_policy="top",
        include_ledger=True,
    )
    unfiltered = track_passive_fills(
        df,
        flags,
        side="B",
        queue_col="q1",
        cancellation_policy="top",
    )

    assert level1.fills["event_row_pos"].tolist() == [4]
    assert level1.fills["qty"].tolist() == [2]
    assert level1.ledger["event_row_pos"].tolist() == [0, 3, 4]
    assert level1.ledger["top_after"].tolist() == [0, 0, 0]

    # Without the first-level filter, the level-2 limit incorrectly creates
    # top buffer, so the level-1 cancel no longer advances our position.
    assert unfiltered.fills.empty


def test_passive_fill_tracker_preserves_own_order_priority():
    df = pd.DataFrame(
        {
            "ts": pd.to_datetime(
                [
                    "2025-01-01 09:30:00",
                    "2025-01-01 09:30:01",
                    "2025-01-01 09:30:02",
                    "2025-01-01 09:30:03",
                ]
            ),
            "order_type": ["limit", "limit", "market", "market"],
            "side": ["B"] * 4,
            "qty": [1, 1, 2, 5],
            # The second post-event snapshot is deliberately smaller than the
            # first order's tracked position. The tracker must not let order 1
            # jump ahead of order 0.
            "q1": [5, 2, 0, 0],
        }
    )
    flags = np.array([True, True, False, False])

    result = track_passive_fills(df, flags, side="B", queue_col="q1")

    assert result.fills["order_id"].tolist() == [0, 1]
    assert result.fills["event_row_pos"].tolist() == [3, 3]
    assert result.orders["completed_time"].tolist() == [3.0, 3.0]


def test_passive_fill_level_filter_rejects_non_target_flags():
    df = pd.DataFrame(
        {
            "ts": pd.to_datetime(["2025-01-01 09:30:00"]),
            "order_type": ["limit"],
            "side": ["B"],
            "level": [2],
            "qty": [1],
            "q1": [1],
        }
    )

    with pytest.raises(ValueError, match="execution target level"):
        track_passive_fills(
            df,
            np.array([True]),
            side="B",
            queue_col="q1",
            level_col="level",
            target_level=1,
        )


def test_flag_passive_limits_can_select_within_target_level_only():
    df = pd.DataFrame(
        {
            "ts": pd.to_datetime(
                [
                    "2025-01-01 09:30:00",
                    "2025-01-01 09:30:01",
                    "2025-01-01 09:30:02",
                ]
            ),
            "order_type": ["limit", "limit", "limit"],
            "side": ["B"] * 3,
            "level": [2, 1, 1],
            "qty": [1, 1, 1],
        }
    )

    flags = flag_passive_limits(
        df,
        "indices",
        side="B",
        level_col="level",
        target_level=1,
        indices=[1],
    )

    assert np.flatnonzero(flags).tolist() == [1]


def test_native_passive_limit_selector_can_select_target_level_only():
    pytest.importorskip("simproj")
    df = pd.DataFrame(
        {
            "ts": pd.to_datetime(
                [
                    "2025-01-01 09:30:00",
                    "2025-01-01 09:30:01",
                    "2025-01-01 09:30:02",
                ]
            ),
            "order_type": ["limit", "limit", "limit"],
            "side": ["B"] * 3,
            "level": [2, 1, 1],
            "qty": [1, 1, 1],
        }
    )

    flags = select_passive_limit_flags(
        df,
        "indices",
        raw_side="B",
        level_col="level",
        target_level=1,
        indices=[1],
    )

    assert np.flatnonzero(flags).tolist() == [1]


def test_native_passive_fill_tracker_can_ignore_non_target_level_via_dims():
    simproj = pytest.importorskip("simproj")
    event_times = np.arange(5, dtype=np.float64)
    # Non-first-level rows are encoded as -1 before calling the native tracker.
    dims = np.array([0, -1, -1, 1, 2], dtype=np.int32)
    qty = np.array([2, 1, 10, 10, 2], dtype=np.uint32)
    queue_post = np.array([5, 5, 5, 2, 0], dtype=np.float64)
    flags = np.array([True, False, False, False, False])

    native = simproj.track_passive_fills(
        event_times,
        dims,
        qty,
        queue_post,
        flags,
        "top",
        1.0,
        7,
    )

    assert np.asarray(native["fill_event_row_pos"]).tolist() == [4]
    assert np.asarray(native["fill_qtys"]).tolist() == [2]


def test_native_execution_latency_grid_tracks_minute_windows():
    simproj = pytest.importorskip("simproj")
    event_times = np.array([0.1, 1.1, 2.1, 3.0, 4.0, 60.1, 61.1, 62.1, 63.0], dtype=np.float64)
    dims = np.array([0, 0, 0, 2, 2, 0, 0, 0, 2], dtype=np.int32)
    qty = np.array([1, 1, 1, 3, 1, 1, 1, 1, 3], dtype=np.uint32)
    queue_post = np.array([1, 2, 3, 0, 0, 1, 2, 3, 0], dtype=np.float64)
    source_row_pos = np.arange(100, 109, dtype=np.int64)
    minute_starts = np.array([0.0, 60.0], dtype=np.float64)

    out = simproj.simulate_execution_latency_grid(
        event_times,
        dims,
        qty,
        queue_post,
        source_row_pos,
        minute_starts,
        3,
        1.0,
        10.0,
        "top",
        1.0,
        7,
    )

    assert np.asarray(out["orders_posted_by_window"]).tolist() == [3, 3]
    assert np.asarray(out["orders_filled_by_window"]).tolist() == [3, 3]
    assert np.asarray(out["order_window_ids"]).tolist() == [0, 0, 0, 1, 1, 1]
    assert np.asarray(out["order_slots"]).tolist() == [0, 1, 2, 0, 1, 2]
    assert np.asarray(out["order_source_row_pos"]).tolist() == [100, 101, 102, 105, 106, 107]
    np.testing.assert_allclose(
        np.asarray(out["completed_times"]),
        np.array([3.0, 3.0, 3.0, 63.0, 63.0, 63.0]),
    )
    assert np.asarray(out["fill_order_ids"]).tolist() == [0, 1, 2, 3, 4, 5]


def test_first_level_execution_events_are_inferred_from_q1_snapshots():
    df = pd.DataFrame(
        {
            "ts": pd.to_datetime(
                [
                    "2025-01-01 09:30:00",
                    "2025-01-01 09:30:01",
                    "2025-01-01 09:30:02",
                    "2025-01-01 09:30:03",
                    "2025-01-01 09:30:04",
                    "2025-01-01 09:30:05",
                ]
            ),
            "order_type": ["limit", "limit", "cancel", "market", "limit", "cancel"],
            "side": ["B", "B", "B", "B", "A", "B"],
            "qty": [2, 3, 5, 1, 4, 1],
            "a_1": [12, 12, 9, 8, 8, 10],
        }
    )

    out = first_level_execution_events_from_snapshots(
        df,
        raw_side="B",
        q1_col="a_1",
        market_side="B",
        previous_q1=10,
    )

    assert q1_column_for_side(raw_side="B", queue_col="q_a") == "a_1"
    assert q1_column_for_side(raw_side="A", queue_col="q_b") == "b_1"
    assert market_side_for_queue(raw_side="A", queue_col="q_b") == "B"
    assert market_side_for_queue(raw_side="B", queue_col="q_a") == "A"
    assert price_sign_for_queue("q_b") == -1.0
    assert price_sign_for_queue("q_a") == 1.0
    assert out["level"].tolist() == [1, 0, 1, 1, 0, 0]
    assert out["qty"].tolist() == [2, 0, 3, 1, 0, 0]
    assert out["source_qty"].tolist() == [2, 3, 5, 1, 4, 1]
    assert out["q1_delta"].tolist() == [2.0, 0.0, -3.0, -1.0, 0.0, 2.0]


def test_cost_from_fills_uses_left_limit_impact():
    fills = pd.DataFrame({"time": [2.5, 3.0], "qty": [1, 4]})
    impact_times = np.array([0.0, 1.0, 2.0, 3.0])
    impact_values = np.array([0.0, 10.0, 20.0, 30.0])
    assert cost_from_fills(fills, impact_times, impact_values) == 100.0


def test_reduced_form_passive_impact_uses_image_gamma_and_queue_sign():
    q_without_us = np.array([10, 10, 10], dtype=np.uint32)
    q_with_us = np.array([11, 12, 13], dtype=np.uint32)

    bid_impact = passive_reduced_form_impact_from_queue_samples(
        q_without_us,
        q_with_us,
        queue_col="q_b",
    )
    ask_impact = passive_reduced_form_impact_from_queue_samples(
        q_without_us,
        q_with_us,
        queue_col="q_a",
    )

    np.testing.assert_allclose(
        bid_impact,
        -DEFAULT_PROPAGATOR_GAMMA * np.array([1.0, 3.0, 6.0]),
    )
    np.testing.assert_allclose(
        ask_impact,
        DEFAULT_PROPAGATOR_GAMMA * np.array([1.0, 3.0, 6.0]),
    )


def test_passive_impact_path_core_uses_reduced_form_model_config():
    impact = passive_impact_path_from_queue_samples(
        np.array([10, 10], dtype=np.uint32),
        np.array([12, 13], dtype=np.uint32),
        np.array([0.5, 1.0], dtype=np.float64),
        queue_col="q_b",
        cfg=PassiveImpactModelConfig(impact_model="reduced_form"),
    )

    np.testing.assert_allclose(
        impact,
        -DEFAULT_PROPAGATOR_GAMMA * np.array([2.0, 5.0]),
    )


def test_effective_passive_kernel_coefficients_scale_propagator_weights():
    coefficients = ReducedFormPropagator(
        kappa=2.0,
        gamma=-0.5,
        weights=(0.2, -0.4),
        beta=(1.0, 3.0),
    )

    eta = effective_passive_kernel_coefficients(
        coefficients=coefficients,
        c_lambda=1.0,
    )

    np.testing.assert_allclose(eta, [0.05, -0.15])


def test_propagator_tail_passive_impact_updates_event_states():
    coefficients = ReducedFormPropagator(
        kappa=1.0,
        gamma=2.0,
        weights=(2.0,),
        beta=(3.0,),
    )
    market_times = np.array([0.0, 1.0], dtype=np.float64)

    impact = passive_propagator_tail_impact_from_queue_samples(
        np.array([10, 10], dtype=np.uint32),
        np.array([11, 12], dtype=np.uint32),
        market_times,
        queue_col="q_a",
        coefficients=coefficients,
        c_lambda=1.0,
        queue_sensitivity=2.0,
        zeta=0.5,
    )

    eta = 3.0 * 2.0 / (1.0 * (3.0 + 1.0))
    first_forecast = 0.5 + eta
    second_forecast = 0.5 + eta * np.exp(-3.0) + eta
    expected = np.array(
        [
            2.0 * (1.0 + 1.0 * first_forecast),
            2.0 * (3.0 + 2.0 * second_forecast),
        ]
    )
    np.testing.assert_allclose(impact, expected)


def test_passive_impact_path_core_uses_propagator_tail_model_config():
    impact = passive_impact_path_from_queue_samples(
        np.array([10, 10], dtype=np.uint32),
        np.array([11, 12], dtype=np.uint32),
        np.array([0.0, 1.0], dtype=np.float64),
        queue_col="q_a",
        cfg=PassiveImpactModelConfig(
            impact_model="propagator_tail",
            propagator_kappa=1.0,
            propagator_gamma=2.0,
            propagator_weights=(2.0,),
            propagator_beta=(3.0,),
            propagator_tail_zeta=0.5,
            b_l=0.0,
            b_c=1.0,
        ),
    )

    eta = 3.0 * 2.0 / (1.0 * (3.0 + 1.0))
    np.testing.assert_allclose(
        impact,
        [
            2.0 * (1.0 + 1.0 * (0.5 + eta)),
            2.0 * (3.0 + 2.0 * (0.5 + eta * np.exp(-3.0) + eta)),
        ],
    )


def test_tail_propagator_ignores_structural_only_fields():
    q = np.array([10, 10], dtype=np.uint32)
    q_bar = np.array([11, 12], dtype=np.uint32)
    market_times = np.array([0.0, 1.0], dtype=np.float64)
    common = {
        "impact_model": "tail_propagator",
        "propagator_kappa": 1.0,
        "propagator_gamma": 2.0,
        "propagator_weights": (2.0,),
        "propagator_beta": (3.0,),
        "propagator_tail_zeta": 0.5,
        "b_l": 0.0,
        "b_c": 1.0,
    }

    base = passive_impact_path_from_queue_samples(
        q,
        q_bar,
        market_times,
        queue_col="q_a",
        cfg=PassiveImpactModelConfig(**common),
    )
    noisy_structural_fields = passive_impact_path_from_queue_samples(
        q,
        q_bar,
        market_times,
        queue_col="q_a",
        cfg=PassiveImpactModelConfig(
            **common,
            c_kappa=123.0,
            hawkes_mu=999.0,
            hawkes_alpha=(1.0,),
            hawkes_beta=(1.0, 2.0),
        ),
    )

    np.testing.assert_allclose(noisy_structural_fields, base)


def test_structural_model_validates_structural_hawkes_fields():
    with pytest.raises(ValueError, match="hawkes_alpha"):
        passive_impact_path_from_queue_samples(
            np.array([10], dtype=np.uint32),
            np.array([11], dtype=np.uint32),
            np.array([0.0], dtype=np.float64),
            queue_col="q_a",
            cfg=PassiveImpactModelConfig(
                impact_model="structural",
                hawkes_alpha=(1.0,),
                hawkes_beta=(1.0, 2.0),
            ),
        )


def test_native_tail_propagator_matches_python_helper():
    simproj = pytest.importorskip("simproj")
    q = np.array([10, 10], dtype=np.uint32)
    q_bar = np.array([11, 12], dtype=np.uint32)
    market_times = np.array([0.0, 1.0], dtype=np.float64)
    coefficients = ReducedFormPropagator(
        kappa=1.0,
        gamma=2.0,
        weights=(2.0,),
        beta=(3.0,),
    )

    native = simproj.passive_tail_propagator_impact_from_queue_samples(
        q,
        q_bar,
        market_times,
        1.0,
        [2.0],
        [3.0],
        0.0,
        1.0,
        2.0,
        0.5,
    )
    python = passive_propagator_tail_impact_from_queue_samples(
        q,
        q_bar,
        market_times,
        queue_col="q_a",
        coefficients=coefficients,
        c_lambda=1.0,
        queue_sensitivity=2.0,
        zeta=0.5,
    )

    np.testing.assert_allclose(np.asarray(native), python)


def test_execution_cost_jump_series_preserves_fill_labels_and_cumulative_cost():
    fills = pd.DataFrame(
        {
            "order_id": [10, 11],
            "order_slot": [0, 1],
            "fill_time_s": [2.5, 3.0],
            "qty": [1, 4],
        }
    )
    jumps = execution_cost_jump_series(
        fills,
        market_times=np.array([0.0, 1.0, 2.0, 3.0]),
        impact=np.array([0.0, 10.0, 20.0, 30.0]),
        copy_columns=("order_id", "order_slot"),
    )

    assert jumps["order_id"].tolist() == [10, 11]
    assert jumps["order_slot"].tolist() == [0, 1]
    assert jumps["impact_left"].tolist() == [20.0, 20.0]
    assert jumps["cost_jump"].tolist() == [20.0, 80.0]
    assert jumps["contribution"].tolist() == [20.0, 80.0]
    assert jumps["cumulative_cost"].tolist() == [20.0, 100.0]


def test_reduced_form_direct_propagator_uses_image_coefficients():
    coefficients = ReducedFormPropagator()
    impact = propagator_impact_from_events(
        np.array([0.0, 1.0]),
        np.array([0.0]),
        coefficients=coefficients,
    )

    expected_at_zero = DEFAULT_PROPAGATOR_KAPPA + sum(DEFAULT_PROPAGATOR_WEIGHTS)
    expected_at_one = DEFAULT_PROPAGATOR_KAPPA + sum(
        w * np.exp(-b) for w, b in zip(coefficients.weights, coefficients.beta)
    )
    np.testing.assert_allclose(impact, [expected_at_zero, expected_at_one])


def test_latency_filter_can_keep_only_fast_completed_orders():
    orders = pd.DataFrame(
        {
            "window_id": [0, 0, 0, 1, 1, 1],
            "window_start": ["t0", "t0", "t0", "t1", "t1", "t1"],
            "order_id": [0, 1, 2, 3, 4, 5],
            "order_slot": [0, 1, 2, 0, 1, 2],
            "post_qty": [1, 1, 1, 1, 1, 1],
            "latency_s": [10.0, 31.0, np.nan, 8.0, 9.0, 11.0],
            "completed_time_s": [10.5, 32.0, np.nan, 8.5, 9.5, 11.5],
        }
    )

    selected, windows = select_latency_orders(
        orders,
        LatencyFilterConfig(max_latency_seconds=30.0, selection_mode="orders"),
    )
    assert selected["order_id"].tolist() == [0, 3, 4, 5]
    assert windows["window_id"].tolist() == [0, 1]

    selected_all, windows_all = select_latency_orders(
        orders,
        LatencyFilterConfig(
            max_latency_seconds=30.0,
            selection_mode="window_all",
            min_orders=3,
        ),
    )
    assert selected_all["order_id"].tolist() == [3, 4, 5]
    assert windows_all["window_id"].tolist() == [1]


def test_native_passive_flow_impact_from_queue_samples_zero_when_paths_match():
    simproj = pytest.importorskip("simproj")
    q = np.array([10, 11, 9], dtype=np.uint32)
    market_times = np.array([0.5, 1.0, 2.0], dtype=np.float64)

    impact = simproj.passive_flow_impact_from_queue_samples(
        q,
        q.copy(),
        market_times,
        1.0,
        [0.065, 0.2],
        [0.15, 0.60],
        -0.0097,
        0.00989,
        -2.1766e-6,
    )

    np.testing.assert_allclose(np.asarray(impact), np.zeros_like(market_times))
