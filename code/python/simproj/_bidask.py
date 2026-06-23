"""Shared helpers for bid-ask Python facades."""
from __future__ import annotations

import numpy as np

from . import _native

TOTAL_DIMS = 6
ASK_LIMIT_DIM = 0
ASK_MARKET_DIM = 2
BID_LIMIT_DIM = 3
BID_MARKET_DIM = 5
MARKET_DIMS = {ASK_MARKET_DIM, BID_MARKET_DIM}


def initial_sizes(cfg) -> tuple[int, int]:
    ask = cfg.initial_ask_queue_size
    bid = cfg.initial_bid_queue_size
    fallback = int(cfg.initial_queue_size)
    return int(ask if ask is not None else fallback), int(bid if bid is not None else fallback)


def metaorder_dim(side: str) -> int:
    normalized = side.lower()
    if normalized == "ask":
        return ASK_LIMIT_DIM
    if normalized == "bid":
        return BID_LIMIT_DIM
    raise ValueError(f"metaorder_side must be 'ask' or 'bid'; got {side!r}")


def make_bidask_process(cfg):
    q0_a, q0_b = initial_sizes(cfg)
    return _native.AffineBidAskQueueProcess.new_queue(
        float(q0_a),
        float(q0_b),
        cfg.a_l,
        cfg.b_l,
        cfg.b_l_cross,
        cfg.a_c,
        cfg.b_c,
        cfg.b_c_cross,
        cfg.a_l,
        cfg.b_l_cross,
        cfg.b_l,
        cfg.a_c,
        cfg.b_c_cross,
        cfg.b_c,
    )


def make_bidask_market_orders(cfg):
    hawkes_ask = _native.MultiExponentialHawkes.with_stationary_state(
        cfg.mu, list(cfg.alpha), list(cfg.beta),
    )
    hawkes_bid = _native.MultiExponentialHawkes.with_stationary_state(
        cfg.mu, list(cfg.alpha), list(cfg.beta),
    )
    ask_hawkes = _native.simulate_hawkes_result(hawkes_ask, cfg.time_horizon, cfg.seed)
    bid_hawkes = _native.simulate_hawkes_result(hawkes_bid, cfg.time_horizon, cfg.seed + 1)
    ask_market = _native.events_to_dim(
        ask_hawkes, target_dim=ASK_MARKET_DIM, total_dims=TOTAL_DIMS,
    )
    bid_market = _native.events_to_dim(
        bid_hawkes, target_dim=BID_MARKET_DIM, total_dims=TOTAL_DIMS,
    )
    return ask_market, bid_market, _native.merge_events(ask_market, bid_market)


def make_bidask_meta_orders(cfg):
    target_dim = metaorder_dim(cfg.metaorder_side)
    if isinstance(cfg.metaorder, int):
        meta = _native.create_meta_orders(cfg.metaorder, *cfg.metaorder_window)
        return _native.events_to_dim(meta, target_dim=target_dim, total_dims=TOTAL_DIMS)
    times = np.asarray(cfg.metaorder, dtype=np.float64)
    return _native.create_meta_orders_from_times(
        times, target_dim=target_dim, total_dims=TOTAL_DIMS,
    )


def conditioning_events_without_market(internal_events):
    by_dim = [
        list(arr) for arr in _native.extract_events_by_dim(
            internal_events, TOTAL_DIMS, None,
        )
    ]
    for dim in MARKET_DIMS:
        by_dim[dim] = []
    return by_dim


def sample_bidask_at_times(events, initial_q_a: int, initial_q_b: int, times):
    samples = _native.sample_bidask_queue_at_times(
        events, int(initial_q_a), int(initial_q_b), np.asarray(times, dtype=np.float64),
    )
    return samples["ask"], samples["bid"]
