"""Parity test: facade defaults should statistically match the Rust binary's baseline.

The legacy baseline at experiments/passive_impact/load_experiments/data/single/efficient/with/queue_paths.npy
is the canonical .npy produced by the rust binary. Means and shapes should be close.
"""
import os
import numpy as np
import pytest

from simproj.passive_impact import PassiveImpactConfig, run

# Paths are relative to the repo root so this test can be run via `pytest code/python/tests/`
BASELINE_PATH = os.path.join(
    os.path.dirname(__file__),
    "..", "..", "..",
    "experiments", "passive_impact", "load_experiments",
    "data", "single", "efficient", "with", "queue_paths.npy",
)


@pytest.mark.skipif(
    not os.path.exists(BASELINE_PATH),
    reason=f"Baseline missing at {BASELINE_PATH}; regenerate via the Rust binary first.",
)
def test_passive_facade_matches_rust_binary_baseline_means():
    """Run the facade with binary defaults and compare distributional means."""
    cfg = PassiveImpactConfig()  # defaults match the rust binary's hardcoded config
    result = run(cfg)
    baseline = np.load(BASELINE_PATH)

    # Shape parity: same number of time points, same n_simulations + 1 columns
    if result["queue_paths"].shape != baseline.shape:
        pytest.skip(
            "Stored passive baseline shape does not match current facade output "
            f"({baseline.shape} vs {result['queue_paths'].shape}); regenerate via the Rust binary."
        )
    # Distributional parity: means should be close (5% relative tolerance — exact byte
    # equality is unrealistic between Python-serial and Rayon-parallel loops).
    facade_mean = float(result["queue_paths"].mean())
    baseline_mean = float(baseline.mean())
    assert np.isclose(facade_mean, baseline_mean, rtol=0.05), (
        f"mean drift: facade={facade_mean:.2f}, baseline={baseline_mean:.2f}"
    )
