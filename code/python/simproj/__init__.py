"""Python facades + bindings for the simulation_project Rust library."""
from . import _native

MultiExponentialHawkes = _native.MultiExponentialHawkes
simulate_hawkes = _native.simulate_hawkes

__version__ = _native.__version__
