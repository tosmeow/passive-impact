"""Python facades + bindings for the simulation_project Rust library."""
from . import _native  # noqa: F401

__version__ = _native.__version__
