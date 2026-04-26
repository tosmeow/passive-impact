"""Python facades + bindings for the simulation_project Rust library."""
from . import _native

MultiExponentialHawkes = _native.MultiExponentialHawkes
SimulationResult = _native.SimulationResult
AffineQueueProcess = _native.AffineQueueProcess
AffineBidAskQueueProcess = _native.AffineBidAskQueueProcess

simulate_hawkes = _native.simulate_hawkes
simulate_with_externals = _native.simulate_with_externals
simulate_hawkes_as_market_orders = _native.simulate_hawkes_as_market_orders
merge_events = _native.merge_events
create_meta_orders = _native.create_meta_orders
create_meta_orders_from_times = _native.create_meta_orders_from_times
events_to_dim = _native.events_to_dim
extract_events_by_dim = _native.extract_events_by_dim
sample_queue_at_times = _native.sample_queue_at_times
ConditionalSimulationContext = _native.ConditionalSimulationContext
TailImpact = _native.TailImpact
AggressiveImpactPath = _native.AggressiveImpactPath
aggressive_impact_from_queue_samples = _native.aggressive_impact_from_queue_samples
compute_impact_path = _native.compute_impact_path

__version__ = _native.__version__

from . import passive_impact
from . import agressive_impact
