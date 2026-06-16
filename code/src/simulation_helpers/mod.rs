mod multi_queue;
mod rng;
mod single_queue;

pub const DEFAULT_PASSIVE_C_KAPPA_EFFECTIVE: f64 = -0.0001;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PassiveSingleQueueSide {
    Ask,
    Bid,
}

impl PassiveSingleQueueSide {
    pub fn impact_sign(self) -> f64 {
        match self {
            Self::Ask => 1.0,
            Self::Bid => -1.0,
        }
    }
}

pub fn passive_c_kappa_effective_from_env() -> f64 {
    let value = std::env::var("C_KAPPA_EFFECTIVE")
        .ok()
        .map(|raw| {
            raw.parse::<f64>()
                .unwrap_or_else(|_| panic!("C_KAPPA_EFFECTIVE must be a finite f64, got {raw:?}"))
        })
        .unwrap_or(DEFAULT_PASSIVE_C_KAPPA_EFFECTIVE);
    if !value.is_finite() {
        panic!("C_KAPPA_EFFECTIVE must be finite, got {value}");
    }
    value
}

pub fn passive_single_queue_side_from_env() -> PassiveSingleQueueSide {
    match std::env::var("SINGLE_QUEUE_SIDE") {
        Ok(raw) => match raw.trim().to_ascii_lowercase().as_str() {
            "ask" => PassiveSingleQueueSide::Ask,
            "bid" => PassiveSingleQueueSide::Bid,
            other => panic!("SINGLE_QUEUE_SIDE must be 'ask' or 'bid', got {other:?}"),
        },
        Err(_) => PassiveSingleQueueSide::Bid,
    }
}

// Re-export rng utilities
pub use rng::{create_rng, sample_exponential, sample_uniform};

// Re-export everything at the simulation_helpers level for flat API
pub use multi_queue::*;
pub use single_queue::*;
