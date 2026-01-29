//! Abstract point process trait.
//!
//! Defines the interface that all point process models must implement.

/// Abstract trait for point process models.
///
/// All point process implementations must provide methods for:
/// - Computing the conditional intensity at a given time
/// - Computing intensity contributions from historical events
/// - Providing an upper bound on intensity (for thinning algorithms)
///
/// # Example
///
/// ```rust
/// use simulation_project::models::PointProcess;
///
/// fn simulate_generic<P: PointProcess>(model: &P, t_max: f64) -> Vec<f64> {
///     // Generic simulation using the trait interface
///     todo!()
/// }
/// ```
pub trait PointProcess: Send + Sync {
    /// Compute the conditional intensity λ(t | H_t) given history.
    ///
    /// # Arguments
    /// * `t` - Time at which to compute intensity
    /// * `events` - Historical event times (must be < t)
    ///
    /// # Returns
    /// The conditional intensity value at time t.
    fn intensity(&self, t: f64, events: &[f64]) -> f64;

    /// Compute an upper bound on the intensity for the thinning algorithm.
    ///
    /// Given current state, returns λ* such that λ(s) ≤ λ* for s ∈ [t, t + dt]
    /// where dt is reasonably small.
    ///
    /// # Arguments
    /// * `t` - Current time
    /// * `events` - Historical event times
    ///
    /// # Returns
    /// Upper bound on intensity.
    fn intensity_upper_bound(&self, t: f64, events: &[f64]) -> f64;

    /// Get the baseline (background) intensity μ.
    fn baseline_intensity(&self) -> f64;

    /// Get the number of kernel components (M).
    fn num_components(&self) -> usize;
}

/// Trait for models that support Markovian (recursive) intensity computation.
///
/// This is more efficient than recomputing from scratch at each event.
pub trait MarkovianIntensity: PointProcess {
    /// State type for Markovian recursion (typically Vec<f64> for component intensities).
    type State: Clone;

    /// Initialize the Markovian state.
    fn initial_state(&self) -> Self::State;

    /// Update state after an event at time t.
    ///
    /// # Arguments
    /// * `state` - Current Markovian state (modified in place)
    /// * `t` - Time of the new event
    /// * `t_prev` - Time of the previous event (or 0 if first event)
    fn update_state(&self, state: &mut Self::State, t: f64, t_prev: f64);

    /// Compute intensity from Markovian state.
    ///
    /// # Arguments
    /// * `state` - Current Markovian state
    /// * `t` - Time at which to compute intensity
    /// * `t_last` - Time of the last event
    fn intensity_from_state(&self, state: &Self::State, t: f64, t_last: f64) -> f64;
}
