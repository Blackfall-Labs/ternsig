//! Core Traits - Facade API for SNN Models
//!
//! ## SpikingNeuron Trait
//!
//! The ONLY interface for external code to interact with neurons.
//! `activation()` is the contract - all region logic uses this.
//!
//! ## NeuronIntrospection Trait
//!
//! Debug/telemetry only - NOT in SpikingNeuron trait.
//! Accessible only for tests, debug instrumentation, telemetry layers.
//! NOT in region logic. NOT in convergence logic.

use crate::Signal;
use super::ChemicalAxes;

/// How membrane potential maps to Signal polarity
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ActivationMode {
    /// Only depolarization produces output (most regions)
    /// rest → Signal(0,0), depol → +Signal, hyperpol → Signal(0,0)
    #[default]
    DepolarOnly,

    /// Full bipolar output (emotional/regulatory regions)
    /// rest → Signal(0,0), depol → +Signal, hyperpol → -Signal
    Bipolar,
}

/// Unified trait for all neuron models
///
/// NOTE: membrane() is deliberately NOT in this trait - see NeuronIntrospection
pub trait SpikingNeuron: Send + Sync {
    /// Step the neuron with input current and chemical axes
    ///
    /// Returns spike event for INSTRUMENTATION ONLY.
    /// Real output is `activation()` - never use this bool for decisions.
    fn step(&mut self, current: i32, axes: &ChemicalAxes) -> bool;

    /// Get activation level as Signal (the ONLY output for scheduling/convergence)
    ///
    /// This is the contract - all region logic uses this.
    fn activation(&self) -> Signal;

    /// Apply chemical axis state (model-specific effects)
    fn modulate(&mut self, axes: &ChemicalAxes);

    /// How this neuron maps membrane to Signal polarity
    fn activation_mode(&self) -> ActivationMode;

    /// Reset to initial state
    fn reset(&mut self);

    /// Get the timestep (dt) in milliseconds
    fn dt(&self) -> f32;

    /// Set the timestep (dt) in milliseconds
    fn set_dt(&mut self, dt: f32);
}

/// Gating state for HH neurons (debug/telemetry)
#[derive(Clone, Copy, Debug, Default)]
pub struct GatingState {
    /// Sodium activation gate (m)
    pub m: f32,
    /// Sodium inactivation gate (h)
    pub h: f32,
    /// Potassium activation gate (n)
    pub n: f32,
    /// Calcium gate (s) if applicable
    pub s: Option<f32>,
}

/// Debug/telemetry access ONLY - not in SpikingNeuron trait
///
/// This trait provides access to internal state for:
/// - Tests
/// - Debug instrumentation
/// - Telemetry layers
///
/// NOT for region logic. NOT for convergence logic.
pub trait NeuronIntrospection {
    /// Get membrane potential (mV)
    fn membrane(&self) -> f32;

    /// Get membrane potential as scaled i16 (×256)
    fn membrane_i16(&self) -> i16 {
        (self.membrane() * 256.0).clamp(-32768.0, 32767.0) as i16
    }

    /// Get recovery variable (Izhikevich u, or equivalent)
    fn recovery(&self) -> f32;

    /// Get gating state (HH only, None for Izhikevich)
    fn gating_state(&self) -> Option<GatingState>;

    /// Get current spike state
    fn is_spiking(&self) -> bool;

    /// Get last spike time (timesteps since last spike, None if never spiked)
    fn last_spike_time(&self) -> Option<u64>;

    /// Get model name for debugging
    fn model_name(&self) -> &'static str;
}
