//! Izhikevich Neuron Models
//!
//! Three variants:
//! - **Izhikevich**: Standard 4-parameter model
//! - **LeakyIzhikevich**: + leak term toward e_l
//! - **BCMIzhikevich**: + BCM activity-dependent plasticity
//!
//! ## Equations
//!
//! ```text
//! dv/dt = 0.04v² + 5v + 140 - u + I
//! du/dt = a(bv - u)
//! if v >= 30: v = c, u = u + d
//! ```

use crate::Signal;
use super::{ActivationMode, ChemicalAxes, GatingState, IzhProfile, NeuronIntrospection, SpikingNeuron};

/// Standard Izhikevich neuron (4 parameters)
#[derive(Clone, Debug)]
pub struct IzhikevichNeuron {
    /// Membrane potential (mV)
    pub v: f32,
    /// Recovery variable
    pub u: f32,
    /// Time scale of recovery
    pub a: f32,
    /// Sensitivity of u to v
    pub b: f32,
    /// After-spike reset value for v
    pub c: f32,
    /// After-spike reset value for u
    pub d: f32,
    /// Spike threshold (mV)
    pub v_th: f32,
    /// Timestep (ms)
    pub dt: f32,
    /// Membrane capacitance scale
    pub c_m: f32,
    /// Whether currently spiking
    pub is_spiking: bool,
    /// Timestep counter
    pub timestep: u64,
    /// Last spike time
    pub last_spike: Option<u64>,
    /// Activation mode
    pub activation_mode: ActivationMode,
    /// Rest potential for activation calculation
    pub v_rest: f32,
}

impl Default for IzhikevichNeuron {
    fn default() -> Self {
        Self::from_profile(IzhProfile::RegularSpiking)
    }
}

impl IzhikevichNeuron {
    /// Rest potential constant
    const V_REST: f32 = -65.0;
    /// Voltage range for activation mapping
    const V_RANGE: f32 = 95.0; // -65 to +30

    /// Create from profile
    pub fn from_profile(profile: IzhProfile) -> Self {
        let (a, b, c, d) = profile.params();
        Self {
            v: Self::V_REST,
            u: b * Self::V_REST,
            a,
            b,
            c,
            d,
            v_th: 30.0,
            dt: 1.0,
            c_m: 1.0,
            is_spiking: false,
            timestep: 0,
            last_spike: None,
            activation_mode: ActivationMode::DepolarOnly,
            v_rest: Self::V_REST,
        }
    }

    /// Compute dv/dt
    fn dv(&self, i: f32) -> f32 {
        (0.04 * self.v * self.v + 5.0 * self.v + 140.0 - self.u + i) * (self.dt / self.c_m)
    }

    /// Compute du/dt
    fn du(&self) -> f32 {
        self.a * (self.b * self.v - self.u) * self.dt
    }
}

impl SpikingNeuron for IzhikevichNeuron {
    fn step(&mut self, current: i32, axes: &ChemicalAxes) -> bool {
        self.timestep += 1;

        // Apply chemical axes modulation to input
        let current_f = current as f32 / 256.0; // Assume current is scaled ×256
        let modulated_current = current_f * axes.background_current_modifier();

        // Izhikevich dynamics
        let dv = self.dv(modulated_current);
        let du = self.du();

        self.v += dv;
        self.u += du;

        // Spike detection and reset
        self.is_spiking = false;
        if self.v >= self.v_th {
            self.is_spiking = true;
            self.v = self.c;
            self.u += self.d;
            self.last_spike = Some(self.timestep);
        }

        self.is_spiking
    }

    fn activation(&self) -> Signal {
        let delta_v = self.v - self.v_rest;

        match self.activation_mode {
            ActivationMode::Bipolar => {
                if delta_v > 0.5 {
                    let magnitude = ((delta_v / Self::V_RANGE) * 255.0).clamp(0.0, 255.0) as u8;
                    Signal::positive(magnitude)
                } else if delta_v < -0.5 {
                    let magnitude = ((-delta_v / Self::V_RANGE) * 255.0).clamp(0.0, 255.0) as u8;
                    Signal::negative(magnitude)
                } else {
                    Signal::ZERO
                }
            }
            ActivationMode::DepolarOnly => {
                if delta_v > 0.5 {
                    let magnitude = ((delta_v / Self::V_RANGE) * 255.0).clamp(0.0, 255.0) as u8;
                    Signal::positive(magnitude)
                } else {
                    Signal::ZERO
                }
            }
        }
    }

    fn modulate(&mut self, _axes: &ChemicalAxes) {
        // Izhikevich modulation is applied during step()
        // Could adjust a, b, c, d within bounded ranges here
    }

    fn activation_mode(&self) -> ActivationMode {
        self.activation_mode
    }

    fn reset(&mut self) {
        self.v = self.v_rest;
        self.u = self.b * self.v_rest;
        self.is_spiking = false;
        self.timestep = 0;
        self.last_spike = None;
    }

    fn dt(&self) -> f32 {
        self.dt
    }

    fn set_dt(&mut self, dt: f32) {
        self.dt = dt;
    }
}

impl NeuronIntrospection for IzhikevichNeuron {
    fn membrane(&self) -> f32 {
        self.v
    }

    fn recovery(&self) -> f32 {
        self.u
    }

    fn gating_state(&self) -> Option<GatingState> {
        None // Izhikevich has no gating variables
    }

    fn is_spiking(&self) -> bool {
        self.is_spiking
    }

    fn last_spike_time(&self) -> Option<u64> {
        self.last_spike
    }

    fn model_name(&self) -> &'static str {
        "Izhikevich"
    }
}

// ============================================================================
// LEAKY IZHIKEVICH
// ============================================================================

/// Leaky Izhikevich neuron (5 parameters)
///
/// Adds leak term toward e_l for better resting stability.
#[derive(Clone, Debug)]
pub struct LeakyIzhikevichNeuron {
    /// Base Izhikevich neuron
    pub base: IzhikevichNeuron,
    /// Leak reversal potential (mV)
    pub e_l: f32,
}

impl Default for LeakyIzhikevichNeuron {
    fn default() -> Self {
        Self::from_profile(IzhProfile::RegularSpiking)
    }
}

impl LeakyIzhikevichNeuron {
    /// Create from profile
    pub fn from_profile(profile: IzhProfile) -> Self {
        Self {
            base: IzhikevichNeuron::from_profile(profile),
            e_l: -65.0,
        }
    }

    /// Compute dv/dt with leak term
    fn dv_leaky(&self, i: f32) -> f32 {
        let leak = self.base.u * (self.base.v - self.e_l);
        (0.04 * self.base.v * self.base.v + 5.0 * self.base.v + 140.0 - leak + i)
            * (self.base.dt / self.base.c_m)
    }
}

impl SpikingNeuron for LeakyIzhikevichNeuron {
    fn step(&mut self, current: i32, axes: &ChemicalAxes) -> bool {
        self.base.timestep += 1;

        let current_f = current as f32 / 256.0;
        let modulated_current = current_f * axes.background_current_modifier();

        // Leaky Izhikevich dynamics
        let dv = self.dv_leaky(modulated_current);
        let du = self.base.a * (self.base.b * self.base.v - self.base.u) * self.base.dt;

        self.base.v += dv;
        self.base.u += du;

        // Spike detection and reset
        self.base.is_spiking = false;
        if self.base.v >= self.base.v_th {
            self.base.is_spiking = true;
            self.base.v = self.base.c;
            self.base.u += self.base.d;
            self.base.last_spike = Some(self.base.timestep);
        }

        self.base.is_spiking
    }

    fn activation(&self) -> Signal {
        self.base.activation()
    }

    fn modulate(&mut self, axes: &ChemicalAxes) {
        self.base.modulate(axes);
    }

    fn activation_mode(&self) -> ActivationMode {
        self.base.activation_mode
    }

    fn reset(&mut self) {
        self.base.reset();
    }

    fn dt(&self) -> f32 {
        self.base.dt
    }

    fn set_dt(&mut self, dt: f32) {
        self.base.dt = dt;
    }
}

impl NeuronIntrospection for LeakyIzhikevichNeuron {
    fn membrane(&self) -> f32 {
        self.base.v
    }

    fn recovery(&self) -> f32 {
        self.base.u
    }

    fn gating_state(&self) -> Option<GatingState> {
        None
    }

    fn is_spiking(&self) -> bool {
        self.base.is_spiking
    }

    fn last_spike_time(&self) -> Option<u64> {
        self.base.last_spike
    }

    fn model_name(&self) -> &'static str {
        "LeakyIzhikevich"
    }
}

// ============================================================================
// BCM IZHIKEVICH
// ============================================================================

/// BCM Izhikevich neuron (activity-dependent plasticity)
///
/// Adds Bienenstock-Cooper-Munro plasticity for learning regions.
#[derive(Clone, Debug)]
pub struct BCMIzhikevichNeuron {
    /// Base Izhikevich neuron
    pub base: IzhikevichNeuron,
    /// Activity trace (exponential moving average of spikes)
    pub activity: f32,
    /// Sliding threshold (BCM theta)
    pub theta: f32,
    /// Activity decay rate
    pub activity_decay: f32,
    /// Theta adaptation rate
    pub theta_rate: f32,
}

impl Default for BCMIzhikevichNeuron {
    fn default() -> Self {
        Self::from_profile(IzhProfile::RegularSpiking)
    }
}

impl BCMIzhikevichNeuron {
    /// Create from profile
    pub fn from_profile(profile: IzhProfile) -> Self {
        Self {
            base: IzhikevichNeuron::from_profile(profile),
            activity: 0.0,
            theta: 0.5,
            activity_decay: 0.99,
            theta_rate: 0.001,
        }
    }

    /// Get BCM learning rate (activity - theta)
    /// Positive when activity > theta (LTP), negative when < (LTD)
    pub fn bcm_rate(&self) -> f32 {
        self.activity - self.theta
    }

    /// Update BCM state after spike
    fn update_bcm(&mut self, spiked: bool) {
        // Update activity trace
        self.activity *= self.activity_decay;
        if spiked {
            self.activity += 1.0 - self.activity_decay;
        }

        // Update sliding threshold
        self.theta += self.theta_rate * (self.activity * self.activity - self.theta);
    }
}

impl SpikingNeuron for BCMIzhikevichNeuron {
    fn step(&mut self, current: i32, axes: &ChemicalAxes) -> bool {
        let spiked = self.base.step(current, axes);
        self.update_bcm(spiked);
        spiked
    }

    fn activation(&self) -> Signal {
        self.base.activation()
    }

    fn modulate(&mut self, axes: &ChemicalAxes) {
        self.base.modulate(axes);
    }

    fn activation_mode(&self) -> ActivationMode {
        self.base.activation_mode
    }

    fn reset(&mut self) {
        self.base.reset();
        self.activity = 0.0;
        self.theta = 0.5;
    }

    fn dt(&self) -> f32 {
        self.base.dt
    }

    fn set_dt(&mut self, dt: f32) {
        self.base.dt = dt;
    }
}

impl NeuronIntrospection for BCMIzhikevichNeuron {
    fn membrane(&self) -> f32 {
        self.base.v
    }

    fn recovery(&self) -> f32 {
        self.base.u
    }

    fn gating_state(&self) -> Option<GatingState> {
        None
    }

    fn is_spiking(&self) -> bool {
        self.base.is_spiking
    }

    fn last_spike_time(&self) -> Option<u64> {
        self.base.last_spike
    }

    fn model_name(&self) -> &'static str {
        "BCMIzhikevich"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_izhikevich_spike() {
        let mut neuron = IzhikevichNeuron::from_profile(IzhProfile::RegularSpiking);
        let axes = ChemicalAxes::baseline();

        // Apply strong current
        let mut spiked = false;
        for _ in 0..100 {
            if neuron.step(10000, &axes) {
                // 10000/256 ≈ 39 input
                spiked = true;
                break;
            }
        }

        assert!(spiked, "Neuron should spike with sufficient input");
    }

    #[test]
    fn test_izhikevich_no_spike_low_input() {
        let mut neuron = IzhikevichNeuron::from_profile(IzhProfile::RegularSpiking);
        let axes = ChemicalAxes::baseline();

        // Apply weak current
        for _ in 0..100 {
            assert!(
                !neuron.step(100, &axes),
                "Should not spike with low input"
            );
        }
    }

    #[test]
    fn test_bcm_activity_increases() {
        let mut neuron = BCMIzhikevichNeuron::from_profile(IzhProfile::FastSpiking);
        let axes = ChemicalAxes::high_excitability();

        let initial_activity = neuron.activity;

        // Drive to spike
        for _ in 0..100 {
            neuron.step(15000, &axes);
        }

        assert!(
            neuron.activity > initial_activity,
            "Activity should increase after spiking"
        );
    }

    #[test]
    fn test_activation_signal() {
        let mut neuron = IzhikevichNeuron::from_profile(IzhProfile::RegularSpiking);

        // At rest
        let sig = neuron.activation();
        assert!(sig.magnitude < 10, "Should be near zero at rest");

        // After spike
        neuron.v = 20.0; // Depolarized
        let sig = neuron.activation();
        assert!(sig.is_positive(), "Should be positive when depolarized");
        assert!(sig.magnitude > 100, "Should have significant magnitude");
    }
}
