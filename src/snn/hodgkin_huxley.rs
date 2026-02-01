//! Hodgkin-Huxley Neuron Model
//!
//! Full ion channel model for deep chemical range.
//! Used for RegulationRegion and PersonalityRegion.
//!
//! ## Equations
//!
//! ```text
//! C_m * dV/dt = I - I_Na - I_K - I_leak - I_Ca
//! I_Na = g_Na * m³h * (V - E_Na)
//! I_K = g_K * n⁴ * (V - E_K)
//! ```
//!
//! ## Note on f32
//!
//! HH gating kinetics use exponentials (α, β rates).
//! We use f32 internally for numerical stability, but:
//! - Never export raw f32 values
//! - Never store f32 in persistent state
//! - Collapse all outputs to Signal at integration boundary

use crate::Signal;
use super::{
    ActivationMode, ChemicalAxes, GatingState, HHProfile, NeuronIntrospection,
    SpikingNeuron,
};

/// Gating variable with α and β rates
#[derive(Clone, Copy, Debug)]
struct Gate {
    /// Current state (0 to 1)
    state: f32,
    /// Alpha rate
    alpha: f32,
    /// Beta rate
    beta: f32,
}

impl Default for Gate {
    fn default() -> Self {
        Self {
            state: 0.0,
            alpha: 0.0,
            beta: 0.0,
        }
    }
}

impl Gate {
    /// Initialize state to steady-state value
    fn init_steady_state(&mut self) {
        if self.alpha + self.beta > 0.0 {
            self.state = self.alpha / (self.alpha + self.beta);
        }
    }

    /// Update gate state for one timestep
    fn update(&mut self, dt: f32) {
        let d_state = (self.alpha * (1.0 - self.state) - self.beta * self.state) * dt;
        self.state = (self.state + d_state).clamp(0.0, 1.0);
    }
}

/// Hodgkin-Huxley neuron with full ion channel dynamics
#[derive(Clone, Debug)]
pub struct HodgkinHuxleyNeuron {
    /// Membrane potential (mV)
    pub v: f32,
    /// Membrane capacitance (nF)
    pub c_m: f32,

    // === Ion Channel Conductances ===
    /// Sodium conductance (nS)
    pub g_na: f32,
    /// Potassium conductance (nS)
    pub g_k: f32,
    /// Leak conductance (nS)
    pub g_leak: f32,
    /// Calcium conductance (nS)
    pub g_ca: f32,

    // === Reversal Potentials ===
    /// Sodium reversal (mV)
    pub e_na: f32,
    /// Potassium reversal (mV)
    pub e_k: f32,
    /// Leak reversal (mV)
    pub e_leak: f32,
    /// Calcium reversal (mV)
    pub e_ca: f32,

    // === Gating Variables ===
    /// Sodium activation (m)
    m: Gate,
    /// Sodium inactivation (h)
    h: Gate,
    /// Potassium activation (n)
    n: Gate,
    /// Calcium activation (s)
    s: Gate,

    // === State ===
    /// Timestep (ms)
    pub dt: f32,
    /// Spike threshold (mV)
    pub v_th: f32,
    /// Whether currently spiking
    pub is_spiking: bool,
    /// Was voltage increasing last step
    was_increasing: bool,
    /// Timestep counter
    pub timestep: u64,
    /// Last spike time
    pub last_spike: Option<u64>,
    /// Activation mode
    pub activation_mode: ActivationMode,
    /// Rest potential
    pub v_rest: f32,

    // === Profile modifiers (from chemical axes) ===
    /// Excitability modifier (from axes)
    excitability_mod: f32,
    /// Inhibition modifier (from axes)
    inhibition_mod: f32,
}

impl Default for HodgkinHuxleyNeuron {
    fn default() -> Self {
        Self::from_profile(HHProfile::Standard)
    }
}

impl HodgkinHuxleyNeuron {
    /// Rest potential constant
    const V_REST: f32 = -65.0;
    /// Voltage range for activation mapping
    const V_RANGE: f32 = 125.0; // -65 to +60

    /// Create from profile
    pub fn from_profile(profile: HHProfile) -> Self {
        let (g_na, g_k, g_leak, g_ca) = profile.conductances();

        let mut neuron = Self {
            v: Self::V_REST,
            c_m: 1.0,
            g_na,
            g_k,
            g_leak,
            g_ca,
            e_na: 50.0,
            e_k: -77.0,
            e_leak: -54.4,
            e_ca: 120.0,
            m: Gate::default(),
            h: Gate::default(),
            n: Gate::default(),
            s: Gate::default(),
            dt: 0.01, // HH needs small dt for stability
            v_th: 0.0,
            is_spiking: false,
            was_increasing: false,
            timestep: 0,
            last_spike: None,
            activation_mode: ActivationMode::Bipolar, // HH regions use bipolar
            v_rest: Self::V_REST,
            excitability_mod: 1.0,
            inhibition_mod: 1.0,
        };

        // Initialize gates to steady state at rest potential
        neuron.update_gate_rates();
        neuron.m.init_steady_state();
        neuron.h.init_steady_state();
        neuron.n.init_steady_state();
        neuron.s.init_steady_state();

        neuron
    }

    /// Update α and β rates for all gates based on voltage
    fn update_gate_rates(&mut self) {
        let v = self.v;

        // Sodium activation (m)
        self.m.alpha = if (v + 40.0).abs() < 0.001 {
            1.0
        } else {
            0.1 * (v + 40.0) / (1.0 - (-0.1 * (v + 40.0)).exp())
        };
        self.m.beta = 4.0 * (-0.0556 * (v + 65.0)).exp();

        // Sodium inactivation (h)
        self.h.alpha = 0.07 * (-0.05 * (v + 65.0)).exp();
        self.h.beta = 1.0 / (1.0 + (-0.1 * (v + 35.0)).exp());

        // Potassium activation (n)
        self.n.alpha = if (v + 55.0).abs() < 0.001 {
            0.1
        } else {
            0.01 * (v + 55.0) / (1.0 - (-0.1 * (v + 55.0)).exp())
        };
        self.n.beta = 0.125 * (-0.0125 * (v + 65.0)).exp();

        // Calcium activation (s)
        self.s.alpha = 1.6 / (1.0 + (-0.072 * (v - 5.0)).exp());
        let denom = ((v + 8.9) / 5.0).exp() - 1.0;
        self.s.beta = if denom.abs() > 0.001 {
            0.02 * (v + 8.9) / denom
        } else {
            0.1
        };
    }

    /// Calculate ionic currents
    fn ionic_currents(&self) -> (f32, f32, f32, f32) {
        let i_na = self.g_na
            * self.excitability_mod
            * self.m.state.powi(3)
            * self.h.state
            * (self.v - self.e_na);

        let i_k = self.g_k * self.inhibition_mod * self.n.state.powi(4) * (self.v - self.e_k);

        let i_leak = self.g_leak * (self.v - self.e_leak);

        let i_ca = self.g_ca * self.excitability_mod * self.s.state.powi(2) * (self.v - self.e_ca);

        (i_na, i_k, i_leak, i_ca)
    }
}

impl SpikingNeuron for HodgkinHuxleyNeuron {
    fn step(&mut self, current: i32, axes: &ChemicalAxes) -> bool {
        self.timestep += 1;

        // Apply chemical axes modulation
        self.modulate(axes);

        // Convert input current
        let i_ext = current as f32 / 256.0 * axes.background_current_modifier();

        // Update gate rates
        self.update_gate_rates();

        // Update gate states
        self.m.update(self.dt);
        self.h.update(self.dt);
        self.n.update(self.dt);
        self.s.update(self.dt);

        // Calculate ionic currents
        let (i_na, i_k, i_leak, i_ca) = self.ionic_currents();

        // Membrane equation: C dV/dt = I_ext - I_ion
        let i_total = i_ext - i_na - i_k - i_leak - i_ca;
        let dv = (i_total / self.c_m) * self.dt;

        let last_v = self.v;
        self.v += dv;

        // Spike detection (peak detection)
        let increasing_now = self.v > last_v;
        let crossed_threshold = self.v > self.v_th;
        let is_peak = crossed_threshold && self.was_increasing && !increasing_now;

        self.is_spiking = is_peak;
        self.was_increasing = increasing_now;

        if is_peak {
            self.last_spike = Some(self.timestep);
        }

        self.is_spiking
    }

    fn activation(&self) -> Signal {
        let delta_v = self.v - self.v_rest;

        match self.activation_mode {
            ActivationMode::Bipolar => {
                // Full bipolar: hyperpolarization = -Signal, depolarization = +Signal
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

    fn modulate(&mut self, axes: &ChemicalAxes) {
        // Map chemical axes to conductance modifiers
        // Excitability: increases Na, Ca conductance
        self.excitability_mod = 1.0 + 0.5 * axes.excitability_f32();
        self.excitability_mod = self.excitability_mod.clamp(0.5, 2.0);

        // Inhibition: increases K conductance
        self.inhibition_mod = 1.0 + 0.5 * axes.inhibition_f32();
        self.inhibition_mod = self.inhibition_mod.clamp(0.5, 2.0);
    }

    fn activation_mode(&self) -> ActivationMode {
        self.activation_mode
    }

    fn reset(&mut self) {
        self.v = self.v_rest;
        self.update_gate_rates();
        self.m.init_steady_state();
        self.h.init_steady_state();
        self.n.init_steady_state();
        self.s.init_steady_state();
        self.is_spiking = false;
        self.was_increasing = false;
        self.timestep = 0;
        self.last_spike = None;
        self.excitability_mod = 1.0;
        self.inhibition_mod = 1.0;
    }

    fn dt(&self) -> f32 {
        self.dt
    }

    fn set_dt(&mut self, dt: f32) {
        self.dt = dt;
    }
}

impl NeuronIntrospection for HodgkinHuxleyNeuron {
    fn membrane(&self) -> f32 {
        self.v
    }

    fn recovery(&self) -> f32 {
        // HH doesn't have a single recovery variable like Izhikevich
        // Return h (inactivation) as a proxy for recovery
        self.h.state
    }

    fn gating_state(&self) -> Option<GatingState> {
        Some(GatingState {
            m: self.m.state,
            h: self.h.state,
            n: self.n.state,
            s: Some(self.s.state),
        })
    }

    fn is_spiking(&self) -> bool {
        self.is_spiking
    }

    fn last_spike_time(&self) -> Option<u64> {
        self.last_spike
    }

    fn model_name(&self) -> &'static str {
        "HodgkinHuxley"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hh_resting_potential() {
        let neuron = HodgkinHuxleyNeuron::from_profile(HHProfile::Standard);
        assert!(
            (neuron.v - HodgkinHuxleyNeuron::V_REST).abs() < 1.0,
            "Should start at rest"
        );
    }

    #[test]
    fn test_hh_gates_initialized() {
        let neuron = HodgkinHuxleyNeuron::from_profile(HHProfile::Standard);
        assert!(neuron.m.state >= 0.0 && neuron.m.state <= 1.0);
        assert!(neuron.h.state >= 0.0 && neuron.h.state <= 1.0);
        assert!(neuron.n.state >= 0.0 && neuron.n.state <= 1.0);
    }

    #[test]
    fn test_hh_spike() {
        let mut neuron = HodgkinHuxleyNeuron::from_profile(HHProfile::HighExcitability);
        let axes = ChemicalAxes::high_excitability();

        // HH needs many small timesteps
        let mut spiked = false;
        for _ in 0..10000 {
            if neuron.step(5000, &axes) {
                spiked = true;
                break;
            }
        }

        assert!(spiked, "HH should spike with sufficient input");
    }

    #[test]
    fn test_hh_bipolar_activation() {
        let mut neuron = HodgkinHuxleyNeuron::from_profile(HHProfile::Standard);
        neuron.activation_mode = ActivationMode::Bipolar;

        // Depolarized
        neuron.v = 0.0;
        let sig = neuron.activation();
        assert!(sig.is_positive(), "Should be positive when depolarized");

        // Hyperpolarized
        neuron.v = -80.0;
        let sig = neuron.activation();
        assert!(sig.is_negative(), "Should be negative when hyperpolarized");
    }

    #[test]
    fn test_chemical_axes_modulation() {
        let mut neuron = HodgkinHuxleyNeuron::from_profile(HHProfile::Standard);
        let high_excite = ChemicalAxes::high_excitability();

        neuron.modulate(&high_excite);
        assert!(
            neuron.excitability_mod > 1.0,
            "Excitability mod should increase"
        );
    }
}
