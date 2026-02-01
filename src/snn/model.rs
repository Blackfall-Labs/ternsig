//! NeuronModel - Facade Enum
//!
//! The ONLY way external code accesses neuron models.
//! No direct access to internal structs (IzhikevichNeuron, HHNeuron, etc.)

use crate::Signal;
use super::{
    izhikevich::{BCMIzhikevichNeuron, IzhikevichNeuron, LeakyIzhikevichNeuron},
    hodgkin_huxley::HodgkinHuxleyNeuron,
    ActivationMode, ChemicalAxes, GatingState, HHProfile, IzhProfile, NeuronIntrospection,
    SpikingNeuron,
};

/// Neuron model variants - the facade for all models
///
/// External code should ONLY use this enum to work with neurons.
/// Direct access to internal types is forbidden by the vendor facade rule.
#[derive(Clone, Debug)]
pub enum NeuronModel {
    /// Standard Izhikevich (4 params)
    Izhikevich(IzhikevichNeuron),
    /// Leaky Izhikevich (5 params)
    LeakyIzhikevich(LeakyIzhikevichNeuron),
    /// BCM Izhikevich (activity-dependent plasticity)
    BCMIzhikevich(BCMIzhikevichNeuron),
    /// Full Hodgkin-Huxley (deep chemical range)
    HodgkinHuxley(HodgkinHuxleyNeuron),
}

impl NeuronModel {
    /// Create Izhikevich neuron from profile
    pub fn izhikevich(profile: IzhProfile) -> Self {
        Self::Izhikevich(IzhikevichNeuron::from_profile(profile))
    }

    /// Create LeakyIzhikevich neuron from profile
    pub fn leaky_izhikevich(profile: IzhProfile) -> Self {
        Self::LeakyIzhikevich(LeakyIzhikevichNeuron::from_profile(profile))
    }

    /// Create BCMIzhikevich neuron from profile
    pub fn bcm_izhikevich(profile: IzhProfile) -> Self {
        Self::BCMIzhikevich(BCMIzhikevichNeuron::from_profile(profile))
    }

    /// Create HodgkinHuxley neuron from profile
    pub fn hodgkin_huxley(profile: HHProfile) -> Self {
        Self::HodgkinHuxley(HodgkinHuxleyNeuron::from_profile(profile))
    }

    /// Get model type name
    pub fn model_type(&self) -> &'static str {
        match self {
            Self::Izhikevich(_) => "Izhikevich",
            Self::LeakyIzhikevich(_) => "LeakyIzhikevich",
            Self::BCMIzhikevich(_) => "BCMIzhikevich",
            Self::HodgkinHuxley(_) => "HodgkinHuxley",
        }
    }

    /// Check if this is an Izhikevich variant
    pub fn is_izhikevich(&self) -> bool {
        matches!(
            self,
            Self::Izhikevich(_) | Self::LeakyIzhikevich(_) | Self::BCMIzhikevich(_)
        )
    }

    /// Check if this is HodgkinHuxley
    pub fn is_hodgkin_huxley(&self) -> bool {
        matches!(self, Self::HodgkinHuxley(_))
    }

    /// Set activation mode
    pub fn set_activation_mode(&mut self, mode: ActivationMode) {
        match self {
            Self::Izhikevich(n) => n.activation_mode = mode,
            Self::LeakyIzhikevich(n) => n.base.activation_mode = mode,
            Self::BCMIzhikevich(n) => n.base.activation_mode = mode,
            Self::HodgkinHuxley(n) => n.activation_mode = mode,
        }
    }
}

impl SpikingNeuron for NeuronModel {
    fn step(&mut self, current: i32, axes: &ChemicalAxes) -> bool {
        match self {
            Self::Izhikevich(n) => n.step(current, axes),
            Self::LeakyIzhikevich(n) => n.step(current, axes),
            Self::BCMIzhikevich(n) => n.step(current, axes),
            Self::HodgkinHuxley(n) => n.step(current, axes),
        }
    }

    fn activation(&self) -> Signal {
        match self {
            Self::Izhikevich(n) => n.activation(),
            Self::LeakyIzhikevich(n) => n.activation(),
            Self::BCMIzhikevich(n) => n.activation(),
            Self::HodgkinHuxley(n) => n.activation(),
        }
    }

    fn modulate(&mut self, axes: &ChemicalAxes) {
        match self {
            Self::Izhikevich(n) => n.modulate(axes),
            Self::LeakyIzhikevich(n) => n.modulate(axes),
            Self::BCMIzhikevich(n) => n.modulate(axes),
            Self::HodgkinHuxley(n) => n.modulate(axes),
        }
    }

    fn activation_mode(&self) -> ActivationMode {
        match self {
            Self::Izhikevich(n) => n.activation_mode(),
            Self::LeakyIzhikevich(n) => n.activation_mode(),
            Self::BCMIzhikevich(n) => n.activation_mode(),
            Self::HodgkinHuxley(n) => n.activation_mode(),
        }
    }

    fn reset(&mut self) {
        match self {
            Self::Izhikevich(n) => n.reset(),
            Self::LeakyIzhikevich(n) => n.reset(),
            Self::BCMIzhikevich(n) => n.reset(),
            Self::HodgkinHuxley(n) => n.reset(),
        }
    }

    fn dt(&self) -> f32 {
        match self {
            Self::Izhikevich(n) => n.dt(),
            Self::LeakyIzhikevich(n) => n.dt(),
            Self::BCMIzhikevich(n) => n.dt(),
            Self::HodgkinHuxley(n) => n.dt(),
        }
    }

    fn set_dt(&mut self, dt: f32) {
        match self {
            Self::Izhikevich(n) => n.set_dt(dt),
            Self::LeakyIzhikevich(n) => n.set_dt(dt),
            Self::BCMIzhikevich(n) => n.set_dt(dt),
            Self::HodgkinHuxley(n) => n.set_dt(dt),
        }
    }
}

impl NeuronIntrospection for NeuronModel {
    fn membrane(&self) -> f32 {
        match self {
            Self::Izhikevich(n) => n.membrane(),
            Self::LeakyIzhikevich(n) => n.membrane(),
            Self::BCMIzhikevich(n) => n.membrane(),
            Self::HodgkinHuxley(n) => n.membrane(),
        }
    }

    fn recovery(&self) -> f32 {
        match self {
            Self::Izhikevich(n) => n.recovery(),
            Self::LeakyIzhikevich(n) => n.recovery(),
            Self::BCMIzhikevich(n) => n.recovery(),
            Self::HodgkinHuxley(n) => n.recovery(),
        }
    }

    fn gating_state(&self) -> Option<GatingState> {
        match self {
            Self::Izhikevich(n) => n.gating_state(),
            Self::LeakyIzhikevich(n) => n.gating_state(),
            Self::BCMIzhikevich(n) => n.gating_state(),
            Self::HodgkinHuxley(n) => n.gating_state(),
        }
    }

    fn is_spiking(&self) -> bool {
        match self {
            Self::Izhikevich(n) => n.is_spiking(),
            Self::LeakyIzhikevich(n) => n.is_spiking(),
            Self::BCMIzhikevich(n) => n.is_spiking(),
            Self::HodgkinHuxley(n) => n.is_spiking(),
        }
    }

    fn last_spike_time(&self) -> Option<u64> {
        match self {
            Self::Izhikevich(n) => n.last_spike_time(),
            Self::LeakyIzhikevich(n) => n.last_spike_time(),
            Self::BCMIzhikevich(n) => n.last_spike_time(),
            Self::HodgkinHuxley(n) => n.last_spike_time(),
        }
    }

    fn model_name(&self) -> &'static str {
        match self {
            Self::Izhikevich(n) => n.model_name(),
            Self::LeakyIzhikevich(n) => n.model_name(),
            Self::BCMIzhikevich(n) => n.model_name(),
            Self::HodgkinHuxley(n) => n.model_name(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_facade_izhikevich() {
        let mut neuron = NeuronModel::izhikevich(IzhProfile::RegularSpiking);
        let axes = ChemicalAxes::baseline();

        assert_eq!(neuron.model_type(), "Izhikevich");
        assert!(neuron.is_izhikevich());
        assert!(!neuron.is_hodgkin_huxley());

        // Should be able to step
        let _ = neuron.step(5000, &axes);
    }

    #[test]
    fn test_facade_hodgkin_huxley() {
        let mut neuron = NeuronModel::hodgkin_huxley(HHProfile::Standard);
        let axes = ChemicalAxes::baseline();

        assert_eq!(neuron.model_type(), "HodgkinHuxley");
        assert!(!neuron.is_izhikevich());
        assert!(neuron.is_hodgkin_huxley());
        assert!(neuron.gating_state().is_some());

        let _ = neuron.step(5000, &axes);
    }
}
