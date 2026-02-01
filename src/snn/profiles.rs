//! Named Neuron Profiles - First-Class Config
//!
//! Profiles are SERIALIZABLE IDENTIFIERS, not raw parameter sets.
//! Raw params are dev-only. Cartridge never ships 30 knobs.
//!
//! ## Rule
//!
//! Cartridge config uses profile identifiers only:
//! ```toml
//! [regions.personality]
//! model = "HodgkinHuxley"
//! profile = "SlowIntegrating"
//! activation_mode = "Bipolar"
//! ```

use serde::{Deserialize, Serialize};

/// Izhikevich neuron profiles
///
/// Each profile maps to specific (a, b, c, d) parameters.
/// Chemical axes adjust these within bounded ranges.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum IzhProfile {
    /// Regular spiking (a=0.02, b=0.2, c=-65, d=8)
    /// Most common cortical excitatory neuron pattern
    #[default]
    RegularSpiking,

    /// Fast spiking (a=0.1, b=0.2, c=-65, d=2)
    /// Cortical inhibitory interneurons
    FastSpiking,

    /// Intrinsically bursting (a=0.02, b=0.2, c=-55, d=4)
    /// Layer 5 pyramidal neurons
    IntrinsicBursting,

    /// Chattering (a=0.02, b=0.2, c=-50, d=2)
    /// Layer 4 spiny stellate neurons
    Chattering,

    /// Low-threshold spiking (a=0.02, b=0.25, c=-65, d=2)
    /// Some GABAergic interneurons
    LowThreshold,

    /// Thalamic relay (a=0.02, b=0.25, c=-65, d=0.05)
    /// Thalamocortical relay cells
    ThalamicRelay,

    /// Resonator (a=0.1, b=0.26, c=-60, d=-1)
    /// Subthreshold oscillations
    Resonator,
}

impl IzhProfile {
    /// Get Izhikevich parameters (a, b, c, d)
    pub fn params(&self) -> (f32, f32, f32, f32) {
        match self {
            Self::RegularSpiking => (0.02, 0.2, -65.0, 8.0),
            Self::FastSpiking => (0.1, 0.2, -65.0, 2.0),
            Self::IntrinsicBursting => (0.02, 0.2, -55.0, 4.0),
            Self::Chattering => (0.02, 0.2, -50.0, 2.0),
            Self::LowThreshold => (0.02, 0.25, -65.0, 2.0),
            Self::ThalamicRelay => (0.02, 0.25, -65.0, 0.05),
            Self::Resonator => (0.1, 0.26, -60.0, -1.0),
        }
    }

    /// Human-readable description
    pub fn description(&self) -> &'static str {
        match self {
            Self::RegularSpiking => "Regular spiking - most common excitatory pattern",
            Self::FastSpiking => "Fast spiking - inhibitory interneurons",
            Self::IntrinsicBursting => "Intrinsically bursting - layer 5 pyramidal",
            Self::Chattering => "Chattering - spiny stellate neurons",
            Self::LowThreshold => "Low threshold - some GABAergic interneurons",
            Self::ThalamicRelay => "Thalamic relay - thalamocortical cells",
            Self::Resonator => "Resonator - subthreshold oscillations",
        }
    }
}

/// Hodgkin-Huxley neuron profiles
///
/// Each profile maps to specific ion channel configurations.
/// Chemical axes adjust within bounded ranges.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum HHProfile {
    /// Standard HH parameters (classic squid axon)
    #[default]
    Standard,

    /// High excitability (↑ g_Na, ↑ g_Ca)
    /// More sensitive to input, lower threshold
    HighExcitability,

    /// Strong inhibition (↑ g_GABA, ↑ g_K)
    /// Harder to excite, more stable
    StrongInhibition,

    /// Slow integrating (high NMDA/AMPA ratio)
    /// Longer integration window, good for persistence
    SlowIntegrating,

    /// Fast transient (low NMDA/AMPA ratio)
    /// Quick responses, short memory
    FastTransient,
}

impl HHProfile {
    /// Get base conductances (g_Na, g_K, g_leak, g_Ca)
    pub fn conductances(&self) -> (f32, f32, f32, f32) {
        match self {
            Self::Standard => (120.0, 36.0, 0.3, 0.025),
            Self::HighExcitability => (150.0, 30.0, 0.3, 0.05),
            Self::StrongInhibition => (100.0, 50.0, 0.4, 0.02),
            Self::SlowIntegrating => (110.0, 35.0, 0.25, 0.03),
            Self::FastTransient => (130.0, 40.0, 0.35, 0.02),
        }
    }

    /// Get NMDA/AMPA ratio (higher = slower integration)
    pub fn nmda_ampa_ratio(&self) -> f32 {
        match self {
            Self::Standard => 1.0,
            Self::HighExcitability => 0.8,
            Self::StrongInhibition => 1.2,
            Self::SlowIntegrating => 2.0,
            Self::FastTransient => 0.5,
        }
    }

    /// Human-readable description
    pub fn description(&self) -> &'static str {
        match self {
            Self::Standard => "Standard HH - classic parameters",
            Self::HighExcitability => "High excitability - sensitive, low threshold",
            Self::StrongInhibition => "Strong inhibition - stable, hard to excite",
            Self::SlowIntegrating => "Slow integrating - long memory, NMDA-dominant",
            Self::FastTransient => "Fast transient - quick responses, AMPA-dominant",
        }
    }
}

/// Profile identifier for serialization
///
/// Allows storing profile in config without raw parameters.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ProfileId {
    Izhikevich(IzhProfile),
    HodgkinHuxley(HHProfile),
}

impl From<IzhProfile> for ProfileId {
    fn from(p: IzhProfile) -> Self {
        ProfileId::Izhikevich(p)
    }
}

impl From<HHProfile> for ProfileId {
    fn from(p: HHProfile) -> Self {
        ProfileId::HodgkinHuxley(p)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_izh_params() {
        let (a, b, c, d) = IzhProfile::RegularSpiking.params();
        assert!((a - 0.02).abs() < 0.001);
        assert!((b - 0.2).abs() < 0.001);
        assert!((c - (-65.0)).abs() < 0.001);
        assert!((d - 8.0).abs() < 0.001);
    }

    #[test]
    fn test_hh_conductances() {
        let (g_na, g_k, g_l, g_ca) = HHProfile::Standard.conductances();
        assert!((g_na - 120.0).abs() < 0.001);
        assert!((g_k - 36.0).abs() < 0.001);
        assert!((g_l - 0.3).abs() < 0.001);
        assert!((g_ca - 0.025).abs() < 0.001); // Standard has small calcium
    }

    #[test]
    fn test_profile_id_serialization() {
        let id = ProfileId::Izhikevich(IzhProfile::FastSpiking);
        let json = serde_json::to_string(&id).unwrap();
        let restored: ProfileId = serde_json::from_str(&json).unwrap();
        assert_eq!(id, restored);
    }
}
