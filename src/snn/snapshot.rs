//! Snapshot Types for Recovery
//!
//! Recovery MUST restore to the last cohesive balanced state of self.
//! NO region assumes its own independent state on recovery.

use crate::Signal;
use super::ChemicalAxes;
use serde::{Deserialize, Serialize};

/// Recovery state for different neuron models
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum RecoveryState {
    /// Izhikevich recovery variable (u)
    Izhikevich { u: Vec<i16> },
    /// BCM state (u, activity, theta)
    BCM {
        u: Vec<i16>,
        activity: Vec<u8>,
        theta: Vec<u8>,
    },
    /// HH gating states (m, h, n, s as u8 × 255)
    HodgkinHuxley {
        m: Vec<u8>,
        h: Vec<u8>,
        n: Vec<u8>,
        s: Vec<u8>,
    },
}

/// Model-specific snapshot data
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ModelSnapshot {
    /// Izhikevich-family snapshot
    Izhikevich {
        /// Profile identifier (not raw params)
        profile: String,
    },
    /// BCM-specific snapshot
    BCM {
        profile: String,
        /// BCM theta adaptation rate
        theta_rate: u8,
    },
    /// HH-specific snapshot
    HodgkinHuxley {
        profile: String,
        /// Excitability modifier (×128)
        excitability_mod: u8,
        /// Inhibition modifier (×128)
        inhibition_mod: u8,
    },
}

/// Per-SNN state (what each region's SNN stores)
///
/// NOTE: "weights" is forbidden terminology - these are conductances
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RegionSNNSnapshot {
    /// Membrane potentials (scaled i16, ×256)
    pub membranes: Vec<i16>,

    /// Recovery variables (Izhikevich u, or HH gating states)
    pub recovery_state: RecoveryState,

    /// Synaptic conductances (sparse: from, to, strength)
    /// NOTE: "weights" is forbidden terminology - these are conductances
    pub conductances: Vec<(u16, u16, i16)>,

    /// Chemical axes at snapshot time
    pub axes: ChemicalAxes,

    /// Current output Signals (the activation() values)
    pub outputs: Vec<Signal>,

    /// Model-specific state
    pub model_state: ModelSnapshot,

    /// Timestep at snapshot
    pub timestep: u64,

    /// Neuron count
    pub neuron_count: usize,

    /// Per-neuron cumulative spike counts (for evolutionary pruning decisions).
    /// Preserved across snapshots so evolutionary logic has full activity history.
    pub spike_counts: Vec<u32>,
}

impl RegionSNNSnapshot {
    /// Create empty snapshot for given neuron count
    pub fn empty(neuron_count: usize) -> Self {
        Self {
            membranes: vec![0; neuron_count],
            recovery_state: RecoveryState::Izhikevich {
                u: vec![0; neuron_count],
            },
            conductances: Vec::new(),
            axes: ChemicalAxes::baseline(),
            outputs: vec![Signal::ZERO; neuron_count],
            model_state: ModelSnapshot::Izhikevich {
                profile: "RegularSpiking".to_string(),
            },
            timestep: 0,
            neuron_count,
            spike_counts: vec![0; neuron_count],
        }
    }

    /// Get size estimate in bytes
    pub fn size_bytes(&self) -> usize {
        // Rough estimate
        self.membranes.len() * 2
            + match &self.recovery_state {
                RecoveryState::Izhikevich { u } => u.len() * 2,
                RecoveryState::BCM { u, activity, theta } => u.len() * 2 + activity.len() + theta.len(),
                RecoveryState::HodgkinHuxley { m, h, n, s } => m.len() + h.len() + n.len() + s.len(),
            }
            + self.conductances.len() * 6
            + self.outputs.len() * 2
            + self.spike_counts.len() * 4
            + 64 // overhead
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_snapshot_serialization() {
        let snapshot = RegionSNNSnapshot::empty(100);
        let json = serde_json::to_string(&snapshot).unwrap();
        let restored: RegionSNNSnapshot = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.neuron_count, 100);
    }

    #[test]
    fn test_size_estimate() {
        let snapshot = RegionSNNSnapshot::empty(256);
        let size = snapshot.size_bytes();
        assert!(size > 0);
        assert!(size < 10000); // Should be reasonable for 256 neurons
    }
}
