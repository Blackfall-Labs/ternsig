//! # Astromind SNN - Signal-Native Spiking Neural Networks
//!
//! Multi-model SNN substrate for Astromind cognitive regions.
//!
//! ## Models
//!
//! - **Izhikevich**: Standard 4-parameter model (a, b, c, d)
//! - **LeakyIzhikevich**: Izhikevich + leak term toward e_l
//! - **BCMIzhikevich**: Izhikevich + BCM activity-dependent plasticity
//! - **HodgkinHuxley**: Full ion channel model for deep chemical range
//!
//! ## Design Principles
//!
//! 1. **Signal output only**: All models output `Signal` (s = p × m)
//! 2. **Chemical axes**: 4 axes (Excitability, Inhibition, Persistence, Stress)
//!    instead of 30+ raw HH parameters
//! 3. **Named profiles**: Serializable identifiers, not raw params
//! 4. **Facade API**: External code uses only facade types
//! 5. **Microtime internal**: Models run at dt=0.1-1ms internally
//!
//! ## Region-to-Model Mapping
//!
//! | Region | Model | Why |
//! |--------|-------|-----|
//! | RegulationRegion | HodgkinHuxley | Deep neuromodulator response |
//! | PersonalityRegion | HodgkinHuxley | Emotional range |
//! | LearningRegion | BCMIzhikevich | Activity-dependent plasticity |
//! | PlanningRegion | BCMIzhikevich | Goal learning |
//! | DialogRegion | LeakyIzhikevich | Stable, fast |
//! | Others | Izhikevich | Efficient, stable |
//!
//! ## Example
//!
//! ```ignore
//! use astromind_snn::{NeuronModel, IzhProfile, ChemicalAxes, SpikingNeuron};
//!
//! let mut neuron = NeuronModel::izhikevich(IzhProfile::RegularSpiking);
//! let axes = ChemicalAxes::baseline();
//!
//! // Step with input current
//! let spiked = neuron.step(100, &axes);
//!
//! // Get Signal output (the ONLY output contract)
//! let activation = neuron.activation();
//! ```

// Re-export Signal from ternsig top-level (single source of truth)
pub use crate::{Signal, Polarity};

// Chemical modulation axes
mod axes;
pub use axes::ChemicalAxes;

// Core traits (facade API)
mod traits;
pub use traits::{SpikingNeuron, ActivationMode, NeuronIntrospection, GatingState};

// Named profiles (serializable)
mod profiles;
pub use profiles::{IzhProfile, HHProfile, ProfileId};

// Neuron models (internal - access through NeuronModel)
mod izhikevich;
mod hodgkin_huxley;

// Facade enum (the ONLY way to access models externally)
mod model;
pub use model::NeuronModel;

// Snapshot types for recovery
mod snapshot;
pub use snapshot::{RegionSNNSnapshot, RecoveryState, ModelSnapshot};

// Network of neurons (for regions)
mod network;
pub use network::{SNNNetwork, SNNConfig, Synapse};

// Attractor modules (workspace pattern completion)
mod attractor;
pub use attractor::{AttractorModule, HopfieldAttractor, RingAttractor};
