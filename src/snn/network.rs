//! SNN Network - Collection of neurons with synaptic connections
//!
//! A network wraps multiple neurons of the same type with sparse connectivity.

use crate::Signal;
use super::{
    ActivationMode, ChemicalAxes, HHProfile, IzhProfile, NeuronModel, RegionSNNSnapshot,
    RecoveryState, ModelSnapshot, SpikingNeuron, NeuronIntrospection,
};
use rand::Rng;
use serde::{Deserialize, Serialize};

/// Synapse (connection between neurons)
///
/// NOTE: We call these conductances, not weights
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Synapse {
    /// Pre-synaptic neuron index
    pub from: u16,
    /// Post-synaptic neuron index
    pub to: u16,
    /// Conductance strength (scaled i16)
    pub conductance: i16,
}

/// SNN Network configuration
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SNNConfig {
    /// Number of neurons (optional, can be specified at construction)
    pub neuron_count: Option<usize>,
    /// Ratio of excitatory neurons (0.0 to 1.0)
    pub excitatory_ratio: f32,
    /// Connection density (0.0 to 1.0)
    pub connection_density: f32,
    /// Conductance scale for initialization
    pub conductance_scale: f32,
    /// Timestep (ms)
    pub dt: f32,
    /// Optional Izhikevich profile override
    pub izh_profile: Option<IzhProfile>,
    /// Optional HH profile override
    pub hh_profile: Option<HHProfile>,
}

impl Default for SNNConfig {
    fn default() -> Self {
        Self {
            neuron_count: None,
            excitatory_ratio: 0.8,
            connection_density: 0.1,
            conductance_scale: 1.0,
            dt: 1.0,
            izh_profile: None,
            hh_profile: None,
        }
    }
}

impl SNNConfig {
    /// Create config with specified neuron count
    pub fn with_neuron_count(neuron_count: usize) -> Self {
        Self {
            neuron_count: Some(neuron_count),
            ..Default::default()
        }
    }
}

/// Network of spiking neurons
#[derive(Clone, Debug)]
pub struct SNNNetwork {
    /// Neurons in the network
    neurons: Vec<NeuronModel>,
    /// Synaptic connections (sparse)
    synapses: Vec<Synapse>,
    /// Output signals (cached)
    outputs: Vec<Signal>,
    /// Current chemical axes
    axes: ChemicalAxes,
    /// Configuration
    config: SNNConfig,
    /// Global timestep
    timestep: u64,
    /// Per-neuron cumulative spike count (for evolutionary pruning decisions).
    spike_counts: Vec<u32>,
}

impl SNNNetwork {
    /// Create new network with Izhikevich neurons (simple constructor)
    pub fn new_izhikevich(neuron_count: usize, profile: IzhProfile) -> Self {
        Self::new_izhikevich_with_config(SNNConfig::with_neuron_count(neuron_count), profile)
    }

    /// Create new network with Izhikevich neurons (full config)
    pub fn new_izhikevich_with_config(config: SNNConfig, profile: IzhProfile) -> Self {
        let mut rng = rand::thread_rng();
        let n = config.neuron_count.unwrap_or(100);

        // Create neurons
        let n_excitatory = (config.excitatory_ratio * n as f32) as usize;
        let mut neurons = Vec::with_capacity(n);

        for i in 0..n {
            let mut neuron = if i < n_excitatory {
                NeuronModel::izhikevich(profile)
            } else {
                // Inhibitory neurons are fast spiking
                NeuronModel::izhikevich(IzhProfile::FastSpiking)
            };
            neuron.set_dt(config.dt);
            neurons.push(neuron);
        }

        // Create random sparse connections
        let num_connections = (n * n) as f32 * config.connection_density;
        let mut synapses = Vec::with_capacity(num_connections as usize);

        for _ in 0..num_connections as usize {
            let from = rng.gen_range(0..n) as u16;
            let to = rng.gen_range(0..n) as u16;
            if from != to {
                // Inhibitory neurons have negative conductance
                let sign = if (from as usize) < n_excitatory {
                    1.0
                } else {
                    -1.0
                };
                let conductance =
                    (sign * rng.gen::<f32>() * config.conductance_scale * 256.0) as i16;
                synapses.push(Synapse {
                    from,
                    to,
                    conductance,
                });
            }
        }

        Self {
            outputs: vec![Signal::ZERO; n],
            spike_counts: vec![0; n],
            neurons,
            synapses,
            axes: ChemicalAxes::baseline(),
            config,
            timestep: 0,
        }
    }

    /// Create new network with LeakyIzhikevich neurons (simple constructor)
    pub fn new_leaky_izhikevich(neuron_count: usize, profile: IzhProfile) -> Self {
        Self::new_leaky_izhikevich_with_config(SNNConfig::with_neuron_count(neuron_count), profile)
    }

    /// Create new network with LeakyIzhikevich neurons (full config)
    pub fn new_leaky_izhikevich_with_config(config: SNNConfig, profile: IzhProfile) -> Self {
        let mut rng = rand::thread_rng();
        let n = config.neuron_count.unwrap_or(100);

        let n_excitatory = (config.excitatory_ratio * n as f32) as usize;
        let mut neurons = Vec::with_capacity(n);

        for i in 0..n {
            let mut neuron = if i < n_excitatory {
                NeuronModel::leaky_izhikevich(profile)
            } else {
                NeuronModel::leaky_izhikevich(IzhProfile::FastSpiking)
            };
            neuron.set_dt(config.dt);
            neurons.push(neuron);
        }

        let num_connections = (n * n) as f32 * config.connection_density;
        let mut synapses = Vec::with_capacity(num_connections as usize);

        for _ in 0..num_connections as usize {
            let from = rng.gen_range(0..n) as u16;
            let to = rng.gen_range(0..n) as u16;
            if from != to {
                let sign = if (from as usize) < n_excitatory {
                    1.0
                } else {
                    -1.0
                };
                let conductance =
                    (sign * rng.gen::<f32>() * config.conductance_scale * 256.0) as i16;
                synapses.push(Synapse {
                    from,
                    to,
                    conductance,
                });
            }
        }

        Self {
            outputs: vec![Signal::ZERO; n],
            spike_counts: vec![0; n],
            neurons,
            synapses,
            axes: ChemicalAxes::baseline(),
            config,
            timestep: 0,
        }
    }

    /// Create new network with BCM Izhikevich neurons (simple constructor)
    pub fn new_bcm_izhikevich(neuron_count: usize, profile: IzhProfile) -> Self {
        Self::new_bcm_izhikevich_with_config(SNNConfig::with_neuron_count(neuron_count), profile)
    }

    /// Create new network with BCM Izhikevich neurons (full config)
    pub fn new_bcm_izhikevich_with_config(config: SNNConfig, profile: IzhProfile) -> Self {
        let mut rng = rand::thread_rng();
        let n = config.neuron_count.unwrap_or(100);

        let n_excitatory = (config.excitatory_ratio * n as f32) as usize;
        let mut neurons = Vec::with_capacity(n);

        for i in 0..n {
            let mut neuron = if i < n_excitatory {
                NeuronModel::bcm_izhikevich(profile)
            } else {
                NeuronModel::bcm_izhikevich(IzhProfile::FastSpiking)
            };
            neuron.set_dt(config.dt);
            neurons.push(neuron);
        }

        let num_connections = (n * n) as f32 * config.connection_density;
        let mut synapses = Vec::with_capacity(num_connections as usize);

        for _ in 0..num_connections as usize {
            let from = rng.gen_range(0..n) as u16;
            let to = rng.gen_range(0..n) as u16;
            if from != to {
                let sign = if (from as usize) < n_excitatory {
                    1.0
                } else {
                    -1.0
                };
                let conductance =
                    (sign * rng.gen::<f32>() * config.conductance_scale * 256.0) as i16;
                synapses.push(Synapse {
                    from,
                    to,
                    conductance,
                });
            }
        }

        Self {
            outputs: vec![Signal::ZERO; n],
            spike_counts: vec![0; n],
            neurons,
            synapses,
            axes: ChemicalAxes::baseline(),
            config,
            timestep: 0,
        }
    }

    /// Create new network with Hodgkin-Huxley neurons (simple constructor)
    pub fn new_hodgkin_huxley(neuron_count: usize, profile: HHProfile) -> Self {
        Self::new_hodgkin_huxley_with_config(SNNConfig::with_neuron_count(neuron_count), profile)
    }

    /// Create new network with Hodgkin-Huxley neurons (full config)
    pub fn new_hodgkin_huxley_with_config(config: SNNConfig, profile: HHProfile) -> Self {
        let mut rng = rand::thread_rng();
        let n = config.neuron_count.unwrap_or(100);

        // HH networks are typically smaller due to computational cost
        let n_excitatory = (config.excitatory_ratio * n as f32) as usize;
        let mut neurons = Vec::with_capacity(n);

        for i in 0..n {
            let mut neuron = if i < n_excitatory {
                NeuronModel::hodgkin_huxley(profile)
            } else {
                NeuronModel::hodgkin_huxley(HHProfile::StrongInhibition)
            };
            // HH needs smaller dt
            neuron.set_dt(config.dt.min(0.1));
            neurons.push(neuron);
        }

        let num_connections = (n * n) as f32 * config.connection_density;
        let mut synapses = Vec::with_capacity(num_connections as usize);

        for _ in 0..num_connections as usize {
            let from = rng.gen_range(0..n) as u16;
            let to = rng.gen_range(0..n) as u16;
            if from != to {
                let sign = if (from as usize) < n_excitatory {
                    1.0
                } else {
                    -1.0
                };
                let conductance =
                    (sign * rng.gen::<f32>() * config.conductance_scale * 256.0) as i16;
                synapses.push(Synapse {
                    from,
                    to,
                    conductance,
                });
            }
        }

        Self {
            outputs: vec![Signal::ZERO; n],
            spike_counts: vec![0; n],
            neurons,
            synapses,
            axes: ChemicalAxes::baseline(),
            config,
            timestep: 0,
        }
    }

    /// Step the network with external input
    ///
    /// external_input: input current per neuron (scaled ×256)
    pub fn step(&mut self, external_input: &[i32]) -> &[Signal] {
        self.timestep += 1;
        let n = self.neurons.len();

        // Compute synaptic input from previous outputs
        let mut synaptic_input = vec![0i32; n];
        for synapse in &self.synapses {
            let pre_signal = self.outputs[synapse.from as usize];
            // Synaptic current = pre-activation * conductance
            let current = pre_signal.as_signed_i32().saturating_mul(synapse.conductance as i32) / 256;
            synaptic_input[synapse.to as usize] = synaptic_input[synapse.to as usize].saturating_add(current);
        }

        // Step each neuron
        for (i, neuron) in self.neurons.iter_mut().enumerate() {
            let ext = if i < external_input.len() {
                external_input[i]
            } else {
                0
            };
            let total_current = ext.saturating_add(synaptic_input[i]);

            let spiked = neuron.step(total_current, &self.axes);
            if spiked {
                self.spike_counts[i] = self.spike_counts[i].saturating_add(1);
            }
            self.outputs[i] = neuron.activation();
        }

        &self.outputs
    }

    /// Set chemical axes for the network
    pub fn set_axes(&mut self, axes: ChemicalAxes) {
        self.axes = axes;
    }

    /// Step axes toward target (rate-limited)
    pub fn step_axes_toward(&mut self, target: &ChemicalAxes, delta: i8) {
        self.axes.step_toward_by(target, delta);
    }

    /// Get current outputs
    pub fn outputs(&self) -> &[Signal] {
        &self.outputs
    }

    /// Get neuron count
    pub fn neuron_count(&self) -> usize {
        self.neurons.len()
    }

    /// Get synapse count
    pub fn synapse_count(&self) -> usize {
        self.synapses.len()
    }

    /// Get current timestep
    pub fn timestep(&self) -> u64 {
        self.timestep
    }

    /// Set activation mode for all neurons
    pub fn set_activation_mode(&mut self, mode: ActivationMode) {
        for neuron in &mut self.neurons {
            neuron.set_activation_mode(mode);
        }
    }

    /// Reset all neurons
    pub fn reset(&mut self) {
        for neuron in &mut self.neurons {
            neuron.reset();
        }
        for output in &mut self.outputs {
            *output = Signal::ZERO;
        }
        self.timestep = 0;
        self.axes = ChemicalAxes::baseline();
        for c in &mut self.spike_counts {
            *c = 0;
        }
    }

    /// Set dt for all neurons
    pub fn set_dt(&mut self, dt: f32) {
        for neuron in &mut self.neurons {
            neuron.set_dt(dt);
        }
    }

    /// Get current activations (alias for outputs)
    pub fn activations(&self) -> Vec<Signal> {
        self.outputs.clone()
    }

    /// Get mean activation across all neurons
    pub fn mean_activation(&self) -> Signal {
        if self.outputs.is_empty() {
            return Signal::ZERO;
        }

        let sum: i32 = self.outputs.iter().map(|s| s.as_signed_i32()).sum();
        let mean = sum / self.outputs.len() as i32;

        if mean > 0 {
            Signal::positive(mean.min(255) as u8)
        } else if mean < 0 {
            Signal::negative((-mean).min(255) as u8)
        } else {
            Signal::ZERO
        }
    }

    /// Step network with Signal inputs (converts to i32 internally)
    pub fn step_signals(&mut self, input_signals: &[Signal], axes: &ChemicalAxes) -> &[Signal] {
        self.axes = *axes;

        // Convert signals to i32 currents
        let currents: Vec<i32> = input_signals.iter().map(|s| s.as_signed_i32() * 256).collect();

        self.step(&currents)
    }

    // =========================================================================
    // Evolutionary methods (L2 — structural plasticity)
    // =========================================================================

    /// Per-neuron cumulative spike counts since last reset.
    /// Used by evolutionary logic to identify dead (zero-spike) neurons for pruning.
    pub fn neuron_activities(&self) -> &[u32] {
        &self.spike_counts
    }

    /// Reset all spike counts to zero. Called after evolutionary decisions are made.
    pub fn reset_activities(&mut self) {
        for c in &mut self.spike_counts {
            *c = 0;
        }
    }

    /// Grow the network by adding neurons. Preserves all existing state.
    ///
    /// New neurons start at rest potential with zero output and random synapses
    /// connecting them to the existing network at the configured connection density.
    /// Returns the new total neuron count.
    pub fn grow(&mut self, additional: usize) -> usize {
        if additional == 0 {
            return self.neurons.len();
        }

        let old_n = self.neurons.len();
        let new_n = old_n + additional;
        let n_excitatory = (self.config.excitatory_ratio * new_n as f32) as usize;

        // Determine model type from first existing neuron
        let dt = if self.neurons.is_empty() { self.config.dt } else { self.neurons[0].dt() };

        for i in old_n..new_n {
            let neuron = if self.neurons.is_empty() {
                // Fallback: default Izhikevich
                let mut n = NeuronModel::izhikevich(IzhProfile::RegularSpiking);
                n.set_dt(dt);
                n
            } else {
                // Clone model type from existing neurons
                let is_excitatory = i < n_excitatory;
                let template = &self.neurons[0];
                let mut n = match template {
                    NeuronModel::Izhikevich(_) => {
                        if is_excitatory {
                            NeuronModel::izhikevich(IzhProfile::RegularSpiking)
                        } else {
                            NeuronModel::izhikevich(IzhProfile::FastSpiking)
                        }
                    }
                    NeuronModel::LeakyIzhikevich(_) => {
                        if is_excitatory {
                            NeuronModel::leaky_izhikevich(IzhProfile::RegularSpiking)
                        } else {
                            NeuronModel::leaky_izhikevich(IzhProfile::FastSpiking)
                        }
                    }
                    NeuronModel::BCMIzhikevich(_) => {
                        if is_excitatory {
                            NeuronModel::bcm_izhikevich(IzhProfile::RegularSpiking)
                        } else {
                            NeuronModel::bcm_izhikevich(IzhProfile::FastSpiking)
                        }
                    }
                    NeuronModel::HodgkinHuxley(_) => {
                        NeuronModel::hodgkin_huxley(HHProfile::Standard)
                    }
                };
                n.set_dt(dt);
                n
            };

            self.neurons.push(neuron);
            self.outputs.push(Signal::ZERO);
            self.spike_counts.push(0);
        }

        // Add random synapses for new neurons at configured density
        let mut rng = rand::thread_rng();
        let new_connections = (additional as f32 * new_n as f32 * self.config.connection_density) as usize;
        for _ in 0..new_connections {
            // At least one end must be a new neuron
            let (from, to) = if rng.gen_bool(0.5) {
                // New neuron → any neuron
                (rng.gen_range(old_n..new_n) as u16, rng.gen_range(0..new_n) as u16)
            } else {
                // Any neuron → new neuron
                (rng.gen_range(0..new_n) as u16, rng.gen_range(old_n..new_n) as u16)
            };
            if from != to {
                let sign = if (from as usize) < n_excitatory { 1.0 } else { -1.0 };
                let conductance = (sign * rng.gen::<f32>() * self.config.conductance_scale * 256.0) as i16;
                self.synapses.push(Synapse { from, to, conductance });
            }
        }

        // Update config
        self.config.neuron_count = Some(new_n);
        new_n
    }

    /// Prune specified neurons from the network. Indices must be sorted ascending.
    ///
    /// Removes neurons, their synapses, and reindexes all remaining synapse endpoints.
    /// Floor: will not prune if it would leave fewer than 2 neurons.
    /// Returns the new total neuron count.
    pub fn prune(&mut self, indices: &[usize]) -> usize {
        if indices.is_empty() {
            return self.neurons.len();
        }

        let n = self.neurons.len();
        // Safety: never prune below 2 neurons
        if n.saturating_sub(indices.len()) < 2 {
            return n;
        }

        // Build a set for fast lookup and a remap table
        let mut remove_set = vec![false; n];
        for &idx in indices {
            if idx < n {
                remove_set[idx] = true;
            }
        }

        // Build remap: old_index → new_index (or usize::MAX for removed)
        let mut remap = vec![usize::MAX; n];
        let mut new_idx = 0usize;
        for old_idx in 0..n {
            if !remove_set[old_idx] {
                remap[old_idx] = new_idx;
                new_idx += 1;
            }
        }
        let new_n = new_idx;

        // Remove neurons (iterate in reverse to preserve indices)
        let mut new_neurons = Vec::with_capacity(new_n);
        let mut new_outputs = Vec::with_capacity(new_n);
        let mut new_spike_counts = Vec::with_capacity(new_n);
        for (i, neuron) in self.neurons.drain(..).enumerate() {
            if !remove_set[i] {
                new_neurons.push(neuron);
                new_outputs.push(self.outputs[i]);
                new_spike_counts.push(self.spike_counts[i]);
            }
        }
        self.neurons = new_neurons;
        self.outputs = new_outputs;
        self.spike_counts = new_spike_counts;

        // Reindex synapses, removing any that involve pruned neurons
        let mut new_synapses = Vec::with_capacity(self.synapses.len());
        for syn in &self.synapses {
            let from = syn.from as usize;
            let to = syn.to as usize;
            if from < n && to < n && remap[from] != usize::MAX && remap[to] != usize::MAX {
                new_synapses.push(Synapse {
                    from: remap[from] as u16,
                    to: remap[to] as u16,
                    conductance: syn.conductance,
                });
            }
        }
        self.synapses = new_synapses;

        // Update config
        self.config.neuron_count = Some(new_n);
        new_n
    }

    /// Restore from snapshot
    ///
    /// Note: This restores membrane potentials and outputs but not synapses
    /// (synapses are reconstructed from config on network creation).
    pub fn restore(&mut self, snapshot: &RegionSNNSnapshot) {
        // Restore outputs
        if snapshot.outputs.len() == self.outputs.len() {
            self.outputs = snapshot.outputs.clone();
        }

        // Restore axes
        self.axes = snapshot.axes;

        // Restore timestep
        self.timestep = snapshot.timestep;

        // Restore spike counts
        if snapshot.spike_counts.len() == self.spike_counts.len() {
            self.spike_counts = snapshot.spike_counts.clone();
        }

        // Note: Full membrane/recovery state restoration would require
        // internal access to neurons which we don't expose through the facade.
        // For now, we restore outputs, chemical state, and spike counts.
    }

    /// Create snapshot for recovery
    pub fn snapshot(&self) -> RegionSNNSnapshot {
        let n = self.neurons.len();

        // Collect membrane potentials
        let membranes: Vec<i16> = self
            .neurons
            .iter()
            .map(|n| n.membrane_i16())
            .collect();

        // Collect recovery state (depends on model type)
        let recovery_state = if self.neurons.is_empty() {
            RecoveryState::Izhikevich { u: Vec::new() }
        } else {
            match &self.neurons[0] {
                NeuronModel::Izhikevich(_)
                | NeuronModel::LeakyIzhikevich(_) => {
                    let u: Vec<i16> = self
                        .neurons
                        .iter()
                        .map(|n| (n.recovery() * 256.0) as i16)
                        .collect();
                    RecoveryState::Izhikevich { u }
                }
                NeuronModel::BCMIzhikevich(_) => {
                    let u: Vec<i16> = self
                        .neurons
                        .iter()
                        .map(|n| (n.recovery() * 256.0) as i16)
                        .collect();
                    // Would need to extract BCM-specific state
                    RecoveryState::BCM {
                        u,
                        activity: vec![0; n],
                        theta: vec![128; n], // 0.5 * 255
                    }
                }
                NeuronModel::HodgkinHuxley(_) => {
                    let mut m = Vec::with_capacity(n);
                    let mut h = Vec::with_capacity(n);
                    let mut nn = Vec::with_capacity(n);
                    let mut s = Vec::with_capacity(n);

                    for neuron in &self.neurons {
                        if let Some(gs) = neuron.gating_state() {
                            m.push((gs.m * 255.0) as u8);
                            h.push((gs.h * 255.0) as u8);
                            nn.push((gs.n * 255.0) as u8);
                            s.push((gs.s.unwrap_or(0.0) * 255.0) as u8);
                        }
                    }

                    RecoveryState::HodgkinHuxley { m, h, n: nn, s }
                }
            }
        };

        // Collect conductances
        let conductances: Vec<(u16, u16, i16)> = self
            .synapses
            .iter()
            .map(|s| (s.from, s.to, s.conductance))
            .collect();

        // Model snapshot
        let model_state = if self.neurons.is_empty() {
            ModelSnapshot::Izhikevich {
                profile: "RegularSpiking".to_string(),
            }
        } else {
            match &self.neurons[0] {
                NeuronModel::Izhikevich(_) | NeuronModel::LeakyIzhikevich(_) => {
                    ModelSnapshot::Izhikevich {
                        profile: "RegularSpiking".to_string(),
                    }
                }
                NeuronModel::BCMIzhikevich(_) => ModelSnapshot::BCM {
                    profile: "RegularSpiking".to_string(),
                    theta_rate: 1,
                },
                NeuronModel::HodgkinHuxley(_) => ModelSnapshot::HodgkinHuxley {
                    profile: "Standard".to_string(),
                    excitability_mod: 128,
                    inhibition_mod: 128,
                },
            }
        };

        RegionSNNSnapshot {
            membranes,
            recovery_state,
            conductances,
            axes: self.axes,
            outputs: self.outputs.clone(),
            model_state,
            timestep: self.timestep,
            neuron_count: n,
            spike_counts: self.spike_counts.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_network_creation() {
        let network = SNNNetwork::new_izhikevich(50, IzhProfile::RegularSpiking);

        assert_eq!(network.neuron_count(), 50);
        assert!(network.synapse_count() > 0);
    }

    #[test]
    fn test_network_step() {
        let mut network = SNNNetwork::new_izhikevich(10, IzhProfile::RegularSpiking);

        // Apply input
        let input = vec![5000i32; 10];
        let outputs = network.step(&input);

        assert_eq!(outputs.len(), 10);
    }

    #[test]
    fn test_snapshot() {
        let network = SNNNetwork::new_izhikevich(20, IzhProfile::RegularSpiking);
        let snapshot = network.snapshot();

        assert_eq!(snapshot.neuron_count, 20);
        assert_eq!(snapshot.membranes.len(), 20);
        assert_eq!(snapshot.outputs.len(), 20);
    }

    #[test]
    fn test_hh_network() {
        let mut network = SNNNetwork::new_hodgkin_huxley(5, HHProfile::Standard);

        // Step a few times
        let input = vec![3000i32; 5];
        for _ in 0..100 {
            network.step(&input);
        }

        assert_eq!(network.neuron_count(), 5);
    }
}
