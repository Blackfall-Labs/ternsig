//! Thermogram Bridge for Ternsig VM
//!
//! Provides persistence for cold registers (Signal weights) via Thermogram.
//! Weights are stored with thermal state tracking (hot/warm/cold/frozen) and
//! survive restarts.
//!
//! ## Features
//!
//! - All cognitive state persists through crashes via Thermogram
//! - Weights stored as Signal (polarity + magnitude)
//! - Temperature lifecycle: HOT → WARM → COOL → COLD

use crate::vm::{ColdBuffer, Interpreter};
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use crate::Signal;
use thermogram::{Delta, PlasticityRule, ThermalConfig, Thermogram};

/// Content types stored in TensorThermogram
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum TensorContent {
    /// Weight matrix for a layer
    Weights(WeightContent),
    /// Program metadata
    ProgramMeta(ProgramMetaContent),
    /// Learning state (eligibility, etc.)
    LearningState(LearningStateContent),
}

impl TensorContent {
    pub fn key(&self) -> String {
        match self {
            TensorContent::Weights(w) => w.key.clone(),
            TensorContent::ProgramMeta(p) => format!("program.{}", p.name),
            TensorContent::LearningState(l) => format!("learning.{}", l.layer),
        }
    }
}

/// Weight matrix content
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WeightContent {
    /// Thermogram key (e.g., "chip.audio.layer1.weights")
    pub key: String,
    /// Shape of weight matrix
    pub shape: Vec<usize>,
    /// Polarities as i8 (-1, 0, +1)
    pub polarities: Vec<i8>,
    /// Magnitudes as u8 (0-255)
    pub magnitudes: Vec<u8>,
}

impl WeightContent {
    /// Create from ColdBuffer
    pub fn from_cold_buffer(key: &str, buffer: &ColdBuffer) -> Self {
        let polarities: Vec<i8> = buffer.weights.iter().map(|w| w.polarity).collect();
        let magnitudes: Vec<u8> = buffer.weights.iter().map(|w| w.magnitude).collect();

        Self {
            key: key.to_string(),
            shape: buffer.shape.clone(),
            polarities,
            magnitudes,
        }
    }

    /// Convert to Signal array
    pub fn to_ternary_signals(&self) -> Vec<Signal> {
        self.polarities
            .iter()
            .zip(self.magnitudes.iter())
            .map(|(&p, &m)| Signal {
                polarity: p,
                magnitude: m,
            })
            .collect()
    }
}

/// Program metadata content
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProgramMetaContent {
    pub name: String,
    pub version: u32,
    pub checksum: u64,
    pub register_count: usize,
    pub instruction_count: usize,
}

/// Learning state content
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LearningStateContent {
    pub layer: usize,
    pub eligibility_traces: Vec<f32>,
    pub update_count: u64,
    pub last_error: f32,
}

/// Thermal state for weight entries
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ThermalState {
    /// Just added, volatile (10%/tick decay)
    Hot,
    /// Confirmed, session-level (1%/tick decay)
    Warm,
    /// Consolidated, crystallized (0.1%/tick decay)
    Cool,
    /// Permanent, constitutional (0.01%/tick decay)
    Cold,
}

/// Thermogram bridge for TensorISA weight persistence
pub struct TensorThermogram {
    /// The actual thermogram substrate
    substrate: Thermogram,
    /// Tick counter for decay/consolidation
    tick: u64,
    /// Cache dirty flag
    cache_dirty: bool,
}

impl TensorThermogram {
    /// Create a new TensorThermogram
    pub fn new(name: impl Into<String>) -> Self {
        // Use STDP-like plasticity for network weights
        let plasticity = PlasticityRule::stdp_like();

        // Thermal config optimized for neural network weights
        let thermal_config = ThermalConfig {
            decay_rates: [0.10, 0.01, 0.001, 0.0001], // Hot→Warm→Cool→Cold
            promotion_thresholds: [0.5, 0.7, 0.85, 1.0],
            demotion_thresholds: [0.0, 0.2, 0.3, 0.4],
            min_observations: [5, 20, 100, usize::MAX],
            allow_demotion: [false, true, true, false],
            ..Default::default()
        };

        let substrate = Thermogram::with_thermal_config(name, plasticity, thermal_config);

        Self {
            substrate,
            tick: 0,
            cache_dirty: false,
        }
    }

    /// Load from file
    pub fn load(path: impl AsRef<std::path::Path>) -> Result<Self> {
        let substrate =
            Thermogram::load(&path).context("Failed to load TensorThermogram from file")?;

        Ok(Self {
            substrate,
            tick: 0,
            cache_dirty: false,
        })
    }

    /// Save to file
    pub fn save(&self, path: impl AsRef<std::path::Path>) -> Result<()> {
        self.substrate
            .save(&path)
            .context("Failed to save TensorThermogram")?;
        Ok(())
    }

    /// Load a cold register's weights from stored content
    pub fn load_weights(&self, key: &str) -> Result<Option<WeightContent>> {
        match self.substrate.read(key) {
            Ok(Some(data)) => {
                let content: TensorContent = serde_json::from_slice(&data)
                    .context("Failed to deserialize weight content")?;
                match content {
                    TensorContent::Weights(w) => Ok(Some(w)),
                    _ => Ok(None),
                }
            }
            Ok(None) => Ok(None),
            Err(e) => Err(anyhow::anyhow!("Thermogram read error: {:?}", e)),
        }
    }

    /// Store a cold register's weights
    pub fn store_weights(
        &mut self,
        key: &str,
        buffer: &ColdBuffer,
        strength: f32,
    ) -> Result<()> {
        let content = TensorContent::Weights(WeightContent::from_cold_buffer(key, buffer));
        let data = serde_json::to_vec(&content).context("Failed to serialize weight content")?;

        let prev_hash = self.substrate.dirty_chain.head_hash.clone();
        let mut delta = Delta::update(key, data, "tensor_isa", strength, prev_hash);
        delta.metadata.strength = strength.clamp(0.0, 1.0);

        self.substrate
            .apply_delta(delta)
            .map_err(|e| anyhow::anyhow!("Delta apply error: {:?}", e))?;
        self.cache_dirty = true;

        Ok(())
    }

    /// Load all cold registers for an interpreter
    pub fn load_interpreter_weights(&self, interpreter: &mut Interpreter) -> Result<usize> {
        let mut loaded = 0;

        for i in 0..16 {
            if let Some(buffer) = interpreter.cold_reg_mut(i) {
                if let Some(key) = &buffer.thermogram_key.clone() {
                    if let Some(weights) = self.load_weights(key)? {
                        // Verify shape matches
                        let expected_size: usize = buffer.shape.iter().product();
                        let stored_size: usize = weights.shape.iter().product();

                        if expected_size == stored_size {
                            buffer.weights = weights.to_ternary_signals();
                            loaded += 1;
                        } else {
                            log::warn!(
                                "Shape mismatch for {}: expected {}, got {}",
                                key,
                                expected_size,
                                stored_size
                            );
                        }
                    }
                }
            }
        }

        Ok(loaded)
    }

    /// Store all cold registers from an interpreter
    pub fn store_interpreter_weights(
        &mut self,
        interpreter: &Interpreter,
        strength: f32,
    ) -> Result<usize> {
        let mut stored = 0;

        for i in 0..16 {
            if let Some(buffer) = interpreter.cold_reg(i) {
                if let Some(key) = &buffer.thermogram_key {
                    self.store_weights(key, buffer, strength)?;
                    stored += 1;
                }
            }
        }

        Ok(stored)
    }

    /// Advance tick and apply decay/consolidation
    pub fn advance_tick(&mut self) {
        self.tick = self.tick.wrapping_add(1);

        // Apply decay every 10 ticks
        if self.tick % 10 == 0 {
            self.substrate.apply_decay();
            self.cache_dirty = true;
        }

        // Run thermal transitions every 100 ticks
        if self.tick % 100 == 0 {
            let _ = self.substrate.run_thermal_transitions();
            self.cache_dirty = true;
        }
    }

    /// Force consolidation (hot → cold)
    pub fn consolidate(&mut self) -> Result<()> {
        self.substrate
            .consolidate()
            .map_err(|e| anyhow::anyhow!("Consolidation error: {:?}", e))?;
        self.substrate
            .run_thermal_transitions()
            .map_err(|e| anyhow::anyhow!("Transition error: {:?}", e))?;
        self.cache_dirty = true;
        Ok(())
    }

    /// Get statistics
    pub fn stats(&self) -> thermogram::ThermogramStats {
        self.substrate.stats()
    }

    /// Get current tick
    pub fn tick(&self) -> u64 {
        self.tick
    }

    /// Check if cache is dirty
    pub fn is_dirty(&self) -> bool {
        self.cache_dirty
    }

    /// Clear dirty flag
    pub fn clear_dirty(&mut self) {
        self.cache_dirty = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_weight_content_roundtrip() {
        let buffer = ColdBuffer {
            weights: vec![
                Signal {
                    polarity: 1,
                    magnitude: 128,
                },
                Signal {
                    polarity: -1,
                    magnitude: 64,
                },
                Signal {
                    polarity: 0,
                    magnitude: 0,
                },
            ],
            shape: vec![3],
            thermogram_key: Some("test.weights".to_string()),
            frozen: false,
        };

        let content = WeightContent::from_cold_buffer("test.weights", &buffer);
        let recovered = content.to_ternary_signals();

        assert_eq!(recovered.len(), 3);
        assert_eq!(recovered[0].polarity, 1);
        assert_eq!(recovered[0].magnitude, 128);
        assert_eq!(recovered[1].polarity, -1);
        assert_eq!(recovered[1].magnitude, 64);
        assert_eq!(recovered[2].polarity, 0);
    }

    #[test]
    fn test_tensor_thermogram_creation() {
        let thermo = TensorThermogram::new("test_tensor");
        assert_eq!(thermo.tick(), 0);
        assert!(!thermo.is_dirty());
    }
}
