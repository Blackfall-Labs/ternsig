//! Thermogram Bridge for Ternsig VM
//!
//! Provides persistence for cold registers (Signal weights) via Thermogram.
//! Weights are stored as `Vec<Signal>` directly — no JSON serialization.
//!
//! ## Features
//!
//! - All cognitive state persists through crashes via Thermogram
//! - Weights stored as Signal (polarity + magnitude), 2 bytes each
//! - Temperature lifecycle: HOT → WARM → COOL → COLD

use crate::vm::{ColdBuffer, Interpreter};
use anyhow::{Context, Result};
use crate::Signal;
use thermogram::{Delta, PlasticityRule, ThermalConfig, Thermogram};

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
            decay_rates: [
                Signal::positive(26),  // Hot: ~0.10
                Signal::positive(3),   // Warm: ~0.01
                Signal::positive(1),   // Cool: ~0.001 (min non-zero)
                Signal::positive(1),   // Cold: ~0.0001 (min non-zero)
            ],
            promotion_thresholds: [
                Signal::positive(128), // 0.5
                Signal::positive(179), // 0.7
                Signal::positive(217), // 0.85
                Signal::positive(255), // 1.0
            ],
            demotion_thresholds: [
                Signal::zero(),        // 0.0
                Signal::positive(51),  // 0.2
                Signal::positive(77),  // 0.3
                Signal::positive(102), // 0.4
            ],
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
    ///
    /// Returns the stored signals directly (no deserialization needed).
    pub fn load_weights(&self, key: &str) -> Result<Option<Vec<Signal>>> {
        match self.substrate.read(key) {
            Ok(signals) => Ok(signals),
            Err(e) => Err(anyhow::anyhow!("Thermogram read error: {:?}", e)),
        }
    }

    /// Store a cold register's weights
    ///
    /// Stores the weight signals directly into the thermogram entry.
    pub fn store_weights(
        &mut self,
        key: &str,
        buffer: &ColdBuffer,
        strength: Signal,
    ) -> Result<()> {
        let signals: Vec<Signal> = buffer.weights.clone();

        let prev_hash = self.substrate.dirty_chain.head_hash.clone();
        let mut delta = Delta::update(key, signals, "tensor_isa", strength, prev_hash);
        delta.metadata.strength = strength;

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
                    if let Some(signals) = self.load_weights(key)? {
                        // Verify shape matches
                        let expected_size: usize = buffer.shape.iter().product();

                        if expected_size == signals.len() {
                            buffer.weights = signals;
                            loaded += 1;
                        } else {
                            log::warn!(
                                "Shape mismatch for {}: expected {}, got {}",
                                key,
                                expected_size,
                                signals.len()
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
        strength: Signal,
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
    fn test_tensor_thermogram_creation() {
        let thermo = TensorThermogram::new("test_tensor");
        assert_eq!(thermo.tick(), 0);
        assert!(!thermo.is_dirty());
    }

    #[test]
    fn test_store_load_weights() {
        let mut thermo = TensorThermogram::new("test_store_load");

        let mut buffer = ColdBuffer::new(vec![3]);
        buffer.weights[0] = Signal::positive(128);
        buffer.weights[1] = Signal::negative(64);
        buffer.weights[2] = Signal::zero();
        buffer.thermogram_key = Some("test.weights".to_string());

        // Store
        thermo
            .store_weights("test.weights", &buffer, Signal::positive(200))
            .unwrap();

        // Consolidate so read can find it
        thermo.consolidate().unwrap();

        // Load
        let loaded = thermo.load_weights("test.weights").unwrap();
        assert!(loaded.is_some());
        let signals = loaded.unwrap();
        assert_eq!(signals.len(), 3);
        assert_eq!(signals[0].polarity, 1);
        assert_eq!(signals[0].magnitude, 128);
        assert_eq!(signals[1].polarity, -1);
        assert_eq!(signals[1].magnitude, 64);
        assert_eq!(signals[2].polarity, 0);
    }
}
