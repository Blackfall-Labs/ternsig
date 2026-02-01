//! Chemical Axes - Constrained Modulation Interface
//!
//! Instead of modulating 30+ HH parameters directly, we define 4 chemical axes
//! that map to parameters through continuous curves.
//!
//! ## Axes
//!
//! | Axis | What It Controls | Effect on HH |
//! |------|------------------|--------------|
//! | Excitability | How easily neurons fire | ↑ g_Na, ↑ g_Ca, ↓ threshold |
//! | Inhibition | Suppression of activity | ↑ g_GABA, ↑ g_K |
//! | Persistence | Integration vs transience | NMDA/AMPA ratio, τ_decay |
//! | Stress | Metabolic cost, noise | ↑ noise, faster decay |
//!
//! ## Rate Limiting
//!
//! Axes are RATE-LIMITED - they step toward targets, never teleport.
//! This prevents "stress jumps to 127" whiplash.

use serde::{Deserialize, Serialize};

/// Chemical axes - the constrained modulation interface
///
/// All values represent DEVIATION from baseline:
/// - 0 = baseline
/// - ±64 = strong
/// - ±127 = extreme
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChemicalAxes {
    /// How easily neurons fire
    /// Positive = more excitable, negative = less excitable
    pub excitability: i8,

    /// Suppression of activity
    /// Positive = more inhibition, negative = less inhibition
    pub inhibition: i8,

    /// Integration vs transience
    /// Positive = longer integration (NMDA-like), negative = faster transient (AMPA-like)
    pub persistence: i8,

    /// Metabolic cost and noise
    /// Positive = high stress (more noise, faster decay), negative = low stress
    pub stress: i8,
}

impl ChemicalAxes {
    /// Baseline (neutral) state
    pub const fn baseline() -> Self {
        Self {
            excitability: 0,
            inhibition: 0,
            persistence: 0,
            stress: 0,
        }
    }

    /// High excitability preset (dopamine-like)
    pub const fn high_excitability() -> Self {
        Self {
            excitability: 64,
            inhibition: -32,
            persistence: 0,
            stress: 0,
        }
    }

    /// High inhibition preset (GABA-like)
    pub const fn high_inhibition() -> Self {
        Self {
            excitability: -32,
            inhibition: 96,
            persistence: 0,
            stress: -16,
        }
    }

    /// High stress preset (cortisol-like)
    pub const fn high_stress() -> Self {
        Self {
            excitability: 32,
            inhibition: 32,
            persistence: -32,
            stress: 96,
        }
    }

    /// Step toward target by delta per tick (rate-limited, no teleporting)
    pub fn step_toward_by(&mut self, target: &ChemicalAxes, delta: i8) {
        self.excitability = step_i8(self.excitability, target.excitability, delta);
        self.inhibition = step_i8(self.inhibition, target.inhibition, delta);
        self.persistence = step_i8(self.persistence, target.persistence, delta);
        self.stress = step_i8(self.stress, target.stress, delta);
    }

    /// Step toward target with default delta (1 unit per step)
    pub fn step_toward(&mut self, target: &ChemicalAxes) {
        self.step_toward_by(target, 1);
    }

    /// Get excitability as normalized f32 (-1.0 to 1.0) for internal math
    pub fn excitability_f32(&self) -> f32 {
        self.excitability as f32 / 127.0
    }

    /// Get inhibition as normalized f32 (-1.0 to 1.0) for internal math
    pub fn inhibition_f32(&self) -> f32 {
        self.inhibition as f32 / 127.0
    }

    /// Get persistence as normalized f32 (-1.0 to 1.0) for internal math
    pub fn persistence_f32(&self) -> f32 {
        self.persistence as f32 / 127.0
    }

    /// Get stress as normalized f32 (-1.0 to 1.0) for internal math
    pub fn stress_f32(&self) -> f32 {
        self.stress as f32 / 127.0
    }

    /// Compute background current modifier from axes
    /// Higher excitability = more spontaneous activity
    pub fn background_current_modifier(&self) -> f32 {
        // Excitability increases background, inhibition decreases it
        let base = 1.0 + 0.5 * self.excitability_f32() - 0.3 * self.inhibition_f32();
        base.clamp(0.1, 3.0)
    }

    /// Compute noise level from stress
    pub fn noise_level(&self) -> f32 {
        // Higher stress = more noise
        let base = 0.1 + 0.2 * self.stress_f32().max(0.0);
        base.clamp(0.0, 0.5)
    }

    /// Compute decay modifier from persistence
    /// Lower persistence = faster decay
    pub fn decay_modifier(&self) -> f32 {
        // Persistence reduces decay rate
        let base = 1.0 - 0.3 * self.persistence_f32();
        base.clamp(0.5, 1.5)
    }
}

/// Step an i8 value toward target by at most delta
#[inline]
fn step_i8(current: i8, target: i8, delta: i8) -> i8 {
    let diff = target.saturating_sub(current);
    // Use i16 for abs to avoid overflow when diff == i8::MIN (-128)
    let abs_diff = (diff as i16).abs() as u8;
    if abs_diff <= delta as u8 {
        target
    } else if diff > 0 {
        current.saturating_add(delta)
    } else {
        current.saturating_sub(delta)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_step_toward() {
        let mut axes = ChemicalAxes::baseline();
        let target = ChemicalAxes::high_excitability();

        // Should not teleport
        axes.step_toward_by(&target, 10);
        assert_eq!(axes.excitability, 10);
        assert_ne!(axes, target);

        // Keep stepping
        for _ in 0..10 {
            axes.step_toward_by(&target, 10);
        }

        // Should eventually reach target
        assert_eq!(axes.excitability, 64);
    }

    #[test]
    fn test_baseline() {
        let axes = ChemicalAxes::baseline();
        assert_eq!(axes.excitability, 0);
        assert_eq!(axes.inhibition, 0);
        assert_eq!(axes.persistence, 0);
        assert_eq!(axes.stress, 0);
    }

    #[test]
    fn test_normalized_f32() {
        let axes = ChemicalAxes {
            excitability: 127,
            inhibition: -127,
            persistence: 0,
            stress: 64,
        };

        assert!((axes.excitability_f32() - 1.0).abs() < 0.01);
        assert!((axes.inhibition_f32() - (-1.0)).abs() < 0.01);
        assert!((axes.persistence_f32() - 0.0).abs() < 0.01);
        assert!((axes.stress_f32() - 0.504).abs() < 0.01);
    }
}
