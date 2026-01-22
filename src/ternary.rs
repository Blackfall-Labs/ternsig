//! TernarySignal - The fundamental unit of neural communication
//!
//! Compact 2-byte representation: polarity (-1, 0, +1) + magnitude (0-255).
//!
//! # Why Ternary?
//!
//! Biological neurons communicate through:
//! - **Polarity**: Excitatory (+1) or inhibitory (-1) or silent (0)
//! - **Magnitude**: Firing rate / signal strength
//!
//! This maps perfectly to a 2-byte representation that's cache-friendly
//! and avoids floating-point precision issues.
//!
//! # Example
//! ```
//! use ternsig::TernarySignal;
//!
//! // Strong positive signal
//! let excited = TernarySignal::positive(200);
//! assert!(excited.is_positive());
//! assert!(excited.magnitude_f32() > 0.7);
//!
//! // Weak negative signal
//! let inhibited = TernarySignal::negative(50);
//! assert!(inhibited.is_negative());
//!
//! // No signal
//! let neutral = TernarySignal::zero();
//! assert!(!neutral.is_active());
//! ```

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Ternary signal: polarity + magnitude
///
/// The fundamental unit of neural communication.
/// Compact 2-byte representation:
/// - polarity: -1 (inhibited), 0 (neutral), +1 (excited)
/// - magnitude: 0-255 (maps to 0.0-1.0 intensity)
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[repr(C)]
pub struct TernarySignal {
    /// Polarity: -1 (inhibited), 0 (neutral), +1 (excited)
    pub polarity: i8,
    /// Magnitude: 0-255 (intensity, maps to 0.0-1.0)
    pub magnitude: u8,
}

impl TernarySignal {
    /// Zero signal (no activity)
    pub const ZERO: Self = Self { polarity: 0, magnitude: 0 };

    /// Maximum positive signal
    pub const MAX_POSITIVE: Self = Self { polarity: 1, magnitude: 255 };

    /// Maximum negative signal
    pub const MAX_NEGATIVE: Self = Self { polarity: -1, magnitude: 255 };

    /// Create a zero/neutral signal
    #[inline]
    pub const fn zero() -> Self {
        Self::ZERO
    }

    /// Create a positive (excited) signal
    #[inline]
    pub const fn positive(magnitude: u8) -> Self {
        Self { polarity: 1, magnitude }
    }

    /// Create a negative (inhibited) signal
    #[inline]
    pub const fn negative(magnitude: u8) -> Self {
        Self { polarity: -1, magnitude }
    }

    /// Create from polarity and magnitude
    #[inline]
    pub const fn new(polarity: i8, magnitude: u8) -> Self {
        Self { polarity, magnitude }
    }

    /// Create from floating point values (compatibility layer)
    ///
    /// polarity_f: -1.0 to 1.0 (will be quantized to -1, 0, +1)
    /// magnitude_f: 0.0 to 1.0 (will be scaled to 0-255)
    #[inline]
    pub fn from_floats(polarity_f: f32, magnitude_f: f32) -> Self {
        let polarity = if polarity_f > 0.1 {
            1
        } else if polarity_f < -0.1 {
            -1
        } else {
            0
        };
        let magnitude = (magnitude_f.clamp(0.0, 1.0) * 255.0) as u8;
        Self { polarity, magnitude }
    }

    /// Create from a single signed float (-1.0 to 1.0)
    /// Sign becomes polarity, absolute value becomes magnitude
    /// DEPRECATED: Use from_signed_i32 for new code
    #[inline]
    pub fn from_signed(value: f32) -> Self {
        let polarity = if value > 0.01 {
            1
        } else if value < -0.01 {
            -1
        } else {
            0
        };
        let magnitude = (value.abs().min(1.0) * 255.0) as u8;
        Self { polarity, magnitude }
    }

    /// Create from a signed i32 value
    /// Sign becomes polarity, absolute value becomes magnitude (clamped to 0-255)
    #[inline]
    pub fn from_signed_i32(value: i32) -> Self {
        if value == 0 {
            Self::ZERO
        } else if value > 0 {
            Self {
                polarity: 1,
                magnitude: (value.min(255)) as u8,
            }
        } else {
            Self {
                polarity: -1,
                magnitude: ((-value).min(255)) as u8,
            }
        }
    }

    /// Get as signed i32 (polarity * magnitude)
    #[inline]
    pub fn as_signed_i32(&self) -> i32 {
        self.polarity as i32 * self.magnitude as i32
    }

    /// Get magnitude as float (0.0 to 1.0)
    #[inline]
    pub fn magnitude_f32(&self) -> f32 {
        self.magnitude as f32 / 255.0
    }

    /// Get as signed float (-1.0 to 1.0)
    #[inline]
    pub fn as_signed_f32(&self) -> f32 {
        self.polarity as f32 * self.magnitude_f32()
    }

    /// Is this signal active (non-zero)?
    #[inline]
    pub fn is_active(&self) -> bool {
        self.polarity != 0 && self.magnitude > 0
    }

    /// Is this a positive/excitatory signal?
    #[inline]
    pub fn is_positive(&self) -> bool {
        self.polarity > 0 && self.magnitude > 0
    }

    /// Is this a negative/inhibitory signal?
    #[inline]
    pub fn is_negative(&self) -> bool {
        self.polarity < 0 && self.magnitude > 0
    }

    /// Apply decay to magnitude (for temporal fields)
    #[inline]
    pub fn decay(&mut self, retention: f32) {
        let new_mag = (self.magnitude as f32 * retention) as u8;
        self.magnitude = new_mag;
        if self.magnitude == 0 {
            self.polarity = 0;
        }
    }

    /// Return a decayed copy
    #[inline]
    pub fn decayed(&self, retention: f32) -> Self {
        let mut copy = *self;
        copy.decay(retention);
        copy
    }

    /// Add two signals (same polarity adds magnitude, opposite cancels)
    #[inline]
    pub fn add(&self, other: &Self) -> Self {
        let combined = self.as_signed_f32() + other.as_signed_f32();
        Self::from_signed(combined)
    }

    /// Scale magnitude by a factor
    #[inline]
    pub fn scale(&self, factor: f32) -> Self {
        let new_mag = ((self.magnitude as f32 * factor).clamp(0.0, 255.0)) as u8;
        Self {
            polarity: if new_mag > 0 { self.polarity } else { 0 },
            magnitude: new_mag,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ternary_signal_creation() {
        let zero = TernarySignal::zero();
        assert!(!zero.is_active());
        assert_eq!(zero.polarity, 0);
        assert_eq!(zero.magnitude, 0);

        let pos = TernarySignal::positive(200);
        assert!(pos.is_positive());
        assert!(pos.is_active());
        assert!((pos.magnitude_f32() - 0.784).abs() < 0.01);

        let neg = TernarySignal::negative(128);
        assert!(neg.is_negative());
        assert!((neg.as_signed_f32() - (-0.502)).abs() < 0.01);
    }

    #[test]
    fn test_from_signed() {
        let from_pos = TernarySignal::from_signed(0.75);
        assert_eq!(from_pos.polarity, 1);
        assert!((from_pos.magnitude_f32() - 0.75).abs() < 0.01);

        let from_neg = TernarySignal::from_signed(-0.5);
        assert_eq!(from_neg.polarity, -1);
        assert!((from_neg.magnitude_f32() - 0.5).abs() < 0.01);

        let from_zero = TernarySignal::from_signed(0.005);
        assert_eq!(from_zero.polarity, 0);
    }

    #[test]
    fn test_decay() {
        let mut signal = TernarySignal::positive(200);
        signal.decay(0.5);
        assert_eq!(signal.magnitude, 100);
        assert!(signal.is_positive());

        // Decay to zero
        for _ in 0..10 {
            signal.decay(0.5);
        }
        assert!(!signal.is_active());
        assert_eq!(signal.polarity, 0);
    }

    #[test]
    fn test_add() {
        let a = TernarySignal::positive(100);
        let b = TernarySignal::positive(100);
        let sum = a.add(&b);
        assert!(sum.is_positive());
        assert!(sum.magnitude > 150); // ~200

        // Opposite polarities cancel
        let c = TernarySignal::negative(100);
        let cancel = a.add(&c);
        assert!(!cancel.is_active() || cancel.magnitude < 10);
    }

    #[test]
    fn test_size() {
        assert_eq!(std::mem::size_of::<TernarySignal>(), 2);
    }
}
