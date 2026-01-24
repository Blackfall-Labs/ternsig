//! Signal - The fundamental unit of neural communication
//!
//! Everything that flows is a Signal: `s = p × m`
//!
//! Compact 2-byte representation: polarity (-1, 0, +1) + magnitude (0-255).
//!
//! # Why Signal?
//!
//! Biological neurons communicate through:
//! - **Polarity**: Excitatory (+1) or inhibitory (-1) or silent (0)
//! - **Magnitude**: Firing rate / signal strength (0-255)
//!
//! This maps perfectly to a 2-byte representation that's cache-friendly
//! and avoids floating-point precision issues.
//!
//! # Example
//! ```
//! use ternsig::{Signal, Polarity};
//!
//! // Strong positive signal
//! let excited = Signal::positive(200);
//! assert!(excited.is_positive());
//! assert!(excited.magnitude_f32() > 0.7);
//!
//! // Using Polarity enum (type-safe)
//! let inhibited = Signal::with_polarity(Polarity::Negative, 50);
//! assert!(inhibited.is_negative());
//!
//! // No signal
//! let neutral = Signal::zero();
//! assert!(!neutral.is_active());
//! ```

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Polarity of a neural signal - strictly {-1, 0, +1}
///
/// Using this enum instead of raw i8 prevents invalid states like polarity=2.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[repr(i8)]
pub enum Polarity {
    /// Inhibitory signal
    Negative = -1,
    /// No signal / silent
    #[default]
    Zero = 0,
    /// Excitatory signal
    Positive = 1,
}

impl Polarity {
    /// Convert to i8
    #[inline]
    pub const fn as_i8(self) -> i8 {
        self as i8
    }

    /// Try to convert from i8, returns None for invalid values
    #[inline]
    pub const fn from_i8(value: i8) -> Option<Self> {
        match value {
            -1 => Some(Self::Negative),
            0 => Some(Self::Zero),
            1 => Some(Self::Positive),
            _ => None,
        }
    }

    /// Convert from i8, clamping invalid values to the nearest valid polarity
    #[inline]
    pub const fn from_i8_clamped(value: i8) -> Self {
        if value > 0 {
            Self::Positive
        } else if value < 0 {
            Self::Negative
        } else {
            Self::Zero
        }
    }

    /// Is this an active (non-zero) polarity?
    #[inline]
    pub const fn is_active(self) -> bool {
        !matches!(self, Self::Zero)
    }
}

impl From<Polarity> for i8 {
    fn from(p: Polarity) -> i8 {
        p.as_i8()
    }
}

impl TryFrom<i8> for Polarity {
    type Error = &'static str;

    fn try_from(value: i8) -> Result<Self, Self::Error> {
        Polarity::from_i8(value).ok_or("polarity must be -1, 0, or +1")
    }
}

/// Signal: polarity + magnitude (s = p × m)
///
/// The fundamental unit of neural communication.
/// Compact 2-byte representation:
/// - polarity: -1 (inhibited), 0 (neutral), +1 (excited)
/// - magnitude: 0-255 (maps to 0.0-1.0 intensity)
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[repr(C)]
pub struct Signal {
    /// Polarity: -1 (inhibited), 0 (neutral), +1 (excited)
    pub polarity: i8,
    /// Magnitude: 0-255 (intensity, maps to 0.0-1.0)
    pub magnitude: u8,
}

/// Deprecated alias for backwards compatibility
#[deprecated(since = "0.5.0", note = "Use Signal instead")]
pub type TernarySignal = Signal;

impl Signal {
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

    /// Create from polarity enum and magnitude (type-safe)
    #[inline]
    pub const fn with_polarity(polarity: Polarity, magnitude: u8) -> Self {
        Self { polarity: polarity.as_i8(), magnitude }
    }

    /// Create from raw i8 polarity and magnitude
    ///
    /// # Warning
    /// This does not validate that polarity is in {-1, 0, +1}.
    /// Prefer `with_polarity()` for type safety, or use `new_checked()`.
    #[inline]
    pub const fn new(polarity: i8, magnitude: u8) -> Self {
        Self { polarity, magnitude }
    }

    /// Create from raw i8 polarity with validation
    ///
    /// Returns None if polarity is not in {-1, 0, +1}.
    #[inline]
    pub const fn new_checked(polarity: i8, magnitude: u8) -> Option<Self> {
        match Polarity::from_i8(polarity) {
            Some(_) => Some(Self { polarity, magnitude }),
            None => None,
        }
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

    /// Create from spike rate (Hz) - bridges SNN microtime to Signal macrotime
    ///
    /// Converts a firing rate into a Signal:
    /// - rate = 0 Hz → neutral signal
    /// - rate > 0 → positive signal, magnitude scales with rate
    /// - max_rate defines what maps to magnitude 255 (typically 100 Hz)
    ///
    /// # Example
    /// ```ignore
    /// // 50 Hz out of max 100 Hz → positive signal with magnitude ~128
    /// let signal = Signal::from_spike_rate(50.0, 100.0);
    /// assert!(signal.is_positive());
    /// assert!(signal.magnitude > 100 && signal.magnitude < 150);
    /// ```
    #[inline]
    pub fn from_spike_rate(rate_hz: f32, max_rate_hz: f32) -> Self {
        if rate_hz <= 0.0 || max_rate_hz <= 0.0 {
            return Self::ZERO;
        }
        let normalized = (rate_hz / max_rate_hz).clamp(0.0, 1.0);
        let magnitude = (normalized * 255.0) as u8;
        if magnitude == 0 {
            Self::ZERO
        } else {
            Self::positive(magnitude)
        }
    }

    /// Create from spike count in a time window - SNN integration helper
    ///
    /// Given spike_count spikes in window_ms milliseconds,
    /// calculates rate and converts to Signal.
    ///
    /// # Example
    /// ```ignore
    /// // 5 spikes in 50ms window = 100 Hz rate
    /// let signal = Signal::from_spike_count(5, 50.0, 100.0);
    /// ```
    #[inline]
    pub fn from_spike_count(count: u32, window_ms: f32, max_rate_hz: f32) -> Self {
        if count == 0 || window_ms <= 0.0 {
            return Self::ZERO;
        }
        // rate = count / (window_ms / 1000) = count * 1000 / window_ms
        let rate_hz = count as f32 * 1000.0 / window_ms;
        Self::from_spike_rate(rate_hz, max_rate_hz)
    }

    /// Convert to spike rate (Hz) - inverse of from_spike_rate
    #[inline]
    pub fn to_spike_rate(&self, max_rate_hz: f32) -> f32 {
        if self.polarity == 0 || self.magnitude == 0 {
            return 0.0;
        }
        let normalized = self.magnitude as f32 / 255.0;
        normalized * max_rate_hz * self.polarity.signum() as f32
    }

    /// Get as signed i32 (polarity * magnitude)
    #[inline]
    pub fn as_signed_i32(&self) -> i32 {
        self.polarity as i32 * self.magnitude as i32
    }

    /// Create from i16 signed value (for intermediate calculations)
    ///
    /// Sign becomes polarity, absolute value becomes magnitude (clamped to 0-255)
    #[inline]
    pub fn from_i16(value: i16) -> Self {
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

    /// Create from u8 bipolar representation (0-255 maps to -127 to +128)
    ///
    /// Used for chemical levels where 128 = baseline, <128 = below, >128 = above.
    /// This converts a "level" into a "delta from baseline" signal.
    #[inline]
    pub fn from_u8_bipolar(level: u8) -> Self {
        let delta = level as i16 - 128;
        Self::from_i16(delta)
    }

    /// Convert to u8 bipolar representation (maps -127..+128 to 0..255)
    ///
    /// Inverse of from_u8_bipolar. 0 signal → 128, positive → >128, negative → <128.
    #[inline]
    pub fn to_u8_bipolar(&self) -> u8 {
        let signed = self.polarity as i16 * self.magnitude as i16;
        (signed + 128).clamp(0, 255) as u8
    }

    /// Get polarity as enum (type-safe)
    #[inline]
    pub fn get_polarity(&self) -> Polarity {
        // Safety: We trust that polarity is always valid.
        // If not, this returns Zero as fallback.
        Polarity::from_i8(self.polarity).unwrap_or(Polarity::Zero)
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

    // =========================================================================
    // Smooth Transition Methods (Signal Semantics)
    //
    // CRITICAL: Signals step toward targets, NEVER jump.
    // This enforces the fundamental constraint that neural signals are
    // continuous flows, not discrete events.
    // =========================================================================

    /// Step toward target by one unit
    ///
    /// Signals represent continuous neural flow. They MUST step toward
    /// targets smoothly, never teleport. This is enforced at the type level.
    ///
    /// # Example
    /// ```
    /// use ternsig::Signal;
    ///
    /// let current = Signal::positive(100);
    /// let target = Signal::positive(200);
    /// let next = current.step_toward(&target);
    /// assert_eq!(next.magnitude, 101); // Moved by 1
    /// ```
    #[inline]
    pub fn step_toward(&self, target: &Self) -> Self {
        self.step_toward_by(target, 1)
    }

    /// Step toward target by specified delta
    ///
    /// Larger delta = faster approach, but still smooth (no teleporting).
    /// Polarity changes happen smoothly through zero.
    #[inline]
    pub fn step_toward_by(&self, target: &Self, delta: u8) -> Self {
        // Convert to signed representation for smooth transitions
        let current = self.polarity as i16 * self.magnitude as i16;
        let target_val = target.polarity as i16 * target.magnitude as i16;

        let diff = target_val - current;

        if diff.abs() <= delta as i16 {
            // Close enough - snap to target
            *target
        } else if diff > 0 {
            // Need to increase (toward positive or less negative)
            let new_val = current + delta as i16;
            Self::from_signed_i32(new_val as i32)
        } else {
            // Need to decrease (toward negative or less positive)
            let new_val = current - delta as i16;
            Self::from_signed_i32(new_val as i32)
        }
    }

    /// Step toward target by fractional amount (0.0 to 1.0)
    ///
    /// Useful for rate-limited smooth transitions where you want
    /// proportional approach rather than fixed delta.
    #[inline]
    pub fn step_toward_ratio(&self, target: &Self, ratio: f32) -> Self {
        let current = self.polarity as i16 * self.magnitude as i16;
        let target_val = target.polarity as i16 * target.magnitude as i16;

        let diff = target_val - current;
        let delta = ((diff.abs() as f32 * ratio.clamp(0.0, 1.0)).ceil() as i16).max(1);

        self.step_toward_by(target, delta as u8)
    }

    /// Check if this signal has reached the target (within tolerance)
    #[inline]
    pub fn reached(&self, target: &Self, tolerance: u8) -> bool {
        let current = self.polarity as i16 * self.magnitude as i16;
        let target_val = target.polarity as i16 * target.magnitude as i16;
        (current - target_val).abs() <= tolerance as i16
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_signal_creation() {
        let zero = Signal::zero();
        assert!(!zero.is_active());
        assert_eq!(zero.polarity, 0);
        assert_eq!(zero.magnitude, 0);

        let pos = Signal::positive(200);
        assert!(pos.is_positive());
        assert!(pos.is_active());
        assert!((pos.magnitude_f32() - 0.784).abs() < 0.01);

        let neg = Signal::negative(128);
        assert!(neg.is_negative());
        assert!((neg.as_signed_f32() - (-0.502)).abs() < 0.01);
    }

    #[test]
    fn test_from_signed() {
        let from_pos = Signal::from_signed(0.75);
        assert_eq!(from_pos.polarity, 1);
        assert!((from_pos.magnitude_f32() - 0.75).abs() < 0.01);

        let from_neg = Signal::from_signed(-0.5);
        assert_eq!(from_neg.polarity, -1);
        assert!((from_neg.magnitude_f32() - 0.5).abs() < 0.01);

        let from_zero = Signal::from_signed(0.005);
        assert_eq!(from_zero.polarity, 0);
    }

    #[test]
    fn test_decay() {
        let mut signal = Signal::positive(200);
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
        let a = Signal::positive(100);
        let b = Signal::positive(100);
        let sum = a.add(&b);
        assert!(sum.is_positive());
        assert!(sum.magnitude > 150); // ~200

        // Opposite polarities cancel
        let c = Signal::negative(100);
        let cancel = a.add(&c);
        assert!(!cancel.is_active() || cancel.magnitude < 10);
    }

    #[test]
    fn test_size() {
        assert_eq!(std::mem::size_of::<Signal>(), 2);
    }

    #[test]
    fn test_polarity_enum() {
        assert_eq!(Polarity::Negative.as_i8(), -1);
        assert_eq!(Polarity::Zero.as_i8(), 0);
        assert_eq!(Polarity::Positive.as_i8(), 1);

        assert_eq!(Polarity::from_i8(-1), Some(Polarity::Negative));
        assert_eq!(Polarity::from_i8(0), Some(Polarity::Zero));
        assert_eq!(Polarity::from_i8(1), Some(Polarity::Positive));
        assert_eq!(Polarity::from_i8(2), None);
        assert_eq!(Polarity::from_i8(-5), None);

        assert_eq!(Polarity::from_i8_clamped(100), Polarity::Positive);
        assert_eq!(Polarity::from_i8_clamped(-50), Polarity::Negative);
        assert_eq!(Polarity::from_i8_clamped(0), Polarity::Zero);
    }

    #[test]
    fn test_with_polarity() {
        let pos = Signal::with_polarity(Polarity::Positive, 200);
        assert_eq!(pos.polarity, 1);
        assert_eq!(pos.magnitude, 200);

        let neg = Signal::with_polarity(Polarity::Negative, 100);
        assert_eq!(neg.polarity, -1);
        assert_eq!(neg.get_polarity(), Polarity::Negative);
    }

    #[test]
    fn test_new_checked() {
        assert!(Signal::new_checked(1, 100).is_some());
        assert!(Signal::new_checked(0, 0).is_some());
        assert!(Signal::new_checked(-1, 50).is_some());
        assert!(Signal::new_checked(2, 100).is_none());
        assert!(Signal::new_checked(-5, 100).is_none());
    }

    // =========================================================================
    // Smooth Transition Tests (Signal Semantics)
    // =========================================================================

    #[test]
    fn test_step_toward_same() {
        let signal = Signal::positive(100);
        let target = Signal::positive(100);
        let result = signal.step_toward(&target);
        assert_eq!(result.polarity, target.polarity);
        assert_eq!(result.magnitude, target.magnitude);
    }

    #[test]
    fn test_step_toward_increase() {
        let signal = Signal::positive(100);
        let target = Signal::positive(200);
        let result = signal.step_toward(&target);
        assert_eq!(result.polarity, 1);
        assert_eq!(result.magnitude, 101); // Increased by 1
    }

    #[test]
    fn test_step_toward_decrease() {
        let signal = Signal::positive(100);
        let target = Signal::positive(50);
        let result = signal.step_toward(&target);
        assert_eq!(result.polarity, 1);
        assert_eq!(result.magnitude, 99); // Decreased by 1
    }

    #[test]
    fn test_step_toward_polarity_change() {
        // Crossing zero must happen smoothly
        let signal = Signal::positive(2);
        let target = Signal::negative(100);

        // First step decreases toward zero
        let step1 = signal.step_toward(&target);
        assert_eq!(step1.polarity, 1);
        assert_eq!(step1.magnitude, 1);

        // Second step reaches zero
        let step2 = step1.step_toward(&target);
        assert_eq!(step2.polarity, 0);
        assert_eq!(step2.magnitude, 0);

        // Third step goes negative
        let step3 = step2.step_toward(&target);
        assert_eq!(step3.polarity, -1);
        assert_eq!(step3.magnitude, 1);
    }

    #[test]
    fn test_step_toward_by_delta() {
        let signal = Signal::positive(100);
        let target = Signal::positive(200);
        let result = signal.step_toward_by(&target, 10);
        assert_eq!(result.polarity, 1);
        assert_eq!(result.magnitude, 110); // Increased by 10
    }

    #[test]
    fn test_step_toward_snap_when_close() {
        let signal = Signal::positive(100);
        let target = Signal::positive(105);
        let result = signal.step_toward_by(&target, 10);
        // Should snap to target when difference < delta
        assert_eq!(result, target);
    }

    #[test]
    fn test_reached() {
        let signal = Signal::positive(100);
        let target = Signal::positive(102);

        assert!(!signal.reached(&target, 1));
        assert!(signal.reached(&target, 2));
        assert!(signal.reached(&target, 10));
    }

    #[test]
    fn test_step_toward_ratio() {
        let signal = Signal::positive(0);
        let target = Signal::positive(100);

        // 10% approach
        let result = signal.step_toward_ratio(&target, 0.1);
        assert_eq!(result.polarity, 1);
        assert!(result.magnitude >= 10 && result.magnitude <= 11);
    }

    // =========================================================================
    // Spike Rate Conversion Tests (SNN-Ternsig Bridge)
    // =========================================================================

    #[test]
    fn test_from_spike_rate_basic() {
        // 50 Hz out of max 100 Hz → ~50% magnitude
        let signal = Signal::from_spike_rate(50.0, 100.0);
        assert!(signal.is_positive());
        assert!(signal.magnitude > 100 && signal.magnitude < 150);

        // 100 Hz out of max 100 Hz → full magnitude
        let full = Signal::from_spike_rate(100.0, 100.0);
        assert_eq!(full.magnitude, 255);

        // 0 Hz → zero signal
        let zero = Signal::from_spike_rate(0.0, 100.0);
        assert!(!zero.is_active());
    }

    #[test]
    fn test_from_spike_rate_edge_cases() {
        // Negative rate → zero (invalid)
        let neg = Signal::from_spike_rate(-10.0, 100.0);
        assert!(!neg.is_active());

        // Zero max rate → zero (avoid division by zero)
        let zero_max = Signal::from_spike_rate(50.0, 0.0);
        assert!(!zero_max.is_active());

        // Rate exceeds max → clamped to 255
        let over = Signal::from_spike_rate(200.0, 100.0);
        assert_eq!(over.magnitude, 255);
    }

    #[test]
    fn test_from_spike_count() {
        // 10 spikes in 100ms = 100 Hz, max 200 Hz → ~50%
        let signal = Signal::from_spike_count(10, 100.0, 200.0);
        assert!(signal.is_positive());
        assert!(signal.magnitude > 100 && signal.magnitude < 150);

        // 0 spikes → zero
        let zero = Signal::from_spike_count(0, 100.0, 200.0);
        assert!(!zero.is_active());

        // 0 window → zero (avoid division by zero)
        let zero_window = Signal::from_spike_count(10, 0.0, 200.0);
        assert!(!zero_window.is_active());
    }

    #[test]
    fn test_to_spike_rate() {
        // Full positive → max rate
        let full = Signal::positive(255);
        assert!((full.to_spike_rate(100.0) - 100.0).abs() < 0.1);

        // Half magnitude → half rate
        let half = Signal::positive(128);
        assert!((half.to_spike_rate(100.0) - 50.2).abs() < 1.0);

        // Zero signal → 0 Hz
        let zero = Signal::zero();
        assert_eq!(zero.to_spike_rate(100.0), 0.0);

        // Negative signals give negative rate
        let neg = Signal::negative(128);
        let rate = neg.to_spike_rate(100.0);
        assert!(rate < 0.0);
        assert!((rate + 50.2).abs() < 1.0);
    }

    #[test]
    fn test_spike_rate_roundtrip() {
        // from_spike_rate → to_spike_rate should round-trip reasonably
        let original_rate = 75.0;
        let max_rate = 150.0;

        let signal = Signal::from_spike_rate(original_rate, max_rate);
        let recovered_rate = signal.to_spike_rate(max_rate);

        // Allow for quantization error (255 levels)
        assert!((recovered_rate - original_rate).abs() < 1.0);
    }
}
