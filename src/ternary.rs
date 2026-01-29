//! Signal - The fundamental unit of neural communication
//!
//! Re-exports `Signal` and `Polarity` from the canonical `ternary-signal` crate.
//! All methods (constructors, conversions, arithmetic, smooth transitions) live there.

// Canonical types from ternary-signal crate
pub use ternary_signal::{Signal, Polarity};

/// Deprecated alias for backwards compatibility
#[deprecated(since = "0.5.0", note = "Use Signal instead")]
pub type TernarySignal = Signal;

// The clamp_f32 helper is used by callers in this crate that predate Signal's
// own from_floats / from_signed methods.  Keeping it here avoids breaking them.
#[inline]
pub(crate) fn clamp_f32(v: f32, lo: f32, hi: f32) -> f32 {
    if v < lo { lo } else if v > hi { hi } else { v }
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
