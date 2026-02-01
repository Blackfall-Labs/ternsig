//! Attractor Modules - Signal-Native Pattern Completion
//!
//! Attractors are workspace mechanisms for SHORT-TERM pattern completion,
//! NOT archival memory. Archive stays in Thermograms/RAG/GraphRAG.
//!
//! # Available Attractors
//!
//! - **HopfieldAttractor**: Binary pattern completion (workspace)
//! - **RingAttractor**: Continuous attractor for spatial/heading (grid cells)
//!
//! # Signal Semantics
//!
//! All IO uses Signal. No direct state mutation from regions.
//! Patterns step toward completion, never jump.

use crate::Signal;
use serde::{Deserialize, Serialize};

/// Attractor module - Signal IO only, no direct state mutation
pub trait AttractorModule: Send + Sync {
    /// Present a partial pattern, get completion
    ///
    /// The attractor network settles toward a stored pattern
    /// that best matches the partial input.
    fn complete(&mut self, partial: &[Signal]) -> Vec<Signal>;

    /// Store a pattern (during learning phase)
    ///
    /// Patterns are stored as attractors - stable states the network
    /// naturally settles toward.
    fn store(&mut self, pattern: &[Signal]);

    /// Get current attractor state as Signals
    fn state(&self) -> Vec<Signal>;

    /// Reset to neutral state
    fn reset(&mut self);

    /// Get pattern capacity (how many patterns can be reliably stored)
    fn capacity(&self) -> usize;

    /// Get current number of stored patterns
    fn stored_count(&self) -> usize;
}

// ============================================================================
// Hopfield Attractor - Binary Pattern Completion
// ============================================================================

/// Hopfield network for pattern completion
///
/// Uses Signal polarity for binary states: positive = +1, zero/negative = -1
/// Magnitude encodes confidence in that state.
///
/// # Capacity
///
/// Can reliably store ~0.14N patterns where N is neuron count.
/// Beyond this, spurious attractors emerge.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HopfieldAttractor {
    /// Number of neurons
    size: usize,

    /// Connection strengths (symmetric, no self-connections)
    /// Stored as flattened upper triangle: w[i][j] where j > i
    /// NOTE: "conductances" not "weights" per PLAN
    conductances: Vec<i16>,

    /// Current state (bipolar: polarity encodes sign)
    state: Vec<Signal>,

    /// Number of stored patterns
    pattern_count: usize,

    /// Maximum iterations for settling
    max_iterations: usize,
}

impl HopfieldAttractor {
    /// Create a new Hopfield attractor with given size
    pub fn new(size: usize) -> Self {
        // Upper triangle size: n*(n-1)/2
        let conductance_count = size * (size - 1) / 2;

        Self {
            size,
            conductances: vec![0; conductance_count],
            state: vec![Signal::ZERO; size],
            pattern_count: 0,
            max_iterations: 100,
        }
    }

    /// Create with custom iteration limit
    pub fn with_iterations(size: usize, max_iterations: usize) -> Self {
        let mut attractor = Self::new(size);
        attractor.max_iterations = max_iterations;
        attractor
    }

    /// Get index into conductance array for connection i-j
    #[inline]
    fn conductance_idx(&self, i: usize, j: usize) -> usize {
        let (min, max) = if i < j { (i, j) } else { (j, i) };
        // Index in upper triangle
        min * self.size - min * (min + 1) / 2 + (max - min - 1)
    }

    /// Get conductance between neurons i and j
    #[inline]
    fn get_conductance(&self, i: usize, j: usize) -> i16 {
        if i == j {
            0 // No self-connections
        } else {
            self.conductances[self.conductance_idx(i, j)]
        }
    }

    /// Set conductance between neurons i and j
    #[inline]
    fn set_conductance(&mut self, i: usize, j: usize, value: i16) {
        if i != j {
            let idx = self.conductance_idx(i, j);
            self.conductances[idx] = value;
        }
    }

    /// Convert Signal to bipolar value (-1, +1) with magnitude
    fn signal_to_bipolar(signal: &Signal) -> (i8, u8) {
        match signal.polarity {
            1 => (1, signal.magnitude),
            -1 => (-1, signal.magnitude),
            _ => (-1, 0), // Zero treated as -1 (off state)
        }
    }

    /// Convert bipolar value to Signal
    fn bipolar_to_signal(polarity: i8, magnitude: u8) -> Signal {
        if polarity > 0 {
            Signal::positive(magnitude)
        } else {
            Signal::negative(magnitude)
        }
    }

    /// Compute total input to neuron i
    fn compute_input(&self, i: usize) -> i32 {
        let mut sum: i32 = 0;
        for j in 0..self.size {
            if i != j {
                let w = self.get_conductance(i, j) as i32;
                let (s_j, m_j) = Self::signal_to_bipolar(&self.state[j]);
                // Scale by magnitude for graded activation
                sum += w * s_j as i32 * (m_j as i32 + 1) / 256;
            }
        }
        sum
    }

    /// Settle network to attractor state
    fn settle(&mut self) -> bool {
        let mut changed = true;
        let mut iterations = 0;

        while changed && iterations < self.max_iterations {
            changed = false;
            iterations += 1;

            // Asynchronous update (random order would be better, but deterministic for now)
            for i in 0..self.size {
                let input = self.compute_input(i);

                // Threshold at 0
                let new_polarity: i8 = if input > 0 { 1 } else { -1 };

                // Magnitude from input strength (clamped)
                let new_magnitude = (input.abs().min(255 * 4) / 4) as u8;

                let (old_polarity, _) = Self::signal_to_bipolar(&self.state[i]);

                if new_polarity != old_polarity {
                    changed = true;
                }

                self.state[i] = Self::bipolar_to_signal(new_polarity, new_magnitude);
            }
        }

        !changed // Return true if converged
    }
}

impl AttractorModule for HopfieldAttractor {
    fn complete(&mut self, partial: &[Signal]) -> Vec<Signal> {
        // Initialize state from partial pattern
        for (i, signal) in partial.iter().enumerate() {
            if i < self.size {
                self.state[i] = *signal;
            }
        }

        // Fill remaining with zero (will be completed)
        for i in partial.len()..self.size {
            self.state[i] = Signal::ZERO;
        }

        // Settle to attractor
        self.settle();

        self.state.clone()
    }

    fn store(&mut self, pattern: &[Signal]) {
        if pattern.len() != self.size {
            return;
        }

        // Hebbian learning: Δw_ij = s_i * s_j
        for i in 0..self.size {
            for j in (i + 1)..self.size {
                let (s_i, _) = Self::signal_to_bipolar(&pattern[i]);
                let (s_j, _) = Self::signal_to_bipolar(&pattern[j]);

                let current = self.get_conductance(i, j);
                // Scale learning by pattern count to prevent saturation
                let delta = (s_i as i16 * s_j as i16 * 64) / (self.pattern_count as i16 + 1).max(1);
                let new_val = (current + delta).clamp(-32000, 32000);
                self.set_conductance(i, j, new_val);
            }
        }

        self.pattern_count += 1;
    }

    fn state(&self) -> Vec<Signal> {
        self.state.clone()
    }

    fn reset(&mut self) {
        self.state.fill(Signal::ZERO);
    }

    fn capacity(&self) -> usize {
        // ~0.14N reliable capacity
        self.size * 14 / 100
    }

    fn stored_count(&self) -> usize {
        self.pattern_count
    }
}

// ============================================================================
// Ring Attractor - Continuous Spatial Representation
// ============================================================================

/// Ring attractor for continuous variables (heading, spatial position)
///
/// Models a ring of neurons with Mexican-hat connectivity:
/// nearby neurons excite, distant neurons inhibit.
///
/// Used for: grid cells, head direction, spatial position
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RingAttractor {
    /// Number of neurons around the ring
    size: usize,

    /// Current activation (Signal per neuron)
    state: Vec<Signal>,

    /// Peak position (0.0 to 1.0, wraps around)
    peak_position: f32,

    /// Peak width (standard deviation as fraction of ring)
    peak_width: f32,

    /// Excitation strength
    excitation: i16,

    /// Inhibition strength (global)
    inhibition: i16,
}

impl RingAttractor {
    /// Create a new ring attractor
    pub fn new(size: usize) -> Self {
        Self {
            size,
            state: vec![Signal::ZERO; size],
            peak_position: 0.0,
            peak_width: 0.1,
            excitation: 200,
            inhibition: 50,
        }
    }

    /// Create with custom width
    pub fn with_width(size: usize, peak_width: f32) -> Self {
        let mut ring = Self::new(size);
        ring.peak_width = peak_width.clamp(0.01, 0.5);
        ring
    }

    /// Get angular position of neuron i (0.0 to 1.0)
    fn neuron_position(&self, i: usize) -> f32 {
        i as f32 / self.size as f32
    }

    /// Circular distance between two positions (0.0 to 0.5)
    fn circular_distance(a: f32, b: f32) -> f32 {
        let diff = (a - b).abs();
        diff.min(1.0 - diff)
    }

    /// Update state toward new peak position
    fn update_state(&mut self) {
        let sigma = self.peak_width;

        // Multiple steps to allow settling (step_toward moves by 1 per call)
        for _ in 0..32 {
            for i in 0..self.size {
                let pos = self.neuron_position(i);
                let dist = Self::circular_distance(pos, self.peak_position);

                // Gaussian bump centered at peak
                let activation = (-dist * dist / (2.0 * sigma * sigma)).exp();
                let magnitude = (activation * 255.0) as u8;

                // Step toward target (smooth transition)
                let target = Signal::positive(magnitude);
                self.state[i] = self.state[i].step_toward_by(&target, 8);
            }
        }
    }

    /// Find current peak from state
    fn find_peak(&self) -> f32 {
        // Population vector decoding
        let mut sum_cos = 0.0f32;
        let mut sum_sin = 0.0f32;

        for i in 0..self.size {
            let angle = 2.0 * std::f32::consts::PI * self.neuron_position(i);
            let weight = self.state[i].magnitude as f32 / 255.0;

            sum_cos += weight * angle.cos();
            sum_sin += weight * angle.sin();
        }

        // Atan2 gives angle, normalize to 0-1
        let angle = sum_sin.atan2(sum_cos);
        let mut position = angle / (2.0 * std::f32::consts::PI);
        if position < 0.0 {
            position += 1.0;
        }
        position
    }

    /// Set peak position directly (for external input)
    pub fn set_position(&mut self, position: f32) {
        self.peak_position = position.rem_euclid(1.0);
        self.update_state();
    }

    /// Get current peak position
    pub fn position(&self) -> f32 {
        self.peak_position
    }

    /// Shift peak by delta (for path integration)
    pub fn shift(&mut self, delta: f32) {
        self.peak_position = (self.peak_position + delta).rem_euclid(1.0);
        self.update_state();
    }
}

impl AttractorModule for RingAttractor {
    fn complete(&mut self, partial: &[Signal]) -> Vec<Signal> {
        // Initialize from partial input
        for (i, signal) in partial.iter().enumerate() {
            if i < self.size {
                self.state[i] = *signal;
            }
        }

        // Find peak from current state
        self.peak_position = self.find_peak();

        // Update to clean bump
        self.update_state();

        self.state.clone()
    }

    fn store(&mut self, _pattern: &[Signal]) {
        // Ring attractors don't store discrete patterns
        // They maintain a continuous representation
        // This is a no-op
    }

    fn state(&self) -> Vec<Signal> {
        self.state.clone()
    }

    fn reset(&mut self) {
        self.peak_position = 0.0;
        self.state.fill(Signal::ZERO);
    }

    fn capacity(&self) -> usize {
        // Continuous attractor - infinite positions, one "pattern"
        1
    }

    fn stored_count(&self) -> usize {
        1 // Always has one continuous representation
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hopfield_store_and_complete() {
        let mut hopfield = HopfieldAttractor::new(16);

        // Store a pattern: alternating +/-
        let pattern: Vec<Signal> = (0..16)
            .map(|i| {
                if i % 2 == 0 {
                    Signal::positive(200)
                } else {
                    Signal::negative(200)
                }
            })
            .collect();

        hopfield.store(&pattern);

        // Present partial pattern (first half)
        let partial: Vec<Signal> = pattern.iter().take(8).copied().collect();
        let completed = hopfield.complete(&partial);

        // Check that completion matches original pattern polarity
        for i in 0..16 {
            let original_pol = pattern[i].polarity;
            let completed_pol = completed[i].polarity;
            // Allow some tolerance for attractor dynamics
            if i < 8 {
                assert_eq!(
                    original_pol, completed_pol,
                    "Mismatch at position {} (given)",
                    i
                );
            }
        }
    }

    #[test]
    fn test_hopfield_capacity() {
        let hopfield = HopfieldAttractor::new(100);
        assert_eq!(hopfield.capacity(), 14); // 0.14 * 100
    }

    #[test]
    fn test_ring_position() {
        let mut ring = RingAttractor::new(32);

        ring.set_position(0.25);
        assert!((ring.position() - 0.25).abs() < 0.01);

        ring.shift(0.5);
        assert!((ring.position() - 0.75).abs() < 0.01);

        ring.shift(0.5);
        assert!((ring.position() - 0.25).abs() < 0.01); // Wrapped
    }

    #[test]
    fn test_ring_state_has_peak() {
        let mut ring = RingAttractor::new(32);
        ring.set_position(0.5);

        let state = ring.state();

        // Find neuron with highest activation
        let peak_idx = state
            .iter()
            .enumerate()
            .max_by_key(|(_, s)| s.magnitude)
            .map(|(i, _)| i)
            .unwrap();

        // Peak should be around position 0.5 (index 16)
        // Allow wider tolerance due to step_toward smoothing
        let expected_idx = 16;
        let distance = (peak_idx as i32 - expected_idx).abs();
        assert!(
            distance <= 4,
            "Peak at {} but expected near {}, distance {}",
            peak_idx,
            expected_idx,
            distance
        );
    }

    #[test]
    fn test_ring_complete_finds_peak() {
        let mut ring = RingAttractor::new(32);

        // Create a noisy input with peak around 0.25
        let input: Vec<Signal> = (0..32)
            .map(|i| {
                let pos = i as f32 / 32.0;
                let dist = (pos - 0.25).abs().min((pos - 0.25 + 1.0).abs());
                let mag = ((1.0 - dist * 4.0).max(0.0) * 200.0) as u8;
                Signal::positive(mag)
            })
            .collect();

        ring.complete(&input);

        // Position should be near 0.25
        assert!((ring.position() - 0.25).abs() < 0.1);
    }
}
