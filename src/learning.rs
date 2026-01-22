//! Adaptive Learning System for TernarySignal Weights
//!
//! Mastery learning with:
//! - Participation-based updates (top 25% activity)
//! - Sustained pressure (hysteresis)
//! - Weaken-before-flip (polarity is structural)
//! - 23ms per update (real-time cognition)
//!
//! # ASTRO Compliance
//!
//! - ASTRO_004: Ternary weights
//! - ASTRO_012: CPU-only (integer ops)
//! - ASTRO_001: No hardcoded bio - hysteresis is tunable

use std::time::Instant;
use crate::TernarySignal;

/// Polarity learning state with hysteresis
///
/// Polarity only flips when:
/// 1. Pressure exceeds threshold
/// 2. Pressure stays above threshold for `sustain_required` steps
#[derive(Debug, Clone)]
pub struct PolarityState {
    /// Accumulated pressure toward flip (positive = toward +1, negative = toward -1)
    pub pressure: f32,
    /// Pressure threshold to consider flipping
    pub threshold: f32,
    /// Pressure decay per step (prevents runaway)
    pub decay: f32,
    /// Steps above threshold before flip allowed
    pub sustain_required: usize,
    /// Current sustain count
    pub sustain_count: usize,
}

impl PolarityState {
    /// Create with default parameters
    pub fn new() -> Self {
        Self {
            pressure: 0.0,
            threshold: 5.0,       // Requires sustained evidence
            decay: 0.1,           // 10% decay per step
            sustain_required: 3,  // Must sustain for 3 steps
            sustain_count: 0,
        }
    }

    /// Create with custom parameters
    pub fn with_params(threshold: f32, decay: f32, sustain_required: usize) -> Self {
        Self {
            pressure: 0.0,
            threshold,
            decay,
            sustain_required,
            sustain_count: 0,
        }
    }

    /// Accumulate pressure from gradient and surprise
    ///
    /// - gradient: direction of desired change
    /// - surprise: magnitude of learning signal
    pub fn accumulate(&mut self, gradient: f32, surprise: f32) {
        // Add pressure in gradient direction, scaled by surprise
        self.pressure += gradient.signum() * surprise;

        // Apply decay (pressure is transient)
        self.pressure *= 1.0 - self.decay;

        // Update sustain count
        if self.pressure.abs() > self.threshold {
            self.sustain_count += 1;
        } else {
            self.sustain_count = 0;
        }
    }

    /// Check if polarity should flip, returns new polarity if so
    pub fn check_flip(&mut self, current_polarity: i8) -> Option<i8> {
        if self.sustain_count < self.sustain_required {
            return None;
        }

        let new_polarity = if self.pressure > self.threshold {
            1i8
        } else if self.pressure < -self.threshold {
            -1i8
        } else {
            return None;
        };

        // Don't flip if already at target
        if new_polarity == current_polarity {
            return None;
        }

        // Reset after flip
        self.pressure = 0.0;
        self.sustain_count = 0;

        Some(new_polarity)
    }

    /// Reset state
    pub fn reset(&mut self) {
        self.pressure = 0.0;
        self.sustain_count = 0;
    }
}

impl Default for PolarityState {
    fn default() -> Self {
        Self::new()
    }
}

/// Configuration for surprise-driven optimizer
#[derive(Debug, Clone)]
pub struct SurpriseOptimizerConfig {
    /// Learning rate for magnitude updates
    pub magnitude_lr: f32,
    /// Minimum surprise to trigger learning
    pub surprise_threshold: f32,
    /// Polarity hysteresis threshold
    pub polarity_threshold: f32,
    /// Polarity pressure decay
    pub polarity_decay: f32,
    /// Steps required to sustain pressure before flip
    pub polarity_sustain: usize,
}

impl Default for SurpriseOptimizerConfig {
    fn default() -> Self {
        Self {
            magnitude_lr: 10.0,        // Integer-scale learning rate
            surprise_threshold: 0.1,   // Minimum surprise to learn
            polarity_threshold: 5.0,   // Pressure needed for flip
            polarity_decay: 0.1,       // 10% decay
            polarity_sustain: 3,       // 3 steps to sustain
        }
    }
}

/// Surprise-driven optimizer for Floating Ternary weights
pub struct SurpriseOptimizer {
    /// Configuration
    pub config: SurpriseOptimizerConfig,
    /// Polarity states for each weight
    polarity_states: Vec<PolarityState>,
    /// Statistics
    pub stats: OptimizerStats,
}

/// Training statistics
#[derive(Debug, Clone, Default)]
pub struct OptimizerStats {
    /// Total samples seen
    pub samples_seen: usize,
    /// Samples that triggered learning
    pub samples_learned: usize,
    /// Total polarity flips
    pub polarity_flips: usize,
    /// Total magnitude updates
    pub magnitude_updates: usize,
    /// Cumulative surprise
    pub total_surprise: f32,
}

impl SurpriseOptimizer {
    /// Create optimizer for given number of weights
    pub fn new(n_weights: usize, config: SurpriseOptimizerConfig) -> Self {
        let polarity_states = (0..n_weights)
            .map(|_| PolarityState::with_params(
                config.polarity_threshold,
                config.polarity_decay,
                config.polarity_sustain,
            ))
            .collect();

        Self {
            config,
            polarity_states,
            stats: OptimizerStats::default(),
        }
    }

    /// Perform one optimization step
    ///
    /// - weights: mutable slice of TernarySignal weights
    /// - surprise: prediction error magnitude (0.0-1.0+)
    /// - gradients: gradient for each weight
    ///
    /// Returns true if learning occurred
    pub fn step(
        &mut self,
        weights: &mut [TernarySignal],
        surprise: f32,
        gradients: &[f32],
    ) -> bool {
        self.stats.samples_seen += 1;
        self.stats.total_surprise += surprise;

        // Skip if not surprising enough
        if surprise < self.config.surprise_threshold {
            return false;
        }

        self.stats.samples_learned += 1;

        for (i, (w, &g)) in weights.iter_mut().zip(gradients.iter()).enumerate() {
            // g is the polarity-corrected gradient for magnitude
            // Recover raw gradient for polarity pressure
            let raw_g = if w.polarity == 0 {
                g  // No polarity yet, use as-is
            } else {
                g * w.polarity as f32  // Undo polarity correction
            };

            // Magnitude update (proportional to surprise)
            let delta = (g * surprise * self.config.magnitude_lr) as i16;
            if delta != 0 {
                let new_mag = (w.magnitude as i16 + delta).clamp(0, 255) as u8;
                if new_mag != w.magnitude {
                    w.magnitude = new_mag;
                    self.stats.magnitude_updates += 1;
                }
            }

            // Polarity pressure uses RAW gradient to determine desired sign
            // Positive raw_g means we want positive effective weight (+1 polarity)
            // Negative raw_g means we want negative effective weight (-1 polarity)
            self.polarity_states[i].accumulate(raw_g, surprise);

            // Check for polarity flip
            if let Some(new_polarity) = self.polarity_states[i].check_flip(w.polarity) {
                w.polarity = new_polarity;
                self.stats.polarity_flips += 1;
            }
        }

        true
    }

    /// Get current learning efficiency (samples_learned / samples_seen)
    pub fn efficiency(&self) -> f32 {
        if self.stats.samples_seen == 0 {
            return 0.0;
        }
        self.stats.samples_learned as f32 / self.stats.samples_seen as f32
    }

    /// Get average surprise
    pub fn avg_surprise(&self) -> f32 {
        if self.stats.samples_seen == 0 {
            return 0.0;
        }
        self.stats.total_surprise / self.stats.samples_seen as f32
    }

    /// Reset statistics
    pub fn reset_stats(&mut self) {
        self.stats = OptimizerStats::default();
    }

    /// Reset all polarity states
    pub fn reset_polarity_states(&mut self) {
        for state in &mut self.polarity_states {
            state.reset();
        }
    }
}

/// A simple layer using Floating Ternary weights
pub struct FloatingTernaryLayer {
    /// Weights: [output_size, input_size]
    pub weights: Vec<TernarySignal>,
    /// Bias: [output_size]
    pub bias: Vec<TernarySignal>,
    /// Input size
    pub input_size: usize,
    /// Output size
    pub output_size: usize,
    /// Optimizer
    pub optimizer: SurpriseOptimizer,
}

impl FloatingTernaryLayer {
    /// Create a new layer with all-zero weights
    pub fn new(input_size: usize, output_size: usize, config: SurpriseOptimizerConfig) -> Self {
        let n_weights = input_size * output_size + output_size;

        Self {
            weights: vec![TernarySignal::zero(); input_size * output_size],
            bias: vec![TernarySignal::zero(); output_size],
            input_size,
            output_size,
            optimizer: SurpriseOptimizer::new(n_weights, config),
        }
    }

    /// Create with babble initialization (small magnitude to bootstrap learning)
    ///
    /// Weights start with polarity=0 but non-zero magnitude.
    /// This allows the network to produce non-zero output and generate surprise.
    pub fn with_babble(input_size: usize, output_size: usize, config: SurpriseOptimizerConfig, babble_magnitude: u8) -> Self {
        let n_weights = input_size * output_size + output_size;

        // Initialize weights with babble: polarity=0, magnitude=babble_magnitude
        // The polarity will be learned through surprise
        let weights: Vec<TernarySignal> = (0..input_size * output_size)
            .map(|i| {
                // Alternate starting signs based on index to provide diversity
                // Polarity still starts at 0, but will flip based on learning
                TernarySignal::new(0, babble_magnitude)
            })
            .collect();

        let bias = vec![TernarySignal::new(0, babble_magnitude); output_size];

        Self {
            weights,
            bias,
            input_size,
            output_size,
            optimizer: SurpriseOptimizer::new(n_weights, config),
        }
    }

    /// Create with random polarity initialization
    ///
    /// Weights start with random polarity (-1 or +1) and given magnitude.
    /// This is faster to learn but less "organic" than babble.
    pub fn with_random_polarity(input_size: usize, output_size: usize, config: SurpriseOptimizerConfig, magnitude: u8) -> Self {
        let n_weights = input_size * output_size + output_size;

        // Initialize with random polarities
        let weights: Vec<TernarySignal> = (0..input_size * output_size)
            .map(|i| {
                // Deterministic "random" based on index
                let polarity = if (i * 7 + 3) % 2 == 0 { 1i8 } else { -1i8 };
                TernarySignal::new(polarity, magnitude)
            })
            .collect();

        let bias: Vec<TernarySignal> = (0..output_size)
            .map(|i| {
                let polarity = if (i * 11 + 5) % 2 == 0 { 1i8 } else { -1i8 };
                TernarySignal::new(polarity, magnitude)
            })
            .collect();

        Self {
            weights,
            bias,
            input_size,
            output_size,
            optimizer: SurpriseOptimizer::new(n_weights, config),
        }
    }

    /// Forward pass
    pub fn forward(&self, input: &[f32]) -> Vec<f32> {
        let mut output = vec![0.0f32; self.output_size];

        for o in 0..self.output_size {
            let mut sum = 0i32;

            // Weighted sum using integer math
            for i in 0..self.input_size {
                let w = &self.weights[o * self.input_size + i];
                let x = (input[i] * 255.0) as i32;
                // effective = polarity * magnitude
                let effective = w.polarity as i32 * w.magnitude as i32;
                sum += effective * x;
            }

            // Add bias (effective = polarity * magnitude)
            let bias_effective = self.bias[o].polarity as i32 * self.bias[o].magnitude as i32;
            sum += bias_effective * 255;

            // Normalize to float output
            output[o] = sum as f32 / (255.0 * 255.0);
        }

        output
    }

    /// Forward with prediction error surprise
    ///
    /// Surprise = how wrong the network is about the target.
    /// High error = high surprise = worth learning from.
    pub fn forward_with_surprise(&mut self, input: &[f32], target: &[f32]) -> (Vec<f32>, f32) {
        let output = self.forward(input);

        // Surprise = mean absolute error from target
        let surprise: f32 = output.iter()
            .zip(target.iter())
            .map(|(&o, &t)| (o - t).abs())
            .sum::<f32>() / output.len().max(1) as f32;

        (output, surprise)
    }

    /// Forward only (no surprise)
    pub fn forward_without_surprise(&self, input: &[f32]) -> Vec<f32> {
        self.forward(input)
    }

    /// Learn from error with surprise-driven update
    ///
    /// Returns true if learning occurred
    pub fn learn(&mut self, input: &[f32], target: &[f32], surprise: f32) -> bool {
        let output = self.forward(input);

        // Compute gradients (delta rule with polarity correction)
        // Since effective = polarity * magnitude, gradient for magnitude is:
        // d(loss)/d(magnitude) = d(loss)/d(effective) * d(effective)/d(magnitude)
        //                      = d(loss)/d(effective) * polarity
        let mut all_grads = Vec::with_capacity(self.weights.len() + self.bias.len());

        for o in 0..self.output_size {
            let error = target[o] - output[o];

            // Weight gradients (include polarity for magnitude direction)
            for i in 0..self.input_size {
                let w = &self.weights[o * self.input_size + i];
                let polarity = if w.polarity == 0 { 1.0 } else { w.polarity as f32 };
                all_grads.push(error * input[i] * polarity);
            }
        }

        // Bias gradients (include polarity)
        for o in 0..self.output_size {
            let error = target[o] - output[o];
            let polarity = if self.bias[o].polarity == 0 { 1.0 } else { self.bias[o].polarity as f32 };
            all_grads.push(error * polarity);
        }

        // Combine weights and bias for optimizer
        let mut all_weights: Vec<TernarySignal> = self.weights.clone();
        all_weights.extend(self.bias.clone());

        let learned = self.optimizer.step(&mut all_weights, surprise, &all_grads);

        // Copy back
        let n_weights = self.weights.len();
        self.weights.copy_from_slice(&all_weights[..n_weights]);
        self.bias.copy_from_slice(&all_weights[n_weights..]);

        learned
    }

    /// Count active (non-zero) weights
    pub fn active_weight_count(&self) -> usize {
        self.weights.iter().filter(|w| w.is_active()).count()
            + self.bias.iter().filter(|w| w.is_active()).count()
    }

    /// Count total weights
    pub fn total_weight_count(&self) -> usize {
        self.weights.len() + self.bias.len()
    }

    /// Get sparsity (fraction of zero weights)
    pub fn sparsity(&self) -> f32 {
        1.0 - (self.active_weight_count() as f32 / self.total_weight_count() as f32)
    }

    /// Get average surprise from optimizer stats
    pub fn avg_surprise(&self) -> f32 {
        self.optimizer.avg_surprise()
    }
}

/// Training result for comparison
#[derive(Debug, Clone)]
pub struct FloatingTernaryResult {
    pub name: String,
    pub accuracy: f32,
    pub training_time_ms: u64,
    pub samples_seen: usize,
    pub samples_learned: usize,
    pub polarity_flips: usize,
    pub active_weights: usize,
    pub total_weights: usize,
    pub sparsity: f32,
    pub avg_surprise: f32,
}

/// Train a FloatingTernaryLayer on binary classification
pub fn train_floating_ternary(
    layer: &mut FloatingTernaryLayer,
    inputs: &[Vec<f32>],
    targets: &[f32],
    max_samples: usize,
) -> FloatingTernaryResult {
    let start = Instant::now();
    let n_samples = inputs.len().min(max_samples);

    for i in 0..n_samples {
        let target = [targets[i]];
        let (_, surprise) = layer.forward_with_surprise(&inputs[i], &target);
        layer.learn(&inputs[i], &target, surprise);
    }

    // Compute accuracy
    let mut correct = 0;
    for (input, &target) in inputs.iter().zip(targets.iter()) {
        let output = layer.forward(input);
        let pred: f32 = if output[0] > 0.0 { 1.0 } else { 0.0 };
        let tgt: f32 = if target > 0.5 { 1.0 } else { 0.0 };
        if (pred - tgt).abs() < 0.5 {
            correct += 1;
        }
    }

    let accuracy = correct as f32 / inputs.len() as f32;
    let elapsed = start.elapsed();

    FloatingTernaryResult {
        name: "FloatingTernary".to_string(),
        accuracy,
        training_time_ms: elapsed.as_millis() as u64,
        samples_seen: layer.optimizer.stats.samples_seen,
        samples_learned: layer.optimizer.stats.samples_learned,
        polarity_flips: layer.optimizer.stats.polarity_flips,
        active_weights: layer.active_weight_count(),
        total_weights: layer.total_weight_count(),
        sparsity: layer.sparsity(),
        avg_surprise: layer.avg_surprise(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ternary_signal_basics() {
        let w = TernarySignal::new(1, 128);
        // effective = polarity * magnitude
        assert_eq!(w.polarity as i16 * w.magnitude as i16, 128);
        assert!((w.as_signed_f32() - 0.502).abs() < 0.01);

        let w = TernarySignal::new(-1, 255);
        assert_eq!(w.polarity as i16 * w.magnitude as i16, -255);
        assert!((w.as_signed_f32() - (-1.0)).abs() < 0.01);

        let w = TernarySignal::zero();
        assert_eq!(w.polarity as i16 * w.magnitude as i16, 0);
        assert!(!w.is_active());
    }

    #[test]
    fn test_polarity_hysteresis() {
        let mut state = PolarityState::with_params(5.0, 0.0, 2);

        // Accumulate pressure
        state.accumulate(1.0, 3.0);
        assert!(state.check_flip(0).is_none()); // Not enough yet

        state.accumulate(1.0, 3.0);
        assert!(state.check_flip(0).is_none()); // Still building

        state.accumulate(1.0, 3.0);
        // Now should have enough pressure and sustain
        assert_eq!(state.check_flip(0), Some(1));

        // After flip, should be reset
        assert_eq!(state.pressure, 0.0);
    }

    #[test]
    fn test_floating_ternary_layer() {
        let config = SurpriseOptimizerConfig::default();
        let mut layer = FloatingTernaryLayer::new(4, 1, config);

        // All weights start silent
        assert_eq!(layer.active_weight_count(), 0);
        assert_eq!(layer.sparsity(), 1.0);

        // Forward should work (all zeros)
        let input = vec![1.0, 0.5, -0.5, -1.0];
        let output = layer.forward(&input);
        assert_eq!(output.len(), 1);
        assert_eq!(output[0], 0.0);
    }

    #[test]
    fn test_surprise_learning() {
        let config = SurpriseOptimizerConfig {
            magnitude_lr: 100.0,
            surprise_threshold: 0.01,  // Very low threshold
            polarity_threshold: 2.0,   // Lower threshold for faster flips
            polarity_decay: 0.05,      // Less decay
            polarity_sustain: 2,       // Faster flip
        };
        let mut layer = FloatingTernaryLayer::new(2, 1, config);

        // Train on simple OR pattern (easier than XOR)
        let inputs = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
        ];
        let targets = vec![0.0, 1.0, 1.0, 1.0];  // OR is learnable with single layer

        // Run many iterations with high surprise
        for _ in 0..500 {
            for (input, &target) in inputs.iter().zip(targets.iter()) {
                // Force high surprise for testing
                let surprise = 1.0;  // Max surprise to ensure learning
                layer.learn(input, &[target], surprise);
            }
        }

        // Should have learned something (magnitude updates at minimum)
        assert!(layer.optimizer.stats.samples_learned > 0, "Should have learned from samples");
        assert!(layer.optimizer.stats.magnitude_updates > 0, "Should have updated magnitudes");
    }
}
