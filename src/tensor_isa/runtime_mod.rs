//! Runtime Architecture Modification for TensorISA
//!
//! Enables self-modification: grow/prune layers, fork specialists at runtime.
//! This is the core enabler for genuine machine autonomy at the architectural level.
//!
//! ## Architecture Modification Operations
//!
//! ```text
//! ; Runtime growth example
//! alloc       H6, i32[64]              ; Allocate new activation register
//! alloc       C6, ternary[64, 32]      ; Allocate new weight matrix
//! wire        H6, C6, H2               ; Wire: H6 = C6 @ H2 (dynamic matmul)
//! ```
//!
//! ## Self-Modification Loop
//!
//! ```text
//! 1. Brain encounters unknown pattern (novelty signal high)
//! 2. Learning ISA: IF_NOVELTY_HIGH → CALL grow_detector
//! 3. TensorISA: ALLOC new registers, WIRE to existing mesh
//! 4. Adaptive Learning: Train new detector
//! 5. Thermogram: Track in HOT state during learning
//! 6. Success: CONSOLIDATE → move to COLD
//! 7. Failure: FREE registers, prune pathway
//! ```

use super::{
    ColdBuffer, HotBuffer, StepResult, TensorAction, TensorDtype, TensorInstruction,
    TensorInterpreter, TensorRegister,
};
use crate::Signal;

/// Shape specification for runtime allocation
#[derive(Debug, Clone)]
pub struct ShapeSpec {
    /// Dimensions
    pub dims: Vec<usize>,
    /// Data type
    pub dtype: TensorDtype,
    /// Optional thermogram key for persistence
    pub thermogram_key: Option<String>,
}

impl ShapeSpec {
    /// Create a hot buffer shape (i32 activations)
    pub fn hot(dims: Vec<usize>) -> Self {
        Self {
            dims,
            dtype: TensorDtype::I32,
            thermogram_key: None,
        }
    }

    /// Create a cold buffer shape (ternary weights)
    pub fn cold(dims: Vec<usize>) -> Self {
        Self {
            dims,
            dtype: TensorDtype::Ternary,
            thermogram_key: None,
        }
    }

    /// Create with thermogram key for persistence
    pub fn with_key(mut self, key: impl Into<String>) -> Self {
        self.thermogram_key = Some(key.into());
        self
    }

    /// Total number of elements
    pub fn numel(&self) -> usize {
        self.dims.iter().product()
    }
}

/// Dynamic wiring specification
#[derive(Debug, Clone)]
pub struct WireSpec {
    /// Output register
    pub output: TensorRegister,
    /// Weight register (cold)
    pub weights: TensorRegister,
    /// Input register
    pub input: TensorRegister,
    /// Wire type
    pub wire_type: WireType,
}

/// Type of wiring connection
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WireType {
    /// Standard forward connection (matmul)
    Forward,
    /// Skip/residual connection (add)
    Skip,
    /// Concatenation
    Concat,
    /// Gated connection
    Gated,
}

/// Runtime modification event
#[derive(Debug, Clone)]
pub enum ModEvent {
    /// Register allocated
    Allocated {
        register: TensorRegister,
        shape: Vec<usize>,
    },
    /// Register freed
    Freed { register: TensorRegister },
    /// Connection wired
    Wired { spec: WireSpec },
    /// Connection removed
    Unwired {
        output: TensorRegister,
        input: TensorRegister,
    },
}

/// Runtime modification extensions for TensorInterpreter
impl TensorInterpreter {
    /// Allocate a new hot register at runtime
    ///
    /// Returns the register index if successful.
    pub fn alloc_hot(&mut self, shape: Vec<usize>) -> Option<usize> {
        // Find first free hot register slot
        for i in 0..16 {
            if self.hot_reg(i).is_none() {
                // Allocate it
                self.set_hot_reg(i, HotBuffer::new(shape));
                return Some(i);
            }
        }
        None // No free slots
    }

    /// Allocate a new cold register at runtime
    ///
    /// Returns the register index if successful.
    pub fn alloc_cold(&mut self, shape: Vec<usize>, key: Option<String>) -> Option<usize> {
        // Find first free cold register slot
        for i in 0..16 {
            if self.cold_reg(i).is_none() {
                let mut buf = ColdBuffer::new(shape);
                if let Some(k) = key {
                    buf = buf.with_key(k);
                }
                self.set_cold_reg(i, buf);
                return Some(i);
            }
        }
        None
    }

    /// Free a hot register
    pub fn free_hot(&mut self, index: usize) -> bool {
        self.clear_hot_reg(index)
    }

    /// Free a cold register
    pub fn free_cold(&mut self, index: usize) -> bool {
        self.clear_cold_reg(index)
    }

    /// Execute a dynamic forward wiring (matmul)
    ///
    /// output = weights @ input
    pub fn wire_forward(
        &mut self,
        output_idx: usize,
        weights_idx: usize,
        input_idx: usize,
    ) -> StepResult {
        // Get weight dimensions
        let (out_dim, in_dim) = {
            let weights = match self.cold_reg(weights_idx) {
                Some(buf) => buf,
                None => {
                    return StepResult::Error(format!(
                        "Cold register C{} not allocated",
                        weights_idx
                    ))
                }
            };

            if weights.shape.len() < 2 {
                return StepResult::Error("Weights must be 2D".to_string());
            }
            (weights.shape[0], weights.shape[1])
        };

        // Get input
        let input_data = {
            let input = match self.hot_reg(input_idx) {
                Some(buf) => buf,
                None => {
                    return StepResult::Error(format!(
                        "Hot register H{} not allocated",
                        input_idx
                    ))
                }
            };
            input.data.clone()
        };

        // Get weights for computation
        let weights_data = {
            let weights = self.cold_reg(weights_idx).unwrap();
            weights.weights.clone()
        };

        // Compute matmul
        let mut output_data = vec![0i64; out_dim];
        for o in 0..out_dim {
            let mut sum = 0i64;
            for i in 0..in_dim.min(input_data.len()) {
                let w = &weights_data[o * in_dim + i];
                let effective = w.polarity as i64 * w.magnitude as i64;
                sum += effective * input_data[i] as i64;
            }
            output_data[o] = sum;
        }

        // Store result
        let output_i32: Vec<i32> = output_data
            .iter()
            .map(|&v| v.clamp(i32::MIN as i64, i32::MAX as i64) as i32)
            .collect();

        self.set_hot_reg(
            output_idx,
            HotBuffer {
                data: output_i32,
                shape: vec![out_dim],
            },
        );

        StepResult::Continue
    }

    /// Execute a skip connection (element-wise add)
    ///
    /// output = input1 + input2
    pub fn wire_skip(&mut self, output_idx: usize, input1_idx: usize, input2_idx: usize) -> StepResult {
        let (data1, shape1) = {
            let input1 = match self.hot_reg(input1_idx) {
                Some(buf) => buf,
                None => {
                    return StepResult::Error(format!(
                        "Hot register H{} not allocated",
                        input1_idx
                    ))
                }
            };
            (input1.data.clone(), input1.shape.clone())
        };

        let data2 = {
            let input2 = match self.hot_reg(input2_idx) {
                Some(buf) => buf,
                None => {
                    return StepResult::Error(format!(
                        "Hot register H{} not allocated",
                        input2_idx
                    ))
                }
            };
            input2.data.clone()
        };

        // Element-wise add with broadcasting
        let len = data1.len().max(data2.len());
        let mut result = vec![0i32; len];
        for i in 0..len {
            let v1 = data1.get(i % data1.len()).copied().unwrap_or(0);
            let v2 = data2.get(i % data2.len()).copied().unwrap_or(0);
            result[i] = v1.saturating_add(v2);
        }

        self.set_hot_reg(
            output_idx,
            HotBuffer {
                data: result,
                shape: shape1,
            },
        );

        StepResult::Continue
    }

    /// Initialize cold register with random-ish ternary weights
    pub fn init_cold_random(&mut self, index: usize, seed: u64) {
        if let Some(cold) = self.cold_reg_mut(index) {
            let mut rng_state = seed;
            for w in cold.weights.iter_mut() {
                // Simple PRNG
                rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
                let r = (rng_state >> 33) as u32;

                w.polarity = match r % 3 {
                    0 => -1,
                    1 => 0,
                    _ => 1,
                };
                w.magnitude = ((r >> 2) % 200 + 30) as u8;
            }
        }
    }

    /// Clone a cold register to a new slot
    pub fn clone_cold(&mut self, src_idx: usize, dst_idx: usize) -> bool {
        if let Some(src) = self.cold_reg(src_idx) {
            let cloned = src.clone();
            self.set_cold_reg(dst_idx, cloned);
            true
        } else {
            false
        }
    }

    /// Get architecture statistics
    pub fn arch_stats(&self) -> ArchStats {
        let mut hot_count = 0;
        let mut cold_count = 0;
        let mut total_hot_elements = 0;
        let mut total_cold_elements = 0;

        for i in 0..16 {
            if let Some(buf) = self.hot_reg(i) {
                hot_count += 1;
                total_hot_elements += buf.numel();
            }
            if let Some(buf) = self.cold_reg(i) {
                cold_count += 1;
                total_cold_elements += buf.numel();
            }
        }

        ArchStats {
            hot_count,
            cold_count,
            total_hot_elements,
            total_cold_elements,
        }
    }

    // === Private helpers for register manipulation ===

    fn set_hot_reg(&mut self, index: usize, buf: HotBuffer) {
        // Access through the mutable API we'll add
        self.set_hot_reg_internal(index, Some(buf));
    }

    fn set_cold_reg(&mut self, index: usize, buf: ColdBuffer) {
        self.set_cold_reg_internal(index, Some(buf));
    }

    fn clear_hot_reg(&mut self, index: usize) -> bool {
        if self.hot_reg(index).is_some() {
            self.set_hot_reg_internal(index, None);
            true
        } else {
            false
        }
    }

    fn clear_cold_reg(&mut self, index: usize) -> bool {
        if self.cold_reg(index).is_some() {
            self.set_cold_reg_internal(index, None);
            true
        } else {
            false
        }
    }
}

/// Architecture statistics
#[derive(Debug, Clone)]
pub struct ArchStats {
    /// Number of allocated hot registers
    pub hot_count: usize,
    /// Number of allocated cold registers
    pub cold_count: usize,
    /// Total elements in hot registers
    pub total_hot_elements: usize,
    /// Total elements in cold registers
    pub total_cold_elements: usize,
}

impl ArchStats {
    /// Estimated memory usage in bytes
    pub fn memory_bytes(&self) -> usize {
        // Hot: i32 per element = 4 bytes
        // Cold: Signal per element = 2 bytes
        self.total_hot_elements * 4 + self.total_cold_elements * 2
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor_isa::assemble;

    #[test]
    fn test_alloc_and_free() {
        let source = r#"
.registers
    H0: i32[4]

.program
    halt
"#;

        let program = assemble(source).unwrap();
        let mut interp = TensorInterpreter::from_program(&program);

        // Allocate new hot register
        let idx = interp.alloc_hot(vec![8]);
        assert!(idx.is_some());
        let idx = idx.unwrap();
        assert!(idx > 0); // H0 is taken

        // Verify allocation
        assert!(interp.hot_reg(idx).is_some());
        assert_eq!(interp.hot_reg(idx).unwrap().shape, vec![8]);

        // Free it
        assert!(interp.free_hot(idx));
        assert!(interp.hot_reg(idx).is_none());
    }

    #[test]
    fn test_alloc_cold_with_key() {
        let mut interp = TensorInterpreter::new();

        let idx = interp.alloc_cold(vec![4, 2], Some("test.weights".to_string()));
        assert!(idx.is_some());
        let idx = idx.unwrap();

        let cold = interp.cold_reg(idx).unwrap();
        assert_eq!(cold.shape, vec![4, 2]);
        assert_eq!(cold.thermogram_key, Some("test.weights".to_string()));
    }

    #[test]
    fn test_wire_forward() {
        let source = r#"
.registers
    C0: ternary[4, 2]
    H0: i32[2]
    H1: i32[4]

.program
    halt
"#;

        let program = assemble(source).unwrap();
        let mut interp = TensorInterpreter::from_program(&program);

        // Set up input
        if let Some(h0) = interp.hot_reg_mut(0) {
            h0.data = vec![100, 50];
        }

        // Set up weights
        if let Some(c0) = interp.cold_reg_mut(0) {
            for (i, w) in c0.weights.iter_mut().enumerate() {
                w.polarity = if i % 2 == 0 { 1 } else { -1 };
                w.magnitude = 100;
            }
        }

        // Wire forward
        let result = interp.wire_forward(1, 0, 0);
        assert!(matches!(result, StepResult::Continue));

        // Check output
        let h1 = interp.hot_reg(1).unwrap();
        assert_eq!(h1.shape, vec![4]);
        // Values depend on weight pattern
    }

    #[test]
    fn test_arch_stats() {
        let source = r#"
.registers
    C0: ternary[4, 2]
    C1: ternary[8, 4]
    H0: i32[2]
    H1: i32[4]
    H2: i32[8]

.program
    halt
"#;

        let program = assemble(source).unwrap();
        let interp = TensorInterpreter::from_program(&program);

        let stats = interp.arch_stats();
        assert_eq!(stats.hot_count, 3);
        assert_eq!(stats.cold_count, 2);
        assert_eq!(stats.total_hot_elements, 2 + 4 + 8);
        assert_eq!(stats.total_cold_elements, 4 * 2 + 8 * 4);
    }
}
