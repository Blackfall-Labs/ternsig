//! Runtime Architecture Modification Operations
//!
//! Implements opcodes for self-modifying neural architecture:
//! - ALLOC_TENSOR: Allocate registers at runtime
//! - FREE_TENSOR: Free registers
//! - WIRE_FORWARD: Create forward connections
//! - WIRE_SKIP: Create skip connections
//! - GROW_NEURON: Add neurons to a layer
//! - PRUNE_NEURON: Remove a neuron from a layer
//! - INIT_RANDOM: Initialize weights with random values
//!
//! These enable Phase 6 Structural Plasticity: the brain can grow.

use super::{ColdBuffer, HotBuffer, Instruction, Interpreter, StepResult};
use crate::Signal;

impl Interpreter {
    /// Execute ALLOC_TENSOR: Allocate a register at runtime
    ///
    /// Instruction format:
    /// - target: Register to allocate (H0-H15 or C0-C15)
    /// - modifier[0..2]: Encoded shape (dim0 in [0], dim1 in [1..2] as u16)
    ///
    /// For hot registers: allocates i32 buffer
    /// For cold registers: allocates Signal buffer with optional thermogram key
    pub(super) fn execute_alloc_tensor(&mut self, instr: Instruction) -> StepResult {
        let target = instr.target();
        let idx = target.index();

        // TVMR operand layout: [target, dim0, dim1_hi, dim1_lo]
        let dim0 = instr.operands[1] as usize;
        let dim1 = ((instr.operands[2] as usize) << 8) | (instr.operands[3] as usize);

        let shape = if dim1 == 0 {
            vec![dim0]
        } else {
            vec![dim0, dim1]
        };

        if target.is_hot() {
            // Allocate hot register
            if self.hot_regs[idx].is_some() {
                return StepResult::Error(format!(
                    "Hot register H{} already allocated",
                    idx
                ));
            }
            self.hot_regs[idx] = Some(HotBuffer::new(shape));
            StepResult::Continue
        } else if target.is_cold() {
            // Allocate cold register
            if self.cold_regs[idx].is_some() {
                return StepResult::Error(format!(
                    "Cold register C{} already allocated",
                    idx
                ));
            }
            self.cold_regs[idx] = Some(ColdBuffer::new(shape));
            StepResult::Continue
        } else {
            StepResult::Error(format!(
                "Cannot allocate register {:?} - must be Hot or Cold",
                target
            ))
        }
    }

    /// Execute FREE_TENSOR: Free a register
    ///
    /// Instruction format:
    /// - target: Register to free (H0-H15 or C0-C15)
    pub(super) fn execute_free_tensor(&mut self, instr: Instruction) -> StepResult {
        let target = instr.target();
        let idx = target.index();

        if target.is_hot() {
            if self.hot_regs[idx].is_none() {
                return StepResult::Error(format!(
                    "Hot register H{} not allocated",
                    idx
                ));
            }
            self.hot_regs[idx] = None;
            StepResult::Continue
        } else if target.is_cold() {
            if self.cold_regs[idx].is_none() {
                return StepResult::Error(format!(
                    "Cold register C{} not allocated",
                    idx
                ));
            }
            self.cold_regs[idx] = None;
            StepResult::Continue
        } else {
            StepResult::Error(format!(
                "Cannot free register {:?} - must be Hot or Cold",
                target
            ))
        }
    }

    /// Execute WIRE_FORWARD: Dynamic forward connection (matmul)
    ///
    /// output = weights @ input
    ///
    /// Instruction format:
    /// - target: Output hot register
    /// - source: Weights cold register
    /// - aux: Input hot register index
    pub(super) fn execute_wire_forward(&mut self, instr: Instruction) -> StepResult {
        let output_idx = instr.target().index();
        let weights_idx = instr.source().index();
        let input_idx = instr.aux() as usize;

        // Use the runtime_mod implementation
        self.wire_forward(output_idx, weights_idx, input_idx)
    }

    /// Execute WIRE_SKIP: Dynamic skip connection (element-wise add)
    ///
    /// output = input1 + input2
    ///
    /// Instruction format:
    /// - target: Output hot register
    /// - source: Input1 hot register
    /// - aux: Input2 hot register index
    pub(super) fn execute_wire_skip(&mut self, instr: Instruction) -> StepResult {
        let output_idx = instr.target().index();
        let input1_idx = instr.source().index();
        let input2_idx = instr.aux() as usize;

        // Use the runtime_mod implementation
        self.wire_skip(output_idx, input1_idx, input2_idx)
    }

    /// Execute GROW_NEURON: Add neurons to a cold register
    ///
    /// Grows the output dimension of a weight matrix by adding new neurons.
    /// New weights are initialized with random ternary values.
    ///
    /// Instruction format:
    /// - target: Cold register to grow
    /// - aux: Number of neurons to add (1-255)
    /// - modifier[0..2]: Random seed for initialization
    pub(super) fn execute_grow_neuron(&mut self, instr: Instruction) -> StepResult {
        let idx = instr.target().index();
        let neurons_to_add = instr.aux() as usize;

        if neurons_to_add == 0 {
            return StepResult::Error("Cannot grow by 0 neurons".to_string());
        }

        let cold = match self.cold_regs[idx].as_mut() {
            Some(c) => c,
            None => {
                return StepResult::Error(format!(
                    "Cold register C{} not allocated",
                    idx
                ));
            }
        };

        // Get current dimensions
        // For 2D: shape = [out_dim, in_dim]
        // For 1D: shape = [size]
        if cold.shape.len() < 1 {
            return StepResult::Error("Cannot grow empty register".to_string());
        }

        // TVMR: seed from modifier byte (operands[3])
        let seed = instr.modifier()[0] as u64;

        if cold.shape.len() == 1 {
            // 1D: just extend
            let new_size = cold.shape[0] + neurons_to_add;
            let mut new_weights = cold.weights.clone();

            // Add new random weights
            let mut rng_state = seed.wrapping_add(new_weights.len() as u64);
            for _ in 0..neurons_to_add {
                rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
                let r = (rng_state >> 33) as u32;
                let polarity = match r % 3 {
                    0 => -1,
                    1 => 0,
                    _ => 1,
                };
                let magnitude = ((r >> 2) % 200 + 30) as u8;
                new_weights.push(Signal { polarity, magnitude });
            }

            cold.weights = new_weights;
            cold.shape = vec![new_size];
        } else if cold.shape.len() == 2 {
            // 2D: [out_dim, in_dim] - add output neurons
            let out_dim = cold.shape[0];
            let in_dim = cold.shape[1];
            let new_out_dim = out_dim + neurons_to_add;

            let mut new_weights = cold.weights.clone();

            // Add new rows (one row per new neuron)
            let mut rng_state = seed.wrapping_add(new_weights.len() as u64);
            for _ in 0..neurons_to_add {
                for _ in 0..in_dim {
                    rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
                    let r = (rng_state >> 33) as u32;
                    let polarity = match r % 3 {
                        0 => -1,
                        1 => 0,
                        _ => 1,
                    };
                    let magnitude = ((r >> 2) % 200 + 30) as u8;
                    new_weights.push(Signal { polarity, magnitude });
                }
            }

            cold.weights = new_weights;
            cold.shape = vec![new_out_dim, in_dim];
        } else {
            return StepResult::Error(format!(
                "Cannot grow {}D register, only 1D or 2D supported",
                cold.shape.len()
            ));
        }

        // Extend temperatures if tracking
        if let Some(temps) = &mut cold.temperatures {
            let new_count = cold.weights.len() - temps.len();
            temps.extend(std::iter::repeat(super::SignalTemperature::Hot).take(new_count));
        }

        StepResult::Continue
    }

    /// Execute PRUNE_NEURON: Remove a neuron from a cold register
    ///
    /// Removes a neuron (row) from a weight matrix.
    ///
    /// Instruction format:
    /// - target: Cold register to prune
    /// - aux: Index of neuron to remove
    pub(super) fn execute_prune_neuron(&mut self, instr: Instruction) -> StepResult {
        let idx = instr.target().index();
        let neuron_idx = instr.aux() as usize;

        let cold = match self.cold_regs[idx].as_mut() {
            Some(c) => c,
            None => {
                return StepResult::Error(format!(
                    "Cold register C{} not allocated",
                    idx
                ));
            }
        };

        if cold.shape.len() == 1 {
            // 1D: remove single element
            if neuron_idx >= cold.shape[0] {
                return StepResult::Error(format!(
                    "Neuron index {} out of bounds (size {})",
                    neuron_idx, cold.shape[0]
                ));
            }
            if cold.shape[0] <= 1 {
                return StepResult::Error("Cannot prune last neuron".to_string());
            }

            cold.weights.remove(neuron_idx);
            cold.shape[0] -= 1;

            if let Some(temps) = &mut cold.temperatures {
                if neuron_idx < temps.len() {
                    temps.remove(neuron_idx);
                }
            }
        } else if cold.shape.len() == 2 {
            // 2D: [out_dim, in_dim] - remove a row
            let out_dim = cold.shape[0];
            let in_dim = cold.shape[1];

            if neuron_idx >= out_dim {
                return StepResult::Error(format!(
                    "Neuron index {} out of bounds (out_dim {})",
                    neuron_idx, out_dim
                ));
            }
            if out_dim <= 1 {
                return StepResult::Error("Cannot prune last neuron".to_string());
            }

            // Remove row: indices [neuron_idx * in_dim .. (neuron_idx + 1) * in_dim]
            let start = neuron_idx * in_dim;
            let end = start + in_dim;

            // Create new weights without the removed row
            let mut new_weights = Vec::with_capacity(cold.weights.len() - in_dim);
            new_weights.extend_from_slice(&cold.weights[..start]);
            new_weights.extend_from_slice(&cold.weights[end..]);

            cold.weights = new_weights;
            cold.shape = vec![out_dim - 1, in_dim];

            // Update temperatures
            if let Some(temps) = &mut cold.temperatures {
                let mut new_temps = Vec::with_capacity(temps.len() - in_dim);
                new_temps.extend_from_slice(&temps[..start]);
                new_temps.extend_from_slice(&temps[end..]);
                *temps = new_temps;
            }
        } else {
            return StepResult::Error(format!(
                "Cannot prune {}D register, only 1D or 2D supported",
                cold.shape.len()
            ));
        }

        StepResult::Continue
    }

    /// Execute INIT_RANDOM: Initialize cold register with random weights
    ///
    /// Instruction format:
    /// - target: Cold register to initialize
    /// - modifier[0..2]: Random seed
    pub(super) fn execute_init_random(&mut self, instr: Instruction) -> StepResult {
        let idx = instr.target().index();

        // TVMR: seed from modifier byte (operands[3])
        let seed = instr.modifier()[0] as u64;

        // Use the runtime_mod implementation
        self.init_cold_random(idx, seed);

        StepResult::Continue
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vm::{assemble, Action, Register, Instruction};

    #[test]
    fn test_alloc_free_hot() {
        let source = r#"
.registers
    H0: i32[4]

.program
    halt
"#;

        let program = assemble(source).unwrap();
        let mut interp = Interpreter::from_program(&program);

        // H0 is allocated, H1 is not
        assert!(interp.hot_reg(0).is_some());
        assert!(interp.hot_reg(1).is_none());

        // Allocate H1 with shape [8]
        // TVMR format: [target, dim0, dim1_hi, dim1_lo]
        let alloc_instr = Instruction::core(
            Action::ALLOC_TENSOR.0,
            [Register::hot(1).0, 8, 0, 0], // target=H1, dim0=8, dim1=0 (1D)
        );
        let result = interp.execute_alloc_tensor(alloc_instr);
        assert!(matches!(result, StepResult::Continue));
        assert!(interp.hot_reg(1).is_some());
        assert_eq!(interp.hot_reg(1).unwrap().shape, vec![8]);

        // Free H1
        let free_instr = Instruction::core(
            Action::FREE_TENSOR.0,
            [Register::hot(1).0, 0, 0, 0],
        );
        let result = interp.execute_free_tensor(free_instr);
        assert!(matches!(result, StepResult::Continue));
        assert!(interp.hot_reg(1).is_none());
    }

    #[test]
    fn test_alloc_free_cold() {
        let mut interp = Interpreter::new();

        // Allocate C0 with shape [4, 2]
        // TVMR format: [target, dim0, dim1_hi, dim1_lo]
        let alloc_instr = Instruction::core(
            Action::ALLOC_TENSOR.0,
            [Register::cold(0).0, 4, 0, 2], // target=C0, dim0=4, dim1=2
        );
        let result = interp.execute_alloc_tensor(alloc_instr);
        assert!(matches!(result, StepResult::Continue));
        assert!(interp.cold_reg(0).is_some());
        assert_eq!(interp.cold_reg(0).unwrap().shape, vec![4, 2]);

        // Free C0
        let free_instr = Instruction::core(
            Action::FREE_TENSOR.0,
            [Register::cold(0).0, 0, 0, 0],
        );
        let result = interp.execute_free_tensor(free_instr);
        assert!(matches!(result, StepResult::Continue));
        assert!(interp.cold_reg(0).is_none());
    }

    #[test]
    fn test_alloc_already_allocated_error() {
        let source = r#"
.registers
    H0: i32[4]

.program
    halt
"#;

        let program = assemble(source).unwrap();
        let mut interp = Interpreter::from_program(&program);

        // Try to allocate H0 again - should error
        let alloc_instr = Instruction::core(
            Action::ALLOC_TENSOR.0,
            [Register::hot(0).0, 8, 0, 0],
        );
        let result = interp.execute_alloc_tensor(alloc_instr);
        assert!(matches!(result, StepResult::Error(_)));
    }

    #[test]
    fn test_free_not_allocated_error() {
        let mut interp = Interpreter::new();

        // Try to free H5 which was never allocated
        let free_instr = Instruction::core(
            Action::FREE_TENSOR.0,
            [Register::hot(5).0, 0, 0, 0],
        );
        let result = interp.execute_free_tensor(free_instr);
        assert!(matches!(result, StepResult::Error(_)));
    }

    #[test]
    fn test_wire_forward_opcode() {
        let source = r#"
.registers
    C0: ternary[4, 2]
    H0: i32[2]
    H1: i32[4]

.program
    halt
"#;

        let program = assemble(source).unwrap();
        let mut interp = Interpreter::from_program(&program);

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

        // Wire forward: H1 = C0 @ H0
        let wire_instr = Instruction::core(
            Action::WIRE_FORWARD.0,
            [Register::hot(1).0, Register::cold(0).0, 0, 0], // target=H1, source=C0, aux=H0(idx 0)
        );
        let result = interp.execute_wire_forward(wire_instr);
        assert!(matches!(result, StepResult::Continue));

        // Check output has values
        let h1 = interp.hot_reg(1).unwrap();
        assert_eq!(h1.shape, vec![4]);
    }

    #[test]
    fn test_wire_skip_opcode() {
        let source = r#"
.registers
    H0: i32[4]
    H1: i32[4]
    H2: i32[4]

.program
    halt
"#;

        let program = assemble(source).unwrap();
        let mut interp = Interpreter::from_program(&program);

        // Set up inputs
        if let Some(h0) = interp.hot_reg_mut(0) {
            h0.data = vec![10, 20, 30, 40];
        }
        if let Some(h1) = interp.hot_reg_mut(1) {
            h1.data = vec![1, 2, 3, 4];
        }

        // Wire skip: H2 = H0 + H1
        let wire_instr = Instruction::core(
            Action::WIRE_SKIP.0,
            [Register::hot(2).0, Register::hot(0).0, 1, 0], // target=H2, source=H0, aux=1 (H1)
        );
        let result = interp.execute_wire_skip(wire_instr);
        assert!(matches!(result, StepResult::Continue));

        // Check output
        let h2 = interp.hot_reg(2).unwrap();
        assert_eq!(h2.data, vec![11, 22, 33, 44]);
    }

    #[test]
    fn test_grow_neuron_1d() {
        let source = r#"
.registers
    C0: ternary[4]

.program
    halt
"#;

        let program = assemble(source).unwrap();
        let mut interp = Interpreter::from_program(&program);

        // Initial size is 4
        assert_eq!(interp.cold_reg(0).unwrap().shape, vec![4]);
        assert_eq!(interp.cold_reg(0).unwrap().weights.len(), 4);

        // Grow by 3 neurons
        // TVMR format: [target, aux(neurons_to_add), seed_hi, seed_lo]
        let grow_instr = Instruction::core(
            Action::GROW_NEURON.0,
            [Register::cold(0).0, 0xFF, 3, 42], // target=C0, unused, aux=3, modifier[0]=42
        );
        let result = interp.execute_grow_neuron(grow_instr);
        assert!(matches!(result, StepResult::Continue));

        // New size should be 7
        assert_eq!(interp.cold_reg(0).unwrap().shape, vec![7]);
        assert_eq!(interp.cold_reg(0).unwrap().weights.len(), 7);
    }

    #[test]
    fn test_grow_neuron_2d() {
        let source = r#"
.registers
    C0: ternary[4, 2]

.program
    halt
"#;

        let program = assemble(source).unwrap();
        let mut interp = Interpreter::from_program(&program);

        // Initial: [4, 2] = 8 signals
        assert_eq!(interp.cold_reg(0).unwrap().shape, vec![4, 2]);
        assert_eq!(interp.cold_reg(0).unwrap().weights.len(), 8);

        // Grow by 2 neurons
        let grow_instr = Instruction::core(
            Action::GROW_NEURON.0,
            [Register::cold(0).0, 0xFF, 2, 123], // target=C0, unused, aux=2, modifier[0]=123
        );
        let result = interp.execute_grow_neuron(grow_instr);
        assert!(matches!(result, StepResult::Continue));

        // New: [6, 2] = 12 signals
        assert_eq!(interp.cold_reg(0).unwrap().shape, vec![6, 2]);
        assert_eq!(interp.cold_reg(0).unwrap().weights.len(), 12);
    }

    #[test]
    fn test_prune_neuron_1d() {
        let source = r#"
.registers
    C0: ternary[5]

.program
    halt
"#;

        let program = assemble(source).unwrap();
        let mut interp = Interpreter::from_program(&program);

        // Set up signals with distinct values
        if let Some(c0) = interp.cold_reg_mut(0) {
            for (i, w) in c0.weights.iter_mut().enumerate() {
                w.polarity = 1;
                w.magnitude = (i * 10 + 10) as u8; // 10, 20, 30, 40, 50
            }
        }

        // Prune index 2 (value 30)
        let prune_instr = Instruction::core(
            Action::PRUNE_NEURON.0,
            [Register::cold(0).0, 0xFF, 2, 0], // target=C0, unused, aux=2
        );
        let result = interp.execute_prune_neuron(prune_instr);
        assert!(matches!(result, StepResult::Continue));

        // New size should be 4
        let c0 = interp.cold_reg(0).unwrap();
        assert_eq!(c0.shape, vec![4]);
        assert_eq!(c0.weights.len(), 4);

        // Values should be 10, 20, 40, 50 (30 removed)
        let mags: Vec<u8> = c0.weights.iter().map(|s| s.magnitude).collect();
        assert_eq!(mags, vec![10, 20, 40, 50]);
    }

    #[test]
    fn test_prune_neuron_2d() {
        let source = r#"
.registers
    C0: ternary[3, 2]

.program
    halt
"#;

        let program = assemble(source).unwrap();
        let mut interp = Interpreter::from_program(&program);

        // Initial: [3, 2] = 6 signals
        assert_eq!(interp.cold_reg(0).unwrap().shape, vec![3, 2]);
        assert_eq!(interp.cold_reg(0).unwrap().weights.len(), 6);

        // Prune neuron 1 (second row)
        let prune_instr = Instruction::core(
            Action::PRUNE_NEURON.0,
            [Register::cold(0).0, 0xFF, 1, 0], // target=C0, unused, aux=1
        );
        let result = interp.execute_prune_neuron(prune_instr);
        assert!(matches!(result, StepResult::Continue));

        // New: [2, 2] = 4 signals
        assert_eq!(interp.cold_reg(0).unwrap().shape, vec![2, 2]);
        assert_eq!(interp.cold_reg(0).unwrap().weights.len(), 4);
    }

    #[test]
    fn test_prune_last_neuron_error() {
        let source = r#"
.registers
    C0: ternary[1]

.program
    halt
"#;

        let program = assemble(source).unwrap();
        let mut interp = Interpreter::from_program(&program);

        // Try to prune the only neuron - should error
        let prune_instr = Instruction::core(
            Action::PRUNE_NEURON.0,
            [Register::cold(0).0, 0xFF, 0, 0],
        );
        let result = interp.execute_prune_neuron(prune_instr);
        assert!(matches!(result, StepResult::Error(_)));
    }

    #[test]
    fn test_init_random() {
        let source = r#"
.registers
    C0: ternary[8]

.program
    halt
"#;

        let program = assemble(source).unwrap();
        let mut interp = Interpreter::from_program(&program);

        // Initially all zeros
        let initial: Vec<u8> = interp.cold_reg(0).unwrap().weights.iter().map(|s| s.magnitude).collect();
        assert!(initial.iter().all(|&m| m == 0));

        // Initialize with random
        // TVMR format: [target, unused, unused, seed_byte]
        let init_instr = Instruction::core(
            Action::INIT_RANDOM.0,
            [Register::cold(0).0, 0xFF, 0, 42], // target=C0, modifier[0]=42 (seed)
        );
        let result = interp.execute_init_random(init_instr);
        assert!(matches!(result, StepResult::Continue));

        // Now should have non-zero values
        let after: Vec<u8> = interp.cold_reg(0).unwrap().weights.iter().map(|s| s.magnitude).collect();
        assert!(after.iter().any(|&m| m > 0));
    }

    #[test]
    fn test_grow_zero_error() {
        let source = r#"
.registers
    C0: ternary[4]

.program
    halt
"#;

        let program = assemble(source).unwrap();
        let mut interp = Interpreter::from_program(&program);

        // Try to grow by 0 - should error
        // aux=0 means 0 neurons
        let grow_instr = Instruction::core(
            Action::GROW_NEURON.0,
            [Register::cold(0).0, 0xFF, 0, 0],
        );
        let result = interp.execute_grow_neuron(grow_instr);
        assert!(matches!(result, StepResult::Error(_)));
    }
}
