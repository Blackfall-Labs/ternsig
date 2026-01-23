//! Interpreter - Runtime execution engine for Ternsig VM
//!
//! Executes Ternsig programs against a typed register file.
//! Integrates with Thermograms for weight persistence and Learning for training.
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────┐
//! │                  Interpreter                     │
//! ├─────────────────────────────────────────────────┤
//! │  Hot Regs [H0-HF]:  Vec<i32> activations        │
//! │  Cold Regs [C0-CF]: Vec<Signal> weights         │
//! │  Param Regs [P0-PF]: i32 scalars                │
//! │  Shape Regs [S0-SF]: Vec<usize> shapes          │
//! ├─────────────────────────────────────────────────┤
//! │  Program Counter, Call Stack, Loop Stack        │
//! │  Input/Output Buffers (i32 only, NO FLOATS)     │
//! └─────────────────────────────────────────────────┘
//! ```

use super::{AssembledProgram, RegisterMeta, Action, Dtype, Instruction, Register};
use std::collections::VecDeque;
use crate::Signal;

/// Result of executing a single instruction
#[derive(Debug, Clone)]
pub enum StepResult {
    /// Continue to next instruction
    Continue,
    /// Program halted normally
    Halt,
    /// Break out of current loop
    Break,
    /// Return from subroutine
    Return,
    /// Execution yielded to Learning ISA
    Yield(DomainOp),
    /// Program ended (PC past end)
    Ended,
    /// Runtime error
    Error(String),
}

/// Domain operations that require external execution
#[derive(Debug, Clone)]
pub enum DomainOp {
    /// Load weights from Thermogram
    LoadWeights { register: Register, key: String },
    /// Store weights to Thermogram
    StoreWeights { register: Register, key: String },
    /// Consolidate hot → cold
    Consolidate,
    /// Compute error signal (i32 scaled by 256)
    ComputeError { target: i32, output: i32 },
}

/// Loop state for LOOP instruction
#[derive(Debug, Clone)]
struct LoopState {
    /// PC of the LOOP instruction
    start_pc: usize,
    /// Remaining iterations
    remaining: u16,
}

/// Hot register (activation buffer)
#[derive(Debug, Clone)]
pub struct HotBuffer {
    /// Data as i32 (quantized activations)
    pub data: Vec<i32>,
    /// Shape
    pub shape: Vec<usize>,
}

impl HotBuffer {
    pub fn new(shape: Vec<usize>) -> Self {
        let size: usize = shape.iter().product();
        Self {
            data: vec![0; size],
            shape,
        }
    }

    pub fn zeros(shape: Vec<usize>) -> Self {
        Self::new(shape)
    }

    /// Create from Signal slice
    pub fn from_ternary(signals: &[Signal], shape: Vec<usize>) -> Self {
        let data: Vec<i32> = signals
            .iter()
            .map(|s| s.polarity as i32 * s.magnitude as i32)
            .collect();
        Self { data, shape }
    }

    /// Convert to Signal vec
    pub fn to_ternary(&self) -> Vec<Signal> {
        self.data.iter().map(|&v| Signal::from_signed_i32(v)).collect()
    }

    pub fn numel(&self) -> usize {
        self.data.len()
    }
}

/// Cold register (weight buffer)
#[derive(Debug, Clone)]
pub struct ColdBuffer {
    /// Weights as Signal
    pub weights: Vec<Signal>,
    /// Shape
    pub shape: Vec<usize>,
    /// Thermogram key for persistence
    pub thermogram_key: Option<String>,
    /// Whether this buffer is frozen (non-trainable)
    pub frozen: bool,
}

impl ColdBuffer {
    pub fn new(shape: Vec<usize>) -> Self {
        let size: usize = shape.iter().product();
        Self {
            weights: vec![Signal::zero(); size],
            shape,
            thermogram_key: None,
            frozen: false,
        }
    }

    pub fn with_key(mut self, key: impl Into<String>) -> Self {
        self.thermogram_key = Some(key.into());
        self
    }

    /// Check if this buffer is frozen (non-trainable).
    pub fn is_frozen(&self) -> bool {
        self.frozen
    }

    /// Get immutable reference to weights.
    pub fn weights(&self) -> &[Signal] {
        &self.weights
    }

    /// Get mutable reference to weights.
    pub fn weights_mut(&mut self) -> &mut [Signal] {
        &mut self.weights
    }

    pub fn numel(&self) -> usize {
        self.weights.len()
    }
}

/// TensorISA Interpreter - NO FLOATS, pure integer/ternary
pub struct Interpreter {
    /// Program instructions
    program: Vec<Instruction>,
    /// Program counter
    pc: usize,
    /// Hot registers (activations)
    hot_regs: Vec<Option<HotBuffer>>,
    /// Cold registers (weights)
    cold_regs: Vec<Option<ColdBuffer>>,
    /// Param registers (i32 scalars, scaled by 256)
    param_regs: Vec<i32>,
    /// Shape registers (dimensions)
    shape_regs: Vec<Vec<usize>>,
    /// Call stack for subroutines
    call_stack: Vec<usize>,
    /// Loop stack
    loop_stack: Vec<LoopState>,
    /// Input buffer (i32)
    input_buffer: Vec<i32>,
    /// Output buffer (i32)
    output_buffer: Vec<i32>,
    /// Target buffer (for learning, i32)
    target_buffer: Vec<i32>,
    /// Pressure accumulators (for mastery learning) - indexed by cold register
    pressure_regs: Vec<Option<Vec<i32>>>,
    /// Current babble scale (i32, 0-255 range)
    babble_scale: i32,
    /// Babble phase (deterministic pattern)
    babble_phase: usize,
    /// Current error (from last learning step, scaled by 256)
    current_error: i32,
}

impl Interpreter {
    /// Create a new interpreter with empty program
    pub fn new() -> Self {
        Self {
            program: Vec::new(),
            pc: 0,
            hot_regs: vec![None; 16],
            cold_regs: vec![None; 16],
            param_regs: vec![0; 16],
            shape_regs: vec![Vec::new(); 16],
            call_stack: Vec::with_capacity(16),
            loop_stack: Vec::with_capacity(8),
            input_buffer: Vec::new(),
            output_buffer: Vec::new(),
            target_buffer: Vec::new(),
            pressure_regs: vec![None; 16],
            babble_scale: 5, // ~2% of 255
            babble_phase: 0,
            current_error: 0,
        }
    }

    /// Create from assembled program
    pub fn from_program(program: &AssembledProgram) -> Self {
        let mut interp = Self::new();
        interp.load_program(program);
        interp
    }

    /// Load a program
    pub fn load_program(&mut self, program: &AssembledProgram) {
        self.program = program.instructions.clone();
        self.pc = 0;
        self.call_stack.clear();
        self.loop_stack.clear();

        // Initialize registers from definitions
        for reg_meta in &program.registers {
            match reg_meta.id.bank() {
                super::RegisterBank::Hot => {
                    let idx = reg_meta.id.index();
                    self.hot_regs[idx] = Some(HotBuffer::new(reg_meta.shape.clone()));
                }
                super::RegisterBank::Cold => {
                    let idx = reg_meta.id.index();
                    let mut buf = ColdBuffer::new(reg_meta.shape.clone());
                    buf.thermogram_key = reg_meta.thermogram_key.clone();
                    buf.frozen = reg_meta.frozen;
                    self.cold_regs[idx] = Some(buf);
                }
                super::RegisterBank::Shape => {
                    let idx = reg_meta.id.index();
                    self.shape_regs[idx] = reg_meta.shape.clone();
                }
                _ => {}
            }
        }
    }

    /// Set input buffer from Signal (primary API)
    pub fn set_input(&mut self, input: &[Signal]) {
        self.input_buffer = input
            .iter()
            .map(|s| s.polarity as i32 * s.magnitude as i32)
            .collect();
    }

    /// Set input buffer with i32 values directly
    pub fn set_input_i32(&mut self, input: &[i32]) {
        self.input_buffer = input.to_vec();
    }

    /// Set target buffer (called before learning pass)
    pub fn set_target(&mut self, target: &[Signal]) {
        self.target_buffer = target
            .iter()
            .map(|s| s.polarity as i32 * s.magnitude as i32)
            .collect();
    }

    /// Set target buffer with i32 values directly
    pub fn set_target_i32(&mut self, target: &[i32]) {
        self.target_buffer = target.to_vec();
    }

    /// Get output buffer as i32 (called after forward pass)
    pub fn output_i32(&self) -> &[i32] {
        &self.output_buffer
    }

    /// Get output buffer as Signal
    pub fn output(&self) -> Vec<Signal> {
        self.output_buffer
            .iter()
            .map(|&v| Signal::from_signed_i32(v))
            .collect()
    }

    /// Get immutable iterator over cold buffers.
    pub fn cold_buffers(&self) -> impl Iterator<Item = &ColdBuffer> {
        self.cold_regs.iter().filter_map(|r| r.as_ref())
    }

    /// Get mutable iterator over cold buffers.
    pub fn cold_buffers_mut(&mut self) -> impl Iterator<Item = &mut ColdBuffer> {
        self.cold_regs.iter_mut().filter_map(|r| r.as_mut())
    }

    /// Get a specific cold buffer by index.
    pub fn cold_buffer(&self, idx: usize) -> Option<&ColdBuffer> {
        self.cold_regs.get(idx).and_then(|r| r.as_ref())
    }

    /// Get a specific cold buffer mutably by index.
    pub fn cold_buffer_mut(&mut self, idx: usize) -> Option<&mut ColdBuffer> {
        self.cold_regs.get_mut(idx).and_then(|r| r.as_mut())
    }

    /// Reset program counter to start
    pub fn reset(&mut self) {
        self.pc = 0;
        self.call_stack.clear();
        self.loop_stack.clear();
    }

    /// Execute single instruction
    pub fn step(&mut self) -> StepResult {
        if self.pc >= self.program.len() {
            return StepResult::Ended;
        }

        let instr = self.program[self.pc];
        let result = self.execute(instr);

        // Advance PC unless instruction modified it
        if !instr.modifies_pc() && !matches!(result, StepResult::Break | StepResult::Return) {
            self.pc += 1;
        }

        result
    }

    /// Run until halt or error
    pub fn run(&mut self) -> StepResult {
        loop {
            match self.step() {
                StepResult::Continue => continue,
                StepResult::Halt => return StepResult::Halt,
                StepResult::Ended => return StepResult::Ended,
                StepResult::Error(e) => return StepResult::Error(e),
                StepResult::Yield(op) => return StepResult::Yield(op),
                StepResult::Break | StepResult::Return => continue,
            }
        }
    }

    /// Forward pass: set input, run, get output (Signal API)
    pub fn forward(&mut self, input: &[Signal]) -> Result<Vec<Signal>, String> {
        self.set_input(input);
        self.reset();

        match self.run() {
            StepResult::Halt | StepResult::Ended => Ok(self.output()),
            StepResult::Error(e) => Err(e),
            _ => Err("Unexpected termination".to_string()),
        }
    }

    /// Forward pass with i32 input/output
    pub fn forward_i32(&mut self, input: &[i32]) -> Result<Vec<i32>, String> {
        self.set_input_i32(input);
        self.reset();

        match self.run() {
            StepResult::Halt | StepResult::Ended => Ok(self.output_buffer.clone()),
            StepResult::Error(e) => Err(e),
            _ => Err("Unexpected termination".to_string()),
        }
    }

    /// Execute a single instruction
    fn execute(&mut self, instr: Instruction) -> StepResult {
        match instr.action {
            // === System ===
            Action::NOP => StepResult::Continue,
            Action::HALT => StepResult::Halt,
            Action::RESET => {
                self.reset();
                StepResult::Continue
            }

            // === Register Management ===
            Action::LOAD_INPUT => {
                let idx = instr.target.index();
                if let Some(buf) = &mut self.hot_regs[idx] {
                    let len = buf.data.len().min(self.input_buffer.len());
                    buf.data[..len].copy_from_slice(&self.input_buffer[..len]);
                }
                StepResult::Continue
            }
            Action::STORE_OUTPUT => {
                let idx = instr.source.index();
                if let Some(buf) = &self.hot_regs[idx] {
                    self.output_buffer = buf.data.clone();
                }
                StepResult::Continue
            }
            Action::ZERO_REG => {
                let idx = instr.target.index();
                if instr.target.is_hot() {
                    if let Some(buf) = &mut self.hot_regs[idx] {
                        buf.data.fill(0);
                    }
                }
                StepResult::Continue
            }
            Action::COPY_REG => {
                let src_idx = instr.source.index();
                let dst_idx = instr.target.index();
                if instr.source.is_hot() && instr.target.is_hot() {
                    if let Some(src) = self.hot_regs[src_idx].clone() {
                        self.hot_regs[dst_idx] = Some(src);
                    }
                }
                StepResult::Continue
            }

            // === Forward Ops ===
            Action::TERNARY_MATMUL => self.execute_ternary_matmul(instr),
            Action::ADD => self.execute_add(instr),
            Action::SUB => self.execute_sub(instr),
            Action::MUL => self.execute_mul(instr),
            Action::RELU => self.execute_relu(instr),
            Action::SIGMOID => self.execute_sigmoid(instr),
            Action::SHIFT => self.execute_shift(instr),
            Action::CMP_GT => self.execute_cmp_gt(instr),
            Action::MAX_REDUCE => self.execute_max_reduce(instr),

            // === Ternary Ops ===
            Action::DEQUANTIZE => self.execute_dequantize(instr),
            Action::TERNARY_ADD_BIAS => self.execute_ternary_add_bias(instr),
            Action::EMBED_LOOKUP => self.execute_embed_lookup(instr),
            Action::REDUCE_AVG => self.execute_reduce_avg(instr),
            Action::SLICE => self.execute_slice(instr),
            Action::ARGMAX => self.execute_argmax(instr),
            Action::CONCAT => self.execute_concat(instr),
            Action::SQUEEZE => self.execute_squeeze(instr),
            Action::UNSQUEEZE => self.execute_unsqueeze(instr),
            Action::TRANSPOSE => self.execute_transpose(instr),
            Action::GATE_UPDATE => self.execute_gate_update(instr),

            // === Learning Ops ===
            Action::ADD_BABBLE => self.execute_add_babble(instr),
            Action::MARK_ELIGIBILITY => {
                // Mark eligibility is a no-op in inference mode
                StepResult::Continue
            }
            Action::LOAD_TARGET => self.execute_load_target(instr),
            Action::MASTERY_UPDATE => self.execute_mastery_update(instr),
            Action::MASTERY_COMMIT => self.execute_mastery_commit(instr),

            // === Control Flow ===
            Action::LOOP => self.execute_loop(instr),
            Action::END_LOOP => self.execute_end_loop(),
            Action::BREAK => self.execute_break(),
            Action::JUMP => {
                self.pc = instr.count() as usize;
                StepResult::Continue
            }
            Action::CALL => {
                self.call_stack.push(self.pc + 1);
                self.pc = instr.count() as usize;
                StepResult::Continue
            }
            Action::RETURN => {
                if let Some(ret_pc) = self.call_stack.pop() {
                    self.pc = ret_pc;
                    StepResult::Continue
                } else {
                    StepResult::Return
                }
            }

            _ => StepResult::Error(format!("Unimplemented opcode: {}", instr.action)),
        }
    }

    // === Operation Implementations ===

    fn execute_ternary_matmul(&mut self, instr: Instruction) -> StepResult {
        let weights_idx = instr.source.index();
        let input_idx = instr.aux as usize & 0x0F;
        let output_idx = instr.target.index();

        // Get weight and input
        let weights = match &self.cold_regs[weights_idx] {
            Some(buf) => buf,
            None => return StepResult::Error(format!("Cold register C{} not allocated", weights_idx)),
        };

        let input = match &self.hot_regs[input_idx] {
            Some(buf) => buf,
            None => return StepResult::Error(format!("Hot register H{} not allocated", input_idx)),
        };

        // Compute output dimensions
        let (out_dim, in_dim) = if weights.shape.len() >= 2 {
            (weights.shape[0], weights.shape[1])
        } else {
            return StepResult::Error("Weights must be 2D".to_string());
        };

        // Integer matrix multiply
        let mut output_data = vec![0i64; out_dim];
        for o in 0..out_dim {
            let mut sum = 0i64;
            for i in 0..in_dim.min(input.data.len()) {
                let w = &weights.weights[o * in_dim + i];
                let effective = w.polarity as i64 * w.magnitude as i64;
                sum += effective * input.data[i] as i64;
            }
            output_data[o] = sum;
        }

        // Store result (as i32, clamped)
        let output_i32: Vec<i32> = output_data
            .iter()
            .map(|&v| v.clamp(i32::MIN as i64, i32::MAX as i64) as i32)
            .collect();

        self.hot_regs[output_idx] = Some(HotBuffer {
            data: output_i32,
            shape: vec![out_dim],
        });

        StepResult::Continue
    }

    fn execute_add(&mut self, instr: Instruction) -> StepResult {
        let src_idx = instr.source.index();
        let other_idx = instr.aux as usize & 0x0F;
        let dst_idx = instr.target.index();

        // Handle bias addition (cold + hot)
        if instr.source.is_hot() && Register(instr.aux).is_cold() {
            // Hot + Cold bias
            let src = match &self.hot_regs[src_idx] {
                Some(buf) => buf.clone(),
                None => return StepResult::Error("Source not allocated".to_string()),
            };

            let bias = match &self.cold_regs[other_idx] {
                Some(buf) => buf,
                None => return StepResult::Error("Bias not allocated".to_string()),
            };

            let mut result = src.data.clone();
            for (i, val) in result.iter_mut().enumerate() {
                if i < bias.weights.len() {
                    let b = &bias.weights[i];
                    let bias_val = b.polarity as i64 * b.magnitude as i64 * 255;
                    *val = (*val as i64 + bias_val).clamp(i32::MIN as i64, i32::MAX as i64) as i32;
                }
            }

            self.hot_regs[dst_idx] = Some(HotBuffer {
                data: result,
                shape: src.shape.clone(),
            });
        } else if instr.source.is_hot() && Register(instr.aux).is_hot() {
            // Hot + Hot
            let src = match &self.hot_regs[src_idx] {
                Some(buf) => buf.clone(),
                None => return StepResult::Error("Source not allocated".to_string()),
            };

            let other = match &self.hot_regs[other_idx] {
                Some(buf) => buf,
                None => return StepResult::Error("Other not allocated".to_string()),
            };

            let mut result = src.data.clone();
            for (i, val) in result.iter_mut().enumerate() {
                if i < other.data.len() {
                    *val = val.saturating_add(other.data[i]);
                }
            }

            self.hot_regs[dst_idx] = Some(HotBuffer {
                data: result,
                shape: src.shape.clone(),
            });
        }

        StepResult::Continue
    }

    fn execute_relu(&mut self, instr: Instruction) -> StepResult {
        let src_idx = instr.source.index();
        let dst_idx = instr.target.index();

        let src = match &self.hot_regs[src_idx] {
            Some(buf) => buf.clone(),
            None => return StepResult::Error("Source not allocated".to_string()),
        };

        let result: Vec<i32> = src.data.iter().map(|&v| v.max(0)).collect();

        self.hot_regs[dst_idx] = Some(HotBuffer {
            data: result,
            shape: src.shape.clone(),
        });

        StepResult::Continue
    }

    fn execute_shift(&mut self, instr: Instruction) -> StepResult {
        let src_idx = instr.source.index();
        let dst_idx = instr.target.index();
        let shift_amount = instr.aux as u32;

        let src = match &self.hot_regs[src_idx] {
            Some(buf) => buf.clone(),
            None => return StepResult::Error("Source not allocated".to_string()),
        };

        let result: Vec<i32> = src.data.iter().map(|&v| v >> shift_amount).collect();

        self.hot_regs[dst_idx] = Some(HotBuffer {
            data: result,
            shape: src.shape.clone(),
        });

        StepResult::Continue
    }

    fn execute_sub(&mut self, instr: Instruction) -> StepResult {
        let src_idx = instr.source.index();
        let other_idx = instr.aux as usize & 0x0F;
        let dst_idx = instr.target.index();

        let src = match &self.hot_regs[src_idx] {
            Some(buf) => buf.clone(),
            None => return StepResult::Error("Source not allocated".to_string()),
        };

        let other = match &self.hot_regs[other_idx] {
            Some(buf) => buf,
            None => return StepResult::Error("Other not allocated".to_string()),
        };

        let result: Vec<i32> = src
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(&a, &b)| a.saturating_sub(b))
            .collect();

        self.hot_regs[dst_idx] = Some(HotBuffer {
            data: result,
            shape: src.shape.clone(),
        });

        StepResult::Continue
    }

    fn execute_mul(&mut self, instr: Instruction) -> StepResult {
        let src_idx = instr.source.index();
        let other_idx = instr.aux as usize & 0x0F;
        let dst_idx = instr.target.index();

        let src = match &self.hot_regs[src_idx] {
            Some(buf) => buf.clone(),
            None => return StepResult::Error("Source not allocated".to_string()),
        };

        let other = match &self.hot_regs[other_idx] {
            Some(buf) => buf,
            None => return StepResult::Error("Other not allocated".to_string()),
        };

        // Multiply with scaling (divide by 256 to keep in range)
        let result: Vec<i32> = src
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(&a, &b)| ((a as i64 * b as i64) >> 8) as i32)
            .collect();

        self.hot_regs[dst_idx] = Some(HotBuffer {
            data: result,
            shape: src.shape.clone(),
        });

        StepResult::Continue
    }

    fn execute_sigmoid(&mut self, instr: Instruction) -> StepResult {
        let src_idx = instr.source.index();
        let dst_idx = instr.target.index();
        // Gain from modifier (default 64 = 4.0 * 16)
        let gain = if instr.modifier[0] > 0 {
            instr.modifier[0] as i32
        } else {
            64
        };

        let src = match &self.hot_regs[src_idx] {
            Some(buf) => buf.clone(),
            None => return StepResult::Error("Source not allocated".to_string()),
        };

        // Integer sigmoid approximation using piecewise linear
        // Input is i32, output is i32 (0-255 range)
        // sigmoid(x) ≈ clamp(x * gain / 256 + 128, 0, 255)
        let result: Vec<i32> = src
            .data
            .iter()
            .map(|&v| {
                // Scale by gain, shift to 0-255 range
                let scaled = (v as i64 * gain as i64) >> 10;
                (scaled + 128).clamp(0, 255) as i32
            })
            .collect();

        self.hot_regs[dst_idx] = Some(HotBuffer {
            data: result,
            shape: src.shape.clone(),
        });

        StepResult::Continue
    }

    fn execute_cmp_gt(&mut self, instr: Instruction) -> StepResult {
        let src_idx = instr.source.index();
        let other_idx = instr.aux as usize & 0x0F;
        let dst_idx = instr.target.index();

        let src = match &self.hot_regs[src_idx] {
            Some(buf) => buf.clone(),
            None => return StepResult::Error("Source not allocated".to_string()),
        };

        let other = match &self.hot_regs[other_idx] {
            Some(buf) => buf,
            None => return StepResult::Error("Other not allocated".to_string()),
        };

        let result: Vec<i32> = src
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(&a, &b)| if a > b { 255 } else { 0 })
            .collect();

        self.hot_regs[dst_idx] = Some(HotBuffer {
            data: result,
            shape: src.shape.clone(),
        });

        StepResult::Continue
    }

    fn execute_max_reduce(&mut self, instr: Instruction) -> StepResult {
        let src_idx = instr.source.index();
        let dst_idx = instr.target.index();

        let src = match &self.hot_regs[src_idx] {
            Some(buf) => buf.clone(),
            None => return StepResult::Error("Source not allocated".to_string()),
        };

        let max_val = src.data.iter().cloned().max().unwrap_or(0);

        self.hot_regs[dst_idx] = Some(HotBuffer {
            data: vec![max_val],
            shape: vec![1],
        });

        StepResult::Continue
    }

    fn execute_ternary_add_bias(&mut self, instr: Instruction) -> StepResult {
        let src_idx = instr.source.index();
        let bias_idx = instr.aux as usize & 0x0F;
        let dst_idx = instr.target.index();

        let src = match &self.hot_regs[src_idx] {
            Some(buf) => buf.clone(),
            None => return StepResult::Error("Source not allocated".to_string()),
        };

        let bias = match &self.cold_regs[bias_idx] {
            Some(buf) => buf,
            None => return StepResult::Error("Bias not allocated".to_string()),
        };

        let mut result = src.data.clone();
        for (i, val) in result.iter_mut().enumerate() {
            if i < bias.weights.len() {
                let b = &bias.weights[i];
                // Bias contribution: polarity * magnitude * 256 (scaling factor)
                let bias_val = b.polarity as i64 * b.magnitude as i64 * 256;
                *val = (*val as i64 + bias_val).clamp(i32::MIN as i64, i32::MAX as i64) as i32;
            }
        }

        self.hot_regs[dst_idx] = Some(HotBuffer {
            data: result,
            shape: src.shape.clone(),
        });

        StepResult::Continue
    }

    fn execute_embed_lookup(&mut self, instr: Instruction) -> StepResult {
        // EMBED_LOOKUP: target[i] = table[indices[i]]
        // target = output hot register
        // source = embedding table (cold, 2D: num_embeddings x embedding_dim)
        // aux = indices hot register

        let table_idx = instr.source.index();
        let indices_idx = instr.aux as usize & 0x0F;
        let output_idx = instr.target.index();

        // Get embedding table
        let table = match &self.cold_regs[table_idx] {
            Some(buf) => buf,
            None => return StepResult::Error(format!("Embedding table C{} not allocated", table_idx)),
        };

        // Get indices
        let indices = match &self.hot_regs[indices_idx] {
            Some(buf) => &buf.data,
            None => return StepResult::Error(format!("Indices H{} not allocated", indices_idx)),
        };

        // Table shape: [num_embeddings, embedding_dim]
        let (num_embeddings, embedding_dim) = if table.shape.len() >= 2 {
            (table.shape[0], table.shape[1])
        } else if table.shape.len() == 1 {
            // 1D table: treat as single embedding
            (1, table.shape[0])
        } else {
            return StepResult::Error("Embedding table must have shape".to_string());
        };

        // Output shape: [num_indices, embedding_dim]
        let num_indices = indices.len();
        let mut output_data = vec![0i32; num_indices * embedding_dim];

        for (i, &idx_val) in indices.iter().enumerate() {
            let idx = idx_val.max(0) as usize;
            if idx < num_embeddings {
                // Copy embedding for this index
                let table_offset = idx * embedding_dim;
                for d in 0..embedding_dim {
                    if table_offset + d < table.weights.len() {
                        let w = &table.weights[table_offset + d];
                        output_data[i * embedding_dim + d] = w.polarity as i32 * w.magnitude as i32;
                    }
                }
            }
            // Out-of-bounds indices get zeros (already initialized)
        }

        self.hot_regs[output_idx] = Some(HotBuffer {
            data: output_data,
            shape: vec![num_indices, embedding_dim],
        });

        StepResult::Continue
    }

    fn execute_reduce_avg(&mut self, instr: Instruction) -> StepResult {
        // REDUCE_AVG: target[0] = mean(source[start..start+count])
        // target = output hot register (single element)
        // source = input hot register
        // aux = start index
        // modifier[0] = count

        let src_idx = instr.source.index();
        let dst_idx = instr.target.index();
        let start = instr.aux as usize;
        let count = instr.modifier[0] as usize;

        let src = match &self.hot_regs[src_idx] {
            Some(buf) => &buf.data,
            None => return StepResult::Error(format!("Source H{} not allocated", src_idx)),
        };

        if count == 0 {
            // Zero count = zero output
            self.hot_regs[dst_idx] = Some(HotBuffer {
                data: vec![0],
                shape: vec![1],
            });
            return StepResult::Continue;
        }

        // Sum the range
        let end = (start + count).min(src.len());
        let actual_count = end.saturating_sub(start);

        let sum: i32 = src.get(start..end)
            .map(|slice| slice.iter().sum())
            .unwrap_or(0);

        let avg = if actual_count > 0 {
            sum / actual_count as i32
        } else {
            0
        };

        self.hot_regs[dst_idx] = Some(HotBuffer {
            data: vec![avg],
            shape: vec![1],
        });

        StepResult::Continue
    }

    fn execute_slice(&mut self, instr: Instruction) -> StepResult {
        // SLICE: target = source[start..start+len]
        let src_idx = instr.source.index();
        let dst_idx = instr.target.index();
        let start = instr.aux as usize;
        let len = instr.modifier[0] as usize;

        let src = match &self.hot_regs[src_idx] {
            Some(buf) => &buf.data,
            None => return StepResult::Error(format!("Source H{} not allocated", src_idx)),
        };

        let end = (start + len).min(src.len());
        let slice_data: Vec<i32> = src.get(start..end)
            .map(|s| s.to_vec())
            .unwrap_or_default();

        let slice_len = slice_data.len();
        self.hot_regs[dst_idx] = Some(HotBuffer {
            data: slice_data,
            shape: vec![slice_len],
        });

        StepResult::Continue
    }

    fn execute_argmax(&mut self, instr: Instruction) -> StepResult {
        // ARGMAX: target[0] = index of max value in source
        let src_idx = instr.source.index();
        let dst_idx = instr.target.index();

        let src = match &self.hot_regs[src_idx] {
            Some(buf) => &buf.data,
            None => return StepResult::Error(format!("Source H{} not allocated", src_idx)),
        };

        let max_idx = if src.is_empty() {
            0
        } else {
            src.iter()
                .enumerate()
                .max_by_key(|(_, v)| *v)
                .map(|(i, _)| i)
                .unwrap_or(0)
        };

        self.hot_regs[dst_idx] = Some(HotBuffer {
            data: vec![max_idx as i32],
            shape: vec![1],
        });

        StepResult::Continue
    }

    fn execute_concat(&mut self, instr: Instruction) -> StepResult {
        // CONCAT: target = concat(source, other)
        let src_idx = instr.source.index();
        let other_idx = instr.aux as usize & 0x0F;
        let dst_idx = instr.target.index();

        let src = match &self.hot_regs[src_idx] {
            Some(buf) => buf.data.clone(),
            None => return StepResult::Error(format!("Source H{} not allocated", src_idx)),
        };

        let other = match &self.hot_regs[other_idx] {
            Some(buf) => &buf.data,
            None => return StepResult::Error(format!("Other H{} not allocated", other_idx)),
        };

        let mut result = src;
        result.extend_from_slice(other);
        let len = result.len();

        self.hot_regs[dst_idx] = Some(HotBuffer {
            data: result,
            shape: vec![len],
        });

        StepResult::Continue
    }

    fn execute_squeeze(&mut self, instr: Instruction) -> StepResult {
        // SQUEEZE: For 1D signals, this is effectively a copy
        // In higher-dimensional contexts, removes dim of size 1
        let src_idx = instr.source.index();
        let dst_idx = instr.target.index();

        if let Some(buf) = self.hot_regs[src_idx].clone() {
            self.hot_regs[dst_idx] = Some(buf);
        }

        StepResult::Continue
    }

    fn execute_unsqueeze(&mut self, instr: Instruction) -> StepResult {
        // UNSQUEEZE: For 1D signals, this is effectively a copy
        // In higher-dimensional contexts, adds dim of size 1
        let src_idx = instr.source.index();
        let dst_idx = instr.target.index();

        if let Some(buf) = self.hot_regs[src_idx].clone() {
            self.hot_regs[dst_idx] = Some(buf);
        }

        StepResult::Continue
    }

    fn execute_transpose(&mut self, instr: Instruction) -> StepResult {
        // TRANSPOSE: For 1D signals, this is effectively a copy
        // In higher-dimensional contexts, swaps dimensions
        let src_idx = instr.source.index();
        let dst_idx = instr.target.index();

        if let Some(buf) = self.hot_regs[src_idx].clone() {
            self.hot_regs[dst_idx] = Some(buf);
        }

        StepResult::Continue
    }

    fn execute_gate_update(&mut self, instr: Instruction) -> StepResult {
        // GATE_UPDATE: target = gate * update + (1 - gate) * state
        // Fused operation for GRU-style gated updates
        // source = gate register (hot)
        // aux = update register (hot)
        // modifier[0] = state register (hot)
        let gate_idx = instr.source.index();
        let update_idx = instr.aux as usize & 0x0F;
        let state_idx = instr.modifier[0] as usize & 0x0F;
        let dst_idx = instr.target.index();

        let gate = match &self.hot_regs[gate_idx] {
            Some(buf) => &buf.data,
            None => return StepResult::Error(format!("Gate H{} not allocated", gate_idx)),
        };

        let update = match &self.hot_regs[update_idx] {
            Some(buf) => &buf.data,
            None => return StepResult::Error(format!("Update H{} not allocated", update_idx)),
        };

        let state = match &self.hot_regs[state_idx] {
            Some(buf) => &buf.data,
            None => return StepResult::Error(format!("State H{} not allocated", state_idx)),
        };

        // gate values are in integer range after sigmoid (0-255 scaled)
        // For proper gating: result = (gate * update + (255 - gate) * state) / 255
        let len = gate.len().min(update.len()).min(state.len());
        let mut result = Vec::with_capacity(len);

        for i in 0..len {
            let g = gate[i].clamp(0, 255) as i64;
            let u = update[i] as i64;
            let s = state[i] as i64;
            // Fused: gate * update + (255 - gate) * state, scaled back
            let val = (g * u + (255 - g) * s) / 255;
            result.push(val as i32);
        }

        self.hot_regs[dst_idx] = Some(HotBuffer {
            data: result,
            shape: vec![len],
        });

        StepResult::Continue
    }

    fn execute_load_target(&mut self, instr: Instruction) -> StepResult {
        let idx = instr.target.index();

        // Load target into a hot register (already i32)
        let data = self.target_buffer.clone();
        let len = data.len();
        self.hot_regs[idx] = Some(HotBuffer {
            data,
            shape: vec![len],
        });

        StepResult::Continue
    }

    fn execute_mastery_update(&mut self, instr: Instruction) -> StepResult {
        // MASTERY_UPDATE: pressure[i] += direction * activity[i] * scale
        // target = weight register (cold)
        // source = activity register (hot)
        // aux = direction register (hot, single value)
        // modifier[0] = scale factor (default 15)
        // modifier[1] = activity threshold divisor (default 4 = top 25%)

        let weights_idx = instr.target.index();
        let activity_idx = instr.source.index();
        let direction_idx = instr.aux as usize & 0x0F;
        let scale = if instr.modifier[0] > 0 { instr.modifier[0] as i32 } else { 15 };
        let threshold_div = if instr.modifier[1] > 0 { instr.modifier[1] as i32 } else { 4 };

        // Get activity
        let activity = match &self.hot_regs[activity_idx] {
            Some(buf) => buf.data.clone(),
            None => return StepResult::Error("Activity register not allocated".to_string()),
        };

        // Get direction (single value, positive or negative)
        let direction = match &self.hot_regs[direction_idx] {
            Some(buf) => {
                if buf.data.is_empty() { 0 } else { buf.data[0] }
            }
            None => return StepResult::Error("Direction register not allocated".to_string()),
        };

        // Get or create pressure register
        let pressure = self.pressure_regs[weights_idx].get_or_insert_with(|| {
            vec![0i32; activity.len()]
        });

        // Ensure pressure is right size
        if pressure.len() != activity.len() {
            *pressure = vec![0i32; activity.len()];
        }

        // Find activity threshold (max / threshold_div)
        let max_activity = activity.iter().cloned().max().unwrap_or(1).max(1);
        let threshold = max_activity / threshold_div;

        // Update pressure for participating neurons
        for (i, &act) in activity.iter().enumerate() {
            if act > threshold {
                let activity_strength = (act - threshold) as i64 * 256 / max_activity as i64;
                let delta = (direction as i64 * activity_strength * scale as i64 / 256) as i32;
                pressure[i] = pressure[i].saturating_add(delta);
            }
        }

        StepResult::Continue
    }

    fn execute_mastery_commit(&mut self, instr: Instruction) -> StepResult {
        // MASTERY_COMMIT: if |pressure| > threshold, update weights
        // target = weight register (cold)
        // modifier[0] = pressure threshold (default 50)
        // modifier[1] = magnitude step (default 5)

        let weights_idx = instr.target.index();
        let pressure_threshold = if instr.modifier[0] > 0 { instr.modifier[0] as i32 } else { 50 };
        let mag_step = if instr.modifier[1] > 0 { instr.modifier[1] } else { 5 };

        // Get pressure
        let pressure = match &self.pressure_regs[weights_idx] {
            Some(p) => p.clone(),
            None => return StepResult::Continue, // No pressure accumulated
        };

        // Get weights (mutable)
        let weights = match &mut self.cold_regs[weights_idx] {
            Some(buf) => &mut buf.weights,
            None => return StepResult::Error("Weights not allocated".to_string()),
        };

        // Apply pressure to weights
        for (i, &p) in pressure.iter().enumerate() {
            if i >= weights.len() { break; }

            if p.abs() >= pressure_threshold {
                let needed_polarity = if p > 0 { 1i8 } else { -1i8 };
                let w = &mut weights[i];

                if w.polarity == needed_polarity {
                    // Same direction: strengthen
                    w.magnitude = w.magnitude.saturating_add(mag_step);
                } else if w.polarity == 0 {
                    // Initialize
                    w.polarity = needed_polarity;
                    w.magnitude = mag_step;
                } else if w.magnitude > mag_step {
                    // Opposing: weaken first
                    w.magnitude -= mag_step;
                } else {
                    // Flip polarity when magnitude depletes
                    w.polarity = needed_polarity;
                    w.magnitude = mag_step;
                }
            }
        }

        // Clear pressure after commit
        if let Some(p) = &mut self.pressure_regs[weights_idx] {
            p.fill(0);
        }

        StepResult::Continue
    }

    fn execute_dequantize(&mut self, instr: Instruction) -> StepResult {
        let src_idx = instr.source.index();
        let shift = instr.scale() as u32;

        let src = match &self.hot_regs[src_idx] {
            Some(buf) => buf,
            None => return StepResult::Error("Source not allocated".to_string()),
        };

        // Shift values and store in output buffer (stays i32)
        let shift_amt = if shift > 0 { shift } else { 8 };
        self.output_buffer = src.data.iter().map(|&v| v >> shift_amt).collect();

        StepResult::Continue
    }

    fn execute_add_babble(&mut self, instr: Instruction) -> StepResult {
        let idx = instr.target.index();

        if let Some(buf) = &mut self.hot_regs[idx] {
            // babble_scale is 0-255, multiply by 50 for babble magnitude
            let babble_base = self.babble_scale * 50;

            for (i, val) in buf.data.iter_mut().enumerate() {
                // Deterministic pseudo-random pattern using phase and index
                let phase = self.babble_phase;
                // Simple integer-based magnitude variation (30%-100% range)
                let magnitude_factor = 77 + ((phase * 7 + i * 13) % 179); // 77-255 range
                let sign = if (i * 7 + phase / 10) % 5 < 3 { 1i32 } else { -1i32 };
                let babble = sign * (babble_base * magnitude_factor as i32 / 255);
                let positive_bias = babble_base / 3;
                *val = val.saturating_add(babble + positive_bias);
            }

            self.babble_phase += 1;
        }

        StepResult::Continue
    }

    fn execute_loop(&mut self, instr: Instruction) -> StepResult {
        let count = instr.count();
        if count > 0 {
            self.loop_stack.push(LoopState {
                start_pc: self.pc,
                remaining: count,
            });
            self.pc += 1;
        } else {
            // Zero iterations - skip to END_LOOP
            // For simplicity, just advance (real impl would scan for END_LOOP)
            self.pc += 1;
        }
        StepResult::Continue
    }

    fn execute_end_loop(&mut self) -> StepResult {
        if let Some(state) = self.loop_stack.last_mut() {
            state.remaining -= 1;
            if state.remaining > 0 {
                let start = state.start_pc;
                self.pc = start + 1;
            } else {
                self.loop_stack.pop();
                self.pc += 1;
            }
        } else {
            self.pc += 1;
        }
        StepResult::Continue
    }

    fn execute_break(&mut self) -> StepResult {
        self.loop_stack.pop();
        self.pc += 1;
        StepResult::Break
    }

    // === Accessors ===

    /// Get current program counter
    pub fn pc(&self) -> usize {
        self.pc
    }

    /// Check if program has ended
    pub fn is_ended(&self) -> bool {
        self.pc >= self.program.len()
    }

    /// Get hot register
    pub fn hot_reg(&self, index: usize) -> Option<&HotBuffer> {
        self.hot_regs.get(index).and_then(|r| r.as_ref())
    }

    /// Get cold register
    pub fn cold_reg(&self, index: usize) -> Option<&ColdBuffer> {
        self.cold_regs.get(index).and_then(|r| r.as_ref())
    }

    /// Get mutable cold register
    pub fn cold_reg_mut(&mut self, index: usize) -> Option<&mut ColdBuffer> {
        self.cold_regs.get_mut(index).and_then(|r| r.as_mut())
    }

    /// Set babble scale (0-255 range)
    pub fn set_babble_scale(&mut self, scale: i32) {
        self.babble_scale = scale.clamp(0, 255);
    }

    /// Get current program length
    pub fn program_len(&self) -> usize {
        self.program.len()
    }

    /// Set program counter directly (for testing/debugging)
    pub fn set_pc(&mut self, pc: usize) {
        self.pc = pc;
    }

    /// Get mutable hot register (for runtime modification)
    pub fn hot_reg_mut(&mut self, index: usize) -> Option<&mut HotBuffer> {
        self.hot_regs.get_mut(index).and_then(|r| r.as_mut())
    }

    /// Set hot register directly (internal use)
    pub(super) fn set_hot_reg_internal(&mut self, index: usize, buf: Option<HotBuffer>) {
        if index < 16 {
            self.hot_regs[index] = buf;
        }
    }

    /// Set cold register directly (internal use)
    pub(super) fn set_cold_reg_internal(&mut self, index: usize, buf: Option<ColdBuffer>) {
        if index < 16 {
            self.cold_regs[index] = buf;
        }
    }
}

impl Default for Interpreter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vm::assemble;

    #[test]
    fn test_interpreter_creation() {
        let interp = Interpreter::new();
        assert_eq!(interp.pc(), 0);
        assert!(interp.is_ended());
    }

    #[test]
    fn test_simple_program() {
        let source = r#"
.registers
    H0: i32[4]

.program
    load_input H0
    store_output H0
    halt
"#;

        let program = assemble(source).unwrap();
        let mut interp = Interpreter::from_program(&program);

        // Input as Signal
        let input = vec![
            Signal::positive(128), // ~0.5
            Signal::negative(77),  // ~-0.3
            Signal::positive(204), // ~0.8
            Signal::zero(),        // 0
        ];
        let output = interp.forward(&input).unwrap();

        assert_eq!(output.len(), 4);
        // Values should be preserved
        assert!(output[0].polarity > 0);
        assert!(output[1].polarity < 0);
        assert!(output[2].polarity > 0);
        assert_eq!(output[3].polarity, 0);
    }

    #[test]
    fn test_relu() {
        let source = r#"
.registers
    H0: i32[4]
    H1: i32[4]

.program
    load_input H0
    relu H1, H0
    store_output H1
    halt
"#;

        let program = assemble(source).unwrap();
        let mut interp = Interpreter::from_program(&program);

        // Input as Signal
        let input = vec![
            Signal::positive(128), // positive
            Signal::negative(77),  // negative
            Signal::positive(204), // positive
            Signal::negative(128), // negative
        ];
        let output = interp.forward(&input).unwrap();

        // ReLU: negative values become 0
        assert!(output[0].polarity >= 0);  // positive stays positive
        assert_eq!(output[1].magnitude, 0); // negative becomes 0
        assert!(output[2].polarity >= 0);  // positive stays positive
        assert_eq!(output[3].magnitude, 0); // negative becomes 0
    }

    #[test]
    fn test_forward_i32() {
        let source = r#"
.registers
    H0: i32[4]

.program
    load_input H0
    store_output H0
    halt
"#;

        let program = assemble(source).unwrap();
        let mut interp = Interpreter::from_program(&program);

        let input = vec![100, -50, 200, 0];
        let output = interp.forward_i32(&input).unwrap();

        assert_eq!(output.len(), 4);
        assert_eq!(output[0], 100);
        assert_eq!(output[1], -50);
        assert_eq!(output[2], 200);
        assert_eq!(output[3], 0);
    }
}
