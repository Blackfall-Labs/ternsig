//! TensorInterpreter - Runtime execution engine for TensorISA
//!
//! Executes TensorISA programs against a typed register file.
//! Integrates with Thermograms for weight persistence and Learning ISA for training.
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────┐
//! │              TensorInterpreter                   │
//! ├─────────────────────────────────────────────────┤
//! │  Hot Regs [H0-HF]:  Vec<i32> activations        │
//! │  Cold Regs [C0-CF]: Vec<TernarySignal> weights  │
//! │  Param Regs [P0-PF]: f32 scalars                │
//! │  Shape Regs [S0-SF]: Vec<usize> shapes          │
//! ├─────────────────────────────────────────────────┤
//! │  Program Counter, Call Stack, Loop Stack        │
//! │  Input/Output Buffers                           │
//! └─────────────────────────────────────────────────┘
//! ```

use super::{AssembledProgram, RegisterMeta, TensorAction, TensorDtype, TensorInstruction, TensorRegister};
use std::collections::VecDeque;
use crate::TernarySignal;

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
    LoadWeights { register: TensorRegister, key: String },
    /// Store weights to Thermogram
    StoreWeights { register: TensorRegister, key: String },
    /// Consolidate hot → cold
    Consolidate,
    /// Compute error signal
    ComputeError { target: f32, output: f32 },
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

    pub fn from_f32(values: &[f32], shape: Vec<usize>) -> Self {
        let data: Vec<i32> = values
            .iter()
            .map(|&v| (v.clamp(-1.0, 1.0) * 255.0) as i32)
            .collect();
        Self { data, shape }
    }

    pub fn to_f32(&self) -> Vec<f32> {
        self.data.iter().map(|&v| v as f32 / 255.0).collect()
    }

    pub fn numel(&self) -> usize {
        self.data.len()
    }
}

/// Cold register (weight buffer)
#[derive(Debug, Clone)]
pub struct ColdBuffer {
    /// Weights as TernarySignal
    pub weights: Vec<TernarySignal>,
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
            weights: vec![TernarySignal::zero(); size],
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
    pub fn weights(&self) -> &[TernarySignal] {
        &self.weights
    }

    /// Get mutable reference to weights.
    pub fn weights_mut(&mut self) -> &mut [TernarySignal] {
        &mut self.weights
    }

    pub fn numel(&self) -> usize {
        self.weights.len()
    }
}

/// TensorISA Interpreter
pub struct TensorInterpreter {
    /// Program instructions
    program: Vec<TensorInstruction>,
    /// Program counter
    pc: usize,
    /// Hot registers (activations)
    hot_regs: Vec<Option<HotBuffer>>,
    /// Cold registers (weights)
    cold_regs: Vec<Option<ColdBuffer>>,
    /// Param registers (scalars)
    param_regs: Vec<f32>,
    /// Shape registers (dimensions)
    shape_regs: Vec<Vec<usize>>,
    /// Call stack for subroutines
    call_stack: Vec<usize>,
    /// Loop stack
    loop_stack: Vec<LoopState>,
    /// Input buffer
    input_buffer: Vec<i32>,
    /// Output buffer
    output_buffer: Vec<f32>,
    /// Target buffer (for learning)
    target_buffer: Vec<f32>,
    /// Pressure accumulators (for mastery learning) - indexed by cold register
    pressure_regs: Vec<Option<Vec<i32>>>,
    /// Current babble scale (for learning)
    babble_scale: f32,
    /// Babble phase (deterministic pattern)
    babble_phase: usize,
    /// Current error (from last learning step)
    current_error: f32,
}

impl TensorInterpreter {
    /// Create a new interpreter with empty program
    pub fn new() -> Self {
        Self {
            program: Vec::new(),
            pc: 0,
            hot_regs: vec![None; 16],
            cold_regs: vec![None; 16],
            param_regs: vec![0.0; 16],
            shape_regs: vec![Vec::new(); 16],
            call_stack: Vec::with_capacity(16),
            loop_stack: Vec::with_capacity(8),
            input_buffer: Vec::new(),
            output_buffer: Vec::new(),
            target_buffer: Vec::new(),
            pressure_regs: vec![None; 16],
            babble_scale: 0.02,
            babble_phase: 0,
            current_error: 0.0,
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

    /// Set input buffer (called before forward pass)
    pub fn set_input(&mut self, input: &[f32]) {
        self.input_buffer = input
            .iter()
            .map(|&v| (v.clamp(-1.0, 1.0) * 255.0) as i32)
            .collect();
    }

    /// Set target buffer (called before learning pass)
    pub fn set_target(&mut self, target: &[f32]) {
        self.target_buffer = target.to_vec();
    }

    /// Get output buffer (called after forward pass)
    pub fn output(&self) -> &[f32] {
        &self.output_buffer
    }

    /// Set input buffer with i32 values directly (no scaling).
    pub fn set_input_i32(&mut self, input: &[i32]) {
        self.input_buffer = input.to_vec();
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

    /// Forward pass: set input, run, get output
    pub fn forward(&mut self, input: &[f32]) -> Result<Vec<f32>, String> {
        self.set_input(input);
        self.reset();

        match self.run() {
            StepResult::Halt | StepResult::Ended => Ok(self.output_buffer.clone()),
            StepResult::Error(e) => Err(e),
            _ => Err("Unexpected termination".to_string()),
        }
    }

    /// Execute a single instruction
    fn execute(&mut self, instr: TensorInstruction) -> StepResult {
        match instr.action {
            // === System ===
            TensorAction::NOP => StepResult::Continue,
            TensorAction::HALT => StepResult::Halt,
            TensorAction::RESET => {
                self.reset();
                StepResult::Continue
            }

            // === Register Management ===
            TensorAction::LOAD_INPUT => {
                let idx = instr.target.index();
                if let Some(buf) = &mut self.hot_regs[idx] {
                    let len = buf.data.len().min(self.input_buffer.len());
                    buf.data[..len].copy_from_slice(&self.input_buffer[..len]);
                }
                StepResult::Continue
            }
            TensorAction::STORE_OUTPUT => {
                let idx = instr.source.index();
                if let Some(buf) = &self.hot_regs[idx] {
                    self.output_buffer = buf.to_f32();
                }
                StepResult::Continue
            }
            TensorAction::ZERO_REG => {
                let idx = instr.target.index();
                if instr.target.is_hot() {
                    if let Some(buf) = &mut self.hot_regs[idx] {
                        buf.data.fill(0);
                    }
                }
                StepResult::Continue
            }
            TensorAction::COPY_REG => {
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
            TensorAction::TERNARY_MATMUL => self.execute_ternary_matmul(instr),
            TensorAction::ADD => self.execute_add(instr),
            TensorAction::SUB => self.execute_sub(instr),
            TensorAction::MUL => self.execute_mul(instr),
            TensorAction::RELU => self.execute_relu(instr),
            TensorAction::SIGMOID => self.execute_sigmoid(instr),
            TensorAction::SHIFT => self.execute_shift(instr),
            TensorAction::CMP_GT => self.execute_cmp_gt(instr),
            TensorAction::MAX_REDUCE => self.execute_max_reduce(instr),

            // === Ternary Ops ===
            TensorAction::DEQUANTIZE => self.execute_dequantize(instr),
            TensorAction::TERNARY_ADD_BIAS => self.execute_ternary_add_bias(instr),

            // === Learning Ops ===
            TensorAction::ADD_BABBLE => self.execute_add_babble(instr),
            TensorAction::MARK_ELIGIBILITY => {
                // Mark eligibility is a no-op in inference mode
                StepResult::Continue
            }
            TensorAction::LOAD_TARGET => self.execute_load_target(instr),
            TensorAction::MASTERY_UPDATE => self.execute_mastery_update(instr),
            TensorAction::MASTERY_COMMIT => self.execute_mastery_commit(instr),

            // === Control Flow ===
            TensorAction::LOOP => self.execute_loop(instr),
            TensorAction::END_LOOP => self.execute_end_loop(),
            TensorAction::BREAK => self.execute_break(),
            TensorAction::JUMP => {
                self.pc = instr.count() as usize;
                StepResult::Continue
            }
            TensorAction::CALL => {
                self.call_stack.push(self.pc + 1);
                self.pc = instr.count() as usize;
                StepResult::Continue
            }
            TensorAction::RETURN => {
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

    fn execute_ternary_matmul(&mut self, instr: TensorInstruction) -> StepResult {
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

    fn execute_add(&mut self, instr: TensorInstruction) -> StepResult {
        let src_idx = instr.source.index();
        let other_idx = instr.aux as usize & 0x0F;
        let dst_idx = instr.target.index();

        // Handle bias addition (cold + hot)
        if instr.source.is_hot() && TensorRegister(instr.aux).is_cold() {
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
        } else if instr.source.is_hot() && TensorRegister(instr.aux).is_hot() {
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

    fn execute_relu(&mut self, instr: TensorInstruction) -> StepResult {
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

    fn execute_shift(&mut self, instr: TensorInstruction) -> StepResult {
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

    fn execute_sub(&mut self, instr: TensorInstruction) -> StepResult {
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

    fn execute_mul(&mut self, instr: TensorInstruction) -> StepResult {
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

    fn execute_sigmoid(&mut self, instr: TensorInstruction) -> StepResult {
        let src_idx = instr.source.index();
        let dst_idx = instr.target.index();
        // Gain from modifier (default 4.0)
        let gain = if instr.modifier[0] > 0 {
            instr.modifier[0] as f32 / 16.0
        } else {
            4.0
        };

        let src = match &self.hot_regs[src_idx] {
            Some(buf) => buf.clone(),
            None => return StepResult::Error("Source not allocated".to_string()),
        };

        // Sigmoid: 1 / (1 + exp(-x * gain))
        // Input is i32 (scaled by 256), output is i32 (0-255 range)
        let result: Vec<i32> = src
            .data
            .iter()
            .map(|&v| {
                let x = v as f32 / 256.0;
                let sigmoid = 1.0 / (1.0 + (-x * gain).exp());
                (sigmoid * 255.0) as i32
            })
            .collect();

        self.hot_regs[dst_idx] = Some(HotBuffer {
            data: result,
            shape: src.shape.clone(),
        });

        StepResult::Continue
    }

    fn execute_cmp_gt(&mut self, instr: TensorInstruction) -> StepResult {
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

    fn execute_max_reduce(&mut self, instr: TensorInstruction) -> StepResult {
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

    fn execute_ternary_add_bias(&mut self, instr: TensorInstruction) -> StepResult {
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

    fn execute_load_target(&mut self, instr: TensorInstruction) -> StepResult {
        let idx = instr.target.index();

        // Load target into a hot register (as i32 scaled)
        let data: Vec<i32> = self
            .target_buffer
            .iter()
            .map(|&v| (v * 255.0) as i32)
            .collect();

        let len = data.len();
        self.hot_regs[idx] = Some(HotBuffer {
            data,
            shape: vec![len],
        });

        StepResult::Continue
    }

    fn execute_mastery_update(&mut self, instr: TensorInstruction) -> StepResult {
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

    fn execute_mastery_commit(&mut self, instr: TensorInstruction) -> StepResult {
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

    fn execute_dequantize(&mut self, instr: TensorInstruction) -> StepResult {
        let src_idx = instr.source.index();
        let scale = instr.scale() as f32;

        let src = match &self.hot_regs[src_idx] {
            Some(buf) => buf,
            None => return StepResult::Error("Source not allocated".to_string()),
        };

        // Convert to f32 and store in output buffer
        let scale_factor = if scale > 0.0 { scale } else { 65536.0 };
        self.output_buffer = src.data.iter().map(|&v| v as f32 / scale_factor).collect();

        StepResult::Continue
    }

    fn execute_add_babble(&mut self, instr: TensorInstruction) -> StepResult {
        let idx = instr.target.index();

        if let Some(buf) = &mut self.hot_regs[idx] {
            let babble_base = (self.babble_scale * 255.0 * 50.0) as i32;

            for (i, val) in buf.data.iter_mut().enumerate() {
                let phase = self.babble_phase as f32;
                let magnitude_factor =
                    0.3 + 0.7 * (phase * 0.03 + i as f32 * 0.7).cos().abs();
                let sign = if (i * 7 + (phase as usize / 10)) % 5 < 3 {
                    1
                } else {
                    -1
                };
                let babble = sign * ((babble_base as f32 * magnitude_factor) as i32);
                let positive_bias = babble_base / 3;
                *val = val.saturating_add(babble + positive_bias);
            }

            self.babble_phase += 1;
        }

        StepResult::Continue
    }

    fn execute_loop(&mut self, instr: TensorInstruction) -> StepResult {
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

    /// Set babble scale
    pub fn set_babble_scale(&mut self, scale: f32) {
        self.babble_scale = scale;
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

impl Default for TensorInterpreter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor_isa::assemble;

    #[test]
    fn test_interpreter_creation() {
        let interp = TensorInterpreter::new();
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
        let mut interp = TensorInterpreter::from_program(&program);

        let input = vec![0.5, -0.3, 0.8, 0.0];
        let output = interp.forward(&input).unwrap();

        assert_eq!(output.len(), 4);
        // Values should be approximately preserved (quantization)
        assert!((output[0] - 0.5).abs() < 0.01);
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
        let mut interp = TensorInterpreter::from_program(&program);

        let input = vec![0.5, -0.3, 0.8, -0.5];
        let output = interp.forward(&input).unwrap();

        // ReLU: negative values become 0
        assert!(output[0] > 0.0);
        assert!((output[1] - 0.0).abs() < 0.01);
        assert!(output[2] > 0.0);
        assert!((output[3] - 0.0).abs() < 0.01);
    }
}
