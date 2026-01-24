//! Interpreter - Runtime execution engine for Ternsig VM
//!
//! Executes Ternsig programs against a typed register file.
//! Integrates with Thermograms for weight persistence and Learning for training.

mod ops_forward;
mod ops_ternary;
mod ops_learning;
mod ops_control;

use super::{AssembledProgram, Action, Instruction, Register};
use crate::Signal;

/// Result of executing a single instruction
#[derive(Debug, Clone)]
pub enum StepResult {
    Continue,
    Halt,
    Break,
    Return,
    Yield(DomainOp),
    Ended,
    Error(String),
}

/// Domain operations that require external execution
#[derive(Debug, Clone)]
pub enum DomainOp {
    LoadWeights { register: Register, key: String },
    StoreWeights { register: Register, key: String },
    Consolidate,
    ComputeError { target: i32, output: i32 },
}

/// Loop state for LOOP instruction
#[derive(Debug, Clone)]
struct LoopState {
    start_pc: usize,
    remaining: u16,
}

/// Hot register (activation buffer)
#[derive(Debug, Clone)]
pub struct HotBuffer {
    pub data: Vec<i32>,
    pub shape: Vec<usize>,
}

impl HotBuffer {
    pub fn new(shape: Vec<usize>) -> Self {
        let size: usize = shape.iter().product();
        Self { data: vec![0; size], shape }
    }

    pub fn zeros(shape: Vec<usize>) -> Self {
        Self::new(shape)
    }

    pub fn from_ternary(signals: &[Signal], shape: Vec<usize>) -> Self {
        let data: Vec<i32> = signals
            .iter()
            .map(|s| s.polarity as i32 * s.magnitude as i32)
            .collect();
        Self { data, shape }
    }

    pub fn to_ternary(&self) -> Vec<Signal> {
        self.data.iter().map(|&v| Signal::from_signed_i32(v)).collect()
    }

    pub fn numel(&self) -> usize {
        self.data.len()
    }
}

/// Temperature levels for signal plasticity
/// Maps to thermogram 4-layer temperature model
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum SignalTemperature {
    /// Hot: Working memory, high plasticity (threshold / 2)
    #[default]
    Hot = 0,
    /// Warm: Session learning, normal plasticity (threshold)
    Warm = 1,
    /// Cool: Expertise, low plasticity (threshold * 2)
    Cool = 2,
    /// Cold: Core identity, frozen (no learning)
    Cold = 3,
}

impl SignalTemperature {
    /// Get threshold multiplier for this temperature
    pub fn threshold_multiplier(&self) -> i32 {
        match self {
            Self::Hot => 1,   // threshold / 2 (easy)
            Self::Warm => 2,  // threshold (normal)
            Self::Cool => 4,  // threshold * 2 (hard)
            Self::Cold => i32::MAX, // impossible
        }
    }

    /// Create from u8 value
    pub fn from_u8(v: u8) -> Self {
        match v {
            0 => Self::Hot,
            1 => Self::Warm,
            2 => Self::Cool,
            _ => Self::Cold,
        }
    }

    /// Promote to colder (more stable) temperature
    pub fn promote(&self) -> Self {
        match self {
            Self::Hot => Self::Warm,
            Self::Warm => Self::Cool,
            Self::Cool => Self::Cold,
            Self::Cold => Self::Cold, // Already coldest
        }
    }

    /// Demote to hotter (more plastic) temperature
    pub fn demote(&self) -> Self {
        match self {
            Self::Hot => Self::Hot, // Already hottest
            Self::Warm => Self::Hot,
            Self::Cool => Self::Warm,
            Self::Cold => Self::Cool,
        }
    }
}

/// Cold register (signal buffer with optional per-signal temperature)
#[derive(Debug, Clone)]
pub struct ColdBuffer {
    pub weights: Vec<Signal>,
    pub shape: Vec<usize>,
    pub thermogram_key: Option<String>,
    pub frozen: bool,
    /// Per-signal temperature (None = all HOT)
    pub temperatures: Option<Vec<SignalTemperature>>,
    /// Per-signal usage counts for consolidation (None = not tracked)
    usage_counts: Option<Vec<u32>>,
    /// Last tick when usage was recorded (for decay)
    last_usage_tick: u64,
}

impl ColdBuffer {
    pub fn new(shape: Vec<usize>) -> Self {
        let size: usize = shape.iter().product();
        Self {
            weights: vec![Signal::zero(); size],
            shape,
            thermogram_key: None,
            frozen: false,
            temperatures: None,
            usage_counts: None,
            last_usage_tick: 0,
        }
    }

    pub fn with_key(mut self, key: impl Into<String>) -> Self {
        self.thermogram_key = Some(key.into());
        self
    }

    pub fn is_frozen(&self) -> bool {
        self.frozen
    }

    pub fn weights(&self) -> &[Signal] {
        &self.weights
    }

    pub fn weights_mut(&mut self) -> &mut [Signal] {
        &mut self.weights
    }

    pub fn numel(&self) -> usize {
        self.weights.len()
    }

    /// Get temperature for a specific signal index
    pub fn temperature(&self, idx: usize) -> SignalTemperature {
        match &self.temperatures {
            Some(temps) => temps.get(idx).copied().unwrap_or(SignalTemperature::Hot),
            None => SignalTemperature::Hot,
        }
    }

    /// Set temperature for a specific signal
    pub fn set_temperature(&mut self, idx: usize, temp: SignalTemperature) {
        let temps = self.temperatures.get_or_insert_with(|| {
            vec![SignalTemperature::Hot; self.weights.len()]
        });
        if idx < temps.len() {
            temps[idx] = temp;
        }
    }

    /// Set all temperatures to a single value
    pub fn set_all_temperatures(&mut self, temp: SignalTemperature) {
        self.temperatures = Some(vec![temp; self.weights.len()]);
    }

    /// Get mutable access to temperatures (creates if None)
    pub fn temperatures_mut(&mut self) -> &mut Vec<SignalTemperature> {
        self.temperatures.get_or_insert_with(|| {
            vec![SignalTemperature::Hot; self.weights.len()]
        })
    }

    // =========================================================================
    // Usage Tracking (for consolidation)
    // =========================================================================

    /// Enable usage tracking for this buffer
    pub fn enable_usage_tracking(&mut self) {
        if self.usage_counts.is_none() {
            self.usage_counts = Some(vec![0u32; self.weights.len()]);
        }
    }

    /// Record usage for signals that were active during forward pass
    /// Called by interpreter when signals participate in computation
    pub fn record_usage(&mut self, active_indices: &[usize], tick: u64) {
        let counts = match &mut self.usage_counts {
            Some(c) => c,
            None => return, // Tracking not enabled
        };

        for &idx in active_indices {
            if idx < counts.len() {
                counts[idx] = counts[idx].saturating_add(1);
            }
        }
        self.last_usage_tick = tick;
    }

    /// Record usage for a single signal
    pub fn record_signal_usage(&mut self, idx: usize) {
        if let Some(counts) = &mut self.usage_counts {
            if idx < counts.len() {
                counts[idx] = counts[idx].saturating_add(1);
            }
        }
    }

    /// Get usage count for a specific signal
    pub fn usage_count(&self, idx: usize) -> u32 {
        self.usage_counts
            .as_ref()
            .and_then(|c| c.get(idx).copied())
            .unwrap_or(0)
    }

    /// Get all usage counts (None if tracking not enabled)
    pub fn usage_counts(&self) -> Option<&[u32]> {
        self.usage_counts.as_deref()
    }

    /// Reset all usage counts to zero
    pub fn reset_usage_counts(&mut self) {
        if let Some(counts) = &mut self.usage_counts {
            counts.fill(0);
        }
    }

    /// Apply decay to usage counts (for periodic consolidation)
    /// decay_factor: 0-255 where 255 = no decay, 128 = 50% retention
    pub fn decay_usage(&mut self, decay_factor: u8) {
        if let Some(counts) = &mut self.usage_counts {
            for count in counts.iter_mut() {
                *count = ((*count as u64 * decay_factor as u64) / 255) as u32;
            }
        }
    }

    /// Get signals that are frequently used (above threshold)
    pub fn frequently_used(&self, threshold: u32) -> Vec<usize> {
        match &self.usage_counts {
            Some(counts) => counts
                .iter()
                .enumerate()
                .filter(|(_, &c)| c >= threshold)
                .map(|(i, _)| i)
                .collect(),
            None => Vec::new(),
        }
    }

    /// Get signals that are rarely used (below threshold)
    pub fn rarely_used(&self, threshold: u32) -> Vec<usize> {
        match &self.usage_counts {
            Some(counts) => counts
                .iter()
                .enumerate()
                .filter(|(_, &c)| c < threshold)
                .map(|(i, _)| i)
                .collect(),
            None => Vec::new(),
        }
    }

    /// Get last tick when usage was recorded
    pub fn last_usage_tick(&self) -> u64 {
        self.last_usage_tick
    }
}

/// Chemical state for neuromodulator gating
///
/// Neuromodulators control WHEN learning happens:
/// - Dopamine: "This is surprising/rewarding" → enable learning
/// - Serotonin: "Things are going well" → moderate learning rate
/// - Norepinephrine: "Pay attention!" → amplify signals
/// - GABA: "Calm down" → inhibit runaway excitation
///
/// All values are 0-255 (like Signal magnitude).
/// Learning requires dopamine > threshold (default: ~76, which is 0.3 * 255).
#[derive(Debug, Clone, Copy, Default)]
pub struct ChemicalState {
    /// Dopamine: gates learning (surprise/reward)
    pub dopamine: u8,
    /// Serotonin: scales learning rate
    pub serotonin: u8,
    /// Norepinephrine: amplifies attention
    pub norepinephrine: u8,
    /// GABA: inhibits excitation
    pub gaba: u8,
}

impl ChemicalState {
    /// Default threshold for dopamine gating (~0.3 on 0-1 scale)
    pub const DOPAMINE_THRESHOLD: u8 = 76;

    /// Create with all neuromodulators at baseline
    pub fn baseline() -> Self {
        Self {
            dopamine: 128,      // Moderate surprise
            serotonin: 128,     // Normal mood
            norepinephrine: 128, // Normal attention
            gaba: 128,          // Normal inhibition
        }
    }

    /// Check if learning is enabled (dopamine above threshold)
    pub fn learning_enabled(&self) -> bool {
        self.dopamine >= Self::DOPAMINE_THRESHOLD
    }

    /// Get dopamine-based learning rate multiplier (0-4)
    pub fn dopamine_scale(&self) -> i32 {
        if self.dopamine < Self::DOPAMINE_THRESHOLD {
            0 // No learning
        } else {
            // Scale from 1 (at threshold) to 4 (at max)
            1 + (self.dopamine.saturating_sub(Self::DOPAMINE_THRESHOLD) as i32 / 64)
        }
    }
}

/// Ternsig Interpreter - NO FLOATS, pure integer/ternary
pub struct Interpreter {
    program: Vec<Instruction>,
    pub(super) pc: usize,
    pub(super) hot_regs: Vec<Option<HotBuffer>>,
    pub(super) cold_regs: Vec<Option<ColdBuffer>>,
    param_regs: Vec<i32>,
    shape_regs: Vec<Vec<usize>>,
    pub(super) call_stack: Vec<usize>,
    pub(super) loop_stack: Vec<LoopState>,
    input_buffer: Vec<i32>,
    pub(super) output_buffer: Vec<i32>,
    pub(super) target_buffer: Vec<i32>,
    pub(super) pressure_regs: Vec<Option<Vec<i32>>>,
    pub(super) babble_scale: i32,
    pub(super) babble_phase: usize,
    current_error: i32,
    /// Chemical state for neuromodulator gating
    pub chemical_state: ChemicalState,
}

impl Interpreter {
    pub fn new() -> Self {
        Self {
            program: Vec::new(),
            pc: 0,
            hot_regs: vec![None; 16],
            cold_regs: vec![None; 16],
            param_regs: vec![0; 16],
            shape_regs: vec![Vec::new(); 16],
            call_stack: Vec::new(),
            loop_stack: Vec::new(),
            input_buffer: Vec::new(),
            output_buffer: Vec::new(),
            target_buffer: Vec::new(),
            pressure_regs: vec![None; 16],
            babble_scale: 0,
            babble_phase: 0,
            current_error: 0,
            chemical_state: ChemicalState::baseline(),
        }
    }

    pub fn from_program(program: &AssembledProgram) -> Self {
        let mut interp = Self::new();
        interp.program = program.instructions.clone();
        interp.allocate_registers(&program.registers);
        interp
    }

    pub fn load_program(&mut self, program: &AssembledProgram) {
        self.program = program.instructions.clone();
        self.pc = 0;
        self.loop_stack.clear();
        self.call_stack.clear();
        self.allocate_registers(&program.registers);
    }

    fn allocate_registers(&mut self, registers: &[super::RegisterMeta]) {
        for reg in registers {
            let size: usize = reg.shape.iter().product();
            match reg.id.bank() {
                super::RegisterBank::Hot => {
                    let idx = reg.id.index();
                    self.hot_regs[idx] = Some(HotBuffer {
                        data: vec![0; size],
                        shape: reg.shape.clone(),
                    });
                }
                super::RegisterBank::Cold => {
                    let idx = reg.id.index();
                    let mut buf = ColdBuffer::new(reg.shape.clone());
                    buf.thermogram_key = reg.thermogram_key.clone();
                    buf.frozen = reg.frozen;
                    self.cold_regs[idx] = Some(buf);
                }
                super::RegisterBank::Param => {
                    let idx = reg.id.index();
                    self.param_regs[idx] = 0;
                }
                super::RegisterBank::Shape => {
                    let idx = reg.id.index();
                    self.shape_regs[idx] = reg.shape.clone();
                }
            }
        }
    }

    pub fn load_input(&mut self, input: &[Signal]) {
        self.input_buffer = input
            .iter()
            .map(|s| s.polarity as i32 * s.magnitude as i32)
            .collect();
    }

    pub fn load_input_i32(&mut self, input: &[i32]) {
        self.input_buffer = input.to_vec();
    }

    pub fn load_target(&mut self, target: &[Signal]) {
        self.target_buffer = target
            .iter()
            .map(|s| s.polarity as i32 * s.magnitude as i32)
            .collect();
    }

    pub fn get_output(&self) -> Vec<Signal> {
        self.output_buffer
            .iter()
            .map(|&v| Signal::from_signed_i32(v))
            .collect()
    }

    pub fn get_output_i32(&self) -> &[i32] {
        &self.output_buffer
    }

    pub fn step(&mut self) -> StepResult {
        if self.pc >= self.program.len() {
            return StepResult::Ended;
        }

        let instr = self.program[self.pc].clone();
        self.pc += 1;

        match instr.action {
            Action::NOP => StepResult::Continue,
            Action::HALT => StepResult::Halt,
            Action::RESET => {
                self.pc = 0;
                self.loop_stack.clear();
                StepResult::Continue
            }
            Action::LOAD_INPUT => {
                let idx = instr.target.index();
                let data = self.input_buffer.clone();
                let len = data.len();
                self.hot_regs[idx] = Some(HotBuffer { data, shape: vec![len] });
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

            // Forward ops (ops_forward.rs)
            Action::TERNARY_MATMUL => self.execute_ternary_matmul(instr),
            Action::ADD => self.execute_add(instr),
            Action::SUB => self.execute_sub(instr),
            Action::MUL => self.execute_mul(instr),
            Action::RELU => self.execute_relu(instr),
            Action::SIGMOID => self.execute_sigmoid(instr),
            Action::GELU => self.execute_gelu(instr),
            Action::SOFTMAX => self.execute_softmax(instr),
            Action::SHIFT => self.execute_shift(instr),
            Action::CMP_GT => self.execute_cmp_gt(instr),
            Action::MAX_REDUCE => self.execute_max_reduce(instr),

            // Ternary ops (ops_ternary.rs)
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

            // Learning ops (ops_learning.rs)
            Action::ADD_BABBLE => self.execute_add_babble(instr),
            Action::MARK_ELIGIBILITY => StepResult::Continue,
            Action::LOAD_TARGET => self.execute_load_target(instr),
            Action::MASTERY_UPDATE => self.execute_mastery_update(instr),
            Action::MASTERY_COMMIT => self.execute_mastery_commit(instr),

            // Control flow (ops_control.rs)
            Action::LOOP => self.execute_loop(instr),
            Action::END_LOOP => self.execute_end_loop(),
            Action::BREAK => self.execute_break(),
            Action::JUMP => {
                self.pc = instr.count() as usize;
                StepResult::Continue
            }
            Action::CALL => {
                self.call_stack.push(self.pc);
                self.pc = instr.count() as usize;
                StepResult::Continue
            }
            Action::RETURN => {
                if let Some(ret_pc) = self.call_stack.pop() {
                    self.pc = ret_pc;
                    StepResult::Continue
                } else {
                    StepResult::Halt
                }
            }

            _ => StepResult::Error(format!("Unknown action: {:?}", instr.action)),
        }
    }

    pub fn run(&mut self) -> Result<(), String> {
        loop {
            match self.step() {
                StepResult::Continue => continue,
                StepResult::Halt | StepResult::Ended => return Ok(()),
                StepResult::Error(e) => return Err(e),
                StepResult::Break | StepResult::Return => continue,
                StepResult::Yield(_) => continue,
            }
        }
    }

    pub fn forward(&mut self, input: &[Signal]) -> Result<Vec<Signal>, String> {
        self.pc = 0;
        self.load_input(input);
        self.run()?;
        Ok(self.get_output())
    }

    pub fn forward_i32(&mut self, input: &[i32]) -> Result<Vec<i32>, String> {
        self.pc = 0;
        self.load_input_i32(input);
        self.run()?;
        Ok(self.output_buffer.clone())
    }

    // === Accessors ===

    pub fn pc(&self) -> usize {
        self.pc
    }

    pub fn is_ended(&self) -> bool {
        self.pc >= self.program.len()
    }

    pub fn hot_reg(&self, index: usize) -> Option<&HotBuffer> {
        self.hot_regs.get(index).and_then(|r| r.as_ref())
    }

    pub fn cold_reg(&self, index: usize) -> Option<&ColdBuffer> {
        self.cold_regs.get(index).and_then(|r| r.as_ref())
    }

    pub fn cold_reg_mut(&mut self, index: usize) -> Option<&mut ColdBuffer> {
        self.cold_regs.get_mut(index).and_then(|r| r.as_mut())
    }

    pub fn set_babble_scale(&mut self, scale: i32) {
        self.babble_scale = scale.clamp(0, 255);
    }

    pub fn program_len(&self) -> usize {
        self.program.len()
    }

    pub fn set_pc(&mut self, pc: usize) {
        self.pc = pc;
    }

    pub fn hot_reg_mut(&mut self, index: usize) -> Option<&mut HotBuffer> {
        self.hot_regs.get_mut(index).and_then(|r| r.as_mut())
    }

    pub(crate) fn set_hot_reg_internal(&mut self, index: usize, buf: Option<HotBuffer>) {
        if index < 16 {
            self.hot_regs[index] = buf;
        }
    }

    pub(crate) fn set_cold_reg_internal(&mut self, index: usize, buf: Option<ColdBuffer>) {
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

        let input = vec![
            Signal::positive(128),
            Signal::negative(77),
            Signal::positive(204),
            Signal::zero(),
        ];
        let output = interp.forward(&input).unwrap();

        assert_eq!(output.len(), 4);
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

        let input = vec![
            Signal::positive(128),
            Signal::negative(77),
            Signal::positive(204),
            Signal::negative(128),
        ];
        let output = interp.forward(&input).unwrap();

        assert!(output[0].polarity >= 0);
        assert_eq!(output[1].magnitude, 0);
        assert!(output[2].polarity >= 0);
        assert_eq!(output[3].magnitude, 0);
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
