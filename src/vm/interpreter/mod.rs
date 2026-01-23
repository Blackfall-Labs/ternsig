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

/// Cold register (weight buffer)
#[derive(Debug, Clone)]
pub struct ColdBuffer {
    pub weights: Vec<Signal>,
    pub shape: Vec<usize>,
    pub thermogram_key: Option<String>,
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
