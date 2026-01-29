//! TVMR Extension System — Pluggable Instruction Set Extensions
//!
//! Extensions add domain-specific instructions to the TVMR without modifying
//! the core runtime. Each extension is assigned a 2-byte ID (u16) and gets
//! its own 65,536-opcode address space.
//!
//! ## Extension ID Ranges
//!
//! ```text
//! 0x0000          Core ISA (built-in, not an extension)
//! 0x0001-0x00FF   Standard extensions (tensor, ternary, learning, etc.)
//! 0x0100-0xFFFE   User-defined extensions
//! 0xFFFF          Reserved
//! ```

use super::interpreter::{ColdBuffer, HotBuffer, ChemicalState};
use super::register::Register;
use std::fmt;

/// Result of executing a single instruction.
#[derive(Debug, Clone)]
pub enum StepResult {
    /// Continue to next instruction.
    Continue,
    /// Program halted normally.
    Halt,
    /// Break from current loop.
    Break,
    /// Return from subroutine.
    Return,
    /// Yield control to the host with a domain operation.
    Yield(DomainOp),
    /// Reached end of program (PC past last instruction).
    Ended,
    /// Execution error.
    Error(String),
}

/// Domain operations yielded to the host for external handling.
#[derive(Debug, Clone)]
pub enum DomainOp {
    /// Load weights from persistent storage.
    LoadWeights { register: Register, key: String },
    /// Store weights to persistent storage.
    StoreWeights { register: Register, key: String },
    /// Trigger consolidation.
    Consolidate,
    /// Compute error between target and output.
    ComputeError { target: i32, output: i32 },
}

/// Describes how an instruction's 4 operand bytes are interpreted.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum OperandPattern {
    /// No operands used.
    None,
    /// `[reg:1][_:3]` — single register.
    Reg,
    /// `[dst:1][src:1][_:2]` — two registers.
    RegReg,
    /// `[dst:1][a:1][b:1][_:1]` — three registers.
    RegRegReg,
    /// `[dst:1][a:1][b:1][flags:1]` — three registers + flags byte.
    RegRegRegFlags,
    /// `[dst:1][src:1][imm16:2]` — register + 16-bit immediate.
    RegRegImm16,
    /// `[dst:1][_:1][imm16:2]` — register + 16-bit immediate.
    RegImm16,
    /// `[reg:1][imm8:1][_:2]` — register + 8-bit immediate.
    RegImm8,
    /// `[imm32:4]` — 32-bit immediate (e.g., jump target).
    Imm32,
    /// `[imm16:2][_:2]` — 16-bit immediate.
    Imm16,
    /// `[imm8:1][_:3]` — 8-bit immediate.
    Imm8,
    /// `[dst:1][cond:1][a:1][b:1]` — conditional select.
    RegCondRegReg,
    /// Custom pattern described by a string.
    Custom(&'static str),
}

/// Metadata for a single instruction within an extension.
#[derive(Debug, Clone)]
pub struct InstructionMeta {
    /// Extension-local opcode (0x0000-0xFFFF).
    pub opcode: u16,
    /// Assembly mnemonic (e.g., "TERNARY_MATMUL").
    pub mnemonic: &'static str,
    /// How the 4 operand bytes are interpreted.
    pub operand_pattern: OperandPattern,
    /// Human-readable description.
    pub description: &'static str,
}

/// Execution context passed to extension `execute()` calls.
///
/// Provides controlled, mutable access to the VM's register file,
/// program counter, stacks, and I/O buffers.
pub struct ExecutionContext<'a> {
    // Register banks
    pub hot_regs: &'a mut Vec<Option<HotBuffer>>,
    pub cold_regs: &'a mut Vec<Option<ColdBuffer>>,
    pub param_regs: &'a mut Vec<i32>,
    pub shape_regs: &'a mut Vec<Vec<usize>>,

    // Program counter & stacks
    pub pc: &'a mut usize,
    pub call_stack: &'a mut Vec<usize>,
    pub loop_stack: &'a mut Vec<LoopState>,

    // I/O buffers
    pub input_buffer: &'a [i32],
    pub output_buffer: &'a mut Vec<i32>,
    pub target_buffer: &'a [i32],

    // Learning state
    pub chemical_state: &'a mut ChemicalState,
    pub current_error: &'a mut i32,
    pub babble_scale: &'a mut i32,
    pub babble_phase: &'a mut usize,

    // Pressure registers (for mastery learning)
    pub pressure_regs: &'a mut Vec<Option<Vec<i32>>>,
}

/// Loop state for LOOP/ENDLOOP tracking.
#[derive(Debug, Clone)]
pub struct LoopState {
    /// PC of the instruction after LOOP (loop body start).
    pub start_pc: usize,
    /// Remaining iterations.
    pub remaining: u32,
}

/// Extension trait — implement this to add instructions to the TVMR.
///
/// Each extension gets a unique 2-byte ID and its own 65k opcode space.
/// Extensions declare their instruction metadata at registration and
/// receive dispatch calls during execution.
pub trait Extension: Send + Sync {
    /// Unique extension identifier (u16).
    fn ext_id(&self) -> u16;

    /// Human-readable name (e.g., "tvmr.tensor").
    fn name(&self) -> &str;

    /// Semantic version (major, minor, patch).
    fn version(&self) -> (u16, u16, u16);

    /// List all instructions this extension provides.
    fn instructions(&self) -> &[InstructionMeta];

    /// Execute an instruction.
    ///
    /// Called by the interpreter when an instruction with this extension's
    /// ID is encountered. The `opcode` and `operands` are the extension-local
    /// opcode and the 4 operand bytes from the instruction.
    fn execute(
        &self,
        opcode: u16,
        operands: [u8; 4],
        ctx: &mut ExecutionContext,
    ) -> StepResult;
}

impl fmt::Debug for dyn Extension {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (maj, min, pat) = self.version();
        write!(f, "Extension(0x{:04X} \"{}\" v{}.{}.{})", self.ext_id(), self.name(), maj, min, pat)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_operand_pattern_eq() {
        assert_eq!(OperandPattern::None, OperandPattern::None);
        assert_eq!(OperandPattern::RegRegReg, OperandPattern::RegRegReg);
        assert_ne!(OperandPattern::Reg, OperandPattern::RegReg);
    }

    #[test]
    fn test_instruction_meta() {
        let meta = InstructionMeta {
            opcode: 0x0000,
            mnemonic: "TEST_OP",
            operand_pattern: OperandPattern::RegReg,
            description: "A test operation",
        };
        assert_eq!(meta.mnemonic, "TEST_OP");
        assert_eq!(meta.operand_pattern, OperandPattern::RegReg);
    }
}
