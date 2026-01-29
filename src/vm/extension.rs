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
///
/// The VM does not own external resources (substrate, fields, regions, files).
/// When an instruction needs external state, it yields a DomainOp.
/// The host fulfills the operation and resumes execution.
///
/// Convention: operations that READ expect the host to write results into
/// the register specified by `target`. Operations that WRITE read from
/// the register specified by `source`.
#[derive(Debug, Clone)]
pub enum DomainOp {
    // =========================================================================
    // Persistence (existing)
    // =========================================================================
    /// Load weights from persistent storage.
    LoadWeights { register: Register, key: String },
    /// Store weights to persistent storage.
    StoreWeights { register: Register, key: String },
    /// Trigger consolidation (hot → cold thermogram).
    Consolidate,
    /// Compute error between target and output.
    ComputeError { target: i32, output: i32 },

    // =========================================================================
    // Neuro: Chemical substrate (0x0005)
    // =========================================================================
    /// Read chemical level. Host writes value into target H[reg][0].
    /// chem_id: 0=DA, 1=5HT, 2=NE, 3=GABA, 4=cortisol, 5=endorphin, 6=ACh, 7=fatigue.
    ChemRead { target: Register, chem_id: u8 },
    /// SET chemical level (authoritative). Host reads value from source H[reg][0].
    ChemSet { source: Register, chem_id: u8 },
    /// ADDITIVE chemical injection (phasic event). Host reads delta from source H[reg][0].
    ChemInject { source: Register, chem_id: u8 },

    // =========================================================================
    // Neuro: Field substrate (0x0005)
    // =========================================================================
    /// Read field region slice into target register.
    /// field_id: 0=activity, 1=chemical, 2=convergence, 3=sensory, etc.
    FieldRead { target: Register, field_id: u8 },
    /// Write source register data to field region slice.
    FieldWrite { source: Register, field_id: u8 },
    /// Advance field by 1 tick (decay, age frames).
    FieldTick { field_id: u8 },
    /// Apply metabolic decay to field: retention factor + fatigue boost.
    FieldDecay { field_id: u8, retention: u8, fatigue_boost: u8 },

    // =========================================================================
    // Neuro: Sensory / convergence (0x0005)
    // =========================================================================
    /// Read all stimulation levels into target register.
    StimRead { target: Register },
    /// Read valence [reward, punish] into target register.
    ValenceRead { target: Register },
    /// Read convergence field state into target register.
    ConvRead { target: Register },
    /// Winner-take-some lateral inhibition on field.
    LateralInhibit { source: Register, strength: u8 },
    /// Apply exhaustion decay boost to sustained activity.
    ExhaustionBoost { source: Register, factor: u8 },
    /// Compute novelty z-scores from region energies.
    NoveltyScore { target: Register, source: Register },

    // =========================================================================
    // Orchestration: Model table (0x0007)
    // =========================================================================
    /// Load a model from firmware path into table slot.
    ModelLoad { target: Register, slot: u8 },
    /// Execute model in slot. Host runs interpreter, writes output.
    ModelExec { target: Register, slot: u8 },
    /// Hot-reload model in slot from disk.
    ModelReload { slot: u8 },

    // =========================================================================
    // Orchestration: Region routing (0x0007)
    // =========================================================================
    /// Route input data to a specific region.
    RouteInput { source: Register, region_id: u8 },
    /// Fire a region's SNN, output into target register.
    RegionFire { target: Register, region_id: u8 },
    /// Aggregate all pending region outputs into target register.
    CollectOutputs { target: Register },
    /// Enable a region for routing.
    RegionEnable { region_id: u8 },
    /// Disable a region.
    RegionDisable { region_id: u8 },

    // =========================================================================
    // Lifecycle (0x0008)
    // =========================================================================
    /// Read current boot phase. Host writes into target H[reg][0].
    PhaseRead { target: Register },
    /// Read current tick count. Host writes into target H[reg][0].
    TickRead { target: Register },
    /// Read current neuronal level. Host writes into target H[reg][0].
    LevelRead { target: Register },
    /// Initialize thermogram for cold register.
    InitThermo { register: Register },
    /// Save cold register to thermogram.
    SaveThermo { register: Register },
    /// Load cold register from thermogram.
    LoadThermo { register: Register },
    /// Log a lifecycle event.
    LogEvent { source: Register, event_type: u8 },
    /// Halt a specific brain region.
    HaltRegion { region_id: u8 },

    // =========================================================================
    // IPC (0x0009)
    // =========================================================================
    /// Send signal to another region.
    SendSignal { source: Register, region_id: u8 },
    /// Receive signal from another region. Host writes into target.
    RecvSignal { target: Register, region_id: u8 },
    /// Broadcast signal to all regions.
    Broadcast { source: Register, channel: u8 },
    /// Subscribe to signals from a region.
    Subscribe { target: Register, region_id: u8 },
    /// Peek at mailbox without consuming.
    MailboxPeek { target: Register, mailbox_id: u8 },
    /// Pop from mailbox (consume).
    MailboxPop { target: Register, mailbox_id: u8 },
    /// Wait at synchronization barrier.
    BarrierWait { barrier_id: u8 },
    /// Atomic compare-and-swap.
    AtomicCas { target: Register, expected: Register, desired: Register },
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
