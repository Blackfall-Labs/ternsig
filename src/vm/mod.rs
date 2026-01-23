//! Ternsig VM - Virtual Machine for Neural Network Programs
//!
//! **THE SELF-MODIFICATION ENABLER:**
//! Ternsig VM + Learning + Thermograms + Adaptive Learning = closed-loop self-modifying brain
//!
//! This VM executes neural network architectures as instruction streams, enabling:
//! - Hot-reload of network structures without Rust recompilation
//! - Runtime architecture modification (grow/prune layers)
//! - Self-directed specialization via Learning triggers
//!
//! ## Format: 8-byte instructions
//!
//! ```text
//! [ACTION:2][TARGET:1][SOURCE:1][AUX:1][MODIFIER:3]
//! ```
//!
//! - ACTION: Action opcode (0x0000-0xFFFF)
//! - TARGET: Destination register (0x00-0xFF)
//! - SOURCE: Source register (0x00-0xFF)
//! - AUX: Auxiliary register/immediate value
//! - MODIFIER: 3 bytes operation-specific data
//!
//! ## Register Banks
//!
//! ```text
//! Hot Bank   (0x00-0x0F): Activations/intermediates (volatile)
//! Cold Bank  (0x10-0x1F): Weights (Signal, persistent via Thermogram)
//! Param Bank (0x20-0x2F): Scalars (learning_rate, babble_scale, etc.)
//! Shape Bank (0x30-0x3F): Dimension metadata
//! ```
//!
//! ## Design Principles
//!
//! - All cold registers store Signal weights (2 bytes each)
//! - TERNARY_MATMUL uses CPU integer arithmetic only (no floats, no GPU)
//! - Weights persist via Thermogram with temperature lifecycle
//! - Programs are hot-reloadable without Rust recompilation
//!
//! ## Example Assembly
//!
//! ```text
//! ; audio_classifier.ternsig
//! .registers
//!     C0: ternary[32, 12]  key="chip.audio.w1"
//!     H0: i32[12]
//!
//! .program
//!     load_input    H0
//!     ternary_matmul H1, C0, H0
//!     relu          H2, H1
//!     halt
//! ```

mod action;
mod assembler;
mod binary;
mod hot_reload;
mod instruction;
mod interpreter;
mod modifier;
mod register;
mod runtime_mod;

// New names (primary exports)
pub use action::Action;
pub use assembler::{assemble, AssembledProgram, AssemblerError, Assembler};
pub use binary::{
    deserialize, load_from_file, save_to_file, serialize,
    BinaryFlags, Header, HEADER_SIZE,
};
pub use instruction::{Instruction, InstructionBuilder};
pub use interpreter::{ColdBuffer, DomainOp, HotBuffer, StepResult, Interpreter};
pub use modifier::{ModifierFlags, Modifier};
pub use register::{RegisterBank, RegisterMeta, Dtype, Register};
pub use hot_reload::{
    HotReloadManager, InterpreterState, ReloadEvent, ReloadableInterpreter,
};
pub use runtime_mod::{ArchStats, ModEvent, ShapeSpec, WireSpec, WireType};

// Re-export Signal for users of ColdBuffer
pub use crate::Signal;

/// Instruction size in bytes
pub const INSTRUCTION_SIZE: usize = 8;

/// Magic bytes for .ternsig binary format
pub const TERNSIG_MAGIC: [u8; 4] = [0x54, 0x45, 0x52, 0x4E]; // "TERN"

/// Current version of the binary format
pub const TERNSIG_VERSION: u16 = 0x0002; // Bumped for rename

// =============================================================================
// Legacy type aliases for backwards compatibility (all deprecated)
// =============================================================================

#[deprecated(note = "Use Action instead")]
pub type TensorAction = Action;
#[deprecated(note = "Use Instruction instead")]
pub type TensorInstruction = Instruction;
#[deprecated(note = "Use InstructionBuilder instead")]
pub type TensorInstructionBuilder = InstructionBuilder;
#[deprecated(note = "Use Interpreter instead")]
pub type TensorInterpreter = Interpreter;
#[deprecated(note = "Use Modifier instead")]
pub type TensorModifier = Modifier;
#[deprecated(note = "Use Register instead")]
pub type TensorRegister = Register;
#[deprecated(note = "Use Dtype instead")]
pub type TensorDtype = Dtype;
#[deprecated(note = "Use Assembler instead")]
pub type TensorAssembler = Assembler;
#[deprecated(note = "Use Header instead")]
pub type TisaHeader = Header;

#[deprecated(note = "Use INSTRUCTION_SIZE instead")]
pub const TENSOR_INSTRUCTION_SIZE: usize = INSTRUCTION_SIZE;
#[deprecated(note = "Use TERNSIG_MAGIC instead")]
pub const TISA_MAGIC: [u8; 4] = [0x54, 0x49, 0x53, 0x41]; // "TISA"
#[deprecated(note = "Use TERNSIG_VERSION instead")]
pub const TISA_VERSION: u16 = 0x0001;
#[deprecated(note = "Use HEADER_SIZE instead")]
pub const TISA_HEADER_SIZE: usize = HEADER_SIZE;

#[deprecated(note = "Use serialize instead")]
pub use binary::serialize as serialize_tisa;
#[deprecated(note = "Use deserialize instead")]
pub use binary::deserialize as deserialize_tisa;
#[deprecated(note = "Use load_from_file instead")]
pub use binary::load_from_file as load_tisa_file;
#[deprecated(note = "Use save_to_file instead")]
pub use binary::save_to_file as save_tisa_file;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_instruction_roundtrip() {
        let instr = Instruction::new(
            Action::TERNARY_MATMUL,
            Register::hot(1),
            Register::cold(0),
            Register::hot(0).0,
            [0, 0, 0],
        );

        let bytes = instr.to_bytes();
        assert_eq!(bytes.len(), INSTRUCTION_SIZE);

        let parsed = Instruction::from_bytes(&bytes);
        assert_eq!(instr.action, parsed.action);
        assert_eq!(instr.target, parsed.target);
        assert_eq!(instr.source, parsed.source);
    }

    #[test]
    fn test_register_banks() {
        let hot = Register::hot(5);
        assert_eq!(hot.bank(), RegisterBank::Hot);
        assert_eq!(hot.index(), 5);

        let cold = Register::cold(3);
        assert_eq!(cold.bank(), RegisterBank::Cold);
        assert_eq!(cold.index(), 3);

        let param = Register::param(7);
        assert_eq!(param.bank(), RegisterBank::Param);
        assert_eq!(param.index(), 7);
    }
}
