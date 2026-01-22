//! Tensor ISA - Instruction Set Architecture for Neural Network Definitions
//!
//! **THE SELF-MODIFICATION ENABLER:**
//! TensorISA + Learning ISA + Thermograms + Adaptive Learning = closed-loop self-modifying brain
//!
//! This ISA defines neural network architectures as instruction streams, enabling:
//! - Hot-reload of network structures without Rust recompilation
//! - Runtime architecture modification (grow/prune layers)
//! - Self-directed specialization via Learning ISA triggers
//!
//! ## Format: 8-byte instructions
//!
//! ```text
//! [ACTION:2][TARGET:1][SOURCE:1][AUX:1][MODIFIER:3]
//! ```
//!
//! - ACTION: TensorAction opcode (0x0000-0xFFFF)
//! - TARGET: Destination register (0x00-0xFF)
//! - SOURCE: Source register (0x00-0xFF)
//! - AUX: Auxiliary register/immediate value
//! - MODIFIER: 3 bytes operation-specific data
//!
//! ## Register Banks
//!
//! ```text
//! Hot Bank   (0x00-0x0F): Activations/intermediates (volatile)
//! Cold Bank  (0x10-0x1F): Weights (TernarySignal, persistent via Thermogram)
//! Param Bank (0x20-0x2F): Scalars (learning_rate, babble_scale, etc.)
//! Shape Bank (0x30-0x3F): Dimension metadata
//! ```
//!
//! ## Design Principles
//!
//! - All cold registers store TernarySignal weights (2 bytes each)
//! - TERNARY_MATMUL uses CPU integer arithmetic only (no floats, no GPU)
//! - Weights persist via Thermogram with temperature lifecycle
//! - Programs are hot-reloadable without Rust recompilation
//!
//! ## Example Assembly
//!
//! ```text
//! ; audio_classifier.tisa.asm
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

pub use action::TensorAction;
pub use assembler::{assemble, AssembledProgram, AssemblerError, TensorAssembler};
pub use binary::{
    deserialize as deserialize_tisa, load_from_file as load_tisa_file,
    save_to_file as save_tisa_file, serialize as serialize_tisa, BinaryFlags, TisaHeader,
    TISA_HEADER_SIZE,
};
pub use instruction::{TensorInstruction, TensorInstructionBuilder};
pub use interpreter::{ColdBuffer, DomainOp, HotBuffer, StepResult, TensorInterpreter};
pub use modifier::{ModifierFlags, TensorModifier};
pub use register::{RegisterBank, RegisterMeta, TensorDtype, TensorRegister};
pub use hot_reload::{
    HotReloadManager, InterpreterState, ReloadEvent, ReloadableInterpreter,
};
pub use runtime_mod::{ArchStats, ModEvent, ShapeSpec, WireSpec, WireType};

// Re-export TernarySignal for users of ColdBuffer
pub use crate::TernarySignal;

/// Instruction size in bytes (extended from 6-byte ISA pattern)
pub const TENSOR_INSTRUCTION_SIZE: usize = 8;

/// Magic bytes for .tisa file format
pub const TISA_MAGIC: [u8; 4] = [0x54, 0x49, 0x53, 0x41]; // "TISA"

/// Current version of the .tisa format
pub const TISA_VERSION: u16 = 0x0001;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_instruction_roundtrip() {
        let instr = TensorInstruction::new(
            TensorAction::TERNARY_MATMUL,
            TensorRegister::hot(1),
            TensorRegister::cold(0),
            TensorRegister::hot(0).0,
            [0, 0, 0],
        );

        let bytes = instr.to_bytes();
        assert_eq!(bytes.len(), TENSOR_INSTRUCTION_SIZE);

        let parsed = TensorInstruction::from_bytes(&bytes);
        assert_eq!(instr.action, parsed.action);
        assert_eq!(instr.target, parsed.target);
        assert_eq!(instr.source, parsed.source);
    }

    #[test]
    fn test_register_banks() {
        let hot = TensorRegister::hot(5);
        assert_eq!(hot.bank(), RegisterBank::Hot);
        assert_eq!(hot.index(), 5);

        let cold = TensorRegister::cold(3);
        assert_eq!(cold.bank(), RegisterBank::Cold);
        assert_eq!(cold.index(), 3);

        let param = TensorRegister::param(7);
        assert_eq!(param.bank(), RegisterBank::Param);
        assert_eq!(param.index(), 7);
    }
}
