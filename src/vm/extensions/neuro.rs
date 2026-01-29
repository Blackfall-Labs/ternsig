//! Neuro Extension (ExtID: 0x0005)
//!
//! Substrate I/O operations: chemical read/write, field read/write,
//! stimulation, convergence, temperature management.
//! These operations require a SubstrateHandle provided by the host.

use crate::vm::extension::{
    ExecutionContext, Extension, InstructionMeta, OperandPattern, StepResult,
};

/// Neuro extension — substrate I/O operations.
///
/// Placeholder: full implementation requires SubstrateHandle trait
/// that bridges the VM to the host brain's field substrate.
pub struct NeuroExtension {
    instructions: Vec<InstructionMeta>,
}

impl NeuroExtension {
    pub fn new() -> Self {
        Self {
            instructions: vec![
                InstructionMeta {
                    opcode: 0x0000,
                    mnemonic: "CHEM_READ",
                    operand_pattern: OperandPattern::RegImm8,
                    description: "Read neuromodulator level into register",
                },
                InstructionMeta {
                    opcode: 0x0001,
                    mnemonic: "CHEM_WRITE",
                    operand_pattern: OperandPattern::RegImm8,
                    description: "Write neuromodulator level from register",
                },
                InstructionMeta {
                    opcode: 0x0002,
                    mnemonic: "FIELD_READ",
                    operand_pattern: OperandPattern::RegImm8,
                    description: "Read from temporal field into register",
                },
                InstructionMeta {
                    opcode: 0x0003,
                    mnemonic: "FIELD_WRITE",
                    operand_pattern: OperandPattern::RegImm8,
                    description: "Write register to temporal field",
                },
                InstructionMeta {
                    opcode: 0x0004,
                    mnemonic: "STIM_READ",
                    operand_pattern: OperandPattern::RegImm8,
                    description: "Read stimulation level",
                },
                InstructionMeta {
                    opcode: 0x0005,
                    mnemonic: "CONV_READ",
                    operand_pattern: OperandPattern::Reg,
                    description: "Read convergence field state",
                },
                InstructionMeta {
                    opcode: 0x0006,
                    mnemonic: "TEMP_READ",
                    operand_pattern: OperandPattern::RegReg,
                    description: "Read temperature of cold register weights",
                },
                InstructionMeta {
                    opcode: 0x0007,
                    mnemonic: "TEMP_WRITE",
                    operand_pattern: OperandPattern::RegReg,
                    description: "Write temperature of cold register weights",
                },
            ],
        }
    }
}

impl Extension for NeuroExtension {
    fn ext_id(&self) -> u16 {
        0x0005
    }

    fn name(&self) -> &str {
        "tvmr.neuro"
    }

    fn version(&self) -> (u16, u16, u16) {
        (1, 0, 0)
    }

    fn instructions(&self) -> &[InstructionMeta] {
        &self.instructions
    }

    fn execute(
        &self,
        opcode: u16,
        _operands: [u8; 4],
        _ctx: &mut ExecutionContext,
    ) -> StepResult {
        // All neuro ops require SubstrateHandle — not yet wired
        match opcode {
            0x0000..=0x0007 => StepResult::Continue,
            _ => StepResult::Error(format!("tvmr.neuro: unknown opcode 0x{:04X}", opcode)),
        }
    }
}
