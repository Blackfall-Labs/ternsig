//! Lifecycle Extension (ExtID: 0x0008)
//!
//! Boot, phase, tick, and thermogram lifecycle operations.
//! Integrates with the brain's boot sequence and persistence layer.

use crate::vm::extension::{
    ExecutionContext, Extension, InstructionMeta, OperandPattern, StepResult,
};

/// Lifecycle extension — boot/phase/persistence operations.
pub struct LifecycleExtension {
    instructions: Vec<InstructionMeta>,
}

impl LifecycleExtension {
    pub fn new() -> Self {
        Self {
            instructions: vec![
                InstructionMeta {
                    opcode: 0x0000,
                    mnemonic: "PHASE_READ",
                    operand_pattern: OperandPattern::Reg,
                    description: "Read current boot phase into register",
                },
                InstructionMeta {
                    opcode: 0x0001,
                    mnemonic: "TICK_READ",
                    operand_pattern: OperandPattern::Reg,
                    description: "Read current tick count into register",
                },
                InstructionMeta {
                    opcode: 0x0002,
                    mnemonic: "LEVEL_READ",
                    operand_pattern: OperandPattern::Reg,
                    description: "Read current neuronal level into register",
                },
                InstructionMeta {
                    opcode: 0x0003,
                    mnemonic: "INIT_THERMO",
                    operand_pattern: OperandPattern::Reg,
                    description: "Initialize thermogram for cold register",
                },
                InstructionMeta {
                    opcode: 0x0004,
                    mnemonic: "SAVE_THERMO",
                    operand_pattern: OperandPattern::Reg,
                    description: "Save cold register to thermogram",
                },
                InstructionMeta {
                    opcode: 0x0005,
                    mnemonic: "LOAD_THERMO",
                    operand_pattern: OperandPattern::Reg,
                    description: "Load cold register from thermogram",
                },
                InstructionMeta {
                    opcode: 0x0006,
                    mnemonic: "LOG_EVENT",
                    operand_pattern: OperandPattern::RegImm8,
                    description: "Log a lifecycle event",
                },
                InstructionMeta {
                    opcode: 0x0007,
                    mnemonic: "HALT_REGION",
                    operand_pattern: OperandPattern::Imm8,
                    description: "Halt a specific brain region",
                },
            ],
        }
    }
}

impl Extension for LifecycleExtension {
    fn ext_id(&self) -> u16 {
        0x0008
    }

    fn name(&self) -> &str {
        "tvmr.lifecycle"
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
        match opcode {
            0x0000..=0x0007 => StepResult::Continue,
            _ => StepResult::Error(format!(
                "tvmr.lifecycle: unknown opcode 0x{:04X}",
                opcode
            )),
        }
    }
}
