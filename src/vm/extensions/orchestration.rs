//! Orchestration Extension (ExtID: 0x0007)
//!
//! Model table operations: load, execute, chain, batch models.
//! Enables multi-model orchestration within a single program.

use crate::vm::extension::{
    ExecutionContext, Extension, InstructionMeta, OperandPattern, StepResult,
};

/// Orchestration extension — multi-model management.
pub struct OrchestrationExtension {
    instructions: Vec<InstructionMeta>,
}

impl OrchestrationExtension {
    pub fn new() -> Self {
        Self {
            instructions: vec![
                InstructionMeta {
                    opcode: 0x0000,
                    mnemonic: "MODEL_LOAD",
                    operand_pattern: OperandPattern::RegImm8,
                    description: "Load a model into model table slot",
                },
                InstructionMeta {
                    opcode: 0x0001,
                    mnemonic: "MODEL_EXEC",
                    operand_pattern: OperandPattern::RegImm8,
                    description: "Execute model from table slot",
                },
                InstructionMeta {
                    opcode: 0x0002,
                    mnemonic: "MODEL_INPUT",
                    operand_pattern: OperandPattern::RegImm8,
                    description: "Set input for model table slot",
                },
                InstructionMeta {
                    opcode: 0x0003,
                    mnemonic: "MODEL_OUTPUT",
                    operand_pattern: OperandPattern::RegImm8,
                    description: "Get output from model table slot",
                },
                InstructionMeta {
                    opcode: 0x0004,
                    mnemonic: "MODEL_UNLOAD",
                    operand_pattern: OperandPattern::Imm8,
                    description: "Unload model from table slot",
                },
                InstructionMeta {
                    opcode: 0x0005,
                    mnemonic: "MODEL_STATUS",
                    operand_pattern: OperandPattern::RegImm8,
                    description: "Get model status from table slot",
                },
                InstructionMeta {
                    opcode: 0x0006,
                    mnemonic: "MODEL_RELOAD",
                    operand_pattern: OperandPattern::Imm8,
                    description: "Hot-reload model in table slot",
                },
                InstructionMeta {
                    opcode: 0x0007,
                    mnemonic: "MODEL_CHAIN",
                    operand_pattern: OperandPattern::Custom("[slot1:1][slot2:1][_:2]"),
                    description: "Chain two models: output of slot1 feeds input of slot2",
                },
            ],
        }
    }
}

impl Extension for OrchestrationExtension {
    fn ext_id(&self) -> u16 {
        0x0007
    }

    fn name(&self) -> &str {
        "tvmr.orchestration"
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
                "tvmr.orchestration: unknown opcode 0x{:04X}",
                opcode
            )),
        }
    }
}
