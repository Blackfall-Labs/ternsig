//! Test Extension (ExtID: 0x000A)
//!
//! Testing and assertion operations for program validation.
//! Used during development and automated testing.

use crate::vm::extension::{
    ExecutionContext, Extension, InstructionMeta, OperandPattern, StepResult,
};

/// Test extension — assertion and validation operations.
pub struct TestExtension {
    instructions: Vec<InstructionMeta>,
}

impl TestExtension {
    pub fn new() -> Self {
        Self {
            instructions: vec![
                InstructionMeta {
                    opcode: 0x0000,
                    mnemonic: "ASSERT_EQ",
                    operand_pattern: OperandPattern::RegReg,
                    description: "Assert two registers are equal",
                },
                InstructionMeta {
                    opcode: 0x0001,
                    mnemonic: "ASSERT_GT",
                    operand_pattern: OperandPattern::RegReg,
                    description: "Assert first register greater than second",
                },
                InstructionMeta {
                    opcode: 0x0002,
                    mnemonic: "ASSERT_ACTIVE",
                    operand_pattern: OperandPattern::Reg,
                    description: "Assert register has non-zero values",
                },
                InstructionMeta {
                    opcode: 0x0003,
                    mnemonic: "ASSERT_RANGE",
                    operand_pattern: OperandPattern::Custom("[reg:1][min:1][max:1][_:1]"),
                    description: "Assert all values in range [min, max]",
                },
                InstructionMeta {
                    opcode: 0x0004,
                    mnemonic: "TEST_BEGIN",
                    operand_pattern: OperandPattern::Imm8,
                    description: "Mark beginning of test case",
                },
                InstructionMeta {
                    opcode: 0x0005,
                    mnemonic: "TEST_END",
                    operand_pattern: OperandPattern::None,
                    description: "Mark end of test case",
                },
                InstructionMeta {
                    opcode: 0x0006,
                    mnemonic: "EXPECT_CHEM",
                    operand_pattern: OperandPattern::Custom("[chemical:1][min:1][max:1][_:1]"),
                    description: "Assert chemical level in range",
                },
                InstructionMeta {
                    opcode: 0x0007,
                    mnemonic: "SNAPSHOT",
                    operand_pattern: OperandPattern::Reg,
                    description: "Snapshot register state for comparison",
                },
            ],
        }
    }
}

impl Extension for TestExtension {
    fn ext_id(&self) -> u16 {
        0x000A
    }

    fn name(&self) -> &str {
        "tvmr.test"
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
            _ => StepResult::Error(format!("tvmr.test: unknown opcode 0x{:04X}", opcode)),
        }
    }
}
