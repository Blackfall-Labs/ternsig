//! IPC Extension (ExtID: 0x0009)
//!
//! Inter-process communication: signals, broadcasts, barriers.
//! Enables coordination between brain regions running separate programs.

use crate::vm::extension::{
    ExecutionContext, Extension, InstructionMeta, OperandPattern, StepResult,
};

/// IPC extension — inter-region communication.
pub struct IpcExtension {
    instructions: Vec<InstructionMeta>,
}

impl IpcExtension {
    pub fn new() -> Self {
        Self {
            instructions: vec![
                InstructionMeta {
                    opcode: 0x0000,
                    mnemonic: "SEND_SIGNAL",
                    operand_pattern: OperandPattern::RegImm8,
                    description: "Send signal to another region",
                },
                InstructionMeta {
                    opcode: 0x0001,
                    mnemonic: "RECV_SIGNAL",
                    operand_pattern: OperandPattern::RegImm8,
                    description: "Receive signal from another region",
                },
                InstructionMeta {
                    opcode: 0x0002,
                    mnemonic: "BROADCAST",
                    operand_pattern: OperandPattern::RegImm8,
                    description: "Broadcast signal to all regions",
                },
                InstructionMeta {
                    opcode: 0x0003,
                    mnemonic: "SUBSCRIBE",
                    operand_pattern: OperandPattern::RegImm8,
                    description: "Subscribe to signals from a region",
                },
                InstructionMeta {
                    opcode: 0x0004,
                    mnemonic: "MAILBOX_PEEK",
                    operand_pattern: OperandPattern::RegImm8,
                    description: "Peek at mailbox without consuming",
                },
                InstructionMeta {
                    opcode: 0x0005,
                    mnemonic: "MAILBOX_POP",
                    operand_pattern: OperandPattern::RegImm8,
                    description: "Pop from mailbox (consume)",
                },
                InstructionMeta {
                    opcode: 0x0006,
                    mnemonic: "BARRIER_WAIT",
                    operand_pattern: OperandPattern::Imm8,
                    description: "Wait at synchronization barrier",
                },
                InstructionMeta {
                    opcode: 0x0007,
                    mnemonic: "ATOMIC_CAS",
                    operand_pattern: OperandPattern::RegRegReg,
                    description: "Atomic compare-and-swap",
                },
            ],
        }
    }
}

impl Extension for IpcExtension {
    fn ext_id(&self) -> u16 {
        0x0009
    }

    fn name(&self) -> &str {
        "tvmr.ipc"
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
            _ => StepResult::Error(format!("tvmr.ipc: unknown opcode 0x{:04X}", opcode)),
        }
    }
}
