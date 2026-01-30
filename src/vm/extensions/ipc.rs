//! IPC Extension (ExtID: 0x0009)
//!
//! Inter-process communication: signals, broadcasts, barriers.
//! Enables coordination between brain regions running separate programs.
//!
//! All operations yield DomainOps — the host owns region mailboxes,
//! signal routing, barriers, and shared atomic state.

use crate::vm::extension::{
    DomainOp, ExecutionContext, Extension, InstructionMeta, OperandPattern, StepResult,
};
use crate::vm::register::Register;

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
        operands: [u8; 4],
        _ctx: &mut ExecutionContext,
    ) -> StepResult {
        match opcode {
            // SEND_SIGNAL [source:1][region_id:1][_:2]
            // Host reads data from source register and delivers to target region.
            0x0000 => {
                let source = Register(operands[0]);
                let region_id = operands[1];
                StepResult::Yield(DomainOp::SendSignal { source, region_id })
            }

            // RECV_SIGNAL [target:1][region_id:1][_:2]
            // Host writes received signal data into target register.
            // Blocks until signal available (host decides blocking semantics).
            0x0001 => {
                let target = Register(operands[0]);
                let region_id = operands[1];
                StepResult::Yield(DomainOp::RecvSignal { target, region_id })
            }

            // BROADCAST [source:1][channel:1][_:2]
            // Host reads data from source and broadcasts to all subscribers on channel.
            0x0002 => {
                let source = Register(operands[0]);
                let channel = operands[1];
                StepResult::Yield(DomainOp::Broadcast { source, channel })
            }

            // SUBSCRIBE [target:1][region_id:1][_:2]
            // Host subscribes this program to signals from region_id.
            // Future signals from that region will be delivered to target register.
            0x0003 => {
                let target = Register(operands[0]);
                let region_id = operands[1];
                StepResult::Yield(DomainOp::Subscribe { target, region_id })
            }

            // MAILBOX_PEEK [target:1][mailbox_id:1][_:2]
            // Host writes front-of-mailbox into target WITHOUT consuming it.
            // If mailbox empty, target[0] = 0 (or host convention).
            0x0004 => {
                let target = Register(operands[0]);
                let mailbox_id = operands[1];
                StepResult::Yield(DomainOp::MailboxPeek { target, mailbox_id })
            }

            // MAILBOX_POP [target:1][mailbox_id:1][_:2]
            // Host writes front-of-mailbox into target AND removes it.
            0x0005 => {
                let target = Register(operands[0]);
                let mailbox_id = operands[1];
                StepResult::Yield(DomainOp::MailboxPop { target, mailbox_id })
            }

            // BARRIER_WAIT [barrier_id:1][_:3]
            // Host blocks this program until all participants reach the barrier.
            0x0006 => {
                let barrier_id = operands[0];
                StepResult::Yield(DomainOp::BarrierWait { barrier_id })
            }

            // ATOMIC_CAS [target:1][expected:1][desired:1][_:1]
            // Host atomically: if shared[target] == expected, set shared[target] = desired.
            // Result written back into target register (old value).
            0x0007 => {
                let target = Register(operands[0]);
                let expected = Register(operands[1]);
                let desired = Register(operands[2]);
                StepResult::Yield(DomainOp::AtomicCas { target, expected, desired })
            }

            _ => StepResult::Error(format!("tvmr.ipc: unknown opcode 0x{:04X}", opcode)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vm::extension::Extension;

    #[test]
    fn test_metadata() {
        let ext = IpcExtension::new();
        assert_eq!(ext.ext_id(), 0x0009);
        assert_eq!(ext.name(), "tvmr.ipc");
        assert_eq!(ext.instructions().len(), 8);
        assert_eq!(ext.instructions()[0].mnemonic, "SEND_SIGNAL");
        assert_eq!(ext.instructions()[7].mnemonic, "ATOMIC_CAS");
    }

    macro_rules! setup_ctx {
        () => {{
            use crate::vm::extension::LoopState;
            use crate::vm::interpreter::{ChemicalState, HotBuffer};

            let mut hot_regs: Vec<Option<HotBuffer>> = vec![None; 64];
            let mut cold_regs = vec![None; 64];
            let mut param_regs = vec![0i32; 64];
            let mut shape_regs: Vec<Vec<usize>> = vec![Vec::new(); 64];
            let mut pc = 0usize;
            let mut call_stack = Vec::new();
            let mut loop_stack: Vec<LoopState> = Vec::new();
            let input_buffer: Vec<i32> = Vec::new();
            let mut output_buffer = Vec::new();
            let target_buffer: Vec<i32> = Vec::new();
            let mut chemical_state = ChemicalState::baseline();
            let mut current_error = 0i32;
            let mut babble_scale = 0i32;
            let mut babble_phase = 0usize;
            let mut pressure_regs: Vec<Option<Vec<i32>>> = vec![None; 16];
            (
                hot_regs, cold_regs, param_regs, shape_regs, pc, call_stack,
                loop_stack, input_buffer, output_buffer, target_buffer,
                chemical_state, current_error, babble_scale, babble_phase,
                pressure_regs,
            )
        }};
    }

    fn make_ctx<'a>(
        hot_regs: &'a mut Vec<Option<crate::vm::interpreter::HotBuffer>>,
        cold_regs: &'a mut Vec<Option<crate::vm::interpreter::ColdBuffer>>,
        param_regs: &'a mut Vec<i32>,
        shape_regs: &'a mut Vec<Vec<usize>>,
        pc: &'a mut usize,
        call_stack: &'a mut Vec<usize>,
        loop_stack: &'a mut Vec<crate::vm::extension::LoopState>,
        input_buffer: &'a [i32],
        output_buffer: &'a mut Vec<i32>,
        target_buffer: &'a [i32],
        chemical_state: &'a mut crate::vm::interpreter::ChemicalState,
        current_error: &'a mut i32,
        babble_scale: &'a mut i32,
        babble_phase: &'a mut usize,
        pressure_regs: &'a mut Vec<Option<Vec<i32>>>,
    ) -> ExecutionContext<'a> {
        ExecutionContext {
            hot_regs,
            cold_regs,
            param_regs,
            shape_regs,
            pc,
            call_stack,
            loop_stack,
            input_buffer,
            output_buffer,
            target_buffer,
            chemical_state,
            current_error,
            babble_scale,
            babble_phase,
            pressure_regs,
            bank_cache: None,
        }
    }

    #[test]
    fn test_send_signal_yields() {
        let (
            mut hot_regs, mut cold_regs, mut param_regs, mut shape_regs,
            mut pc, mut call_stack, mut loop_stack, input_buffer,
            mut output_buffer, target_buffer, mut chemical_state,
            mut current_error, mut babble_scale, mut babble_phase,
            mut pressure_regs,
        ) = setup_ctx!();

        let mut ctx = make_ctx(
            &mut hot_regs, &mut cold_regs, &mut param_regs, &mut shape_regs,
            &mut pc, &mut call_stack, &mut loop_stack, &input_buffer,
            &mut output_buffer, &target_buffer, &mut chemical_state,
            &mut current_error, &mut babble_scale, &mut babble_phase,
            &mut pressure_regs,
        );

        let ext = IpcExtension::new();
        // SEND_SIGNAL H0, region=3
        let result = ext.execute(0x0000, [0x00, 3, 0, 0], &mut ctx);
        match result {
            StepResult::Yield(DomainOp::SendSignal { source, region_id }) => {
                assert_eq!(source, Register(0x00));
                assert_eq!(region_id, 3);
            }
            other => panic!("Expected Yield(SendSignal), got {:?}", other),
        }
    }

    #[test]
    fn test_recv_signal_yields() {
        let (
            mut hot_regs, mut cold_regs, mut param_regs, mut shape_regs,
            mut pc, mut call_stack, mut loop_stack, input_buffer,
            mut output_buffer, target_buffer, mut chemical_state,
            mut current_error, mut babble_scale, mut babble_phase,
            mut pressure_regs,
        ) = setup_ctx!();

        let mut ctx = make_ctx(
            &mut hot_regs, &mut cold_regs, &mut param_regs, &mut shape_regs,
            &mut pc, &mut call_stack, &mut loop_stack, &input_buffer,
            &mut output_buffer, &target_buffer, &mut chemical_state,
            &mut current_error, &mut babble_scale, &mut babble_phase,
            &mut pressure_regs,
        );

        let ext = IpcExtension::new();
        let result = ext.execute(0x0001, [0x01, 7, 0, 0], &mut ctx);
        match result {
            StepResult::Yield(DomainOp::RecvSignal { target, region_id }) => {
                assert_eq!(target, Register(0x01));
                assert_eq!(region_id, 7);
            }
            other => panic!("Expected Yield(RecvSignal), got {:?}", other),
        }
    }

    #[test]
    fn test_broadcast_yields() {
        let (
            mut hot_regs, mut cold_regs, mut param_regs, mut shape_regs,
            mut pc, mut call_stack, mut loop_stack, input_buffer,
            mut output_buffer, target_buffer, mut chemical_state,
            mut current_error, mut babble_scale, mut babble_phase,
            mut pressure_regs,
        ) = setup_ctx!();

        let mut ctx = make_ctx(
            &mut hot_regs, &mut cold_regs, &mut param_regs, &mut shape_regs,
            &mut pc, &mut call_stack, &mut loop_stack, &input_buffer,
            &mut output_buffer, &target_buffer, &mut chemical_state,
            &mut current_error, &mut babble_scale, &mut babble_phase,
            &mut pressure_regs,
        );

        let ext = IpcExtension::new();
        // BROADCAST H2, channel=1
        let result = ext.execute(0x0002, [0x02, 1, 0, 0], &mut ctx);
        match result {
            StepResult::Yield(DomainOp::Broadcast { source, channel }) => {
                assert_eq!(source, Register(0x02));
                assert_eq!(channel, 1);
            }
            other => panic!("Expected Yield(Broadcast), got {:?}", other),
        }
    }

    #[test]
    fn test_barrier_wait_yields() {
        let (
            mut hot_regs, mut cold_regs, mut param_regs, mut shape_regs,
            mut pc, mut call_stack, mut loop_stack, input_buffer,
            mut output_buffer, target_buffer, mut chemical_state,
            mut current_error, mut babble_scale, mut babble_phase,
            mut pressure_regs,
        ) = setup_ctx!();

        let mut ctx = make_ctx(
            &mut hot_regs, &mut cold_regs, &mut param_regs, &mut shape_regs,
            &mut pc, &mut call_stack, &mut loop_stack, &input_buffer,
            &mut output_buffer, &target_buffer, &mut chemical_state,
            &mut current_error, &mut babble_scale, &mut babble_phase,
            &mut pressure_regs,
        );

        let ext = IpcExtension::new();
        // BARRIER_WAIT barrier=2
        let result = ext.execute(0x0006, [2, 0, 0, 0], &mut ctx);
        match result {
            StepResult::Yield(DomainOp::BarrierWait { barrier_id }) => {
                assert_eq!(barrier_id, 2);
            }
            other => panic!("Expected Yield(BarrierWait), got {:?}", other),
        }
    }

    #[test]
    fn test_atomic_cas_yields() {
        let (
            mut hot_regs, mut cold_regs, mut param_regs, mut shape_regs,
            mut pc, mut call_stack, mut loop_stack, input_buffer,
            mut output_buffer, target_buffer, mut chemical_state,
            mut current_error, mut babble_scale, mut babble_phase,
            mut pressure_regs,
        ) = setup_ctx!();

        let mut ctx = make_ctx(
            &mut hot_regs, &mut cold_regs, &mut param_regs, &mut shape_regs,
            &mut pc, &mut call_stack, &mut loop_stack, &input_buffer,
            &mut output_buffer, &target_buffer, &mut chemical_state,
            &mut current_error, &mut babble_scale, &mut babble_phase,
            &mut pressure_regs,
        );

        let ext = IpcExtension::new();
        // ATOMIC_CAS H0, H1, H2
        let result = ext.execute(0x0007, [0x00, 0x01, 0x02, 0], &mut ctx);
        match result {
            StepResult::Yield(DomainOp::AtomicCas { target, expected, desired }) => {
                assert_eq!(target, Register(0x00));
                assert_eq!(expected, Register(0x01));
                assert_eq!(desired, Register(0x02));
            }
            other => panic!("Expected Yield(AtomicCas), got {:?}", other),
        }
    }

    #[test]
    fn test_all_ops_yield() {
        let (
            mut hot_regs, mut cold_regs, mut param_regs, mut shape_regs,
            mut pc, mut call_stack, mut loop_stack, input_buffer,
            mut output_buffer, target_buffer, mut chemical_state,
            mut current_error, mut babble_scale, mut babble_phase,
            mut pressure_regs,
        ) = setup_ctx!();

        let mut ctx = make_ctx(
            &mut hot_regs, &mut cold_regs, &mut param_regs, &mut shape_regs,
            &mut pc, &mut call_stack, &mut loop_stack, &input_buffer,
            &mut output_buffer, &target_buffer, &mut chemical_state,
            &mut current_error, &mut babble_scale, &mut babble_phase,
            &mut pressure_regs,
        );

        let ext = IpcExtension::new();

        // Verify all 8 opcodes yield (none return Continue)
        for opcode in 0x0000..=0x0007u16 {
            let result = ext.execute(opcode, [0x00, 0x01, 0x02, 0], &mut ctx);
            match result {
                StepResult::Yield(_) => {} // correct
                other => panic!("Opcode 0x{:04X} did not yield: {:?}", opcode, other),
            }
        }
    }
}
