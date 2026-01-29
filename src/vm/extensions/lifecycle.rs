//! Lifecycle Extension (ExtID: 0x0008)
//!
//! Boot, phase, tick, and thermogram lifecycle operations.
//! Integrates with the brain's boot sequence and persistence layer.
//!
//! All operations yield DomainOps — the host owns boot phase, tick counter,
//! neuronal level, and thermogram storage.

use crate::vm::extension::{
    DomainOp, ExecutionContext, Extension, InstructionMeta, OperandPattern, StepResult,
};
use crate::vm::register::Register;

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
        operands: [u8; 4],
        _ctx: &mut ExecutionContext,
    ) -> StepResult {
        match opcode {
            // PHASE_READ [target:1][_:3]
            // Host writes boot phase ordinal into H[target][0].
            0x0000 => {
                let target = Register(operands[0]);
                StepResult::Yield(DomainOp::PhaseRead { target })
            }

            // TICK_READ [target:1][_:3]
            // Host writes current tick count into H[target][0].
            0x0001 => {
                let target = Register(operands[0]);
                StepResult::Yield(DomainOp::TickRead { target })
            }

            // LEVEL_READ [target:1][_:3]
            // Host writes neuronal level ordinal into H[target][0].
            0x0002 => {
                let target = Register(operands[0]);
                StepResult::Yield(DomainOp::LevelRead { target })
            }

            // INIT_THERMO [reg:1][_:3]
            // Host initializes thermogram storage for cold register.
            0x0003 => {
                let register = Register(operands[0]);
                StepResult::Yield(DomainOp::InitThermo { register })
            }

            // SAVE_THERMO [reg:1][_:3]
            // Host persists cold register weights to thermogram.
            0x0004 => {
                let register = Register(operands[0]);
                StepResult::Yield(DomainOp::SaveThermo { register })
            }

            // LOAD_THERMO [reg:1][_:3]
            // Host loads persisted weights into cold register.
            0x0005 => {
                let register = Register(operands[0]);
                StepResult::Yield(DomainOp::LoadThermo { register })
            }

            // LOG_EVENT [source:1][event_type:1][_:2]
            // Host logs event with data from source register.
            // event_type: 0=boot, 1=phase_change, 2=error, 3=warning,
            //             4=metric, 5=checkpoint, 6=recovery, 7=shutdown.
            0x0006 => {
                let source = Register(operands[0]);
                let event_type = operands[1];
                StepResult::Yield(DomainOp::LogEvent { source, event_type })
            }

            // HALT_REGION [region_id:1][_:3]
            // Host halts the specified brain region.
            0x0007 => {
                let region_id = operands[0];
                StepResult::Yield(DomainOp::HaltRegion { region_id })
            }

            _ => StepResult::Error(format!(
                "tvmr.lifecycle: unknown opcode 0x{:04X}",
                opcode
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vm::extension::Extension;

    #[test]
    fn test_metadata() {
        let ext = LifecycleExtension::new();
        assert_eq!(ext.ext_id(), 0x0008);
        assert_eq!(ext.name(), "tvmr.lifecycle");
        assert_eq!(ext.instructions().len(), 8);
        assert_eq!(ext.instructions()[0].mnemonic, "PHASE_READ");
        assert_eq!(ext.instructions()[7].mnemonic, "HALT_REGION");
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
        }
    }

    #[test]
    fn test_phase_read_yields() {
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

        let ext = LifecycleExtension::new();
        // PHASE_READ H0
        let result = ext.execute(0x0000, [0x00, 0, 0, 0], &mut ctx);
        match result {
            StepResult::Yield(DomainOp::PhaseRead { target }) => {
                assert_eq!(target, Register(0x00));
            }
            other => panic!("Expected Yield(PhaseRead), got {:?}", other),
        }
    }

    #[test]
    fn test_tick_read_yields() {
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

        let ext = LifecycleExtension::new();
        let result = ext.execute(0x0001, [0x01, 0, 0, 0], &mut ctx);
        match result {
            StepResult::Yield(DomainOp::TickRead { target }) => {
                assert_eq!(target, Register(0x01));
            }
            other => panic!("Expected Yield(TickRead), got {:?}", other),
        }
    }

    #[test]
    fn test_save_thermo_yields() {
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

        let ext = LifecycleExtension::new();
        // SAVE_THERMO C0 (cold register = 0x40)
        let result = ext.execute(0x0004, [0x40, 0, 0, 0], &mut ctx);
        match result {
            StepResult::Yield(DomainOp::SaveThermo { register }) => {
                assert_eq!(register, Register(0x40));
            }
            other => panic!("Expected Yield(SaveThermo), got {:?}", other),
        }
    }

    #[test]
    fn test_log_event_yields() {
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

        let ext = LifecycleExtension::new();
        // LOG_EVENT H2, 3 (warning)
        let result = ext.execute(0x0006, [0x02, 3, 0, 0], &mut ctx);
        match result {
            StepResult::Yield(DomainOp::LogEvent { source, event_type }) => {
                assert_eq!(source, Register(0x02));
                assert_eq!(event_type, 3);
            }
            other => panic!("Expected Yield(LogEvent), got {:?}", other),
        }
    }

    #[test]
    fn test_halt_region_yields() {
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

        let ext = LifecycleExtension::new();
        // HALT_REGION 5
        let result = ext.execute(0x0007, [5, 0, 0, 0], &mut ctx);
        match result {
            StepResult::Yield(DomainOp::HaltRegion { region_id }) => {
                assert_eq!(region_id, 5);
            }
            other => panic!("Expected Yield(HaltRegion), got {:?}", other),
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

        let ext = LifecycleExtension::new();

        // Verify all 8 opcodes yield (none return Continue)
        for opcode in 0x0000..=0x0007u16 {
            let result = ext.execute(opcode, [0x00, 0x01, 0, 0], &mut ctx);
            match result {
                StepResult::Yield(_) => {} // correct
                other => panic!("Opcode 0x{:04X} did not yield: {:?}", opcode, other),
            }
        }
    }
}
