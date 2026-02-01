//! Pool Extension (ExtID: 0x000C)
//!
//! Biological neuron pool operations: tick, inject, read output, three-factor
//! plasticity, synaptogenesis, pruning, and persistence.
//!
//! All operations yield DomainOps to the host. The VM does not own the pool.

use crate::vm::extension::{
    DomainOp, ExecutionContext, Extension, InstructionMeta, OperandPattern, StepResult,
};
use crate::vm::register::Register;

/// Pool extension — biological neuron pool substrate operations.
pub struct PoolExtension {
    instructions: Vec<InstructionMeta>,
}

impl PoolExtension {
    pub fn new() -> Self {
        Self {
            instructions: vec![
                InstructionMeta {
                    opcode: 0x0000,
                    mnemonic: "POOL_TICK",
                    operand_pattern: OperandPattern::Reg,
                    description: "Step pool dynamics. Input from H[reg], spike count written back.",
                },
                InstructionMeta {
                    opcode: 0x0001,
                    mnemonic: "POOL_SAVE",
                    operand_pattern: OperandPattern::None,
                    description: "Persist pool state to .pool file. Yields PoolSave.",
                },
                InstructionMeta {
                    opcode: 0x0002,
                    mnemonic: "POOL_LOAD",
                    operand_pattern: OperandPattern::None,
                    description: "Restore pool state from .pool file. Yields PoolLoad.",
                },
                InstructionMeta {
                    opcode: 0x0003,
                    mnemonic: "POOL_INJECT",
                    operand_pattern: OperandPattern::Custom("[src:1][_:1][range_start:1][range_end:1]"),
                    description: "Inject signal into pool neuron range. Yields PoolInject.",
                },
                InstructionMeta {
                    opcode: 0x0004,
                    mnemonic: "POOL_READ_OUTPUT",
                    operand_pattern: OperandPattern::Custom("[tgt:1][_:1][range_start:1][range_end:1]"),
                    description: "Read output spikes from pool neuron range. Yields PoolReadOutput.",
                },
                InstructionMeta {
                    opcode: 0x0005,
                    mnemonic: "POOL_READ_STATS",
                    operand_pattern: OperandPattern::Reg,
                    description: "Read pool stats into H[reg]. Yields PoolReadStats.",
                },
                InstructionMeta {
                    opcode: 0x0006,
                    mnemonic: "POOL_MODULATE",
                    operand_pattern: OperandPattern::Reg,
                    description: "Apply three-factor plasticity. Result in H[reg]. Yields PoolModulate.",
                },
                InstructionMeta {
                    opcode: 0x0007,
                    mnemonic: "POOL_PRUNE_DEAD",
                    operand_pattern: OperandPattern::Reg,
                    description: "Prune dead synapses. Count in H[reg]. Yields PoolPruneDead.",
                },
                InstructionMeta {
                    opcode: 0x0008,
                    mnemonic: "POOL_SYNAPTOGENESIS",
                    operand_pattern: OperandPattern::Reg,
                    description: "Create new synapses (ACh-gated). Count in H[reg]. Yields PoolSynaptogenesis.",
                },
                InstructionMeta {
                    opcode: 0x0009,
                    mnemonic: "POOL_INJECT_DIRECT",
                    operand_pattern: OperandPattern::Custom("[src:1][_:1][start_lo:1][end_lo:1]"),
                    description: "Inject signal into pool neurons [start..end] (direct 0-255). Yields PoolInject.",
                },
                InstructionMeta {
                    opcode: 0x000A,
                    mnemonic: "POOL_READ_OUTPUT_DIRECT",
                    operand_pattern: OperandPattern::Custom("[tgt:1][_:1][start_lo:1][end_lo:1]"),
                    description: "Read output spikes from pool neurons [start..end] (direct 0-255). Yields PoolReadOutput.",
                },
            ],
        }
    }
}

impl Extension for PoolExtension {
    fn ext_id(&self) -> u16 {
        0x000C
    }

    fn name(&self) -> &str {
        "tvmr.pool"
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
            // POOL_TICK: [reg:1][_:3]
            0x0000 => {
                let reg = Register(operands[0]);
                StepResult::Yield(DomainOp::PoolTick { source: reg })
            }
            // POOL_SAVE: [_:4]
            0x0001 => {
                StepResult::Yield(DomainOp::PoolSave)
            }
            // POOL_LOAD: [_:4]
            0x0002 => {
                StepResult::Yield(DomainOp::PoolLoad)
            }
            // POOL_INJECT: [src:1][_:1][range_start:1][range_end:1]
            // range values are multiplied by 256 to allow addressing up to 65536 neurons
            // e.g., operand 0 = neuron 0, operand 255 = neuron 65280
            0x0003 => {
                let src = Register(operands[0]);
                let range_start = operands[2] as u16 * 256;
                let range_end = operands[3] as u16 * 256;
                StepResult::Yield(DomainOp::PoolInject { source: src, range_start, range_end })
            }
            // POOL_READ_OUTPUT: [tgt:1][_:1][range_start:1][range_end:1]
            0x0004 => {
                let tgt = Register(operands[0]);
                let range_start = operands[2] as u16 * 256;
                let range_end = operands[3] as u16 * 256;
                StepResult::Yield(DomainOp::PoolReadOutput { target: tgt, range_start, range_end })
            }
            // POOL_READ_STATS: [reg:1][_:3]
            0x0005 => {
                let reg = Register(operands[0]);
                StepResult::Yield(DomainOp::PoolReadStats { target: reg })
            }
            // POOL_MODULATE: [reg:1][_:3]
            0x0006 => {
                let reg = Register(operands[0]);
                StepResult::Yield(DomainOp::PoolModulate { result: reg })
            }
            // POOL_PRUNE_DEAD: [reg:1][_:3]
            0x0007 => {
                let reg = Register(operands[0]);
                StepResult::Yield(DomainOp::PoolPruneDead { result: reg })
            }
            // POOL_SYNAPTOGENESIS: [reg:1][_:3]
            0x0008 => {
                let reg = Register(operands[0]);
                StepResult::Yield(DomainOp::PoolSynaptogenesis { result: reg })
            }
            // POOL_INJECT_DIRECT: [src:1][_:1][start_lo:1][end_lo:1]
            // Direct neuron indices 0-255 — for sub-256 neuron pools
            0x0009 => {
                let src = Register(operands[0]);
                let range_start = operands[2] as u16;
                let range_end = operands[3] as u16;
                StepResult::Yield(DomainOp::PoolInject { source: src, range_start, range_end })
            }
            // POOL_READ_OUTPUT_DIRECT: [tgt:1][_:1][start_lo:1][end_lo:1]
            // Direct neuron indices 0-255 — for sub-256 neuron pools
            0x000A => {
                let tgt = Register(operands[0]);
                let range_start = operands[2] as u16;
                let range_end = operands[3] as u16;
                StepResult::Yield(DomainOp::PoolReadOutput { target: tgt, range_start, range_end })
            }
            _ => StepResult::Error(format!("tvmr.pool: unknown opcode 0x{:04X}", opcode)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vm::interpreter::{ChemicalState, HotBuffer};
    use crate::vm::extension::LoopState;

    macro_rules! setup_ctx {
        () => {{
            let hot_regs: Vec<Option<HotBuffer>> = vec![None; 64];
            let cold_regs = vec![None; 64];
            let param_regs = vec![0i32; 64];
            let shape_regs: Vec<Vec<usize>> = vec![Vec::new(); 64];
            let pc = 0usize;
            let call_stack = Vec::new();
            let loop_stack: Vec<LoopState> = Vec::new();
            let input_buffer = vec![];
            let output_buffer = Vec::new();
            let target_buffer = vec![];
            let chemical_state = ChemicalState::baseline();
            let current_error = 0i32;
            let babble_scale = 0i32;
            let babble_phase = 0usize;
            let pressure_regs: Vec<Option<Vec<i32>>> = vec![None; 16];
            (
                hot_regs, cold_regs, param_regs, shape_regs,
                pc, call_stack, loop_stack, input_buffer,
                output_buffer, target_buffer, chemical_state,
                current_error, babble_scale, babble_phase, pressure_regs,
            )
        }};
    }

    fn make_ctx<'a>(
        hot_regs: &'a mut Vec<Option<HotBuffer>>,
        cold_regs: &'a mut Vec<Option<crate::vm::interpreter::ColdBuffer>>,
        param_regs: &'a mut Vec<i32>,
        shape_regs: &'a mut Vec<Vec<usize>>,
        pc: &'a mut usize,
        call_stack: &'a mut Vec<usize>,
        loop_stack: &'a mut Vec<LoopState>,
        input_buffer: &'a [i32],
        output_buffer: &'a mut Vec<i32>,
        target_buffer: &'a [i32],
        chemical_state: &'a mut ChemicalState,
        current_error: &'a mut i32,
        babble_scale: &'a mut i32,
        babble_phase: &'a mut usize,
        pressure_regs: &'a mut Vec<Option<Vec<i32>>>,
    ) -> ExecutionContext<'a> {
        ExecutionContext {
            hot_regs, cold_regs, param_regs, shape_regs,
            pc, call_stack, loop_stack, input_buffer,
            output_buffer, target_buffer, chemical_state,
            current_error, babble_scale, babble_phase, pressure_regs,
            bank_cache: None,
        }
    }

    #[test]
    fn test_metadata() {
        let ext = PoolExtension::new();
        assert_eq!(ext.ext_id(), 0x000C);
        assert_eq!(ext.name(), "tvmr.pool");
        assert_eq!(ext.instructions().len(), 11);
    }

    #[test]
    fn test_pool_tick_yields() {
        let ext = PoolExtension::new();
        let (
            mut hot_regs, mut cold_regs, mut param_regs, mut shape_regs,
            mut pc, mut call_stack, mut loop_stack, input_buffer,
            mut output_buffer, target_buffer, mut chemical_state,
            mut current_error, mut babble_scale, mut babble_phase, mut pressure_regs,
        ) = setup_ctx!();
        let mut ctx = make_ctx(
            &mut hot_regs, &mut cold_regs, &mut param_regs, &mut shape_regs,
            &mut pc, &mut call_stack, &mut loop_stack, &input_buffer,
            &mut output_buffer, &target_buffer, &mut chemical_state,
            &mut current_error, &mut babble_scale, &mut babble_phase, &mut pressure_regs,
        );

        let result = ext.execute(0x0000, [0x00, 0x00, 0x00, 0x00], &mut ctx);
        match result {
            StepResult::Yield(DomainOp::PoolTick { source }) => {
                assert_eq!(source, Register(0x00));
            }
            other => panic!("Expected Yield(PoolTick), got {:?}", other),
        }
    }

    #[test]
    fn test_pool_save_yields() {
        let ext = PoolExtension::new();
        let (
            mut hot_regs, mut cold_regs, mut param_regs, mut shape_regs,
            mut pc, mut call_stack, mut loop_stack, input_buffer,
            mut output_buffer, target_buffer, mut chemical_state,
            mut current_error, mut babble_scale, mut babble_phase, mut pressure_regs,
        ) = setup_ctx!();
        let mut ctx = make_ctx(
            &mut hot_regs, &mut cold_regs, &mut param_regs, &mut shape_regs,
            &mut pc, &mut call_stack, &mut loop_stack, &input_buffer,
            &mut output_buffer, &target_buffer, &mut chemical_state,
            &mut current_error, &mut babble_scale, &mut babble_phase, &mut pressure_regs,
        );

        let result = ext.execute(0x0001, [0x00, 0x00, 0x00, 0x00], &mut ctx);
        assert!(matches!(result, StepResult::Yield(DomainOp::PoolSave)));
    }

    #[test]
    fn test_pool_load_yields() {
        let ext = PoolExtension::new();
        let (
            mut hot_regs, mut cold_regs, mut param_regs, mut shape_regs,
            mut pc, mut call_stack, mut loop_stack, input_buffer,
            mut output_buffer, target_buffer, mut chemical_state,
            mut current_error, mut babble_scale, mut babble_phase, mut pressure_regs,
        ) = setup_ctx!();
        let mut ctx = make_ctx(
            &mut hot_regs, &mut cold_regs, &mut param_regs, &mut shape_regs,
            &mut pc, &mut call_stack, &mut loop_stack, &input_buffer,
            &mut output_buffer, &target_buffer, &mut chemical_state,
            &mut current_error, &mut babble_scale, &mut babble_phase, &mut pressure_regs,
        );

        let result = ext.execute(0x0002, [0x00, 0x00, 0x00, 0x00], &mut ctx);
        assert!(matches!(result, StepResult::Yield(DomainOp::PoolLoad)));
    }

    #[test]
    fn test_pool_inject_yields() {
        let ext = PoolExtension::new();
        let (
            mut hot_regs, mut cold_regs, mut param_regs, mut shape_regs,
            mut pc, mut call_stack, mut loop_stack, input_buffer,
            mut output_buffer, target_buffer, mut chemical_state,
            mut current_error, mut babble_scale, mut babble_phase, mut pressure_regs,
        ) = setup_ctx!();
        let mut ctx = make_ctx(
            &mut hot_regs, &mut cold_regs, &mut param_regs, &mut shape_regs,
            &mut pc, &mut call_stack, &mut loop_stack, &input_buffer,
            &mut output_buffer, &target_buffer, &mut chemical_state,
            &mut current_error, &mut babble_scale, &mut babble_phase, &mut pressure_regs,
        );

        // POOL_INJECT H0, range=[0*256, 1*256] = [0, 256]
        let result = ext.execute(0x0003, [0x00, 0x00, 0x00, 0x01], &mut ctx);
        match result {
            StepResult::Yield(DomainOp::PoolInject { source, range_start, range_end }) => {
                assert_eq!(source, Register(0x00));
                assert_eq!(range_start, 0);
                assert_eq!(range_end, 256);
            }
            other => panic!("Expected Yield(PoolInject), got {:?}", other),
        }
    }

    #[test]
    fn test_pool_modulate_yields() {
        let ext = PoolExtension::new();
        let (
            mut hot_regs, mut cold_regs, mut param_regs, mut shape_regs,
            mut pc, mut call_stack, mut loop_stack, input_buffer,
            mut output_buffer, target_buffer, mut chemical_state,
            mut current_error, mut babble_scale, mut babble_phase, mut pressure_regs,
        ) = setup_ctx!();
        let mut ctx = make_ctx(
            &mut hot_regs, &mut cold_regs, &mut param_regs, &mut shape_regs,
            &mut pc, &mut call_stack, &mut loop_stack, &input_buffer,
            &mut output_buffer, &target_buffer, &mut chemical_state,
            &mut current_error, &mut babble_scale, &mut babble_phase, &mut pressure_regs,
        );

        let result = ext.execute(0x0006, [0x01, 0x00, 0x00, 0x00], &mut ctx);
        match result {
            StepResult::Yield(DomainOp::PoolModulate { result: reg }) => {
                assert_eq!(reg, Register(0x01));
            }
            other => panic!("Expected Yield(PoolModulate), got {:?}", other),
        }
    }

    #[test]
    fn test_all_pool_ops_yield() {
        let ext = PoolExtension::new();
        let (
            mut hot_regs, mut cold_regs, mut param_regs, mut shape_regs,
            mut pc, mut call_stack, mut loop_stack, input_buffer,
            mut output_buffer, target_buffer, mut chemical_state,
            mut current_error, mut babble_scale, mut babble_phase, mut pressure_regs,
        ) = setup_ctx!();
        let mut ctx = make_ctx(
            &mut hot_regs, &mut cold_regs, &mut param_regs, &mut shape_regs,
            &mut pc, &mut call_stack, &mut loop_stack, &input_buffer,
            &mut output_buffer, &target_buffer, &mut chemical_state,
            &mut current_error, &mut babble_scale, &mut babble_phase, &mut pressure_regs,
        );

        for opcode in 0x0000..=0x000Au16 {
            let result = ext.execute(opcode, [0x00, 0x00, 0x00, 0x00], &mut ctx);
            assert!(
                matches!(result, StepResult::Yield(_)),
                "Opcode 0x{:04X} should yield, got {:?}", opcode, result
            );
        }
    }

    #[test]
    fn test_unknown_opcode_errors() {
        let ext = PoolExtension::new();
        let (
            mut hot_regs, mut cold_regs, mut param_regs, mut shape_regs,
            mut pc, mut call_stack, mut loop_stack, input_buffer,
            mut output_buffer, target_buffer, mut chemical_state,
            mut current_error, mut babble_scale, mut babble_phase, mut pressure_regs,
        ) = setup_ctx!();
        let mut ctx = make_ctx(
            &mut hot_regs, &mut cold_regs, &mut param_regs, &mut shape_regs,
            &mut pc, &mut call_stack, &mut loop_stack, &input_buffer,
            &mut output_buffer, &target_buffer, &mut chemical_state,
            &mut current_error, &mut babble_scale, &mut babble_phase, &mut pressure_regs,
        );

        let result = ext.execute(0x000B, [0x00, 0x00, 0x00, 0x00], &mut ctx);
        assert!(matches!(result, StepResult::Error(_)));
    }
}
