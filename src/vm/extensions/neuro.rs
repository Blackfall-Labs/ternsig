//! Neuro Extension (ExtID: 0x0005)
//!
//! Substrate I/O operations: chemical read/set/inject, field read/write/tick,
//! stimulation, convergence, temperature management, metabolic constraints.
//!
//! All substrate-touching operations yield DomainOps to the host.
//! The VM does not own the substrate — the host does.
//! TEMP_READ and TEMP_WRITE operate on cold register temperature arrays
//! already in the ExecutionContext, so they execute locally.

use crate::vm::extension::{
    DomainOp, ExecutionContext, Extension, InstructionMeta, OperandPattern, StepResult,
};
use crate::vm::interpreter::{HotBuffer, SignalTemperature};
use crate::vm::register::Register;

/// Neuro extension — substrate I/O operations.
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
                    description: "Read chemical level into H[reg][0]. Yields ChemRead.",
                },
                InstructionMeta {
                    opcode: 0x0001,
                    mnemonic: "CHEM_SET",
                    operand_pattern: OperandPattern::RegImm8,
                    description: "SET chemical level from H[reg][0] (authoritative). Yields ChemSet.",
                },
                InstructionMeta {
                    opcode: 0x0002,
                    mnemonic: "CHEM_INJECT",
                    operand_pattern: OperandPattern::RegImm8,
                    description: "Additive chemical injection (phasic). Yields ChemInject.",
                },
                InstructionMeta {
                    opcode: 0x0003,
                    mnemonic: "FIELD_READ",
                    operand_pattern: OperandPattern::RegImm8,
                    description: "Read field region slice into H[reg]. Yields FieldRead.",
                },
                InstructionMeta {
                    opcode: 0x0004,
                    mnemonic: "FIELD_WRITE",
                    operand_pattern: OperandPattern::RegImm8,
                    description: "Write H[reg] to field region slice. Yields FieldWrite.",
                },
                InstructionMeta {
                    opcode: 0x0005,
                    mnemonic: "FIELD_TICK",
                    operand_pattern: OperandPattern::Imm8,
                    description: "Advance field by 1 tick (decay, age). Yields FieldTick.",
                },
                InstructionMeta {
                    opcode: 0x0006,
                    mnemonic: "STIM_READ",
                    operand_pattern: OperandPattern::Reg,
                    description: "Read all stimulation levels into H[reg]. Yields StimRead.",
                },
                InstructionMeta {
                    opcode: 0x0007,
                    mnemonic: "VALENCE_READ",
                    operand_pattern: OperandPattern::Reg,
                    description: "Read valence [reward, punish] into H[reg]. Yields ValenceRead.",
                },
                InstructionMeta {
                    opcode: 0x0008,
                    mnemonic: "CONV_READ",
                    operand_pattern: OperandPattern::Reg,
                    description: "Read convergence field state into H[reg]. Yields ConvRead.",
                },
                InstructionMeta {
                    opcode: 0x0009,
                    mnemonic: "TEMP_READ",
                    operand_pattern: OperandPattern::RegReg,
                    description: "Read cold register temperatures into H[dst].",
                },
                InstructionMeta {
                    opcode: 0x000A,
                    mnemonic: "TEMP_WRITE",
                    operand_pattern: OperandPattern::RegReg,
                    description: "Write temperatures from H[src] to cold register.",
                },
                InstructionMeta {
                    opcode: 0x000B,
                    mnemonic: "FIELD_DECAY",
                    operand_pattern: OperandPattern::Custom("[field_id:1][retention:1][fatigue:1][_:1]"),
                    description: "Apply metabolic decay to field. Yields FieldDecay.",
                },
                InstructionMeta {
                    opcode: 0x000C,
                    mnemonic: "LATERAL_INHIBIT",
                    operand_pattern: OperandPattern::RegImm8,
                    description: "Winner-take-some: dominant suppresses others. Yields LateralInhibit.",
                },
                InstructionMeta {
                    opcode: 0x000D,
                    mnemonic: "EXHAUSTION_BOOST",
                    operand_pattern: OperandPattern::RegImm8,
                    description: "Apply exhaustion decay boost. Yields ExhaustionBoost.",
                },
                InstructionMeta {
                    opcode: 0x000E,
                    mnemonic: "NOVELTY_SCORE",
                    operand_pattern: OperandPattern::RegReg,
                    description: "Compute novelty z-scores from region energies. Yields NoveltyScore.",
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
        operands: [u8; 4],
        ctx: &mut ExecutionContext,
    ) -> StepResult {
        match opcode {
            // CHEM_READ: [reg:1][chem_id:1][_:2]
            0x0000 => {
                let reg = Register(operands[0]);
                let chem_id = operands[1];
                StepResult::Yield(DomainOp::ChemRead { target: reg, chem_id })
            }
            // CHEM_SET: [reg:1][chem_id:1][_:2]
            0x0001 => {
                let reg = Register(operands[0]);
                let chem_id = operands[1];
                StepResult::Yield(DomainOp::ChemSet { source: reg, chem_id })
            }
            // CHEM_INJECT: [reg:1][chem_id:1][_:2]
            0x0002 => {
                let reg = Register(operands[0]);
                let chem_id = operands[1];
                StepResult::Yield(DomainOp::ChemInject { source: reg, chem_id })
            }
            // FIELD_READ: [reg:1][field_id:1][_:2]
            0x0003 => {
                let reg = Register(operands[0]);
                let field_id = operands[1];
                StepResult::Yield(DomainOp::FieldRead { target: reg, field_id })
            }
            // FIELD_WRITE: [reg:1][field_id:1][_:2]
            0x0004 => {
                let reg = Register(operands[0]);
                let field_id = operands[1];
                StepResult::Yield(DomainOp::FieldWrite { source: reg, field_id })
            }
            // FIELD_TICK: [field_id:1][_:3]
            0x0005 => {
                let field_id = operands[0];
                StepResult::Yield(DomainOp::FieldTick { field_id })
            }
            // STIM_READ: [reg:1][_:3]
            0x0006 => {
                let reg = Register(operands[0]);
                StepResult::Yield(DomainOp::StimRead { target: reg })
            }
            // VALENCE_READ: [reg:1][_:3]
            0x0007 => {
                let reg = Register(operands[0]);
                StepResult::Yield(DomainOp::ValenceRead { target: reg })
            }
            // CONV_READ: [reg:1][_:3]
            0x0008 => {
                let reg = Register(operands[0]);
                StepResult::Yield(DomainOp::ConvRead { target: reg })
            }
            // TEMP_READ: [dst:1][cold_src:1][_:2] — local execution
            0x0009 => execute_temp_read(operands, ctx),
            // TEMP_WRITE: [hot_src:1][cold_dst:1][_:2] — local execution
            0x000A => execute_temp_write(operands, ctx),
            // FIELD_DECAY: [field_id:1][retention:1][fatigue:1][_:1]
            0x000B => {
                let field_id = operands[0];
                let retention = operands[1];
                let fatigue_boost = operands[2];
                StepResult::Yield(DomainOp::FieldDecay { field_id, retention, fatigue_boost })
            }
            // LATERAL_INHIBIT: [reg:1][strength:1][_:2]
            0x000C => {
                let reg = Register(operands[0]);
                let strength = operands[1];
                StepResult::Yield(DomainOp::LateralInhibit { source: reg, strength })
            }
            // EXHAUSTION_BOOST: [reg:1][factor:1][_:2]
            0x000D => {
                let reg = Register(operands[0]);
                let factor = operands[1];
                StepResult::Yield(DomainOp::ExhaustionBoost { source: reg, factor })
            }
            // NOVELTY_SCORE: [dst:1][src:1][_:2]
            0x000E => {
                let dst = Register(operands[0]);
                let src = Register(operands[1]);
                StepResult::Yield(DomainOp::NoveltyScore { target: dst, source: src })
            }
            _ => StepResult::Error(format!("tvmr.neuro: unknown opcode 0x{:04X}", opcode)),
        }
    }
}

// =============================================================================
// Local execution: temperature read/write (operates on cold register arrays)
// =============================================================================

/// TEMP_READ: Read cold register temperatures into hot register.
/// Operands: [dst_hot:1][src_cold:1][_:2]
/// Writes temperature values as i32 (0=Hot, 1=Warm, 2=Cool, 3=Cold) into H[dst].
fn execute_temp_read(ops: [u8; 4], ctx: &mut ExecutionContext) -> StepResult {
    let dst_idx = Register(ops[0]).index();
    let cold_idx = Register(ops[1]).index();

    let cold = match &ctx.cold_regs[cold_idx] {
        Some(buf) => buf,
        None => return StepResult::Error(format!("TEMP_READ: C{} not allocated", cold_idx)),
    };

    let data: Vec<i32> = match &cold.temperatures {
        Some(temps) => temps.iter().map(|t| *t as u8 as i32).collect(),
        None => vec![0i32; cold.weights.len()], // All HOT if no temperature array
    };
    let len = data.len();

    ctx.hot_regs[dst_idx] = Some(HotBuffer {
        data,
        shape: vec![len],
    });

    StepResult::Continue
}

/// TEMP_WRITE: Write hot register values as temperatures to cold register.
/// Operands: [src_hot:1][dst_cold:1][_:2]
/// Reads i32 values from H[src] and maps: 0=Hot, 1=Warm, 2=Cool, 3+=Cold.
fn execute_temp_write(ops: [u8; 4], ctx: &mut ExecutionContext) -> StepResult {
    let src_idx = Register(ops[0]).index();
    let cold_idx = Register(ops[1]).index();

    let src_data = match &ctx.hot_regs[src_idx] {
        Some(buf) => buf.data.clone(),
        None => return StepResult::Error(format!("TEMP_WRITE: H{} not allocated", src_idx)),
    };

    let cold = match &mut ctx.cold_regs[cold_idx] {
        Some(buf) => buf,
        None => return StepResult::Error(format!("TEMP_WRITE: C{} not allocated", cold_idx)),
    };

    let temps: Vec<SignalTemperature> = src_data
        .iter()
        .map(|&v| SignalTemperature::from_u8(v.clamp(0, 3) as u8))
        .collect();

    // Pad or truncate to match weight count
    let weight_count = cold.weights.len();
    let mut final_temps = temps;
    final_temps.resize(weight_count, SignalTemperature::Hot);

    cold.temperatures = Some(final_temps);

    StepResult::Continue
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vm::interpreter::{ChemicalState, ColdBuffer};
    use crate::vm::extension::LoopState;
    use crate::Signal;

    macro_rules! setup_ctx {
        () => {{
            let hot_regs: Vec<Option<HotBuffer>> = vec![None; 64];
            let cold_regs: Vec<Option<ColdBuffer>> = vec![None; 64];
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
        cold_regs: &'a mut Vec<Option<ColdBuffer>>,
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
        }
    }

    #[test]
    fn test_metadata() {
        let ext = NeuroExtension::new();
        assert_eq!(ext.ext_id(), 0x0005);
        assert_eq!(ext.name(), "tvmr.neuro");
        assert_eq!(ext.instructions().len(), 15);
    }

    #[test]
    fn test_chem_read_yields_domain_op() {
        let ext = NeuroExtension::new();
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

        // CHEM_READ H0, dopamine(0)
        let result = ext.execute(0x0000, [0x00, 0x00, 0x00, 0x00], &mut ctx);
        match result {
            StepResult::Yield(DomainOp::ChemRead { target, chem_id }) => {
                assert_eq!(target, Register(0x00));
                assert_eq!(chem_id, 0);
            }
            other => panic!("Expected Yield(ChemRead), got {:?}", other),
        }
    }

    #[test]
    fn test_chem_set_yields_domain_op() {
        let ext = NeuroExtension::new();
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

        // CHEM_SET H1, serotonin(1)
        let result = ext.execute(0x0001, [0x01, 0x01, 0x00, 0x00], &mut ctx);
        match result {
            StepResult::Yield(DomainOp::ChemSet { source, chem_id }) => {
                assert_eq!(source, Register(0x01));
                assert_eq!(chem_id, 1);
            }
            other => panic!("Expected Yield(ChemSet), got {:?}", other),
        }
    }

    #[test]
    fn test_field_tick_yields_domain_op() {
        let ext = NeuroExtension::new();
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

        // FIELD_TICK activity_field(0)
        let result = ext.execute(0x0005, [0x00, 0x00, 0x00, 0x00], &mut ctx);
        match result {
            StepResult::Yield(DomainOp::FieldTick { field_id }) => {
                assert_eq!(field_id, 0);
            }
            other => panic!("Expected Yield(FieldTick), got {:?}", other),
        }
    }

    #[test]
    fn test_temp_read_local() {
        let ext = NeuroExtension::new();
        let (
            mut hot_regs, mut cold_regs, mut param_regs, mut shape_regs,
            mut pc, mut call_stack, mut loop_stack, input_buffer,
            mut output_buffer, target_buffer, mut chemical_state,
            mut current_error, mut babble_scale, mut babble_phase, mut pressure_regs,
        ) = setup_ctx!();

        // Set up a cold register with temperatures
        let mut cold = ColdBuffer::new(vec![4]);
        cold.temperatures = Some(vec![
            SignalTemperature::Hot,
            SignalTemperature::Warm,
            SignalTemperature::Cool,
            SignalTemperature::Cold,
        ]);
        cold_regs[0] = Some(cold);

        let mut ctx = make_ctx(
            &mut hot_regs, &mut cold_regs, &mut param_regs, &mut shape_regs,
            &mut pc, &mut call_stack, &mut loop_stack, &input_buffer,
            &mut output_buffer, &target_buffer, &mut chemical_state,
            &mut current_error, &mut babble_scale, &mut babble_phase, &mut pressure_regs,
        );

        // TEMP_READ H0, C0 — operands: [dst=H0(0x00), src=C0(0x40), _, _]
        let result = ext.execute(0x0009, [0x00, 0x40, 0x00, 0x00], &mut ctx);
        assert!(matches!(result, StepResult::Continue));

        let h0 = ctx.hot_regs[0].as_ref().unwrap();
        assert_eq!(h0.data, vec![0, 1, 2, 3]); // Hot=0, Warm=1, Cool=2, Cold=3
    }

    #[test]
    fn test_temp_write_local() {
        let ext = NeuroExtension::new();
        let (
            mut hot_regs, mut cold_regs, mut param_regs, mut shape_regs,
            mut pc, mut call_stack, mut loop_stack, input_buffer,
            mut output_buffer, target_buffer, mut chemical_state,
            mut current_error, mut babble_scale, mut babble_phase, mut pressure_regs,
        ) = setup_ctx!();

        // Set up source hot register with temperature values
        hot_regs[0] = Some(HotBuffer { data: vec![0, 2, 3, 1], shape: vec![4] });
        // Set up target cold register
        cold_regs[0] = Some(ColdBuffer::new(vec![4]));

        let mut ctx = make_ctx(
            &mut hot_regs, &mut cold_regs, &mut param_regs, &mut shape_regs,
            &mut pc, &mut call_stack, &mut loop_stack, &input_buffer,
            &mut output_buffer, &target_buffer, &mut chemical_state,
            &mut current_error, &mut babble_scale, &mut babble_phase, &mut pressure_regs,
        );

        // TEMP_WRITE H0, C0 — operands: [src=H0(0x00), dst=C0(0x40), _, _]
        let result = ext.execute(0x000A, [0x00, 0x40, 0x00, 0x00], &mut ctx);
        assert!(matches!(result, StepResult::Continue));

        let cold = ctx.cold_regs[0].as_ref().unwrap();
        let temps = cold.temperatures.as_ref().unwrap();
        assert_eq!(temps[0], SignalTemperature::Hot);
        assert_eq!(temps[1], SignalTemperature::Cool);
        assert_eq!(temps[2], SignalTemperature::Cold);
        assert_eq!(temps[3], SignalTemperature::Warm);
    }

    #[test]
    fn test_lateral_inhibit_yields() {
        let ext = NeuroExtension::new();
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

        // LATERAL_INHIBIT H0, strength=128
        let result = ext.execute(0x000C, [0x00, 128, 0x00, 0x00], &mut ctx);
        match result {
            StepResult::Yield(DomainOp::LateralInhibit { source, strength }) => {
                assert_eq!(source, Register(0x00));
                assert_eq!(strength, 128);
            }
            other => panic!("Expected Yield(LateralInhibit), got {:?}", other),
        }
    }

    #[test]
    fn test_all_yield_ops_produce_correct_variants() {
        let ext = NeuroExtension::new();
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

        // Verify each yield-producing opcode returns Yield (not Continue or Error)
        let yield_opcodes = [
            0x0000, 0x0001, 0x0002, 0x0003, 0x0004,
            0x0005, 0x0006, 0x0007, 0x0008,
            0x000B, 0x000C, 0x000D, 0x000E,
        ];
        for &opcode in &yield_opcodes {
            let result = ext.execute(opcode, [0x00, 0x00, 0x00, 0x00], &mut ctx);
            assert!(
                matches!(result, StepResult::Yield(_)),
                "Opcode 0x{:04X} should yield, got {:?}", opcode, result
            );
        }
    }
}
