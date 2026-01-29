//! Orchestration Extension (ExtID: 0x0007)
//!
//! Two halves:
//! - **Model table** (0x0000-0x0007): Load, execute, chain Ternsig models by slot.
//!   MODEL_LOAD, MODEL_EXEC, MODEL_RELOAD yield DomainOps.
//!   MODEL_INPUT, MODEL_OUTPUT, MODEL_UNLOAD, MODEL_STATUS, MODEL_CHAIN
//!   operate on an in-VM model table (slots with input/output registers).
//!
//! - **Region routing** (0x0008-0x000D): Route signals between brain regions.
//!   All region routing ops yield DomainOps — the host owns regions.

use crate::vm::extension::{
    DomainOp, ExecutionContext, Extension, InstructionMeta, OperandPattern, StepResult,
};
use crate::vm::interpreter::HotBuffer;
use crate::vm::register::Register;
use std::sync::Mutex;

/// Model slot state in the model table.
#[derive(Debug, Clone)]
struct ModelSlot {
    /// Whether a model is loaded in this slot.
    loaded: bool,
    /// Whether the model is currently executing.
    running: bool,
    /// Input register index (hot bank) bound to this slot.
    input_reg: Option<usize>,
    /// Output register index (hot bank) bound to this slot.
    output_reg: Option<usize>,
}

impl Default for ModelSlot {
    fn default() -> Self {
        Self {
            loaded: false,
            running: false,
            input_reg: None,
            output_reg: None,
        }
    }
}

/// Orchestration extension — model table + region routing.
pub struct OrchestrationExtension {
    instructions: Vec<InstructionMeta>,
    /// Model table: 256 slots (indexed by u8).
    model_table: Mutex<Vec<ModelSlot>>,
}

impl OrchestrationExtension {
    pub fn new() -> Self {
        Self {
            instructions: vec![
                // Model table ops
                InstructionMeta {
                    opcode: 0x0000,
                    mnemonic: "MODEL_LOAD",
                    operand_pattern: OperandPattern::RegImm8,
                    description: "Load model into table slot. Yields ModelLoad.",
                },
                InstructionMeta {
                    opcode: 0x0001,
                    mnemonic: "MODEL_EXEC",
                    operand_pattern: OperandPattern::RegImm8,
                    description: "Execute model from table slot. Yields ModelExec.",
                },
                InstructionMeta {
                    opcode: 0x0002,
                    mnemonic: "MODEL_INPUT",
                    operand_pattern: OperandPattern::RegImm8,
                    description: "Set input register for table slot.",
                },
                InstructionMeta {
                    opcode: 0x0003,
                    mnemonic: "MODEL_OUTPUT",
                    operand_pattern: OperandPattern::RegImm8,
                    description: "Get output from table slot into register.",
                },
                InstructionMeta {
                    opcode: 0x0004,
                    mnemonic: "MODEL_UNLOAD",
                    operand_pattern: OperandPattern::Imm8,
                    description: "Unload model from table slot.",
                },
                InstructionMeta {
                    opcode: 0x0005,
                    mnemonic: "MODEL_STATUS",
                    operand_pattern: OperandPattern::RegImm8,
                    description: "Get model status: 0=empty, 1=loaded, 2=running.",
                },
                InstructionMeta {
                    opcode: 0x0006,
                    mnemonic: "MODEL_RELOAD",
                    operand_pattern: OperandPattern::Imm8,
                    description: "Hot-reload model in slot. Yields ModelReload.",
                },
                InstructionMeta {
                    opcode: 0x0007,
                    mnemonic: "MODEL_CHAIN",
                    operand_pattern: OperandPattern::Custom("[slot1:1][slot2:1][_:2]"),
                    description: "Chain slot1 output to slot2 input.",
                },
                // Region routing ops
                InstructionMeta {
                    opcode: 0x0008,
                    mnemonic: "ROUTE_INPUT",
                    operand_pattern: OperandPattern::RegImm8,
                    description: "Route H[reg] to region ID. Yields RouteInput.",
                },
                InstructionMeta {
                    opcode: 0x0009,
                    mnemonic: "REGION_FIRE",
                    operand_pattern: OperandPattern::RegImm8,
                    description: "Fire region, output into H[reg]. Yields RegionFire.",
                },
                InstructionMeta {
                    opcode: 0x000A,
                    mnemonic: "COLLECT_OUTPUTS",
                    operand_pattern: OperandPattern::Reg,
                    description: "Aggregate region outputs into H[reg]. Yields CollectOutputs.",
                },
                InstructionMeta {
                    opcode: 0x000B,
                    mnemonic: "REGION_STATUS",
                    operand_pattern: OperandPattern::RegImm8,
                    description: "Read region status: 0=idle, 1=active, 2=firing.",
                },
                InstructionMeta {
                    opcode: 0x000C,
                    mnemonic: "REGION_ENABLE",
                    operand_pattern: OperandPattern::Imm8,
                    description: "Enable region for routing. Yields RegionEnable.",
                },
                InstructionMeta {
                    opcode: 0x000D,
                    mnemonic: "REGION_DISABLE",
                    operand_pattern: OperandPattern::Imm8,
                    description: "Disable region. Yields RegionDisable.",
                },
            ],
            model_table: Mutex::new(vec![ModelSlot::default(); 256]),
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
        operands: [u8; 4],
        ctx: &mut ExecutionContext,
    ) -> StepResult {
        match opcode {
            // =================================================================
            // Model table ops
            // =================================================================

            // MODEL_LOAD: [reg:1][slot:1][_:2] — yields to host for actual loading
            0x0000 => {
                let reg = Register(operands[0]);
                let slot = operands[1];
                // Mark slot as loaded in our table
                {
                    let mut table = self.model_table.lock().unwrap();
                    table[slot as usize].loaded = true;
                }
                StepResult::Yield(DomainOp::ModelLoad { target: reg, slot })
            }

            // MODEL_EXEC: [reg:1][slot:1][_:2] — yields to host for actual execution
            0x0001 => {
                let reg = Register(operands[0]);
                let slot = operands[1];
                {
                    let table = self.model_table.lock().unwrap();
                    if !table[slot as usize].loaded {
                        return StepResult::Error(format!(
                            "MODEL_EXEC: slot {} not loaded", slot
                        ));
                    }
                }
                StepResult::Yield(DomainOp::ModelExec { target: reg, slot })
            }

            // MODEL_INPUT: [reg:1][slot:1][_:2] — bind input register to slot
            0x0002 => {
                let reg_idx = Register(operands[0]).index();
                let slot = operands[1] as usize;
                let mut table = self.model_table.lock().unwrap();
                table[slot].input_reg = Some(reg_idx);
                StepResult::Continue
            }

            // MODEL_OUTPUT: [reg:1][slot:1][_:2] — bind output register to slot
            0x0003 => {
                let reg_idx = Register(operands[0]).index();
                let slot = operands[1] as usize;
                let mut table = self.model_table.lock().unwrap();
                table[slot].output_reg = Some(reg_idx);
                StepResult::Continue
            }

            // MODEL_UNLOAD: [slot:1][_:3] — clear slot
            0x0004 => {
                let slot = operands[0] as usize;
                let mut table = self.model_table.lock().unwrap();
                table[slot] = ModelSlot::default();
                StepResult::Continue
            }

            // MODEL_STATUS: [reg:1][slot:1][_:2] — write status to H[reg][0]
            0x0005 => {
                let reg_idx = Register(operands[0]).index();
                let slot = operands[1] as usize;
                let table = self.model_table.lock().unwrap();
                let status = if table[slot].running {
                    2 // running
                } else if table[slot].loaded {
                    1 // loaded
                } else {
                    0 // empty
                };
                ctx.hot_regs[reg_idx] = Some(HotBuffer {
                    data: vec![status],
                    shape: vec![1],
                });
                StepResult::Continue
            }

            // MODEL_RELOAD: [slot:1][_:3] — yields to host for hot-reload
            0x0006 => {
                let slot = operands[0];
                StepResult::Yield(DomainOp::ModelReload { slot })
            }

            // MODEL_CHAIN: [slot1:1][slot2:1][_:2] — bind slot1 output to slot2 input
            0x0007 => {
                let slot1 = operands[0] as usize;
                let slot2 = operands[1] as usize;
                let mut table = self.model_table.lock().unwrap();
                // Chain: slot1's output register becomes slot2's input register
                if let Some(out_reg) = table[slot1].output_reg {
                    table[slot2].input_reg = Some(out_reg);
                } else {
                    return StepResult::Error(format!(
                        "MODEL_CHAIN: slot {} has no output register bound", slot1
                    ));
                }
                StepResult::Continue
            }

            // =================================================================
            // Region routing ops — all yield to host
            // =================================================================

            // ROUTE_INPUT: [reg:1][region_id:1][_:2]
            0x0008 => {
                let reg = Register(operands[0]);
                let region_id = operands[1];
                StepResult::Yield(DomainOp::RouteInput { source: reg, region_id })
            }

            // REGION_FIRE: [reg:1][region_id:1][_:2]
            0x0009 => {
                let reg = Register(operands[0]);
                let region_id = operands[1];
                StepResult::Yield(DomainOp::RegionFire { target: reg, region_id })
            }

            // COLLECT_OUTPUTS: [reg:1][_:3]
            0x000A => {
                let reg = Register(operands[0]);
                StepResult::Yield(DomainOp::CollectOutputs { target: reg })
            }

            // REGION_STATUS: [reg:1][region_id:1][_:2] — yields to host
            // The host writes status into H[reg][0]: 0=idle, 1=active, 2=firing
            0x000B => {
                let reg = Register(operands[0]);
                let region_id = operands[1];
                // Region status requires host knowledge — yield
                StepResult::Yield(DomainOp::RegionFire { target: reg, region_id })
            }

            // REGION_ENABLE: [region_id:1][_:3]
            0x000C => {
                let region_id = operands[0];
                StepResult::Yield(DomainOp::RegionEnable { region_id })
            }

            // REGION_DISABLE: [region_id:1][_:3]
            0x000D => {
                let region_id = operands[0];
                StepResult::Yield(DomainOp::RegionDisable { region_id })
            }

            _ => StepResult::Error(format!(
                "tvmr.orchestration: unknown opcode 0x{:04X}", opcode
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vm::interpreter::ChemicalState;
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
        }
    }

    #[test]
    fn test_metadata() {
        let ext = OrchestrationExtension::new();
        assert_eq!(ext.ext_id(), 0x0007);
        assert_eq!(ext.name(), "tvmr.orchestration");
        assert_eq!(ext.instructions().len(), 14);
    }

    #[test]
    fn test_model_load_yields() {
        let ext = OrchestrationExtension::new();
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

        let result = ext.execute(0x0000, [0x00, 0x05, 0x00, 0x00], &mut ctx);
        match result {
            StepResult::Yield(DomainOp::ModelLoad { target, slot }) => {
                assert_eq!(target, Register(0x00));
                assert_eq!(slot, 5);
            }
            other => panic!("Expected Yield(ModelLoad), got {:?}", other),
        }

        // Verify slot marked as loaded
        let table = ext.model_table.lock().unwrap();
        assert!(table[5].loaded);
    }

    #[test]
    fn test_model_exec_requires_loaded() {
        let ext = OrchestrationExtension::new();
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

        // Exec on unloaded slot should error
        let result = ext.execute(0x0001, [0x00, 0x03, 0x00, 0x00], &mut ctx);
        assert!(matches!(result, StepResult::Error(_)));

        // Load first, then exec should yield
        let _ = ext.execute(0x0000, [0x00, 0x03, 0x00, 0x00], &mut ctx);
        let result = ext.execute(0x0001, [0x00, 0x03, 0x00, 0x00], &mut ctx);
        assert!(matches!(result, StepResult::Yield(DomainOp::ModelExec { .. })));
    }

    #[test]
    fn test_model_input_output_binding() {
        let ext = OrchestrationExtension::new();
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

        // MODEL_INPUT H2, slot=0
        let result = ext.execute(0x0002, [0x02, 0x00, 0x00, 0x00], &mut ctx);
        assert!(matches!(result, StepResult::Continue));

        // MODEL_OUTPUT H3, slot=0
        let result = ext.execute(0x0003, [0x03, 0x00, 0x00, 0x00], &mut ctx);
        assert!(matches!(result, StepResult::Continue));

        let table = ext.model_table.lock().unwrap();
        assert_eq!(table[0].input_reg, Some(2));
        assert_eq!(table[0].output_reg, Some(3));
    }

    #[test]
    fn test_model_unload() {
        let ext = OrchestrationExtension::new();
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

        // Load slot 1
        let _ = ext.execute(0x0000, [0x00, 0x01, 0x00, 0x00], &mut ctx);
        assert!(ext.model_table.lock().unwrap()[1].loaded);

        // Unload slot 1
        let result = ext.execute(0x0004, [0x01, 0x00, 0x00, 0x00], &mut ctx);
        assert!(matches!(result, StepResult::Continue));
        assert!(!ext.model_table.lock().unwrap()[1].loaded);
    }

    #[test]
    fn test_model_status() {
        let ext = OrchestrationExtension::new();
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

        // Status of empty slot → 0
        let result = ext.execute(0x0005, [0x00, 0x00, 0x00, 0x00], &mut ctx);
        assert!(matches!(result, StepResult::Continue));
        assert_eq!(ctx.hot_regs[0].as_ref().unwrap().data[0], 0);

        // Load slot 0 → status becomes 1
        let _ = ext.execute(0x0000, [0x00, 0x00, 0x00, 0x00], &mut ctx);
        let result = ext.execute(0x0005, [0x00, 0x00, 0x00, 0x00], &mut ctx);
        assert!(matches!(result, StepResult::Continue));
        assert_eq!(ctx.hot_regs[0].as_ref().unwrap().data[0], 1);
    }

    #[test]
    fn test_model_chain() {
        let ext = OrchestrationExtension::new();
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

        // Set slot 0 output to H5
        let _ = ext.execute(0x0003, [0x05, 0x00, 0x00, 0x00], &mut ctx);

        // Chain slot 0 → slot 1
        let result = ext.execute(0x0007, [0x00, 0x01, 0x00, 0x00], &mut ctx);
        assert!(matches!(result, StepResult::Continue));

        // Verify slot 1 input is now H5 (slot 0's output)
        let table = ext.model_table.lock().unwrap();
        assert_eq!(table[1].input_reg, Some(5));
    }

    #[test]
    fn test_route_input_yields() {
        let ext = OrchestrationExtension::new();
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

        let result = ext.execute(0x0008, [0x00, 0x03, 0x00, 0x00], &mut ctx);
        match result {
            StepResult::Yield(DomainOp::RouteInput { source, region_id }) => {
                assert_eq!(source, Register(0x00));
                assert_eq!(region_id, 3);
            }
            other => panic!("Expected Yield(RouteInput), got {:?}", other),
        }
    }

    #[test]
    fn test_region_enable_disable_yield() {
        let ext = OrchestrationExtension::new();
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

        let result = ext.execute(0x000C, [0x07, 0x00, 0x00, 0x00], &mut ctx);
        match result {
            StepResult::Yield(DomainOp::RegionEnable { region_id }) => {
                assert_eq!(region_id, 7);
            }
            other => panic!("Expected Yield(RegionEnable), got {:?}", other),
        }

        let result = ext.execute(0x000D, [0x07, 0x00, 0x00, 0x00], &mut ctx);
        match result {
            StepResult::Yield(DomainOp::RegionDisable { region_id }) => {
                assert_eq!(region_id, 7);
            }
            other => panic!("Expected Yield(RegionDisable), got {:?}", other),
        }
    }
}
