//! Test Extension (ExtID: 0x000A)
//!
//! Testing and assertion operations for program validation.
//! Used during development and automated testing.
//!
//! Assertions halt execution with Error on failure, providing
//! detailed diagnostics (register contents, expected vs actual).

use crate::vm::extension::{
    ExecutionContext, Extension, InstructionMeta, OperandPattern, StepResult,
};
use crate::vm::interpreter::HotBuffer;
use crate::vm::register::Register;
use std::sync::Mutex;

/// Test extension — assertion and validation operations.
pub struct TestExtension {
    instructions: Vec<InstructionMeta>,
    /// Snapshot storage: indexed by snapshot slot (operand B byte).
    /// Populated by SNAPSHOT, read by assertions that compare against snapshots.
    snapshots: Mutex<Vec<Option<Vec<i32>>>>,
    /// Current test case ID (set by TEST_BEGIN, cleared by TEST_END).
    current_test: Mutex<Option<u8>>,
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
            snapshots: Mutex::new(vec![None; 64]),
            current_test: Mutex::new(None),
        }
    }

    fn test_prefix(&self) -> String {
        match *self.current_test.lock().unwrap() {
            Some(id) => format!("[test {}] ", id),
            None => String::new(),
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
        operands: [u8; 4],
        ctx: &mut ExecutionContext,
    ) -> StepResult {
        match opcode {
            0x0000 => execute_assert_eq(operands, ctx, &self.test_prefix()),
            0x0001 => execute_assert_gt(operands, ctx, &self.test_prefix()),
            0x0002 => execute_assert_active(operands, ctx, &self.test_prefix()),
            0x0003 => execute_assert_range(operands, ctx, &self.test_prefix()),
            0x0004 => {
                // TEST_BEGIN: operands[0] = test case ID
                let test_id = operands[0];
                *self.current_test.lock().unwrap() = Some(test_id);
                StepResult::Continue
            }
            0x0005 => {
                // TEST_END: clear test context
                *self.current_test.lock().unwrap() = None;
                StepResult::Continue
            }
            0x0006 => execute_expect_chem(operands, ctx, &self.test_prefix()),
            0x0007 => execute_snapshot(operands, ctx, &self.snapshots),
            _ => StepResult::Error(format!("tvmr.test: unknown opcode 0x{:04X}", opcode)),
        }
    }
}

// =============================================================================
// Instruction implementations
// =============================================================================

/// ASSERT_EQ: Assert two hot registers have identical data.
/// Operands: [dst:1][src:1][_:2] — compares dst and src element-wise.
fn execute_assert_eq(ops: [u8; 4], ctx: &mut ExecutionContext, prefix: &str) -> StepResult {
    let a_idx = Register(ops[0]).index();
    let b_idx = Register(ops[1]).index();

    let a = match &ctx.hot_regs[a_idx] {
        Some(buf) => buf,
        None => return StepResult::Error(format!("{}ASSERT_EQ: H{} not allocated", prefix, a_idx)),
    };
    let b = match &ctx.hot_regs[b_idx] {
        Some(buf) => buf,
        None => return StepResult::Error(format!("{}ASSERT_EQ: H{} not allocated", prefix, b_idx)),
    };

    if a.data.len() != b.data.len() {
        return StepResult::Error(format!(
            "{}ASSERT_EQ failed: H{} has {} elements, H{} has {} elements",
            prefix, a_idx, a.data.len(), b_idx, b.data.len()
        ));
    }

    for (i, (va, vb)) in a.data.iter().zip(b.data.iter()).enumerate() {
        if va != vb {
            return StepResult::Error(format!(
                "{}ASSERT_EQ failed at index {}: H{}[{}]={} != H{}[{}]={}",
                prefix, i, a_idx, i, va, b_idx, i, vb
            ));
        }
    }

    StepResult::Continue
}

/// ASSERT_GT: Assert first register's sum > second register's sum.
/// Element-wise comparison would be overly strict; sum comparison tests
/// that register A is "more active" than register B.
fn execute_assert_gt(ops: [u8; 4], ctx: &mut ExecutionContext, prefix: &str) -> StepResult {
    let a_idx = Register(ops[0]).index();
    let b_idx = Register(ops[1]).index();

    let a = match &ctx.hot_regs[a_idx] {
        Some(buf) => buf,
        None => return StepResult::Error(format!("{}ASSERT_GT: H{} not allocated", prefix, a_idx)),
    };
    let b = match &ctx.hot_regs[b_idx] {
        Some(buf) => buf,
        None => return StepResult::Error(format!("{}ASSERT_GT: H{} not allocated", prefix, b_idx)),
    };

    let sum_a: i64 = a.data.iter().map(|&v| v as i64).sum();
    let sum_b: i64 = b.data.iter().map(|&v| v as i64).sum();

    if sum_a <= sum_b {
        return StepResult::Error(format!(
            "{}ASSERT_GT failed: sum(H{})={} <= sum(H{})={}",
            prefix, a_idx, sum_a, b_idx, sum_b
        ));
    }

    StepResult::Continue
}

/// ASSERT_ACTIVE: Assert register has at least one non-zero value.
/// Operands: [reg:1][_:3]
fn execute_assert_active(ops: [u8; 4], ctx: &mut ExecutionContext, prefix: &str) -> StepResult {
    let reg_idx = Register(ops[0]).index();

    let buf = match &ctx.hot_regs[reg_idx] {
        Some(buf) => buf,
        None => return StepResult::Error(format!("{}ASSERT_ACTIVE: H{} not allocated", prefix, reg_idx)),
    };

    if buf.data.iter().all(|&v| v == 0) {
        return StepResult::Error(format!(
            "{}ASSERT_ACTIVE failed: H{} is entirely zero ({} elements)",
            prefix, reg_idx, buf.data.len()
        ));
    }

    StepResult::Continue
}

/// ASSERT_RANGE: Assert all values in register fall within [min, max].
/// Operands: [reg:1][min:1][max:1][_:1]
/// min and max are unsigned bytes (0-255), interpreted as i32 range.
/// For signed ranges, use min=0 max=255 (the full byte range).
fn execute_assert_range(ops: [u8; 4], ctx: &mut ExecutionContext, prefix: &str) -> StepResult {
    let reg_idx = Register(ops[0]).index();
    let min_val = ops[1] as i32;
    let max_val = ops[2] as i32;

    let buf = match &ctx.hot_regs[reg_idx] {
        Some(buf) => buf,
        None => return StepResult::Error(format!("{}ASSERT_RANGE: H{} not allocated", prefix, reg_idx)),
    };

    for (i, &v) in buf.data.iter().enumerate() {
        if v < min_val || v > max_val {
            return StepResult::Error(format!(
                "{}ASSERT_RANGE failed: H{}[{}]={} not in [{}, {}]",
                prefix, reg_idx, i, v, min_val, max_val
            ));
        }
    }

    StepResult::Continue
}

/// EXPECT_CHEM: Assert a chemical level is within [min, max].
/// Operands: [chemical:1][min:1][max:1][_:1]
/// chemical: 0=dopamine, 1=serotonin, 2=norepinephrine, 3=gaba
fn execute_expect_chem(ops: [u8; 4], ctx: &mut ExecutionContext, prefix: &str) -> StepResult {
    let chem_id = ops[0];
    let min_val = ops[1];
    let max_val = ops[2];

    let (name, value) = match chem_id {
        0 => ("dopamine", ctx.chemical_state.dopamine),
        1 => ("serotonin", ctx.chemical_state.serotonin),
        2 => ("norepinephrine", ctx.chemical_state.norepinephrine),
        3 => ("gaba", ctx.chemical_state.gaba),
        _ => return StepResult::Error(format!("{}EXPECT_CHEM: unknown chemical {}", prefix, chem_id)),
    };

    if value < min_val || value > max_val {
        return StepResult::Error(format!(
            "{}EXPECT_CHEM failed: {}={} not in [{}, {}]",
            prefix, name, value, min_val, max_val
        ));
    }

    StepResult::Continue
}

/// SNAPSHOT: Save a copy of a hot register's data for later comparison.
/// Operands: [reg:1][_:3]
/// The register index is used as the snapshot slot.
fn execute_snapshot(
    ops: [u8; 4],
    ctx: &mut ExecutionContext,
    snapshots: &Mutex<Vec<Option<Vec<i32>>>>,
) -> StepResult {
    let reg_idx = Register(ops[0]).index();

    let buf = match &ctx.hot_regs[reg_idx] {
        Some(buf) => buf,
        None => return StepResult::Error(format!("SNAPSHOT: H{} not allocated", reg_idx)),
    };

    let mut snaps = snapshots.lock().unwrap();
    if reg_idx >= snaps.len() {
        snaps.resize(reg_idx + 1, None);
    }
    snaps[reg_idx] = Some(buf.data.clone());

    StepResult::Continue
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vm::interpreter::ChemicalState;
    use crate::vm::extension::LoopState;

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

    macro_rules! setup_ctx {
        () => {{
            let mut hot_regs: Vec<Option<HotBuffer>> = vec![None; 64];
            let mut cold_regs = vec![None; 64];
            let mut param_regs = vec![0i32; 64];
            let mut shape_regs: Vec<Vec<usize>> = vec![Vec::new(); 64];
            let mut pc = 0usize;
            let mut call_stack = Vec::new();
            let mut loop_stack = Vec::new();
            let input_buffer = vec![];
            let mut output_buffer = Vec::new();
            let target_buffer = vec![];
            let mut chemical_state = ChemicalState::baseline();
            let mut current_error = 0i32;
            let mut babble_scale = 0i32;
            let mut babble_phase = 0usize;
            let mut pressure_regs: Vec<Option<Vec<i32>>> = vec![None; 16];
            (
                hot_regs, cold_regs, param_regs, shape_regs,
                pc, call_stack, loop_stack, input_buffer,
                output_buffer, target_buffer, chemical_state,
                current_error, babble_scale, babble_phase, pressure_regs,
            )
        }};
    }

    #[test]
    fn test_metadata() {
        let ext = TestExtension::new();
        assert_eq!(ext.ext_id(), 0x000A);
        assert_eq!(ext.name(), "tvmr.test");
        assert_eq!(ext.instructions().len(), 8);
    }

    #[test]
    fn test_assert_eq_pass() {
        let ext = TestExtension::new();
        let (
            mut hot_regs, mut cold_regs, mut param_regs, mut shape_regs,
            mut pc, mut call_stack, mut loop_stack, input_buffer,
            mut output_buffer, target_buffer, mut chemical_state,
            mut current_error, mut babble_scale, mut babble_phase, mut pressure_regs,
        ) = setup_ctx!();

        hot_regs[0] = Some(HotBuffer { data: vec![10, 20, 30], shape: vec![3] });
        hot_regs[1] = Some(HotBuffer { data: vec![10, 20, 30], shape: vec![3] });

        let mut ctx = make_ctx(
            &mut hot_regs, &mut cold_regs, &mut param_regs, &mut shape_regs,
            &mut pc, &mut call_stack, &mut loop_stack, &input_buffer,
            &mut output_buffer, &target_buffer, &mut chemical_state,
            &mut current_error, &mut babble_scale, &mut babble_phase, &mut pressure_regs,
        );

        let result = ext.execute(0x0000, [0x00, 0x01, 0x00, 0x00], &mut ctx);
        assert!(matches!(result, StepResult::Continue));
    }

    #[test]
    fn test_assert_eq_fail() {
        let ext = TestExtension::new();
        let (
            mut hot_regs, mut cold_regs, mut param_regs, mut shape_regs,
            mut pc, mut call_stack, mut loop_stack, input_buffer,
            mut output_buffer, target_buffer, mut chemical_state,
            mut current_error, mut babble_scale, mut babble_phase, mut pressure_regs,
        ) = setup_ctx!();

        hot_regs[0] = Some(HotBuffer { data: vec![10, 20, 30], shape: vec![3] });
        hot_regs[1] = Some(HotBuffer { data: vec![10, 99, 30], shape: vec![3] });

        let mut ctx = make_ctx(
            &mut hot_regs, &mut cold_regs, &mut param_regs, &mut shape_regs,
            &mut pc, &mut call_stack, &mut loop_stack, &input_buffer,
            &mut output_buffer, &target_buffer, &mut chemical_state,
            &mut current_error, &mut babble_scale, &mut babble_phase, &mut pressure_regs,
        );

        let result = ext.execute(0x0000, [0x00, 0x01, 0x00, 0x00], &mut ctx);
        assert!(matches!(result, StepResult::Error(_)));
    }

    #[test]
    fn test_assert_gt_pass() {
        let ext = TestExtension::new();
        let (
            mut hot_regs, mut cold_regs, mut param_regs, mut shape_regs,
            mut pc, mut call_stack, mut loop_stack, input_buffer,
            mut output_buffer, target_buffer, mut chemical_state,
            mut current_error, mut babble_scale, mut babble_phase, mut pressure_regs,
        ) = setup_ctx!();

        hot_regs[0] = Some(HotBuffer { data: vec![100, 200, 300], shape: vec![3] });
        hot_regs[1] = Some(HotBuffer { data: vec![10, 20, 30], shape: vec![3] });

        let mut ctx = make_ctx(
            &mut hot_regs, &mut cold_regs, &mut param_regs, &mut shape_regs,
            &mut pc, &mut call_stack, &mut loop_stack, &input_buffer,
            &mut output_buffer, &target_buffer, &mut chemical_state,
            &mut current_error, &mut babble_scale, &mut babble_phase, &mut pressure_regs,
        );

        let result = ext.execute(0x0001, [0x00, 0x01, 0x00, 0x00], &mut ctx);
        assert!(matches!(result, StepResult::Continue));
    }

    #[test]
    fn test_assert_active_pass() {
        let ext = TestExtension::new();
        let (
            mut hot_regs, mut cold_regs, mut param_regs, mut shape_regs,
            mut pc, mut call_stack, mut loop_stack, input_buffer,
            mut output_buffer, target_buffer, mut chemical_state,
            mut current_error, mut babble_scale, mut babble_phase, mut pressure_regs,
        ) = setup_ctx!();

        hot_regs[0] = Some(HotBuffer { data: vec![0, 0, 42, 0], shape: vec![4] });

        let mut ctx = make_ctx(
            &mut hot_regs, &mut cold_regs, &mut param_regs, &mut shape_regs,
            &mut pc, &mut call_stack, &mut loop_stack, &input_buffer,
            &mut output_buffer, &target_buffer, &mut chemical_state,
            &mut current_error, &mut babble_scale, &mut babble_phase, &mut pressure_regs,
        );

        let result = ext.execute(0x0002, [0x00, 0x00, 0x00, 0x00], &mut ctx);
        assert!(matches!(result, StepResult::Continue));
    }

    #[test]
    fn test_assert_active_fail_all_zero() {
        let ext = TestExtension::new();
        let (
            mut hot_regs, mut cold_regs, mut param_regs, mut shape_regs,
            mut pc, mut call_stack, mut loop_stack, input_buffer,
            mut output_buffer, target_buffer, mut chemical_state,
            mut current_error, mut babble_scale, mut babble_phase, mut pressure_regs,
        ) = setup_ctx!();

        hot_regs[0] = Some(HotBuffer { data: vec![0, 0, 0, 0], shape: vec![4] });

        let mut ctx = make_ctx(
            &mut hot_regs, &mut cold_regs, &mut param_regs, &mut shape_regs,
            &mut pc, &mut call_stack, &mut loop_stack, &input_buffer,
            &mut output_buffer, &target_buffer, &mut chemical_state,
            &mut current_error, &mut babble_scale, &mut babble_phase, &mut pressure_regs,
        );

        let result = ext.execute(0x0002, [0x00, 0x00, 0x00, 0x00], &mut ctx);
        assert!(matches!(result, StepResult::Error(_)));
    }

    #[test]
    fn test_assert_range_pass() {
        let ext = TestExtension::new();
        let (
            mut hot_regs, mut cold_regs, mut param_regs, mut shape_regs,
            mut pc, mut call_stack, mut loop_stack, input_buffer,
            mut output_buffer, target_buffer, mut chemical_state,
            mut current_error, mut babble_scale, mut babble_phase, mut pressure_regs,
        ) = setup_ctx!();

        hot_regs[0] = Some(HotBuffer { data: vec![10, 50, 100, 200], shape: vec![4] });

        let mut ctx = make_ctx(
            &mut hot_regs, &mut cold_regs, &mut param_regs, &mut shape_regs,
            &mut pc, &mut call_stack, &mut loop_stack, &input_buffer,
            &mut output_buffer, &target_buffer, &mut chemical_state,
            &mut current_error, &mut babble_scale, &mut babble_phase, &mut pressure_regs,
        );

        // ASSERT_RANGE H0, min=0, max=255
        let result = ext.execute(0x0003, [0x00, 0x00, 0xFF, 0x00], &mut ctx);
        assert!(matches!(result, StepResult::Continue));
    }

    #[test]
    fn test_assert_range_fail() {
        let ext = TestExtension::new();
        let (
            mut hot_regs, mut cold_regs, mut param_regs, mut shape_regs,
            mut pc, mut call_stack, mut loop_stack, input_buffer,
            mut output_buffer, target_buffer, mut chemical_state,
            mut current_error, mut babble_scale, mut babble_phase, mut pressure_regs,
        ) = setup_ctx!();

        hot_regs[0] = Some(HotBuffer { data: vec![10, 50, 100, 200], shape: vec![4] });

        let mut ctx = make_ctx(
            &mut hot_regs, &mut cold_regs, &mut param_regs, &mut shape_regs,
            &mut pc, &mut call_stack, &mut loop_stack, &input_buffer,
            &mut output_buffer, &target_buffer, &mut chemical_state,
            &mut current_error, &mut babble_scale, &mut babble_phase, &mut pressure_regs,
        );

        // ASSERT_RANGE H0, min=0, max=100 — 200 will fail
        let result = ext.execute(0x0003, [0x00, 0x00, 0x64, 0x00], &mut ctx);
        assert!(matches!(result, StepResult::Error(_)));
    }

    #[test]
    fn test_expect_chem_pass() {
        let ext = TestExtension::new();
        let (
            mut hot_regs, mut cold_regs, mut param_regs, mut shape_regs,
            mut pc, mut call_stack, mut loop_stack, input_buffer,
            mut output_buffer, target_buffer, mut chemical_state,
            mut current_error, mut babble_scale, mut babble_phase, mut pressure_regs,
        ) = setup_ctx!();

        // Baseline dopamine = 128
        let mut ctx = make_ctx(
            &mut hot_regs, &mut cold_regs, &mut param_regs, &mut shape_regs,
            &mut pc, &mut call_stack, &mut loop_stack, &input_buffer,
            &mut output_buffer, &target_buffer, &mut chemical_state,
            &mut current_error, &mut babble_scale, &mut babble_phase, &mut pressure_regs,
        );

        // EXPECT_CHEM dopamine(0), min=100, max=200
        let result = ext.execute(0x0006, [0x00, 100, 200, 0x00], &mut ctx);
        assert!(matches!(result, StepResult::Continue));
    }

    #[test]
    fn test_expect_chem_fail() {
        let ext = TestExtension::new();
        let (
            mut hot_regs, mut cold_regs, mut param_regs, mut shape_regs,
            mut pc, mut call_stack, mut loop_stack, input_buffer,
            mut output_buffer, target_buffer, mut chemical_state,
            mut current_error, mut babble_scale, mut babble_phase, mut pressure_regs,
        ) = setup_ctx!();

        // Baseline dopamine = 128
        let mut ctx = make_ctx(
            &mut hot_regs, &mut cold_regs, &mut param_regs, &mut shape_regs,
            &mut pc, &mut call_stack, &mut loop_stack, &input_buffer,
            &mut output_buffer, &target_buffer, &mut chemical_state,
            &mut current_error, &mut babble_scale, &mut babble_phase, &mut pressure_regs,
        );

        // EXPECT_CHEM dopamine(0), min=200, max=255 — 128 is below
        let result = ext.execute(0x0006, [0x00, 200, 255, 0x00], &mut ctx);
        assert!(matches!(result, StepResult::Error(_)));
    }

    #[test]
    fn test_snapshot() {
        let ext = TestExtension::new();
        let (
            mut hot_regs, mut cold_regs, mut param_regs, mut shape_regs,
            mut pc, mut call_stack, mut loop_stack, input_buffer,
            mut output_buffer, target_buffer, mut chemical_state,
            mut current_error, mut babble_scale, mut babble_phase, mut pressure_regs,
        ) = setup_ctx!();

        hot_regs[0] = Some(HotBuffer { data: vec![42, 84, 126], shape: vec![3] });

        let mut ctx = make_ctx(
            &mut hot_regs, &mut cold_regs, &mut param_regs, &mut shape_regs,
            &mut pc, &mut call_stack, &mut loop_stack, &input_buffer,
            &mut output_buffer, &target_buffer, &mut chemical_state,
            &mut current_error, &mut babble_scale, &mut babble_phase, &mut pressure_regs,
        );

        let result = ext.execute(0x0007, [0x00, 0x00, 0x00, 0x00], &mut ctx);
        assert!(matches!(result, StepResult::Continue));

        // Verify snapshot was stored
        let snaps = ext.snapshots.lock().unwrap();
        assert_eq!(snaps[0], Some(vec![42, 84, 126]));
    }

    #[test]
    fn test_begin_end() {
        let ext = TestExtension::new();
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

        // TEST_BEGIN 5
        let result = ext.execute(0x0004, [0x05, 0x00, 0x00, 0x00], &mut ctx);
        assert!(matches!(result, StepResult::Continue));
        assert_eq!(*ext.current_test.lock().unwrap(), Some(5));

        // TEST_END
        let result = ext.execute(0x0005, [0x00, 0x00, 0x00, 0x00], &mut ctx);
        assert!(matches!(result, StepResult::Continue));
        assert_eq!(*ext.current_test.lock().unwrap(), None);
    }
}
