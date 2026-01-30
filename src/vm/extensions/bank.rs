//! Bank Extension (ExtID: 0x000B)
//!
//! Distributed representational memory operations. Simple ops execute inline
//! when `ctx.bank_cache` is available; complex ops (LINK, TRAVERSE) always yield.
//! The host maps per-interpreter bank_slot (u8) to global BankId via BankSlotMap.
//!
//! ## Opcodes
//!
//! | Opcode | Mnemonic        | Description                     | Inline? |
//! |--------|-----------------|---------------------------------|---------|
//! | 0x0000 | BANK_QUERY      | Query bank by vector similarity | Yes     |
//! | 0x0001 | BANK_WRITE      | Write entry to bank             | Yes     |
//! | 0x0002 | BANK_LOAD       | Load entry vector by id         | Yes     |
//! | 0x0003 | BANK_LINK       | Add typed edge between entries   | No      |
//! | 0x0004 | BANK_TRAVERSE   | Traverse edges from entry        | No      |
//! | 0x0005 | BANK_TOUCH      | Touch entry (update access)      | Yes     |
//! | 0x0006 | BANK_DELETE      | Delete entry from bank           | Yes     |
//! | 0x0007 | BANK_COUNT      | Get entry count for bank         | Yes     |
//! | 0x0008 | BANK_PROMOTE    | Promote entry temperature        | No      |
//! | 0x0009 | BANK_DEMOTE     | Demote entry temperature         | No      |
//! | 0x000A | BANK_EVICT      | Evict cold entries               | No      |
//! | 0x000B | BANK_COMPACT    | Compact bank after pruning       | No      |

use crate::vm::extension::{
    DomainOp, ExecutionContext, Extension, InstructionMeta, OperandPattern, StepResult,
};
use crate::vm::interpreter::HotBuffer;
use crate::vm::register::Register;

/// Bank extension — distributed representational memory operations.
pub struct BankExtension {
    instructions: Vec<InstructionMeta>,
}

impl BankExtension {
    pub fn new() -> Self {
        Self {
            instructions: vec![
                InstructionMeta {
                    opcode: 0x0000,
                    mnemonic: "BANK_QUERY",
                    operand_pattern: OperandPattern::Custom("[target:1][source:1][bank_slot:1][top_k:1]"),
                    description: "Query bank by vector similarity. Yields BankQuery.",
                },
                InstructionMeta {
                    opcode: 0x0001,
                    mnemonic: "BANK_WRITE",
                    operand_pattern: OperandPattern::Custom("[target:1][source:1][bank_slot:1][_:1]"),
                    description: "Write entry to bank. Yields BankWrite.",
                },
                InstructionMeta {
                    opcode: 0x0002,
                    mnemonic: "BANK_LOAD",
                    operand_pattern: OperandPattern::Custom("[target:1][source:1][bank_slot:1][_:1]"),
                    description: "Load full entry vector by id. Yields BankLoad.",
                },
                InstructionMeta {
                    opcode: 0x0003,
                    mnemonic: "BANK_LINK",
                    operand_pattern: OperandPattern::Custom("[source:1][edge_type:1][bank_slot:1][_:1]"),
                    description: "Add typed edge between entries. Yields BankLink.",
                },
                InstructionMeta {
                    opcode: 0x0004,
                    mnemonic: "BANK_TRAVERSE",
                    operand_pattern: OperandPattern::Custom("[target:1][source:1][bank_slot:1][packed:1]"),
                    description: "Traverse edges from entry. packed=[edge_type:4][depth:4]. Yields BankTraverse.",
                },
                InstructionMeta {
                    opcode: 0x0005,
                    mnemonic: "BANK_TOUCH",
                    operand_pattern: OperandPattern::Custom("[source:1][bank_slot:1][_:2]"),
                    description: "Touch entry to update access tick/count. Yields BankTouch.",
                },
                InstructionMeta {
                    opcode: 0x0006,
                    mnemonic: "BANK_DELETE",
                    operand_pattern: OperandPattern::Custom("[source:1][bank_slot:1][_:2]"),
                    description: "Delete entry from bank. Yields BankDelete.",
                },
                InstructionMeta {
                    opcode: 0x0007,
                    mnemonic: "BANK_COUNT",
                    operand_pattern: OperandPattern::Custom("[target:1][bank_slot:1][_:2]"),
                    description: "Get entry count for bank. Yields BankCount.",
                },
                InstructionMeta {
                    opcode: 0x0008,
                    mnemonic: "BANK_PROMOTE",
                    operand_pattern: OperandPattern::Custom("[source:1][bank_slot:1][_:2]"),
                    description: "Promote entry temperature (Hot→Warm, etc). Yields BankPromote.",
                },
                InstructionMeta {
                    opcode: 0x0009,
                    mnemonic: "BANK_DEMOTE",
                    operand_pattern: OperandPattern::Custom("[source:1][bank_slot:1][_:2]"),
                    description: "Demote entry temperature (Warm→Hot, etc). Yields BankDemote.",
                },
                InstructionMeta {
                    opcode: 0x000A,
                    mnemonic: "BANK_EVICT",
                    operand_pattern: OperandPattern::Custom("[bank_slot:1][count:1][_:2]"),
                    description: "Evict cold/low-scoring entries. Yields BankEvict.",
                },
                InstructionMeta {
                    opcode: 0x000B,
                    mnemonic: "BANK_COMPACT",
                    operand_pattern: OperandPattern::Custom("[bank_slot:1][_:3]"),
                    description: "Compact bank after pruning. Yields BankCompact.",
                },
            ],
        }
    }
}

impl Extension for BankExtension {
    fn ext_id(&self) -> u16 {
        0x000B
    }

    fn name(&self) -> &str {
        "tvmr.bank"
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
            // BANK_QUERY: [target:1][source:1][bank_slot:1][top_k:1]
            0x0000 => {
                let target = Register(operands[0]);
                let source = Register(operands[1]);
                let bank_slot = operands[2];
                let top_k = operands[3];
                // Inline path: execute locally via bank cache
                if let Some(cache) = ctx.bank_cache.as_mut() {
                    let query_vec = match read_hot_data(ctx.hot_regs, source) {
                        Some(v) => v,
                        None => return StepResult::Error(format!("BANK_QUERY: H{} not allocated", source.index())),
                    };
                    if let Some(results) = cache.query(bank_slot, &query_vec, top_k as usize) {
                        let mut out = vec![results.len() as i32];
                        for (entry_id, score) in &results {
                            out.push(*score);
                            out.push((*entry_id >> 32) as i32);
                            out.push(*entry_id as i32);
                        }
                        write_hot(ctx.hot_regs, target, out);
                        return StepResult::Continue;
                    }
                }
                StepResult::Yield(DomainOp::BankQuery { target, source, bank_slot, top_k })
            }
            // BANK_WRITE: [target:1][source:1][bank_slot:1][_:1]
            0x0001 => {
                let target = Register(operands[0]);
                let source = Register(operands[1]);
                let bank_slot = operands[2];
                if let Some(cache) = ctx.bank_cache.as_mut() {
                    let vector = match read_hot_data(ctx.hot_regs, source) {
                        Some(v) => v,
                        None => return StepResult::Error(format!("BANK_WRITE: H{} not allocated", source.index())),
                    };
                    if let Some((high, low)) = cache.write(bank_slot, &vector) {
                        write_hot(ctx.hot_regs, target, vec![high, low]);
                        return StepResult::Continue;
                    }
                }
                StepResult::Yield(DomainOp::BankWrite { target, source, bank_slot })
            }
            // BANK_LOAD: [target:1][source:1][bank_slot:1][_:1]
            0x0002 => {
                let target = Register(operands[0]);
                let source = Register(operands[1]);
                let bank_slot = operands[2];
                if let Some(cache) = ctx.bank_cache.as_mut() {
                    let id_pair = match read_hot_data(ctx.hot_regs, source) {
                        Some(v) if v.len() >= 2 => (v[0], v[1]),
                        _ => return StepResult::Error(format!("BANK_LOAD: H{} needs [id_high, id_low]", source.index())),
                    };
                    if let Some(vector) = cache.load(bank_slot, id_pair.0, id_pair.1) {
                        write_hot(ctx.hot_regs, target, vector);
                        return StepResult::Continue;
                    }
                }
                StepResult::Yield(DomainOp::BankLoad { target, source, bank_slot })
            }
            // BANK_LINK: [source:1][edge_type:1][bank_slot:1][_:1] — ALWAYS yields
            0x0003 => {
                let source = Register(operands[0]);
                let edge_type = operands[1];
                let bank_slot = operands[2];
                StepResult::Yield(DomainOp::BankLink { source, edge_type, bank_slot })
            }
            // BANK_TRAVERSE: [target:1][source:1][bank_slot:1][packed:1] — ALWAYS yields
            0x0004 => {
                let target = Register(operands[0]);
                let source = Register(operands[1]);
                let bank_slot = operands[2];
                let packed = operands[3];
                let edge_type = packed >> 4;
                let depth = packed & 0x0F;
                StepResult::Yield(DomainOp::BankTraverse { target, source, bank_slot, edge_type, depth })
            }
            // BANK_TOUCH: [source:1][bank_slot:1][_:2]
            0x0005 => {
                let source = Register(operands[0]);
                let bank_slot = operands[1];
                if let Some(cache) = ctx.bank_cache.as_mut() {
                    if let Some(id_pair) = read_hot_data(ctx.hot_regs, source) {
                        if id_pair.len() >= 2 {
                            cache.touch(bank_slot, id_pair[0], id_pair[1]);
                            return StepResult::Continue;
                        }
                    }
                }
                StepResult::Yield(DomainOp::BankTouch { source, bank_slot })
            }
            // BANK_DELETE: [source:1][bank_slot:1][_:2]
            0x0006 => {
                let source = Register(operands[0]);
                let bank_slot = operands[1];
                if let Some(cache) = ctx.bank_cache.as_mut() {
                    if let Some(id_pair) = read_hot_data(ctx.hot_regs, source) {
                        if id_pair.len() >= 2 {
                            cache.delete(bank_slot, id_pair[0], id_pair[1]);
                            return StepResult::Continue;
                        }
                    }
                }
                StepResult::Yield(DomainOp::BankDelete { source, bank_slot })
            }
            // BANK_COUNT: [target:1][bank_slot:1][_:2]
            0x0007 => {
                let target = Register(operands[0]);
                let bank_slot = operands[1];
                if let Some(cache) = ctx.bank_cache.as_mut() {
                    if let Some(count) = cache.count(bank_slot) {
                        write_hot(ctx.hot_regs, target, vec![count]);
                        return StepResult::Continue;
                    }
                }
                StepResult::Yield(DomainOp::BankCount { target, bank_slot })
            }
            // BANK_PROMOTE: [source:1][bank_slot:1][_:2] — ALWAYS yields
            0x0008 => {
                let source = Register(operands[0]);
                let bank_slot = operands[1];
                StepResult::Yield(DomainOp::BankPromote { source, bank_slot })
            }
            // BANK_DEMOTE: [source:1][bank_slot:1][_:2] — ALWAYS yields
            0x0009 => {
                let source = Register(operands[0]);
                let bank_slot = operands[1];
                StepResult::Yield(DomainOp::BankDemote { source, bank_slot })
            }
            // BANK_EVICT: [bank_slot:1][count:1][_:2] — ALWAYS yields
            0x000A => {
                let bank_slot = operands[0];
                let count = operands[1];
                StepResult::Yield(DomainOp::BankEvict { bank_slot, count })
            }
            // BANK_COMPACT: [bank_slot:1][_:3] — ALWAYS yields
            0x000B => {
                let bank_slot = operands[0];
                StepResult::Yield(DomainOp::BankCompact { bank_slot })
            }
            _ => StepResult::Error(format!("tvmr.bank: unknown opcode 0x{:04X}", opcode)),
        }
    }
}

// =============================================================================
// Inline execution helpers
// =============================================================================

/// Read hot register data as a clone. Returns None if not allocated.
fn read_hot_data(hot_regs: &[Option<HotBuffer>], reg: Register) -> Option<Vec<i32>> {
    hot_regs.get(reg.index())?.as_ref().map(|b| b.data.clone())
}

/// Write data into a hot register.
fn write_hot(hot_regs: &mut [Option<HotBuffer>], reg: Register, data: Vec<i32>) {
    let len = data.len();
    if let Some(slot) = hot_regs.get_mut(reg.index()) {
        *slot = Some(HotBuffer {
            data,
            shape: vec![len],
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vm::extension::LoopState;
    use crate::vm::interpreter::{ChemicalState, ColdBuffer, HotBuffer};

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
            bank_cache: None,
        }
    }

    #[test]
    fn test_metadata() {
        let ext = BankExtension::new();
        assert_eq!(ext.ext_id(), 0x000B);
        assert_eq!(ext.name(), "tvmr.bank");
        assert_eq!(ext.instructions().len(), 12);

        let mnemonics: Vec<&str> = ext.instructions().iter().map(|m| m.mnemonic).collect();
        assert!(mnemonics.contains(&"BANK_QUERY"));
        assert!(mnemonics.contains(&"BANK_WRITE"));
        assert!(mnemonics.contains(&"BANK_LOAD"));
        assert!(mnemonics.contains(&"BANK_LINK"));
        assert!(mnemonics.contains(&"BANK_TRAVERSE"));
        assert!(mnemonics.contains(&"BANK_TOUCH"));
        assert!(mnemonics.contains(&"BANK_DELETE"));
        assert!(mnemonics.contains(&"BANK_COUNT"));
        assert!(mnemonics.contains(&"BANK_PROMOTE"));
        assert!(mnemonics.contains(&"BANK_DEMOTE"));
        assert!(mnemonics.contains(&"BANK_EVICT"));
        assert!(mnemonics.contains(&"BANK_COMPACT"));
    }

    #[test]
    fn test_bank_query_yields() {
        let ext = BankExtension::new();
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

        // BANK_QUERY H0, H1, slot=2, top_k=5
        let result = ext.execute(0x0000, [0x00, 0x01, 0x02, 0x05], &mut ctx);
        match result {
            StepResult::Yield(DomainOp::BankQuery { target, source, bank_slot, top_k }) => {
                assert_eq!(target, Register(0x00));
                assert_eq!(source, Register(0x01));
                assert_eq!(bank_slot, 2);
                assert_eq!(top_k, 5);
            }
            other => panic!("Expected Yield(BankQuery), got {:?}", other),
        }
    }

    #[test]
    fn test_bank_write_yields() {
        let ext = BankExtension::new();
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

        // BANK_WRITE H0, H1, slot=0
        let result = ext.execute(0x0001, [0x00, 0x01, 0x00, 0x00], &mut ctx);
        match result {
            StepResult::Yield(DomainOp::BankWrite { target, source, bank_slot }) => {
                assert_eq!(target, Register(0x00));
                assert_eq!(source, Register(0x01));
                assert_eq!(bank_slot, 0);
            }
            other => panic!("Expected Yield(BankWrite), got {:?}", other),
        }
    }

    #[test]
    fn test_bank_load_yields() {
        let ext = BankExtension::new();
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

        // BANK_LOAD H2, H0, slot=1
        let result = ext.execute(0x0002, [0x02, 0x00, 0x01, 0x00], &mut ctx);
        match result {
            StepResult::Yield(DomainOp::BankLoad { target, source, bank_slot }) => {
                assert_eq!(target, Register(0x02));
                assert_eq!(source, Register(0x00));
                assert_eq!(bank_slot, 1);
            }
            other => panic!("Expected Yield(BankLoad), got {:?}", other),
        }
    }

    #[test]
    fn test_bank_link_yields() {
        let ext = BankExtension::new();
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

        // BANK_LINK H0, edge_type=3(RelatedTo), slot=0
        let result = ext.execute(0x0003, [0x00, 0x03, 0x00, 0x00], &mut ctx);
        match result {
            StepResult::Yield(DomainOp::BankLink { source, edge_type, bank_slot }) => {
                assert_eq!(source, Register(0x00));
                assert_eq!(edge_type, 3);
                assert_eq!(bank_slot, 0);
            }
            other => panic!("Expected Yield(BankLink), got {:?}", other),
        }
    }

    #[test]
    fn test_bank_traverse_packed_yields() {
        let ext = BankExtension::new();
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

        // BANK_TRAVERSE H0, H1, slot=0, packed=0x35 (edge_type=3, depth=5)
        let result = ext.execute(0x0004, [0x00, 0x01, 0x00, 0x35], &mut ctx);
        match result {
            StepResult::Yield(DomainOp::BankTraverse { target, source, bank_slot, edge_type, depth }) => {
                assert_eq!(target, Register(0x00));
                assert_eq!(source, Register(0x01));
                assert_eq!(bank_slot, 0);
                assert_eq!(edge_type, 3);
                assert_eq!(depth, 5);
            }
            other => panic!("Expected Yield(BankTraverse), got {:?}", other),
        }
    }

    #[test]
    fn test_bank_touch_yields() {
        let ext = BankExtension::new();
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

        let result = ext.execute(0x0005, [0x00, 0x02, 0x00, 0x00], &mut ctx);
        match result {
            StepResult::Yield(DomainOp::BankTouch { source, bank_slot }) => {
                assert_eq!(source, Register(0x00));
                assert_eq!(bank_slot, 2);
            }
            other => panic!("Expected Yield(BankTouch), got {:?}", other),
        }
    }

    #[test]
    fn test_bank_delete_yields() {
        let ext = BankExtension::new();
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

        let result = ext.execute(0x0006, [0x01, 0x03, 0x00, 0x00], &mut ctx);
        match result {
            StepResult::Yield(DomainOp::BankDelete { source, bank_slot }) => {
                assert_eq!(source, Register(0x01));
                assert_eq!(bank_slot, 3);
            }
            other => panic!("Expected Yield(BankDelete), got {:?}", other),
        }
    }

    #[test]
    fn test_bank_count_yields() {
        let ext = BankExtension::new();
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

        let result = ext.execute(0x0007, [0x00, 0x05, 0x00, 0x00], &mut ctx);
        match result {
            StepResult::Yield(DomainOp::BankCount { target, bank_slot }) => {
                assert_eq!(target, Register(0x00));
                assert_eq!(bank_slot, 5);
            }
            other => panic!("Expected Yield(BankCount), got {:?}", other),
        }
    }

    #[test]
    fn test_all_ops_yield() {
        let ext = BankExtension::new();
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

        for opcode in 0x0000..=0x000Bu16 {
            let result = ext.execute(opcode, [0x00, 0x01, 0x00, 0x00], &mut ctx);
            assert!(
                matches!(result, StepResult::Yield(_)),
                "Opcode 0x{:04X} should yield, got {:?}", opcode, result
            );
        }
    }

    #[test]
    fn test_unknown_opcode_errors() {
        let ext = BankExtension::new();
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

        let result = ext.execute(0x00FF, [0x00, 0x00, 0x00, 0x00], &mut ctx);
        assert!(matches!(result, StepResult::Error(_)));
    }

    // =========================================================================
    // Consolidation opcode tests
    // =========================================================================

    #[test]
    fn test_bank_promote_yields() {
        let ext = BankExtension::new();
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

        let result = ext.execute(0x0008, [0x00, 0x02, 0x00, 0x00], &mut ctx);
        match result {
            StepResult::Yield(DomainOp::BankPromote { source, bank_slot }) => {
                assert_eq!(source, Register(0x00));
                assert_eq!(bank_slot, 2);
            }
            other => panic!("Expected Yield(BankPromote), got {:?}", other),
        }
    }

    #[test]
    fn test_bank_evict_yields() {
        let ext = BankExtension::new();
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

        let result = ext.execute(0x000A, [0x03, 0x10, 0x00, 0x00], &mut ctx);
        match result {
            StepResult::Yield(DomainOp::BankEvict { bank_slot, count }) => {
                assert_eq!(bank_slot, 3);
                assert_eq!(count, 16);
            }
            other => panic!("Expected Yield(BankEvict), got {:?}", other),
        }
    }

    // =========================================================================
    // Inline execution tests (with mock BankAccess)
    // =========================================================================

    use crate::vm::extension::BankAccess;

    struct MockBankCache;
    impl BankAccess for MockBankCache {
        fn query(&self, _bank_slot: u8, _query: &[i32], top_k: usize) -> Option<Vec<(i64, i32)>> {
            let mut results = Vec::new();
            for i in 0..top_k.min(2) {
                results.push(((i as i64 + 1) * 100, 200 - i as i32 * 50));
            }
            Some(results)
        }
        fn load(&self, _bank_slot: u8, _high: i32, _low: i32) -> Option<Vec<i32>> {
            Some(vec![100, -50, 200])
        }
        fn count(&self, _bank_slot: u8) -> Option<i32> {
            Some(42)
        }
        fn write(&mut self, _bank_slot: u8, _vector: &[i32]) -> Option<(i32, i32)> {
            Some((0, 999))
        }
        fn touch(&mut self, _bank_slot: u8, _high: i32, _low: i32) {}
        fn delete(&mut self, _bank_slot: u8, _high: i32, _low: i32) -> bool { true }
    }

    #[test]
    fn test_inline_bank_count() {
        let ext = BankExtension::new();
        let (
            mut hot_regs, mut cold_regs, mut param_regs, mut shape_regs,
            mut pc, mut call_stack, mut loop_stack, input_buffer,
            mut output_buffer, target_buffer, mut chemical_state,
            mut current_error, mut babble_scale, mut babble_phase, mut pressure_regs,
        ) = setup_ctx!();

        let mut mock = MockBankCache;
        let mut ctx = ExecutionContext {
            hot_regs: &mut hot_regs, cold_regs: &mut cold_regs,
            param_regs: &mut param_regs, shape_regs: &mut shape_regs,
            pc: &mut pc, call_stack: &mut call_stack, loop_stack: &mut loop_stack,
            input_buffer: &input_buffer, output_buffer: &mut output_buffer,
            target_buffer: &target_buffer, chemical_state: &mut chemical_state,
            current_error: &mut current_error, babble_scale: &mut babble_scale,
            babble_phase: &mut babble_phase, pressure_regs: &mut pressure_regs,
            bank_cache: Some(&mut mock),
        };

        // BANK_COUNT H0, slot=0
        let result = ext.execute(0x0007, [0x00, 0x00, 0x00, 0x00], &mut ctx);
        assert!(matches!(result, StepResult::Continue), "Expected Continue, got {:?}", result);
        let h0 = ctx.hot_regs[0].as_ref().unwrap();
        assert_eq!(h0.data, vec![42]);
    }

    #[test]
    fn test_inline_bank_query() {
        let ext = BankExtension::new();
        let (
            mut hot_regs, mut cold_regs, mut param_regs, mut shape_regs,
            mut pc, mut call_stack, mut loop_stack, input_buffer,
            mut output_buffer, target_buffer, mut chemical_state,
            mut current_error, mut babble_scale, mut babble_phase, mut pressure_regs,
        ) = setup_ctx!();

        // Set up source register (query vector)
        hot_regs[1] = Some(HotBuffer { data: vec![100, -50], shape: vec![2] });

        let mut mock = MockBankCache;
        let mut ctx = ExecutionContext {
            hot_regs: &mut hot_regs, cold_regs: &mut cold_regs,
            param_regs: &mut param_regs, shape_regs: &mut shape_regs,
            pc: &mut pc, call_stack: &mut call_stack, loop_stack: &mut loop_stack,
            input_buffer: &input_buffer, output_buffer: &mut output_buffer,
            target_buffer: &target_buffer, chemical_state: &mut chemical_state,
            current_error: &mut current_error, babble_scale: &mut babble_scale,
            babble_phase: &mut babble_phase, pressure_regs: &mut pressure_regs,
            bank_cache: Some(&mut mock),
        };

        // BANK_QUERY H0, H1, slot=0, top_k=2
        let result = ext.execute(0x0000, [0x00, 0x01, 0x00, 0x02], &mut ctx);
        assert!(matches!(result, StepResult::Continue), "Expected Continue, got {:?}", result);
        let h0 = ctx.hot_regs[0].as_ref().unwrap();
        assert_eq!(h0.data[0], 2); // count = 2
    }

    #[test]
    fn test_inline_bank_write_and_load() {
        let ext = BankExtension::new();
        let (
            mut hot_regs, mut cold_regs, mut param_regs, mut shape_regs,
            mut pc, mut call_stack, mut loop_stack, input_buffer,
            mut output_buffer, target_buffer, mut chemical_state,
            mut current_error, mut babble_scale, mut babble_phase, mut pressure_regs,
        ) = setup_ctx!();

        // Source register for BANK_WRITE
        hot_regs[1] = Some(HotBuffer { data: vec![100, 200], shape: vec![2] });

        let mut mock = MockBankCache;
        let mut ctx = ExecutionContext {
            hot_regs: &mut hot_regs, cold_regs: &mut cold_regs,
            param_regs: &mut param_regs, shape_regs: &mut shape_regs,
            pc: &mut pc, call_stack: &mut call_stack, loop_stack: &mut loop_stack,
            input_buffer: &input_buffer, output_buffer: &mut output_buffer,
            target_buffer: &target_buffer, chemical_state: &mut chemical_state,
            current_error: &mut current_error, babble_scale: &mut babble_scale,
            babble_phase: &mut babble_phase, pressure_regs: &mut pressure_regs,
            bank_cache: Some(&mut mock),
        };

        // BANK_WRITE H0, H1, slot=0
        let result = ext.execute(0x0001, [0x00, 0x01, 0x00, 0x00], &mut ctx);
        assert!(matches!(result, StepResult::Continue));
        let h0 = ctx.hot_regs[0].as_ref().unwrap();
        assert_eq!(h0.data, vec![0, 999]); // entry_id = (0, 999)

        // Now use that entry_id to BANK_LOAD
        // H0 already has [0, 999]
        let result2 = ext.execute(0x0002, [0x02, 0x00, 0x00, 0x00], &mut ctx);
        assert!(matches!(result2, StepResult::Continue));
        let h2 = ctx.hot_regs[2].as_ref().unwrap();
        assert_eq!(h2.data, vec![100, -50, 200]); // mock returns this
    }

    #[test]
    fn test_no_cache_still_yields() {
        // Without bank_cache, all ops should yield (backward compatible)
        let ext = BankExtension::new();
        let (
            mut hot_regs, mut cold_regs, mut param_regs, mut shape_regs,
            mut pc, mut call_stack, mut loop_stack, input_buffer,
            mut output_buffer, target_buffer, mut chemical_state,
            mut current_error, mut babble_scale, mut babble_phase, mut pressure_regs,
        ) = setup_ctx!();

        hot_regs[1] = Some(HotBuffer { data: vec![100], shape: vec![1] });

        let mut ctx = make_ctx(
            &mut hot_regs, &mut cold_regs, &mut param_regs, &mut shape_regs,
            &mut pc, &mut call_stack, &mut loop_stack, &input_buffer,
            &mut output_buffer, &target_buffer, &mut chemical_state,
            &mut current_error, &mut babble_scale, &mut babble_phase, &mut pressure_regs,
        );
        // bank_cache is None in make_ctx

        let result = ext.execute(0x0007, [0x00, 0x00, 0x00, 0x00], &mut ctx);
        assert!(matches!(result, StepResult::Yield(DomainOp::BankCount { .. })));
    }
}
