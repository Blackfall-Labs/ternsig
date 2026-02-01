//! Sampling Extension (ExtID: 0x00C0)
//!
//! Token sampling and decoding helpers for realtime inference.

use crate::vm::extension::{
    ExecutionContext, Extension, InstructionMeta, OperandPattern, StepResult,
};
use crate::vm::interpreter::HotBuffer;
use crate::vm::register::Register;

/// Sampling extension — logits filtering and token selection.
pub struct SamplingExtension {
    instructions: Vec<InstructionMeta>,
}

impl SamplingExtension {
    pub fn new() -> Self {
        Self {
            instructions: vec![
                InstructionMeta {
                    opcode: 0x0000,
                    mnemonic: "TEMPERATURE",
                    operand_pattern: OperandPattern::RegRegReg,
                    description: "Scale logits by temperature",
                },
                InstructionMeta {
                    opcode: 0x0001,
                    mnemonic: "TOP_K",
                    operand_pattern: OperandPattern::RegRegReg,
                    description: "Keep top-k logits, zero rest",
                },
                InstructionMeta {
                    opcode: 0x0002,
                    mnemonic: "TOP_P",
                    operand_pattern: OperandPattern::RegRegReg,
                    description: "Nucleus filtering (top-p)",
                },
                InstructionMeta {
                    opcode: 0x0003,
                    mnemonic: "REP_PENALTY",
                    operand_pattern: OperandPattern::Custom("[dst:1][logits:1][history:1][penalty:1]"),
                    description: "Apply repetition penalty",
                },
            ],
        }
    }
}

impl Extension for SamplingExtension {
    fn name(&self) -> &str {
        "tvmr.sampling"
    }

    fn ext_id(&self) -> u16 {
        0x00C0
    }

    fn version(&self) -> (u16, u16, u16) {
        (0, 1, 0)
    }

    fn instructions(&self) -> &[InstructionMeta] {
        &self.instructions
    }

    fn execute(&self, opcode: u16, _ops: [u8; 4], _ctx: &mut ExecutionContext) -> StepResult {
        match opcode {
            0x0000 => execute_temperature(_ops, _ctx),
            0x0001 => execute_top_k(_ops, _ctx),
            0x0002 => execute_top_p(_ops, _ctx),
            0x0003 => execute_rep_penalty(_ops, _ctx),
            _ => StepResult::Error(format!(
                "tvmr.sampling opcode 0x{:04X} not implemented",
                opcode
            )),
        }
    }
}

fn execute_temperature(ops: [u8; 4], ctx: &mut ExecutionContext) -> StepResult {
    let dst = Register(ops[0]).index();
    let src = Register(ops[1]).index();
    let temp_idx = Register(ops[2]).index();

    let logits = match &ctx.hot_regs[src] {
        Some(buf) => buf.clone(),
        None => return StepResult::Error(format!("H{} not allocated", src)),
    };
    let temp = match &ctx.hot_regs[temp_idx] {
        Some(buf) => buf.data.get(0).copied().unwrap_or(1).max(1),
        None => 1,
    };

    let data: Vec<i32> = logits
        .data
        .iter()
        .map(|&v| ((v as i64 * 256) / temp as i64).clamp(i32::MIN as i64, i32::MAX as i64) as i32)
        .collect();

    ctx.hot_regs[dst] = Some(HotBuffer {
        data,
        shape: logits.shape,
    });
    StepResult::Continue
}

fn execute_top_k(ops: [u8; 4], ctx: &mut ExecutionContext) -> StepResult {
    let dst = Register(ops[0]).index();
    let src = Register(ops[1]).index();
    let k_idx = Register(ops[2]).index();

    let logits = match &ctx.hot_regs[src] {
        Some(buf) => buf.clone(),
        None => return StepResult::Error(format!("H{} not allocated", src)),
    };
    let k = match &ctx.hot_regs[k_idx] {
        Some(buf) => buf.data.get(0).copied().unwrap_or(1).max(1) as usize,
        None => 1,
    };

    let mut indices: Vec<usize> = (0..logits.data.len()).collect();
    indices.sort_by_key(|&i| std::cmp::Reverse(logits.data[i]));
    let keep = k.min(indices.len());
    let mut out = vec![i32::MIN / 2; logits.data.len()];
    for &i in indices.iter().take(keep) {
        out[i] = logits.data[i];
    }

    ctx.hot_regs[dst] = Some(HotBuffer {
        data: out,
        shape: logits.shape,
    });
    StepResult::Continue
}

fn execute_top_p(ops: [u8; 4], ctx: &mut ExecutionContext) -> StepResult {
    let dst = Register(ops[0]).index();
    let src = Register(ops[1]).index();
    let p_idx = Register(ops[2]).index();

    let logits = match &ctx.hot_regs[src] {
        Some(buf) => buf.clone(),
        None => return StepResult::Error(format!("H{} not allocated", src)),
    };
    let p = match &ctx.hot_regs[p_idx] {
        Some(buf) => buf.data.get(0).copied().unwrap_or(255).clamp(1, 255),
        None => 255,
    };

    let mut indices: Vec<usize> = (0..logits.data.len()).collect();
    indices.sort_by_key(|&i| std::cmp::Reverse(logits.data[i]));

    let max_val = logits.data.iter().copied().max().unwrap_or(0);
    let exp_vals: Vec<i64> = logits
        .data
        .iter()
        .map(|&v| (256 + (v - max_val).clamp(-1024, 1024) as i64).max(1))
        .collect();
    let total: i64 = exp_vals.iter().sum::<i64>().max(1);
    let threshold = (total * p as i64) / 255;

    let mut cumulative: i64 = 0;
    let mut out = vec![i32::MIN / 2; logits.data.len()];
    for &i in &indices {
        cumulative += exp_vals[i];
        out[i] = logits.data[i];
        if cumulative >= threshold {
            break;
        }
    }

    ctx.hot_regs[dst] = Some(HotBuffer {
        data: out,
        shape: logits.shape,
    });
    StepResult::Continue
}

fn execute_rep_penalty(ops: [u8; 4], ctx: &mut ExecutionContext) -> StepResult {
    let dst = Register(ops[0]).index();
    let src = Register(ops[1]).index();
    let hist_idx = Register(ops[2]).index();
    let pen_idx = Register(ops[3]).index();

    let logits = match &ctx.hot_regs[src] {
        Some(buf) => buf.clone(),
        None => return StepResult::Error(format!("H{} not allocated", src)),
    };
    let history = match &ctx.hot_regs[hist_idx] {
        Some(buf) => buf.data.clone(),
        None => vec![],
    };
    let penalty = match &ctx.hot_regs[pen_idx] {
        Some(buf) => buf.data.get(0).copied().unwrap_or(0),
        None => 0,
    };

    let mut out = logits.data.clone();
    for token in history {
        if token >= 0 {
            let idx = token as usize;
            if idx < out.len() {
                out[idx] = out[idx].saturating_sub(penalty);
            }
        }
    }

    ctx.hot_regs[dst] = Some(HotBuffer {
        data: out,
        shape: logits.shape,
    });
    StepResult::Continue
}
