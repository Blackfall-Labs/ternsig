//! Tensor Extension (ExtID: 0x0001)
//!
//! General-purpose tensor operations: arithmetic, reductions, reshaping.
//! All implementations use integer-only arithmetic.

use crate::vm::extension::{
    ExecutionContext, Extension, InstructionMeta, OperandPattern, StepResult,
};
use crate::vm::interpreter::HotBuffer;
use crate::vm::register::Register;

/// Tensor extension — general-purpose tensor operations.
pub struct TensorExtension {
    instructions: Vec<InstructionMeta>,
}

impl TensorExtension {
    pub fn new() -> Self {
        Self {
            instructions: vec![
                InstructionMeta {
                    opcode: 0x0000,
                    mnemonic: "MATMUL",
                    operand_pattern: OperandPattern::RegRegReg,
                    description: "Matrix multiply: target = source @ aux",
                },
                InstructionMeta {
                    opcode: 0x0001,
                    mnemonic: "ADD",
                    operand_pattern: OperandPattern::RegRegReg,
                    description: "Element-wise add: target = source + aux",
                },
                InstructionMeta {
                    opcode: 0x0002,
                    mnemonic: "SUB",
                    operand_pattern: OperandPattern::RegRegReg,
                    description: "Element-wise subtract: target = source - aux",
                },
                InstructionMeta {
                    opcode: 0x0003,
                    mnemonic: "MUL",
                    operand_pattern: OperandPattern::RegRegReg,
                    description: "Element-wise multiply: target = source * aux",
                },
                InstructionMeta {
                    opcode: 0x0004,
                    mnemonic: "SCALE",
                    operand_pattern: OperandPattern::RegRegImm16,
                    description: "Scale by constant: target = source * scale",
                },
                InstructionMeta {
                    opcode: 0x0005,
                    mnemonic: "SHIFT",
                    operand_pattern: OperandPattern::RegRegReg,
                    description: "Right shift: target = source >> amount",
                },
                InstructionMeta {
                    opcode: 0x0006,
                    mnemonic: "CLAMP",
                    operand_pattern: OperandPattern::RegRegRegFlags,
                    description: "Clamp to range: target = clamp(source, min, max)",
                },
                InstructionMeta {
                    opcode: 0x0007,
                    mnemonic: "CMP_GT",
                    operand_pattern: OperandPattern::RegRegReg,
                    description: "Compare greater: target = source > aux ? 1 : 0",
                },
                InstructionMeta {
                    opcode: 0x0008,
                    mnemonic: "MAX_REDUCE",
                    operand_pattern: OperandPattern::RegReg,
                    description: "Max reduce: target[0] = max(source)",
                },
                InstructionMeta {
                    opcode: 0x0009,
                    mnemonic: "NEGATE",
                    operand_pattern: OperandPattern::RegReg,
                    description: "Negate: target = -source",
                },
                InstructionMeta {
                    opcode: 0x000A,
                    mnemonic: "REDUCE_AVG",
                    operand_pattern: OperandPattern::Custom("[dst:1][src:1][start:1][count:1]"),
                    description: "Reduce average: target[0] = mean(source[start..start+count])",
                },
                InstructionMeta {
                    opcode: 0x000B,
                    mnemonic: "REDUCE_MEAN_DIM",
                    operand_pattern: OperandPattern::RegRegReg,
                    description: "Reduce mean along dimension",
                },
                InstructionMeta {
                    opcode: 0x000C,
                    mnemonic: "SLICE",
                    operand_pattern: OperandPattern::Custom("[dst:1][src:1][start:1][len:1]"),
                    description: "Slice: target = source[start..start+len]",
                },
                InstructionMeta {
                    opcode: 0x000D,
                    mnemonic: "ARGMAX",
                    operand_pattern: OperandPattern::RegReg,
                    description: "Argmax: target[0] = index of max in source",
                },
                InstructionMeta {
                    opcode: 0x000E,
                    mnemonic: "CONCAT",
                    operand_pattern: OperandPattern::RegRegReg,
                    description: "Concatenate: target = concat(source, aux)",
                },
                InstructionMeta {
                    opcode: 0x000F,
                    mnemonic: "SQUEEZE",
                    operand_pattern: OperandPattern::RegReg,
                    description: "Remove dimension (shape op)",
                },
                InstructionMeta {
                    opcode: 0x0010,
                    mnemonic: "UNSQUEEZE",
                    operand_pattern: OperandPattern::RegReg,
                    description: "Add dimension (shape op)",
                },
                InstructionMeta {
                    opcode: 0x0011,
                    mnemonic: "TRANSPOSE",
                    operand_pattern: OperandPattern::RegReg,
                    description: "Transpose dimensions (shape op)",
                },
            ],
        }
    }
}

impl Extension for TensorExtension {
    fn ext_id(&self) -> u16 {
        0x0001
    }

    fn name(&self) -> &str {
        "tvmr.tensor"
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
            0x0000 => execute_matmul(operands, ctx),
            0x0001 => execute_add(operands, ctx),
            0x0002 => execute_sub(operands, ctx),
            0x0003 => execute_mul(operands, ctx),
            0x0004 => execute_scale(operands, ctx),
            0x0005 => execute_shift(operands, ctx),
            0x0006 => execute_clamp(operands, ctx),
            0x0007 => execute_cmp_gt(operands, ctx),
            0x0008 => execute_max_reduce(operands, ctx),
            0x0009 => execute_negate(operands, ctx),
            0x000A => execute_reduce_avg(operands, ctx),
            0x000B => execute_reduce_mean_dim(operands, ctx),
            0x000C => execute_slice(operands, ctx),
            0x000D => execute_argmax(operands, ctx),
            0x000E => execute_concat(operands, ctx),
            0x000F => execute_squeeze(operands, ctx),
            0x0010 => execute_unsqueeze(operands, ctx),
            0x0011 => execute_transpose(operands, ctx),
            _ => StepResult::Error(format!("tvmr.tensor: unknown opcode 0x{:04X}", opcode)),
        }
    }
}

// =============================================================================
// Standalone execute functions
// =============================================================================

fn execute_matmul(_ops: [u8; 4], _ctx: &mut ExecutionContext) -> StepResult {
    // Float matmul — not implemented for integer-only VM
    // The ternary extension provides TERNARY_MATMUL instead
    StepResult::Error("MATMUL not implemented (use ternary.TERNARY_MATMUL)".to_string())
}

fn execute_add(ops: [u8; 4], ctx: &mut ExecutionContext) -> StepResult {
    let dst = Register(ops[0]).index();
    let src = Register(ops[1]).index();
    let other_idx = ops[2] as usize & 0x3F;

    let source = match &ctx.hot_regs[src] {
        Some(buf) => buf.clone(),
        None => return StepResult::Error(format!("H{} not allocated", src)),
    };

    // Check if other is hot register
    let other_reg = Register(ops[2]);
    if other_reg.is_hot() {
        let other = match &ctx.hot_regs[other_idx] {
            Some(buf) => buf,
            None => return StepResult::Error(format!("H{} not allocated", other_idx)),
        };

        let len = source.data.len().max(other.data.len());
        let mut result = vec![0i32; len];
        for i in 0..len {
            let a = source.data.get(i).copied().unwrap_or(0);
            let b = other.data.get(i).copied().unwrap_or(0);
            result[i] = a.saturating_add(b);
        }

        ctx.hot_regs[dst] = Some(HotBuffer {
            data: result,
            shape: source.shape,
        });
    } else if other_reg.is_cold() {
        // Hot + Cold (signal bias)
        let cold_idx = other_reg.index();
        let bias = match &ctx.cold_regs[cold_idx] {
            Some(buf) => buf,
            None => return StepResult::Error(format!("C{} not allocated", cold_idx)),
        };

        let mut result = source.data.clone();
        for (i, val) in result.iter_mut().enumerate() {
            if i < bias.weights.len() {
                let b = &bias.weights[i];
                let bias_val = b.polarity as i64 * b.magnitude as i64 * 256;
                *val = (*val as i64 + bias_val).clamp(i32::MIN as i64, i32::MAX as i64) as i32;
            }
        }

        ctx.hot_regs[dst] = Some(HotBuffer {
            data: result,
            shape: source.shape,
        });
    }

    StepResult::Continue
}

fn execute_sub(ops: [u8; 4], ctx: &mut ExecutionContext) -> StepResult {
    let dst = Register(ops[0]).index();
    let src = Register(ops[1]).index();
    let other_idx = Register(ops[2]).index();

    let source = match &ctx.hot_regs[src] {
        Some(buf) => buf.clone(),
        None => return StepResult::Error(format!("H{} not allocated", src)),
    };

    let other = match &ctx.hot_regs[other_idx] {
        Some(buf) => buf,
        None => return StepResult::Error(format!("H{} not allocated", other_idx)),
    };

    let len = source.data.len().max(other.data.len());
    let mut result = vec![0i32; len];
    for i in 0..len {
        let a = source.data.get(i).copied().unwrap_or(0);
        let b = other.data.get(i).copied().unwrap_or(0);
        result[i] = a.saturating_sub(b);
    }

    ctx.hot_regs[dst] = Some(HotBuffer {
        data: result,
        shape: source.shape,
    });

    StepResult::Continue
}

fn execute_mul(ops: [u8; 4], ctx: &mut ExecutionContext) -> StepResult {
    let dst = Register(ops[0]).index();
    let src = Register(ops[1]).index();
    let other_idx = Register(ops[2]).index();

    let source = match &ctx.hot_regs[src] {
        Some(buf) => buf.clone(),
        None => return StepResult::Error(format!("H{} not allocated", src)),
    };

    let other = match &ctx.hot_regs[other_idx] {
        Some(buf) => buf,
        None => return StepResult::Error(format!("H{} not allocated", other_idx)),
    };

    let len = source.data.len().min(other.data.len());
    let mut result = vec![0i32; len];
    for i in 0..len {
        let a = source.data[i] as i64;
        let b = other.data[i] as i64;
        result[i] = ((a * b) >> 8).clamp(i32::MIN as i64, i32::MAX as i64) as i32;
    }

    ctx.hot_regs[dst] = Some(HotBuffer {
        data: result,
        shape: source.shape,
    });

    StepResult::Continue
}

fn execute_scale(ops: [u8; 4], ctx: &mut ExecutionContext) -> StepResult {
    let dst = Register(ops[0]).index();
    let src = Register(ops[1]).index();
    let scale = ((ops[2] as u16) << 8) | (ops[3] as u16);

    let source = match &ctx.hot_regs[src] {
        Some(buf) => buf.clone(),
        None => return StepResult::Error(format!("H{} not allocated", src)),
    };

    let data: Vec<i32> = source
        .data
        .iter()
        .map(|&v| {
            ((v as i64 * scale as i64) / 256).clamp(i32::MIN as i64, i32::MAX as i64) as i32
        })
        .collect();

    ctx.hot_regs[dst] = Some(HotBuffer {
        data,
        shape: source.shape,
    });

    StepResult::Continue
}

fn execute_shift(ops: [u8; 4], ctx: &mut ExecutionContext) -> StepResult {
    let dst = Register(ops[0]).index();
    let src = Register(ops[1]).index();
    let shift = ops[2] as u32;

    let source = match &ctx.hot_regs[src] {
        Some(buf) => buf.clone(),
        None => return StepResult::Error(format!("H{} not allocated", src)),
    };

    let data: Vec<i32> = source.data.iter().map(|&v| v >> shift).collect();

    ctx.hot_regs[dst] = Some(HotBuffer {
        data,
        shape: source.shape,
    });

    StepResult::Continue
}

fn execute_clamp(ops: [u8; 4], ctx: &mut ExecutionContext) -> StepResult {
    let dst = Register(ops[0]).index();
    let src = Register(ops[1]).index();
    let min_val = ops[2] as i32 - 128; // Signed offset
    let max_val = ops[3] as i32 - 128;

    let source = match &ctx.hot_regs[src] {
        Some(buf) => buf.clone(),
        None => return StepResult::Error(format!("H{} not allocated", src)),
    };

    let effective_min = min_val * 256;
    let effective_max = max_val * 256;

    let data: Vec<i32> = source
        .data
        .iter()
        .map(|&v| v.clamp(effective_min, effective_max))
        .collect();

    ctx.hot_regs[dst] = Some(HotBuffer {
        data,
        shape: source.shape,
    });

    StepResult::Continue
}

fn execute_cmp_gt(ops: [u8; 4], ctx: &mut ExecutionContext) -> StepResult {
    let dst = Register(ops[0]).index();
    let src = Register(ops[1]).index();
    let other_idx = Register(ops[2]).index();

    let source = match &ctx.hot_regs[src] {
        Some(buf) => buf.clone(),
        None => return StepResult::Error(format!("H{} not allocated", src)),
    };

    let other = match &ctx.hot_regs[other_idx] {
        Some(buf) => buf,
        None => return StepResult::Error(format!("H{} not allocated", other_idx)),
    };

    let len = source.data.len().min(other.data.len());
    let mut result = vec![0i32; len];
    for i in 0..len {
        result[i] = if source.data[i] > other.data[i] { 1 } else { 0 };
    }

    ctx.hot_regs[dst] = Some(HotBuffer {
        data: result,
        shape: vec![len],
    });

    StepResult::Continue
}

fn execute_max_reduce(ops: [u8; 4], ctx: &mut ExecutionContext) -> StepResult {
    let dst = Register(ops[0]).index();
    let src = Register(ops[1]).index();

    let source = match &ctx.hot_regs[src] {
        Some(buf) => &buf.data,
        None => return StepResult::Error(format!("H{} not allocated", src)),
    };

    let max_val = source.iter().cloned().max().unwrap_or(0);

    ctx.hot_regs[dst] = Some(HotBuffer {
        data: vec![max_val],
        shape: vec![1],
    });

    StepResult::Continue
}

fn execute_negate(ops: [u8; 4], ctx: &mut ExecutionContext) -> StepResult {
    let dst = Register(ops[0]).index();
    let src = Register(ops[1]).index();

    let source = match &ctx.hot_regs[src] {
        Some(buf) => buf.clone(),
        None => return StepResult::Error(format!("H{} not allocated", src)),
    };

    let data: Vec<i32> = source.data.iter().map(|&v| v.saturating_neg()).collect();

    ctx.hot_regs[dst] = Some(HotBuffer {
        data,
        shape: source.shape,
    });

    StepResult::Continue
}

fn execute_reduce_avg(ops: [u8; 4], ctx: &mut ExecutionContext) -> StepResult {
    let dst = Register(ops[0]).index();
    let src = Register(ops[1]).index();
    let start = ops[2] as usize;
    let count = ops[3] as usize;

    let data = match &ctx.hot_regs[src] {
        Some(buf) => &buf.data,
        None => return StepResult::Error(format!("H{} not allocated", src)),
    };

    if count == 0 {
        ctx.hot_regs[dst] = Some(HotBuffer {
            data: vec![0],
            shape: vec![1],
        });
        return StepResult::Continue;
    }

    let end = (start + count).min(data.len());
    let actual_count = end.saturating_sub(start);
    let sum: i32 = data
        .get(start..end)
        .map(|slice| slice.iter().sum())
        .unwrap_or(0);
    let avg = if actual_count > 0 {
        sum / actual_count as i32
    } else {
        0
    };

    ctx.hot_regs[dst] = Some(HotBuffer {
        data: vec![avg],
        shape: vec![1],
    });

    StepResult::Continue
}

fn execute_reduce_mean_dim(ops: [u8; 4], ctx: &mut ExecutionContext) -> StepResult {
    let dst = Register(ops[0]).index();
    let src = Register(ops[1]).index();
    let dim = ops[2] as usize;

    let source = match &ctx.hot_regs[src] {
        Some(buf) => buf,
        None => return StepResult::Error(format!("H{} not allocated", src)),
    };

    if source.shape.len() < 2 {
        let sum: i64 = source.data.iter().map(|&x| x as i64).sum();
        let count = source.data.len().max(1);
        let mean = (sum / count as i64) as i32;
        ctx.hot_regs[dst] = Some(HotBuffer {
            data: vec![mean],
            shape: vec![1],
        });
    } else {
        let rows = source.shape[0];
        let cols = source.shape[1];

        if dim == 0 {
            let mut output_data = vec![0i32; cols];
            for c in 0..cols {
                let mut sum: i64 = 0;
                for r in 0..rows {
                    let idx = r * cols + c;
                    if idx < source.data.len() {
                        sum += source.data[idx] as i64;
                    }
                }
                output_data[c] = (sum / rows.max(1) as i64) as i32;
            }
            ctx.hot_regs[dst] = Some(HotBuffer {
                data: output_data,
                shape: vec![cols],
            });
        } else {
            let mut output_data = vec![0i32; rows];
            for r in 0..rows {
                let mut sum: i64 = 0;
                for c in 0..cols {
                    let idx = r * cols + c;
                    if idx < source.data.len() {
                        sum += source.data[idx] as i64;
                    }
                }
                output_data[r] = (sum / cols.max(1) as i64) as i32;
            }
            ctx.hot_regs[dst] = Some(HotBuffer {
                data: output_data,
                shape: vec![rows],
            });
        }
    }

    StepResult::Continue
}

fn execute_slice(ops: [u8; 4], ctx: &mut ExecutionContext) -> StepResult {
    let dst = Register(ops[0]).index();
    let src = Register(ops[1]).index();
    let start = ops[2] as usize;
    let len = ops[3] as usize;

    let data = match &ctx.hot_regs[src] {
        Some(buf) => &buf.data,
        None => return StepResult::Error(format!("H{} not allocated", src)),
    };

    let end = (start + len).min(data.len());
    let result: Vec<i32> = data.get(start..end).map(|s| s.to_vec()).unwrap_or_default();
    let result_len = result.len();

    ctx.hot_regs[dst] = Some(HotBuffer {
        data: result,
        shape: vec![result_len],
    });

    StepResult::Continue
}

fn execute_argmax(ops: [u8; 4], ctx: &mut ExecutionContext) -> StepResult {
    let dst = Register(ops[0]).index();
    let src = Register(ops[1]).index();

    let data = match &ctx.hot_regs[src] {
        Some(buf) => &buf.data,
        None => return StepResult::Error(format!("H{} not allocated", src)),
    };

    let argmax = data
        .iter()
        .enumerate()
        .max_by_key(|(_, &v)| v)
        .map(|(i, _)| i as i32)
        .unwrap_or(0);

    ctx.hot_regs[dst] = Some(HotBuffer {
        data: vec![argmax],
        shape: vec![1],
    });

    StepResult::Continue
}

fn execute_concat(ops: [u8; 4], ctx: &mut ExecutionContext) -> StepResult {
    let dst = Register(ops[0]).index();
    let src = Register(ops[1]).index();
    let other_idx = ops[2] as usize & 0x0F;

    let source = match &ctx.hot_regs[src] {
        Some(buf) => buf.clone(),
        None => return StepResult::Error(format!("H{} not allocated", src)),
    };

    let other = match &ctx.hot_regs[other_idx] {
        Some(buf) => buf,
        None => return StepResult::Error(format!("H{} not allocated", other_idx)),
    };

    let mut result = source.data.clone();
    result.extend_from_slice(&other.data);
    let result_len = result.len();

    ctx.hot_regs[dst] = Some(HotBuffer {
        data: result,
        shape: vec![result_len],
    });

    StepResult::Continue
}

fn execute_squeeze(ops: [u8; 4], ctx: &mut ExecutionContext) -> StepResult {
    let dst = Register(ops[0]).index();
    let src = Register(ops[1]).index();

    if let Some(buf) = ctx.hot_regs[src].clone() {
        ctx.hot_regs[dst] = Some(buf);
    }

    StepResult::Continue
}

fn execute_unsqueeze(ops: [u8; 4], ctx: &mut ExecutionContext) -> StepResult {
    let dst = Register(ops[0]).index();
    let src = Register(ops[1]).index();

    if let Some(buf) = ctx.hot_regs[src].clone() {
        ctx.hot_regs[dst] = Some(buf);
    }

    StepResult::Continue
}

fn execute_transpose(ops: [u8; 4], ctx: &mut ExecutionContext) -> StepResult {
    let dst = Register(ops[0]).index();
    let src = Register(ops[1]).index();

    if let Some(buf) = ctx.hot_regs[src].clone() {
        ctx.hot_regs[dst] = Some(buf);
    }

    StepResult::Continue
}
