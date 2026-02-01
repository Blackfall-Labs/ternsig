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
                InstructionMeta {
                    opcode: 0x0012,
                    mnemonic: "ATTN",
                    operand_pattern: OperandPattern::Custom("[q:1][k:1][v:1][out:1]"),
                    description: "Attention: out = softmax(q*k^T) * v",
                },
                InstructionMeta {
                    opcode: 0x0013,
                    mnemonic: "ROPE_APPLY",
                    operand_pattern: OperandPattern::Custom("[reg:1][dim:1][pos:1][_:1]"),
                    description: "Apply rotary positional embedding (noop fallback)",
                },
                InstructionMeta {
                    opcode: 0x0014,
                    mnemonic: "LAYERNORM",
                    operand_pattern: OperandPattern::Custom("[src:1][dst:1][scale:1][bias:1]"),
                    description: "LayerNorm over last dimension",
                },
                InstructionMeta {
                    opcode: 0x0015,
                    mnemonic: "RMSNORM",
                    operand_pattern: OperandPattern::Custom("[src:1][dst:1][scale:1][_:1]"),
                    description: "RMSNorm over last dimension",
                },
                InstructionMeta {
                    opcode: 0x0016,
                    mnemonic: "KV_APPEND",
                    operand_pattern: OperandPattern::Custom("[cache:1][kv:1][pos:1][len:1]"),
                    description: "Append KV slice into cache at position",
                },
                InstructionMeta {
                    opcode: 0x0017,
                    mnemonic: "KV_READ",
                    operand_pattern: OperandPattern::Custom("[cache:1][pos:1][out:1][len:1]"),
                    description: "Read KV slice from cache",
                },
                InstructionMeta {
                    opcode: 0x0018,
                    mnemonic: "KV_CLEAR",
                    operand_pattern: OperandPattern::Reg,
                    description: "Clear KV cache buffer",
                },
                InstructionMeta {
                    opcode: 0x0019,
                    mnemonic: "KV_APPEND_POS",
                    operand_pattern: OperandPattern::Custom("[cache:1][kv:1][pos:1][len:1]"),
                    description: "Append KV slice into cache at position from register",
                },
                InstructionMeta {
                    opcode: 0x001A,
                    mnemonic: "KV_READ_POS",
                    operand_pattern: OperandPattern::Custom("[cache:1][pos:1][out:1][len:1]"),
                    description: "Read KV slice from cache at position from register",
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
            0x0012 => execute_attention(operands, ctx),
            0x0013 => execute_rope_apply(operands, ctx),
            0x0014 => execute_layernorm(operands, ctx),
            0x0015 => execute_rmsnorm(operands, ctx),
            0x0016 => execute_kv_append(operands, ctx),
            0x0017 => execute_kv_read(operands, ctx),
            0x0018 => execute_kv_clear(operands, ctx),
            0x0019 => execute_kv_append_pos(operands, ctx),
            0x001A => execute_kv_read_pos(operands, ctx),
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

    // Preserve 2D shape when both inputs have matching first dimension (row count).
    // This is critical for concat feeding into batch_matmul — without it, the
    // output flattens to 1D and batch_size cannot be computed.
    let shape = if source.shape.len() == 2
        && other.shape.len() == 2
        && source.shape[0] == other.shape[0]
    {
        vec![source.shape[0], source.shape[1] + other.shape[1]]
    } else {
        vec![result_len]
    };

    ctx.hot_regs[dst] = Some(HotBuffer {
        data: result,
        shape,
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

fn execute_attention(ops: [u8; 4], ctx: &mut ExecutionContext) -> StepResult {
    let q_idx = Register(ops[0]).index();
    let k_idx = Register(ops[1]).index();
    let v_idx = Register(ops[2]).index();
    let out_idx = Register(ops[3]).index();

    let q = match &ctx.hot_regs[q_idx] {
        Some(buf) => buf.clone(),
        None => return StepResult::Error(format!("H{} not allocated", q_idx)),
    };
    let k = match &ctx.hot_regs[k_idx] {
        Some(buf) => buf.clone(),
        None => return StepResult::Error(format!("H{} not allocated", k_idx)),
    };
    let v = match &ctx.hot_regs[v_idx] {
        Some(buf) => buf.clone(),
        None => return StepResult::Error(format!("H{} not allocated", v_idx)),
    };

    let q_rows = q.shape.get(0).copied().unwrap_or(1).max(1);
    let q_cols = q.shape.get(1).copied().unwrap_or(q.data.len() / q_rows).max(1);
    let k_rows = k.shape.get(0).copied().unwrap_or(1).max(1);
    let k_cols = k.shape.get(1).copied().unwrap_or(k.data.len() / k_rows).max(1);
    let v_rows = v.shape.get(0).copied().unwrap_or(1).max(1);
    let v_cols = v.shape.get(1).copied().unwrap_or(v.data.len() / v_rows).max(1);

    if q_cols != k_cols || k_rows != v_rows {
        return StepResult::Error("ATTN shape mismatch".to_string());
    }

    let mut out = vec![0i32; q_rows * v_cols];

    for qi in 0..q_rows {
        let q_offset = qi * q_cols;
        let mut scores = vec![0i64; k_rows];
        for kj in 0..k_rows {
            let k_offset = kj * k_cols;
            let mut dot: i64 = 0;
            for d in 0..q_cols {
                let a = q.data.get(q_offset + d).copied().unwrap_or(0) as i64;
                let b = k.data.get(k_offset + d).copied().unwrap_or(0) as i64;
                dot += a * b;
            }
            scores[kj] = dot;
        }

        let max_val = scores.iter().copied().max().unwrap_or(0);
        let exp_vals: Vec<i64> = scores
            .iter()
            .map(|&s| (256 + (s - max_val).clamp(-1024, 1024) / 256).max(1))
            .collect();
        let total: i64 = exp_vals.iter().sum::<i64>().max(1);

        for vj in 0..v_cols {
            let mut acc: i64 = 0;
            for kj in 0..k_rows {
                let weight = exp_vals[kj] * 255 / total;
                let v_val = v.data.get(kj * v_cols + vj).copied().unwrap_or(0) as i64;
                acc += weight * v_val;
            }
            out[qi * v_cols + vj] = (acc / 255) as i32;
        }
    }

    ctx.hot_regs[out_idx] = Some(HotBuffer {
        data: out,
        shape: vec![q_rows, v_cols],
    });
    StepResult::Continue
}

fn execute_rope_apply(_ops: [u8; 4], _ctx: &mut ExecutionContext) -> StepResult {
    let reg = Register(_ops[0]).index();
    let dim = _ops[1] as usize;
    let pos = _ops[2] as usize;

    let mut buf = match &mut _ctx.hot_regs[reg] {
        Some(buf) => buf.clone(),
        None => return StepResult::Error(format!("H{} not allocated", reg)),
    };

    let last_dim = buf.shape.last().copied().unwrap_or(buf.data.len()).max(1);
    let apply_dim = dim.min(last_dim);
    if apply_dim < 2 {
        return StepResult::Continue;
    }

    let (sin, cos) = rope_tables();
    let row_count = buf.data.len() / last_dim;
    let scale = 256i32;

    for row in 0..row_count {
        let base = row * last_dim;
        for i in (0..apply_dim).step_by(2) {
            let idx = (pos + i / 2) % sin.len();
            let s = sin[idx];
            let c = cos[idx];
            let x0 = buf.data[base + i];
            let x1 = buf.data[base + i + 1];

            let y0 = (x0 as i64 * c as i64 - x1 as i64 * s as i64) / scale as i64;
            let y1 = (x0 as i64 * s as i64 + x1 as i64 * c as i64) / scale as i64;

            buf.data[base + i] = y0.clamp(i32::MIN as i64, i32::MAX as i64) as i32;
            buf.data[base + i + 1] = y1.clamp(i32::MIN as i64, i32::MAX as i64) as i32;
        }
    }

    _ctx.hot_regs[reg] = Some(buf);
    StepResult::Continue
}

fn execute_layernorm(ops: [u8; 4], ctx: &mut ExecutionContext) -> StepResult {
    let src = Register(ops[0]).index();
    let dst = Register(ops[1]).index();
    let scale_reg = Register(ops[2]);
    let bias_reg = Register(ops[3]);

    let source = match &ctx.hot_regs[src] {
        Some(buf) => buf.clone(),
        None => return StepResult::Error(format!("H{} not allocated", src)),
    };

    let last_dim = source.shape.last().copied().unwrap_or(source.data.len()).max(1);
    let rows = source.data.len() / last_dim;

    let scale_vals = read_register_i32(scale_reg, ctx, last_dim);
    let bias_vals = read_register_i32(bias_reg, ctx, last_dim);

    let mut out = vec![0i32; source.data.len()];
    for row in 0..rows {
        let start = row * last_dim;
        let end = start + last_dim;
        let slice = &source.data[start..end];
        let sum: i64 = slice.iter().map(|&v| v as i64).sum();
        let mean = sum / last_dim as i64;
        let mut var_sum: i64 = 0;
        for &v in slice {
            let diff = v as i64 - mean;
            var_sum += diff * diff;
        }
        let var = (var_sum / last_dim as i64).max(1);
        let denom = isqrt_i64(var) as i64;

        for i in 0..last_dim {
            let diff = slice[i] as i64 - mean;
            let norm = diff * 256 / denom;
            let scaled = norm * scale_vals[i] as i64 / 255;
            let biased = scaled + bias_vals[i] as i64;
            out[start + i] = biased.clamp(i32::MIN as i64, i32::MAX as i64) as i32;
        }
    }

    ctx.hot_regs[dst] = Some(HotBuffer {
        data: out,
        shape: source.shape,
    });
    StepResult::Continue
}

fn execute_rmsnorm(ops: [u8; 4], ctx: &mut ExecutionContext) -> StepResult {
    let src = Register(ops[0]).index();
    let dst = Register(ops[1]).index();
    let scale_reg = Register(ops[2]);

    let source = match &ctx.hot_regs[src] {
        Some(buf) => buf.clone(),
        None => return StepResult::Error(format!("H{} not allocated", src)),
    };

    let last_dim = source.shape.last().copied().unwrap_or(source.data.len()).max(1);
    let rows = source.data.len() / last_dim;

    let scale_vals = read_register_i32(scale_reg, ctx, last_dim);

    let mut out = vec![0i32; source.data.len()];
    for row in 0..rows {
        let start = row * last_dim;
        let end = start + last_dim;
        let slice = &source.data[start..end];
        let mut sum_sq: i64 = 0;
        for &v in slice {
            let val = v as i64;
            sum_sq += val * val;
        }
        let rms = (sum_sq / last_dim as i64).max(1);
        let denom = isqrt_i64(rms) as i64;

        for i in 0..last_dim {
            let norm = (slice[i] as i64) * 256 / denom;
            let scaled = norm * scale_vals[i] as i64 / 255;
            out[start + i] = scaled.clamp(i32::MIN as i64, i32::MAX as i64) as i32;
        }
    }

    ctx.hot_regs[dst] = Some(HotBuffer {
        data: out,
        shape: source.shape,
    });
    StepResult::Continue
}

fn execute_kv_append(ops: [u8; 4], ctx: &mut ExecutionContext) -> StepResult {
    let cache_idx = Register(ops[0]).index();
    let kv_idx = Register(ops[1]).index();
    let pos = ops[2] as usize;
    let len = ops[3] as usize;

    let kv = match &ctx.hot_regs[kv_idx] {
        Some(buf) => buf.clone(),
        None => return StepResult::Error(format!("H{} not allocated", kv_idx)),
    };

    let kv_slice_len = kv.shape.last().copied().unwrap_or(kv.data.len()).max(1);
    let slice_len = if len == 0 { kv_slice_len } else { len.min(kv_slice_len) };
    let required_len = (pos + 1) * slice_len;

    let mut cache = match &ctx.hot_regs[cache_idx] {
        Some(buf) => buf.clone(),
        None => HotBuffer {
            data: vec![],
            shape: vec![0, slice_len],
        },
    };

    if cache.data.len() < required_len {
        cache.data.resize(required_len, 0);
    }

    let start = pos * slice_len;
    let end = start + slice_len;
    cache.data[start..end].copy_from_slice(&kv.data[..slice_len]);
    cache.shape = vec![pos + 1, slice_len];

    ctx.hot_regs[cache_idx] = Some(cache);
    StepResult::Continue
}

fn execute_kv_read(ops: [u8; 4], ctx: &mut ExecutionContext) -> StepResult {
    let cache_idx = Register(ops[0]).index();
    let pos = ops[1] as usize;
    let out_idx = Register(ops[2]).index();
    let len = ops[3] as usize;

    let cache = match &ctx.hot_regs[cache_idx] {
        Some(buf) => buf.clone(),
        None => return StepResult::Error(format!("H{} not allocated", cache_idx)),
    };

    let cache_slice_len = cache.shape.last().copied().unwrap_or(cache.data.len()).max(1);
    let slice_len = if len == 0 { cache_slice_len } else { len.min(cache_slice_len) };
    let start = pos * slice_len;
    let end = (start + slice_len).min(cache.data.len());

    let mut data = vec![0i32; slice_len];
    if start < cache.data.len() {
        let available = end - start;
        data[..available].copy_from_slice(&cache.data[start..end]);
    }

    ctx.hot_regs[out_idx] = Some(HotBuffer {
        data,
        shape: vec![1, slice_len],
    });
    StepResult::Continue
}

fn execute_kv_append_pos(ops: [u8; 4], ctx: &mut ExecutionContext) -> StepResult {
    let pos_reg = Register(ops[2]);
    let pos = read_register_i32(pos_reg, ctx, 1)
        .get(0)
        .copied()
        .unwrap_or(0)
        .max(0) as usize;
    execute_kv_append([ops[0], ops[1], pos as u8, ops[3]], ctx)
}

fn execute_kv_read_pos(ops: [u8; 4], ctx: &mut ExecutionContext) -> StepResult {
    let pos_reg = Register(ops[1]);
    let pos = read_register_i32(pos_reg, ctx, 1)
        .get(0)
        .copied()
        .unwrap_or(0)
        .max(0) as usize;
    execute_kv_read([ops[0], pos as u8, ops[2], ops[3]], ctx)
}

fn execute_kv_clear(ops: [u8; 4], ctx: &mut ExecutionContext) -> StepResult {
    let cache_idx = Register(ops[0]).index();
    ctx.hot_regs[cache_idx] = Some(HotBuffer {
        data: vec![],
        shape: vec![0, 0],
    });
    StepResult::Continue
}

fn read_register_i32(reg: Register, ctx: &ExecutionContext, len: usize) -> Vec<i32> {
    if reg.is_hot() {
        let idx = reg.index();
        if let Some(buf) = &ctx.hot_regs[idx] {
            if buf.data.len() >= len {
                return buf.data[..len].to_vec();
            }
        }
        vec![0; len]
    } else if reg.is_cold() {
        let idx = reg.index();
        if let Some(buf) = &ctx.cold_regs[idx] {
            let mut out = Vec::with_capacity(len);
            for i in 0..len {
                if i < buf.weights.len() {
                    let s = &buf.weights[i];
                    out.push(s.polarity as i32 * s.magnitude as i32);
                } else {
                    out.push(0);
                }
            }
            return out;
        }
        vec![0; len]
    } else {
        vec![0; len]
    }
}

fn isqrt_i64(value: i64) -> i64 {
    if value <= 1 {
        return value;
    }
    let mut x0 = value / 2;
    let mut x1 = (x0 + value / x0) / 2;
    while (x1 - x0).abs() > 1 {
        x0 = x1;
        x1 = (x0 + value / x0) / 2;
    }
    x1.max(1)
}

fn rope_tables() -> (&'static [i32], &'static [i32]) {
    use std::sync::OnceLock;

    static SIN: OnceLock<Vec<i32>> = OnceLock::new();
    static COS: OnceLock<Vec<i32>> = OnceLock::new();

    let sin = SIN.get_or_init(|| build_rope_table(true));
    let cos = COS.get_or_init(|| build_rope_table(false));
    (sin, cos)
}

fn build_rope_table(is_sin: bool) -> Vec<i32> {
    let size = 1024usize;
    let mut table = Vec::with_capacity(size);
    for i in 0..size {
        let angle = (i as f32) * 0.0005;
        let val = if is_sin { angle.sin() } else { angle.cos() };
        table.push((val * 256.0) as i32);
    }
    table
}
