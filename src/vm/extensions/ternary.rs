//! Ternary Extension (ExtID: 0x0002)
//!
//! Ternary/Signal-specific operations: matmul, bias, embedding, gate update.
//! All operations are integer-only, CPU-only — no floats, no GPU.

use crate::vm::extension::{
    ExecutionContext, Extension, InstructionMeta, OperandPattern, StepResult,
};
use crate::vm::interpreter::HotBuffer;
use crate::vm::register::Register;
use crate::Signal;

/// Ternary extension — Signal-based neural network operations.
pub struct TernaryExtension {
    instructions: Vec<InstructionMeta>,
}

impl TernaryExtension {
    pub fn new() -> Self {
        Self {
            instructions: vec![
                InstructionMeta {
                    opcode: 0x0000,
                    mnemonic: "TERNARY_MATMUL",
                    operand_pattern: OperandPattern::RegRegReg,
                    description: "Ternary matmul: target = cold_weights @ hot_input",
                },
                InstructionMeta {
                    opcode: 0x0001,
                    mnemonic: "TERNARY_BATCH_MATMUL",
                    operand_pattern: OperandPattern::RegRegReg,
                    description: "Batch ternary matmul: target[i] = weights @ input[i]",
                },
                InstructionMeta {
                    opcode: 0x0002,
                    mnemonic: "TERNARY_ADD_BIAS",
                    operand_pattern: OperandPattern::RegRegReg,
                    description: "Add signal bias: target = source + cold_bias",
                },
                InstructionMeta {
                    opcode: 0x0003,
                    mnemonic: "DEQUANTIZE",
                    operand_pattern: OperandPattern::RegRegImm16,
                    description: "Dequantize: target = source >> scale",
                },
                InstructionMeta {
                    opcode: 0x0004,
                    mnemonic: "EMBED_LOOKUP",
                    operand_pattern: OperandPattern::RegRegReg,
                    description: "Embedding lookup: target[i] = table[indices[i]]",
                },
                InstructionMeta {
                    opcode: 0x0005,
                    mnemonic: "EMBED_SEQUENCE",
                    operand_pattern: OperandPattern::Custom("[dst:1][table:1][count:1][_:1]"),
                    description: "Sequential embedding: target[i] = table[i]",
                },
                InstructionMeta {
                    opcode: 0x0006,
                    mnemonic: "GATE_UPDATE",
                    operand_pattern: OperandPattern::Custom("[dst:1][gate:1][update:1][state:1]"),
                    description: "Gated update: target = gate * update + (1-gate) * state",
                },
                InstructionMeta {
                    opcode: 0x0007,
                    mnemonic: "QUANTIZE",
                    operand_pattern: OperandPattern::RegReg,
                    description: "Quantize to ternary signal",
                },
                InstructionMeta {
                    opcode: 0x0008,
                    mnemonic: "PACK_TERNARY",
                    operand_pattern: OperandPattern::RegReg,
                    description: "Pack signals to 2-bit representation",
                },
                InstructionMeta {
                    opcode: 0x0009,
                    mnemonic: "UNPACK_TERNARY",
                    operand_pattern: OperandPattern::RegReg,
                    description: "Unpack 2-bit to Signal",
                },
                InstructionMeta {
                    opcode: 0x000A,
                    mnemonic: "APPLY_POLARITY",
                    operand_pattern: OperandPattern::RegReg,
                    description: "Apply polarity update to weight",
                },
                InstructionMeta {
                    opcode: 0x000B,
                    mnemonic: "APPLY_MAGNITUDE",
                    operand_pattern: OperandPattern::RegReg,
                    description: "Apply magnitude update to weight",
                },
                InstructionMeta {
                    opcode: 0x000C,
                    mnemonic: "THRESHOLD_POLARITY",
                    operand_pattern: OperandPattern::RegReg,
                    description: "Check polarity flip threshold (hysteresis)",
                },
                InstructionMeta {
                    opcode: 0x000D,
                    mnemonic: "ACCUMULATE_PRESSURE",
                    operand_pattern: OperandPattern::RegReg,
                    description: "Accumulate polarity pressure",
                },
            ],
        }
    }
}

impl Extension for TernaryExtension {
    fn ext_id(&self) -> u16 {
        0x0002
    }

    fn name(&self) -> &str {
        "tvmr.ternary"
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
            0x0000 => execute_ternary_matmul(operands, ctx),
            0x0001 => execute_ternary_batch_matmul(operands, ctx),
            0x0002 => execute_ternary_add_bias(operands, ctx),
            0x0003 => execute_dequantize(operands, ctx),
            0x0004 => execute_embed_lookup(operands, ctx),
            0x0005 => execute_embed_sequence(operands, ctx),
            0x0006 => execute_gate_update(operands, ctx),
            0x0007 => execute_quantize(operands, ctx),
            0x0008 => execute_pack_ternary(operands, ctx),
            0x0009 => execute_unpack_ternary(operands, ctx),
            0x000A => execute_apply_polarity(operands, ctx),
            0x000B => execute_apply_magnitude(operands, ctx),
            0x000C => execute_threshold_polarity(operands, ctx),
            0x000D => execute_accumulate_pressure(operands, ctx),
            _ => StepResult::Error(format!("tvmr.ternary: unknown opcode 0x{:04X}", opcode)),
        }
    }
}

// =============================================================================
// Standalone execute functions
// =============================================================================

fn execute_ternary_matmul(ops: [u8; 4], ctx: &mut ExecutionContext) -> StepResult {
    let dst = Register(ops[0]).index();
    let weights_reg = Register(ops[1]);
    let input_reg = Register(ops[2]);

    let weights_idx = weights_reg.index();
    let input_idx = input_reg.index();

    let cold = match &ctx.cold_regs[weights_idx] {
        Some(buf) => buf,
        None => return StepResult::Error(format!("C{} not allocated", weights_idx)),
    };

    let input = match &ctx.hot_regs[input_idx] {
        Some(buf) => &buf.data,
        None => return StepResult::Error(format!("H{} not allocated", input_idx)),
    };

    if cold.shape.len() < 2 {
        return StepResult::Error("Weights must be 2D for matmul".to_string());
    }

    let out_dim = cold.shape[0];
    let in_dim = cold.shape[1];

    let mut output = vec![0i64; out_dim];
    for o in 0..out_dim {
        let mut sum = 0i64;
        for i in 0..in_dim.min(input.len()) {
            let w_idx = o * in_dim + i;
            if w_idx >= cold.weights.len() {
                break;
            }
            let w = &cold.weights[w_idx];

            // Temperature-gated signal flow
            let temp = cold.temperature(w_idx);
            let input_magnitude = (input[i].unsigned_abs() as u8).min(255);

            if !temp.can_conduct(input_magnitude) {
                continue; // Signal below activation threshold
            }

            let conductance = temp.conductance() as i64;
            let effective = w.polarity as i64 * w.magnitude as i64;
            let gated = (effective * conductance) / 255;
            sum += gated * input[i] as i64;
        }
        output[o] = sum;
    }

    let result: Vec<i32> = output
        .iter()
        .map(|&v| v.clamp(i32::MIN as i64, i32::MAX as i64) as i32)
        .collect();

    ctx.hot_regs[dst] = Some(HotBuffer {
        data: result,
        shape: vec![out_dim],
    });

    StepResult::Continue
}

fn execute_ternary_batch_matmul(ops: [u8; 4], ctx: &mut ExecutionContext) -> StepResult {
    let dst = Register(ops[0]).index();
    let weights_idx = Register(ops[1]).index();
    let input_idx = Register(ops[2]).index();

    let cold = match &ctx.cold_regs[weights_idx] {
        Some(buf) => buf,
        None => return StepResult::Error(format!("C{} not allocated", weights_idx)),
    };

    let input = match &ctx.hot_regs[input_idx] {
        Some(buf) => buf,
        None => return StepResult::Error(format!("H{} not allocated", input_idx)),
    };

    if cold.shape.len() < 2 || input.shape.len() < 2 {
        return StepResult::Error("Both weights and input must be 2D for batch matmul".to_string());
    }

    let out_dim = cold.shape[0];
    let in_dim = cold.shape[1];
    let batch_size = input.shape[0];

    let mut result_data = vec![0i32; batch_size * out_dim];

    for b in 0..batch_size {
        for o in 0..out_dim {
            let mut sum = 0i64;
            for i in 0..in_dim {
                let input_val = input.data.get(b * in_dim + i).copied().unwrap_or(0);
                let w_idx = o * in_dim + i;
                if w_idx < cold.weights.len() {
                    let w = &cold.weights[w_idx];
                    let temp = cold.temperature(w_idx);
                    let input_magnitude = (input_val.unsigned_abs() as u8).min(255);

                    if temp.can_conduct(input_magnitude) {
                        let conductance = temp.conductance() as i64;
                        let effective = w.polarity as i64 * w.magnitude as i64;
                        let gated = (effective * conductance) / 255;
                        sum += gated * input_val as i64;
                    }
                }
            }
            result_data[b * out_dim + o] =
                sum.clamp(i32::MIN as i64, i32::MAX as i64) as i32;
        }
    }

    ctx.hot_regs[dst] = Some(HotBuffer {
        data: result_data,
        shape: vec![batch_size, out_dim],
    });

    StepResult::Continue
}

fn execute_ternary_add_bias(ops: [u8; 4], ctx: &mut ExecutionContext) -> StepResult {
    let dst = Register(ops[0]).index();
    let src = Register(ops[1]).index();
    let bias_idx = ops[2] as usize & 0x0F;

    let source = match &ctx.hot_regs[src] {
        Some(buf) => buf.clone(),
        None => return StepResult::Error(format!("H{} not allocated", src)),
    };

    let bias = match &ctx.cold_regs[bias_idx] {
        Some(buf) => buf,
        None => return StepResult::Error(format!("C{} not allocated", bias_idx)),
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

    StepResult::Continue
}

fn execute_dequantize(ops: [u8; 4], ctx: &mut ExecutionContext) -> StepResult {
    let _dst = Register(ops[0]).index();
    let src = Register(ops[1]).index();
    let shift = ((ops[2] as u32) << 8) | (ops[3] as u32);
    let shift_amt = if shift > 0 { shift } else { 8 };

    let source = match &ctx.hot_regs[src] {
        Some(buf) => buf,
        None => return StepResult::Error(format!("H{} not allocated", src)),
    };

    *ctx.output_buffer = source.data.iter().map(|&v| v >> shift_amt).collect();

    StepResult::Continue
}

fn execute_embed_lookup(ops: [u8; 4], ctx: &mut ExecutionContext) -> StepResult {
    let dst = Register(ops[0]).index();
    let table_idx = Register(ops[1]).index();
    let indices_idx = ops[2] as usize & 0x0F;

    let table = match &ctx.cold_regs[table_idx] {
        Some(buf) => buf,
        None => return StepResult::Error(format!("C{} not allocated", table_idx)),
    };

    let indices = match &ctx.hot_regs[indices_idx] {
        Some(buf) => &buf.data,
        None => return StepResult::Error(format!("H{} not allocated", indices_idx)),
    };

    let (num_embeddings, embedding_dim) = if table.shape.len() >= 2 {
        (table.shape[0], table.shape[1])
    } else if table.shape.len() == 1 {
        (1, table.shape[0])
    } else {
        return StepResult::Error("Embedding table must have shape".to_string());
    };

    let num_indices = indices.len();
    let mut output_data = vec![0i32; num_indices * embedding_dim];

    for (i, &idx_val) in indices.iter().enumerate() {
        let idx = idx_val.max(0) as usize;
        if idx < num_embeddings {
            let table_offset = idx * embedding_dim;
            for d in 0..embedding_dim {
                if table_offset + d < table.weights.len() {
                    let w = &table.weights[table_offset + d];
                    output_data[i * embedding_dim + d] = w.polarity as i32 * w.magnitude as i32;
                }
            }
        }
    }

    ctx.hot_regs[dst] = Some(HotBuffer {
        data: output_data,
        shape: vec![num_indices, embedding_dim],
    });

    StepResult::Continue
}

fn execute_embed_sequence(ops: [u8; 4], ctx: &mut ExecutionContext) -> StepResult {
    let dst = Register(ops[0]).index();
    let table_idx = Register(ops[1]).index();
    let count = ops[2] as usize;

    let table = match &ctx.cold_regs[table_idx] {
        Some(buf) => buf,
        None => return StepResult::Error(format!("C{} not allocated", table_idx)),
    };

    let (num_embeddings, embedding_dim) = if table.shape.len() >= 2 {
        (table.shape[0], table.shape[1])
    } else if table.shape.len() == 1 {
        (1, table.shape[0])
    } else {
        return StepResult::Error("Embedding table must have shape".to_string());
    };

    let actual_count = count.min(num_embeddings);
    let mut output_data = vec![0i32; actual_count * embedding_dim];

    for i in 0..actual_count {
        let table_offset = i * embedding_dim;
        for d in 0..embedding_dim {
            if table_offset + d < table.weights.len() {
                let w = &table.weights[table_offset + d];
                output_data[i * embedding_dim + d] = w.polarity as i32 * w.magnitude as i32;
            }
        }
    }

    ctx.hot_regs[dst] = Some(HotBuffer {
        data: output_data,
        shape: vec![actual_count, embedding_dim],
    });

    StepResult::Continue
}

// =============================================================================
// QUANTIZE [dst:1][src:1][_:2]
// Convert i32 activations to ternary Signal representation.
// Polarity = sign(value), Magnitude = clamp(|value| >> 8, 0, 255).
// Writes cold register with quantized weights.
// =============================================================================
fn execute_quantize(ops: [u8; 4], ctx: &mut ExecutionContext) -> StepResult {
    let dst_idx = Register(ops[0]).index();
    let src_idx = Register(ops[1]).index();

    let source = match &ctx.hot_regs[src_idx] {
        Some(buf) => &buf.data,
        None => return StepResult::Error(format!("H{} not allocated", src_idx)),
    };

    let signals: Vec<Signal> = source
        .iter()
        .map(|&v| {
            let polarity = if v > 0 { 1i8 } else if v < 0 { -1i8 } else { 0i8 };
            let magnitude = ((v.unsigned_abs() >> 8) as u8).min(255);
            Signal { polarity, magnitude }
        })
        .collect();

    let len = signals.len();
    let mut cold = crate::vm::interpreter::ColdBuffer::new(vec![len]);
    cold.weights = signals;
    ctx.cold_regs[dst_idx] = Some(cold);

    StepResult::Continue
}

// =============================================================================
// PACK_TERNARY [dst:1][src:1][_:2]
// Pack ternary signals into 2-bit representation: 4 signals per byte.
// Encoding: -1 → 0b10, 0 → 0b00, +1 → 0b01
// Output is i32 values where each holds 4 packed signals (bits 0-7).
// =============================================================================
fn execute_pack_ternary(ops: [u8; 4], ctx: &mut ExecutionContext) -> StepResult {
    let dst_idx = Register(ops[0]).index();
    let src_idx = Register(ops[1]).index();

    let cold = match &ctx.cold_regs[src_idx] {
        Some(buf) => buf,
        None => return StepResult::Error(format!("C{} not allocated", src_idx)),
    };

    let packed_count = (cold.weights.len() + 3) / 4;
    let mut packed = vec![0i32; packed_count];

    for (i, signal) in cold.weights.iter().enumerate() {
        let pack_idx = i / 4;
        let bit_offset = (i % 4) * 2;
        let encoded: u8 = match signal.polarity {
            p if p < 0 => 0b10,
            p if p > 0 => 0b01,
            _ => 0b00,
        };
        packed[pack_idx] |= (encoded as i32) << bit_offset;
    }

    ctx.hot_regs[dst_idx] = Some(HotBuffer {
        data: packed,
        shape: vec![packed_count],
    });

    StepResult::Continue
}

// =============================================================================
// UNPACK_TERNARY [dst:1][src:1][_:2]
// Unpack 2-bit ternary to full Signal representation.
// Input: packed i32 values. Output: cold register with Signal weights.
// =============================================================================
fn execute_unpack_ternary(ops: [u8; 4], ctx: &mut ExecutionContext) -> StepResult {
    let dst_idx = Register(ops[0]).index();
    let src_idx = Register(ops[1]).index();

    let packed = match &ctx.hot_regs[src_idx] {
        Some(buf) => &buf.data,
        None => return StepResult::Error(format!("H{} not allocated", src_idx)),
    };

    let mut signals = Vec::with_capacity(packed.len() * 4);
    for &word in packed.iter() {
        for bit_offset in (0..8).step_by(2) {
            let encoded = ((word >> bit_offset) & 0b11) as u8;
            let polarity = match encoded {
                0b01 => 1i8,
                0b10 => -1i8,
                _ => 0i8,
            };
            signals.push(Signal {
                polarity,
                magnitude: if polarity != 0 { 128 } else { 0 },
            });
        }
    }

    let len = signals.len();
    let mut cold = crate::vm::interpreter::ColdBuffer::new(vec![len]);
    cold.weights = signals;
    ctx.cold_regs[dst_idx] = Some(cold);

    StepResult::Continue
}

// =============================================================================
// APPLY_POLARITY [cold_dst:1][hot_src:1][_:2]
// Apply polarity updates from hot register to cold weights.
// hot_src[i] > 0 → polarity = +1, < 0 → polarity = -1, == 0 → no change.
// =============================================================================
fn execute_apply_polarity(ops: [u8; 4], ctx: &mut ExecutionContext) -> StepResult {
    let cold_idx = Register(ops[0]).index();
    let hot_idx = Register(ops[1]).index();

    let updates = match &ctx.hot_regs[hot_idx] {
        Some(buf) => buf.data.clone(),
        None => return StepResult::Error(format!("H{} not allocated", hot_idx)),
    };

    let cold = match ctx.cold_regs[cold_idx].as_mut() {
        Some(buf) => buf,
        None => return StepResult::Error(format!("C{} not allocated", cold_idx)),
    };

    for (i, &update) in updates.iter().enumerate() {
        if i >= cold.weights.len() {
            break;
        }
        if update > 0 {
            cold.weights[i].polarity = 1;
        } else if update < 0 {
            cold.weights[i].polarity = -1;
        }
        // update == 0 → no change
    }

    StepResult::Continue
}

// =============================================================================
// APPLY_MAGNITUDE [cold_dst:1][hot_src:1][_:2]
// Apply magnitude updates from hot register to cold weights.
// hot_src[i] is the new magnitude (clamped to 0-255).
// =============================================================================
fn execute_apply_magnitude(ops: [u8; 4], ctx: &mut ExecutionContext) -> StepResult {
    let cold_idx = Register(ops[0]).index();
    let hot_idx = Register(ops[1]).index();

    let updates = match &ctx.hot_regs[hot_idx] {
        Some(buf) => buf.data.clone(),
        None => return StepResult::Error(format!("H{} not allocated", hot_idx)),
    };

    let cold = match ctx.cold_regs[cold_idx].as_mut() {
        Some(buf) => buf,
        None => return StepResult::Error(format!("C{} not allocated", cold_idx)),
    };

    for (i, &update) in updates.iter().enumerate() {
        if i >= cold.weights.len() {
            break;
        }
        cold.weights[i].magnitude = update.clamp(0, 255) as u8;
    }

    StepResult::Continue
}

// =============================================================================
// THRESHOLD_POLARITY [cold_reg:1][pressure_src:1][_:2]
// Check if accumulated pressure exceeds hysteresis threshold for polarity flip.
// If |pressure[i]| > magnitude * 2, flip polarity and reset magnitude.
// This implements biological hysteresis — you need sustained opposing evidence
// to flip a synapse's sign.
// =============================================================================
fn execute_threshold_polarity(ops: [u8; 4], ctx: &mut ExecutionContext) -> StepResult {
    let cold_idx = Register(ops[0]).index();
    let pressure_idx = Register(ops[1]).index();

    let pressure = match &ctx.hot_regs[pressure_idx] {
        Some(buf) => buf.data.clone(),
        None => return StepResult::Error(format!("H{} not allocated", pressure_idx)),
    };

    let cold = match ctx.cold_regs[cold_idx].as_mut() {
        Some(buf) => buf,
        None => return StepResult::Error(format!("C{} not allocated", cold_idx)),
    };

    for (i, &p) in pressure.iter().enumerate() {
        if i >= cold.weights.len() {
            break;
        }
        let threshold = cold.weights[i].magnitude as i32 * 2;
        if p.abs() > threshold {
            // Flip polarity
            let new_polarity = if p > 0 { 1i8 } else { -1i8 };
            if cold.weights[i].polarity != new_polarity {
                cold.weights[i].polarity = new_polarity;
                // Reset magnitude to baseline after flip (new synapse)
                cold.weights[i].magnitude = 30;
            }
        }
    }

    StepResult::Continue
}

// =============================================================================
// ACCUMULATE_PRESSURE [dst_hot:1][src_hot:1][_:2]
// Accumulate polarity pressure: dst[i] += src[i]
// Used to build up evidence for polarity flips over time.
// =============================================================================
fn execute_accumulate_pressure(ops: [u8; 4], ctx: &mut ExecutionContext) -> StepResult {
    let dst_idx = Register(ops[0]).index();
    let src_idx = Register(ops[1]).index();

    let source = match &ctx.hot_regs[src_idx] {
        Some(buf) => buf.data.clone(),
        None => return StepResult::Error(format!("H{} not allocated", src_idx)),
    };

    let dst = ctx.hot_regs[dst_idx].get_or_insert_with(|| HotBuffer {
        data: vec![0i32; source.len()],
        shape: vec![source.len()],
    });

    let len = dst.data.len().min(source.len());
    for i in 0..len {
        dst.data[i] = dst.data[i].saturating_add(source[i]);
    }

    StepResult::Continue
}

fn execute_gate_update(ops: [u8; 4], ctx: &mut ExecutionContext) -> StepResult {
    let dst = Register(ops[0]).index();
    let gate_idx = Register(ops[1]).index();
    let update_idx = ops[2] as usize & 0x0F;
    let state_idx = ops[3] as usize & 0x0F;

    let gate = match &ctx.hot_regs[gate_idx] {
        Some(buf) => &buf.data,
        None => return StepResult::Error(format!("H{} not allocated", gate_idx)),
    };

    let update = match &ctx.hot_regs[update_idx] {
        Some(buf) => &buf.data,
        None => return StepResult::Error(format!("H{} not allocated", update_idx)),
    };

    let state = match &ctx.hot_regs[state_idx] {
        Some(buf) => &buf.data,
        None => return StepResult::Error(format!("H{} not allocated", state_idx)),
    };

    let len = gate.len().min(update.len()).min(state.len());
    let mut result = Vec::with_capacity(len);

    for i in 0..len {
        let g = gate[i].clamp(0, 255) as i64;
        let u = update[i] as i64;
        let s = state[i] as i64;
        let val = (g * u + (255 - g) * s) / 255;
        result.push(val as i32);
    }

    ctx.hot_regs[dst] = Some(HotBuffer {
        data: result,
        shape: vec![len],
    });

    StepResult::Continue
}
