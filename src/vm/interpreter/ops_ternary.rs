//! Ternary/tensor operation implementations for the Interpreter

use super::{HotBuffer, Instruction, Interpreter, StepResult};

impl Interpreter {
    pub(super) fn execute_ternary_add_bias(&mut self, instr: Instruction) -> StepResult {
        let src_idx = instr.source.index();
        let bias_idx = instr.aux as usize & 0x0F;
        let dst_idx = instr.target.index();

        let src = match &self.hot_regs[src_idx] {
            Some(buf) => buf.clone(),
            None => return StepResult::Error("Source not allocated".to_string()),
        };

        let bias = match &self.cold_regs[bias_idx] {
            Some(buf) => buf,
            None => return StepResult::Error("Bias not allocated".to_string()),
        };

        let mut result = src.data.clone();
        for (i, val) in result.iter_mut().enumerate() {
            if i < bias.weights.len() {
                let b = &bias.weights[i];
                let bias_val = b.polarity as i64 * b.magnitude as i64 * 256;
                *val = (*val as i64 + bias_val).clamp(i32::MIN as i64, i32::MAX as i64) as i32;
            }
        }

        self.hot_regs[dst_idx] = Some(HotBuffer {
            data: result,
            shape: src.shape.clone(),
        });

        StepResult::Continue
    }

    pub(super) fn execute_embed_lookup(&mut self, instr: Instruction) -> StepResult {
        let table_idx = instr.source.index();
        let indices_idx = instr.aux as usize & 0x0F;
        let output_idx = instr.target.index();

        let table = match &self.cold_regs[table_idx] {
            Some(buf) => buf,
            None => return StepResult::Error(format!("Embedding table C{} not allocated", table_idx)),
        };

        let indices = match &self.hot_regs[indices_idx] {
            Some(buf) => &buf.data,
            None => return StepResult::Error(format!("Indices H{} not allocated", indices_idx)),
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

        self.hot_regs[output_idx] = Some(HotBuffer {
            data: output_data,
            shape: vec![num_indices, embedding_dim],
        });

        StepResult::Continue
    }

    pub(super) fn execute_reduce_avg(&mut self, instr: Instruction) -> StepResult {
        let src_idx = instr.source.index();
        let dst_idx = instr.target.index();
        let start = instr.aux as usize;
        let count = instr.modifier[0] as usize;

        let src = match &self.hot_regs[src_idx] {
            Some(buf) => &buf.data,
            None => return StepResult::Error(format!("Source H{} not allocated", src_idx)),
        };

        if count == 0 {
            self.hot_regs[dst_idx] = Some(HotBuffer {
                data: vec![0],
                shape: vec![1],
            });
            return StepResult::Continue;
        }

        let end = (start + count).min(src.len());
        let actual_count = end.saturating_sub(start);

        let sum: i32 = src.get(start..end)
            .map(|slice| slice.iter().sum())
            .unwrap_or(0);

        let avg = if actual_count > 0 {
            sum / actual_count as i32
        } else {
            0
        };

        self.hot_regs[dst_idx] = Some(HotBuffer {
            data: vec![avg],
            shape: vec![1],
        });

        StepResult::Continue
    }

    pub(super) fn execute_slice(&mut self, instr: Instruction) -> StepResult {
        let src_idx = instr.source.index();
        let dst_idx = instr.target.index();
        let start = instr.aux as usize;
        let len = instr.modifier[0] as usize;

        let src = match &self.hot_regs[src_idx] {
            Some(buf) => &buf.data,
            None => return StepResult::Error(format!("Source H{} not allocated", src_idx)),
        };

        let end = (start + len).min(src.len());
        let result: Vec<i32> = src.get(start..end).map(|s| s.to_vec()).unwrap_or_default();
        let result_len = result.len();

        self.hot_regs[dst_idx] = Some(HotBuffer {
            data: result,
            shape: vec![result_len],
        });

        StepResult::Continue
    }

    pub(super) fn execute_argmax(&mut self, instr: Instruction) -> StepResult {
        let src_idx = instr.source.index();
        let dst_idx = instr.target.index();

        let src = match &self.hot_regs[src_idx] {
            Some(buf) => &buf.data,
            None => return StepResult::Error(format!("Source H{} not allocated", src_idx)),
        };

        let argmax = src
            .iter()
            .enumerate()
            .max_by_key(|(_, &v)| v)
            .map(|(i, _)| i as i32)
            .unwrap_or(0);

        self.hot_regs[dst_idx] = Some(HotBuffer {
            data: vec![argmax],
            shape: vec![1],
        });

        StepResult::Continue
    }

    pub(super) fn execute_concat(&mut self, instr: Instruction) -> StepResult {
        let src_idx = instr.source.index();
        let other_idx = instr.aux as usize & 0x0F;
        let dst_idx = instr.target.index();

        let src = match &self.hot_regs[src_idx] {
            Some(buf) => buf.clone(),
            None => return StepResult::Error(format!("Source H{} not allocated", src_idx)),
        };

        let other = match &self.hot_regs[other_idx] {
            Some(buf) => buf,
            None => return StepResult::Error(format!("Other H{} not allocated", other_idx)),
        };

        let mut result = src.data.clone();
        result.extend_from_slice(&other.data);
        let result_len = result.len();

        self.hot_regs[dst_idx] = Some(HotBuffer {
            data: result,
            shape: vec![result_len],
        });

        StepResult::Continue
    }

    pub(super) fn execute_squeeze(&mut self, instr: Instruction) -> StepResult {
        let src_idx = instr.source.index();
        let dst_idx = instr.target.index();

        if let Some(buf) = self.hot_regs[src_idx].clone() {
            self.hot_regs[dst_idx] = Some(buf);
        }

        StepResult::Continue
    }

    pub(super) fn execute_unsqueeze(&mut self, instr: Instruction) -> StepResult {
        let src_idx = instr.source.index();
        let dst_idx = instr.target.index();

        if let Some(buf) = self.hot_regs[src_idx].clone() {
            self.hot_regs[dst_idx] = Some(buf);
        }

        StepResult::Continue
    }

    pub(super) fn execute_transpose(&mut self, instr: Instruction) -> StepResult {
        let src_idx = instr.source.index();
        let dst_idx = instr.target.index();

        if let Some(buf) = self.hot_regs[src_idx].clone() {
            self.hot_regs[dst_idx] = Some(buf);
        }

        StepResult::Continue
    }

    pub(super) fn execute_gate_update(&mut self, instr: Instruction) -> StepResult {
        let gate_idx = instr.source.index();
        let update_idx = instr.aux as usize & 0x0F;
        let state_idx = instr.modifier[0] as usize & 0x0F;
        let dst_idx = instr.target.index();

        let gate = match &self.hot_regs[gate_idx] {
            Some(buf) => &buf.data,
            None => return StepResult::Error(format!("Gate H{} not allocated", gate_idx)),
        };

        let update = match &self.hot_regs[update_idx] {
            Some(buf) => &buf.data,
            None => return StepResult::Error(format!("Update H{} not allocated", update_idx)),
        };

        let state = match &self.hot_regs[state_idx] {
            Some(buf) => &buf.data,
            None => return StepResult::Error(format!("State H{} not allocated", state_idx)),
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

        self.hot_regs[dst_idx] = Some(HotBuffer {
            data: result,
            shape: vec![len],
        });

        StepResult::Continue
    }

    pub(super) fn execute_dequantize(&mut self, instr: Instruction) -> StepResult {
        let src_idx = instr.source.index();
        let shift = instr.scale() as u32;

        let src = match &self.hot_regs[src_idx] {
            Some(buf) => buf,
            None => return StepResult::Error("Source not allocated".to_string()),
        };

        let shift_amt = if shift > 0 { shift } else { 8 };
        self.output_buffer = src.data.iter().map(|&v| v >> shift_amt).collect();

        StepResult::Continue
    }
}
