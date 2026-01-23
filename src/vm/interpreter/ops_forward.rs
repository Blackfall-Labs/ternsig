//! Forward operation implementations for the Interpreter

use super::{ColdBuffer, HotBuffer, Instruction, Interpreter, Register, StepResult};

impl Interpreter {
    pub(super) fn execute_ternary_matmul(&mut self, instr: Instruction) -> StepResult {
        let weights_idx = instr.source.index();
        let input_idx = instr.aux as usize & 0x0F;
        let output_idx = instr.target.index();

        let weights = match &self.cold_regs[weights_idx] {
            Some(buf) => buf,
            None => return StepResult::Error(format!("Cold register C{} not allocated", weights_idx)),
        };

        let input = match &self.hot_regs[input_idx] {
            Some(buf) => buf,
            None => return StepResult::Error(format!("Hot register H{} not allocated", input_idx)),
        };

        let (out_dim, in_dim) = if weights.shape.len() >= 2 {
            (weights.shape[0], weights.shape[1])
        } else {
            return StepResult::Error("Weights must be 2D".to_string());
        };

        let mut output_data = vec![0i64; out_dim];
        for o in 0..out_dim {
            let mut sum = 0i64;
            for i in 0..in_dim.min(input.data.len()) {
                let w = &weights.weights[o * in_dim + i];
                let effective = w.polarity as i64 * w.magnitude as i64;
                sum += effective * input.data[i] as i64;
            }
            output_data[o] = sum;
        }

        let output_i32: Vec<i32> = output_data
            .iter()
            .map(|&v| v.clamp(i32::MIN as i64, i32::MAX as i64) as i32)
            .collect();

        self.hot_regs[output_idx] = Some(HotBuffer {
            data: output_i32,
            shape: vec![out_dim],
        });

        StepResult::Continue
    }

    pub(super) fn execute_add(&mut self, instr: Instruction) -> StepResult {
        let src_idx = instr.source.index();
        let other_idx = instr.aux as usize & 0x0F;
        let dst_idx = instr.target.index();

        if instr.source.is_hot() && Register(instr.aux).is_cold() {
            let src = match &self.hot_regs[src_idx] {
                Some(buf) => buf.clone(),
                None => return StepResult::Error("Source not allocated".to_string()),
            };

            let bias = match &self.cold_regs[other_idx] {
                Some(buf) => buf,
                None => return StepResult::Error("Bias not allocated".to_string()),
            };

            let mut result = src.data.clone();
            for (i, val) in result.iter_mut().enumerate() {
                if i < bias.weights.len() {
                    let b = &bias.weights[i];
                    let bias_val = b.polarity as i64 * b.magnitude as i64 * 255;
                    *val = (*val as i64 + bias_val).clamp(i32::MIN as i64, i32::MAX as i64) as i32;
                }
            }

            self.hot_regs[dst_idx] = Some(HotBuffer {
                data: result,
                shape: src.shape.clone(),
            });
        } else if instr.source.is_hot() && Register(instr.aux).is_hot() {
            let src = match &self.hot_regs[src_idx] {
                Some(buf) => buf.clone(),
                None => return StepResult::Error("Source not allocated".to_string()),
            };

            let other = match &self.hot_regs[other_idx] {
                Some(buf) => buf,
                None => return StepResult::Error("Other not allocated".to_string()),
            };

            let mut result = src.data.clone();
            for (i, val) in result.iter_mut().enumerate() {
                if i < other.data.len() {
                    *val = val.saturating_add(other.data[i]);
                }
            }

            self.hot_regs[dst_idx] = Some(HotBuffer {
                data: result,
                shape: src.shape.clone(),
            });
        }

        StepResult::Continue
    }

    pub(super) fn execute_sub(&mut self, instr: Instruction) -> StepResult {
        let src_idx = instr.source.index();
        let other_idx = instr.aux as usize & 0x0F;
        let dst_idx = instr.target.index();

        let src = match &self.hot_regs[src_idx] {
            Some(buf) => buf.clone(),
            None => return StepResult::Error("Source not allocated".to_string()),
        };

        let other = match &self.hot_regs[other_idx] {
            Some(buf) => buf,
            None => return StepResult::Error("Other not allocated".to_string()),
        };

        let result: Vec<i32> = src
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(&a, &b)| a.saturating_sub(b))
            .collect();

        self.hot_regs[dst_idx] = Some(HotBuffer {
            data: result,
            shape: src.shape.clone(),
        });

        StepResult::Continue
    }

    pub(super) fn execute_mul(&mut self, instr: Instruction) -> StepResult {
        let src_idx = instr.source.index();
        let other_idx = instr.aux as usize & 0x0F;
        let dst_idx = instr.target.index();

        let src = match &self.hot_regs[src_idx] {
            Some(buf) => buf.clone(),
            None => return StepResult::Error("Source not allocated".to_string()),
        };

        let other = match &self.hot_regs[other_idx] {
            Some(buf) => buf,
            None => return StepResult::Error("Other not allocated".to_string()),
        };

        let result: Vec<i32> = src
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(&a, &b)| ((a as i64 * b as i64) >> 8) as i32)
            .collect();

        self.hot_regs[dst_idx] = Some(HotBuffer {
            data: result,
            shape: src.shape.clone(),
        });

        StepResult::Continue
    }

    pub(super) fn execute_relu(&mut self, instr: Instruction) -> StepResult {
        let src_idx = instr.source.index();
        let dst_idx = instr.target.index();

        let src = match &self.hot_regs[src_idx] {
            Some(buf) => buf.clone(),
            None => return StepResult::Error("Source not allocated".to_string()),
        };

        let result: Vec<i32> = src.data.iter().map(|&v| v.max(0)).collect();

        self.hot_regs[dst_idx] = Some(HotBuffer {
            data: result,
            shape: src.shape.clone(),
        });

        StepResult::Continue
    }

    pub(super) fn execute_sigmoid(&mut self, instr: Instruction) -> StepResult {
        let src_idx = instr.source.index();
        let dst_idx = instr.target.index();
        let gain = if instr.modifier[0] > 0 {
            instr.modifier[0] as i32
        } else {
            64
        };

        let src = match &self.hot_regs[src_idx] {
            Some(buf) => buf.clone(),
            None => return StepResult::Error("Source not allocated".to_string()),
        };

        let result: Vec<i32> = src
            .data
            .iter()
            .map(|&v| {
                let scaled = (v as i64 * gain as i64) >> 10;
                (scaled + 128).clamp(0, 255) as i32
            })
            .collect();

        self.hot_regs[dst_idx] = Some(HotBuffer {
            data: result,
            shape: src.shape.clone(),
        });

        StepResult::Continue
    }

    pub(super) fn execute_gelu(&mut self, instr: Instruction) -> StepResult {
        let src_idx = instr.source.index();
        let dst_idx = instr.target.index();

        let src = match &self.hot_regs[src_idx] {
            Some(buf) => buf.clone(),
            None => return StepResult::Error("Source not allocated".to_string()),
        };

        // Integer GELU: GELU(x) ≈ x * sigmoid(1.702 * x)
        let result: Vec<i32> = src
            .data
            .iter()
            .map(|&v| {
                let scaled = (v as i64 * 435) >> 8;
                let sig = ((scaled >> 10) + 128).clamp(0, 255) as i64;
                ((v as i64 * sig) >> 8) as i32
            })
            .collect();

        self.hot_regs[dst_idx] = Some(HotBuffer {
            data: result,
            shape: src.shape.clone(),
        });

        StepResult::Continue
    }

    pub(super) fn execute_softmax(&mut self, instr: Instruction) -> StepResult {
        let src_idx = instr.source.index();
        let dst_idx = instr.target.index();

        let src = match &self.hot_regs[src_idx] {
            Some(buf) => buf.clone(),
            None => return StepResult::Error("Source not allocated".to_string()),
        };

        let max_val = src.data.iter().copied().max().unwrap_or(0);

        let exp_vals: Vec<i64> = src
            .data
            .iter()
            .map(|&v| {
                let shifted = v - max_val;
                if shifted >= 0 {
                    256i64
                } else {
                    (256 + shifted.max(-256) as i64).max(1)
                }
            })
            .collect();

        let sum: i64 = exp_vals.iter().sum();

        let result: Vec<i32> = if sum > 0 {
            exp_vals
                .iter()
                .map(|&e| ((e * 255) / sum) as i32)
                .collect()
        } else {
            vec![0; src.data.len()]
        };

        self.hot_regs[dst_idx] = Some(HotBuffer {
            data: result,
            shape: src.shape.clone(),
        });

        StepResult::Continue
    }

    pub(super) fn execute_shift(&mut self, instr: Instruction) -> StepResult {
        let src_idx = instr.source.index();
        let dst_idx = instr.target.index();
        let shift_amount = instr.aux as u32;

        let src = match &self.hot_regs[src_idx] {
            Some(buf) => buf.clone(),
            None => return StepResult::Error("Source not allocated".to_string()),
        };

        let result: Vec<i32> = src.data.iter().map(|&v| v >> shift_amount).collect();

        self.hot_regs[dst_idx] = Some(HotBuffer {
            data: result,
            shape: src.shape.clone(),
        });

        StepResult::Continue
    }

    pub(super) fn execute_cmp_gt(&mut self, instr: Instruction) -> StepResult {
        let src_idx = instr.source.index();
        let other_idx = instr.aux as usize & 0x0F;
        let dst_idx = instr.target.index();

        let src = match &self.hot_regs[src_idx] {
            Some(buf) => buf.clone(),
            None => return StepResult::Error("Source not allocated".to_string()),
        };

        let other = match &self.hot_regs[other_idx] {
            Some(buf) => buf,
            None => return StepResult::Error("Other not allocated".to_string()),
        };

        let result: Vec<i32> = src
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(&a, &b)| if a > b { 255 } else { 0 })
            .collect();

        self.hot_regs[dst_idx] = Some(HotBuffer {
            data: result,
            shape: src.shape.clone(),
        });

        StepResult::Continue
    }

    pub(super) fn execute_max_reduce(&mut self, instr: Instruction) -> StepResult {
        let src_idx = instr.source.index();
        let dst_idx = instr.target.index();

        let src = match &self.hot_regs[src_idx] {
            Some(buf) => buf.clone(),
            None => return StepResult::Error("Source not allocated".to_string()),
        };

        let max_val = src.data.iter().cloned().max().unwrap_or(0);

        self.hot_regs[dst_idx] = Some(HotBuffer {
            data: vec![max_val],
            shape: vec![1],
        });

        StepResult::Continue
    }
}
