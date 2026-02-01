//! Forward operation implementations for the Interpreter

use super::{ColdBuffer, HotBuffer, Instruction, Interpreter, Register, StepResult};

impl Interpreter {
    /// Temperature-gated ternary matrix multiply
    ///
    /// Signal flow is gated by connection temperature:
    /// - Hot connections: full conductance, no barrier
    /// - Cold connections: reduced conductance, need high input intensity
    ///
    /// This creates the "flow through heated paths" behavior:
    /// - Recent/reinforced paths conduct easily
    /// - Dormant paths need strong signals to activate
    /// - Old memories can still be reached with sufficient intensity
    pub(super) fn execute_ternary_matmul(&mut self, instr: Instruction) -> StepResult {
        let weights_idx = instr.source().index();
        let input_idx = instr.aux() as usize & 0x0F;
        let output_idx = instr.target().index();

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

        // Check if this buffer has per-signal temperatures
        let has_temperatures = weights.temperatures.is_some();

        let mut output_data = vec![0i64; out_dim];
        for o in 0..out_dim {
            let mut sum = 0i64;
            for i in 0..in_dim.min(input.data.len()) {
                let weight_idx = o * in_dim + i;
                let w = &weights.weights[weight_idx];
                let input_val = input.data[i];

                // Get input intensity (absolute value, clamped to u8)
                let input_intensity = (input_val.unsigned_abs().min(255)) as u8;

                // Temperature-gated flow
                if has_temperatures {
                    let temp = weights.temperature(weight_idx);

                    // Check activation threshold (cold paths need strong input)
                    if temp.can_conduct(input_intensity) {
                        let effective = w.polarity as i64 * w.magnitude as i64;
                        let conductance = temp.conductance() as i64;

                        // Scale by conductance: hot = full, cold = reduced
                        sum += (effective * input_val as i64 * conductance) / 255;
                    }
                    // Below threshold = signal doesn't pass through this connection
                } else {
                    // No temperatures = all hot (original behavior)
                    let effective = w.polarity as i64 * w.magnitude as i64;
                    sum += effective * input_val as i64;
                }
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
        let src_idx = instr.source().index();
        let other_idx = instr.aux() as usize & 0x0F;
        let dst_idx = instr.target().index();

        if instr.source().is_hot() && Register(instr.aux()).is_cold() {
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
        } else if instr.source().is_hot() && Register(instr.aux()).is_hot() {
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
        let src_idx = instr.source().index();
        let other_idx = instr.aux() as usize & 0x0F;
        let dst_idx = instr.target().index();

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
        let src_idx = instr.source().index();
        let other_idx = instr.aux() as usize & 0x0F;
        let dst_idx = instr.target().index();

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
        let src_idx = instr.source().index();
        let dst_idx = instr.target().index();

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
        let src_idx = instr.source().index();
        let dst_idx = instr.target().index();
        let gain = if instr.modifier()[0] > 0 {
            instr.modifier()[0] as i32
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
        let src_idx = instr.source().index();
        let dst_idx = instr.target().index();

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

    pub(super) fn execute_tanh(&mut self, instr: Instruction) -> StepResult {
        let src_idx = instr.source().index();
        let dst_idx = instr.target().index();

        let src = match &self.hot_regs[src_idx] {
            Some(buf) => buf.clone(),
            None => return StepResult::Error("Source not allocated".to_string()),
        };

        // Integer tanh: tanh(x) ≈ x / (1 + |x|/k) for soft saturation
        // Input is centered at 128 (0 maps to -128, 128 maps to 0, 255 maps to +127)
        // Output is also centered at 128
        let result: Vec<i32> = src
            .data
            .iter()
            .map(|&v| {
                // Clamp to expected [0, 255] range before tanh calculation
                let clamped = v.clamp(0, 255);
                // Center around 0 for tanh calculation
                let centered = clamped - 128;
                // Apply soft saturation: tanh ≈ x / (1 + |x|/64)
                // Use i64 to prevent overflow with out-of-range inputs
                let abs_val = centered.abs() as i64;
                let denom = (64 + (abs_val >> 1)).max(1);
                let tanh_val = ((centered as i64) * 64) / denom;
                // Map back to [0, 255] centered at 128
                (tanh_val as i32 + 128).clamp(0, 255)
            })
            .collect();

        self.hot_regs[dst_idx] = Some(HotBuffer {
            data: result,
            shape: src.shape.clone(),
        });

        StepResult::Continue
    }

    pub(super) fn execute_softmax(&mut self, instr: Instruction) -> StepResult {
        let src_idx = instr.source().index();
        let dst_idx = instr.target().index();

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
        let src_idx = instr.source().index();
        let dst_idx = instr.target().index();
        let shift_amount = instr.aux() as u32;

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
        let src_idx = instr.source().index();
        let other_idx = instr.aux() as usize & 0x0F;
        let dst_idx = instr.target().index();

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
        let src_idx = instr.source().index();
        let dst_idx = instr.target().index();

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

    /// Clamp values to [min, max] range
    ///
    /// Format: clamp target, source, min, max
    /// - aux = min value
    /// - modifier[0] = max value
    pub(super) fn execute_clamp(&mut self, instr: Instruction) -> StepResult {
        let src_idx = instr.source().index();
        let dst_idx = instr.target().index();
        let min_val = instr.aux() as i32;
        let max_val = instr.modifier()[0] as i32;

        let src = match &self.hot_regs[src_idx] {
            Some(buf) => buf.clone(),
            None => return StepResult::Error("Source not allocated".to_string()),
        };

        let clamped: Vec<i32> = src
            .data
            .iter()
            .map(|&v| v.clamp(min_val, max_val))
            .collect();

        self.hot_regs[dst_idx] = Some(HotBuffer {
            data: clamped,
            shape: src.shape.clone(),
        });

        StepResult::Continue
    }

    /// Temperature-gated batch matrix multiply
    ///
    /// Applies same weight matrix to each row of input batch with temperature gating.
    /// Input shape: [batch_size, in_dim] or flat [batch_size * in_dim]
    /// Weights shape: [out_dim, in_dim]
    /// Output shape: [batch_size, out_dim]
    pub(super) fn execute_ternary_batch_matmul(&mut self, instr: Instruction) -> StepResult {
        let weights_idx = instr.source().index();
        let input_idx = instr.aux() as usize & 0x0F;
        let output_idx = instr.target().index();

        let weights = match &self.cold_regs[weights_idx] {
            Some(buf) => buf,
            None => {
                return StepResult::Error(format!(
                    "Cold register C{} not allocated",
                    weights_idx
                ))
            }
        };

        let input = match &self.hot_regs[input_idx] {
            Some(buf) => buf,
            None => {
                return StepResult::Error(format!("Hot register H{} not allocated", input_idx))
            }
        };

        let (out_dim, in_dim) = if weights.shape.len() >= 2 {
            (weights.shape[0], weights.shape[1])
        } else {
            return StepResult::Error("Weights must be 2D".to_string());
        };

        // Determine batch size from input shape
        let batch_size = if input.shape.len() >= 2 {
            input.shape[0]
        } else if input.data.len() % in_dim == 0 {
            input.data.len() / in_dim
        } else {
            return StepResult::Error(format!(
                "Input size {} not divisible by in_dim {}",
                input.data.len(),
                in_dim
            ));
        };

        // Check if this buffer has per-signal temperatures
        let has_temperatures = weights.temperatures.is_some();

        // Apply matmul to each batch row
        let mut output_data = Vec::with_capacity(batch_size * out_dim);

        for b in 0..batch_size {
            let row_start = b * in_dim;

            for o in 0..out_dim {
                let mut sum = 0i64;
                for i in 0..in_dim {
                    let input_val = input.data.get(row_start + i).copied().unwrap_or(0);
                    let weight_idx = o * in_dim + i;
                    let w = &weights.weights[weight_idx];

                    // Get input intensity
                    let input_intensity = (input_val.unsigned_abs().min(255)) as u8;

                    // Temperature-gated flow
                    if has_temperatures {
                        let temp = weights.temperature(weight_idx);

                        if temp.can_conduct(input_intensity) {
                            let effective = w.polarity as i64 * w.magnitude as i64;
                            let conductance = temp.conductance() as i64;
                            sum += (effective * input_val as i64 * conductance) / 255;
                        }
                    } else {
                        let effective = w.polarity as i64 * w.magnitude as i64;
                        sum += effective * input_val as i64;
                    }
                }
                output_data.push(sum.clamp(i32::MIN as i64, i32::MAX as i64) as i32);
            }
        }

        self.hot_regs[output_idx] = Some(HotBuffer {
            data: output_data,
            shape: vec![batch_size, out_dim],
        });

        StepResult::Continue
    }
}
