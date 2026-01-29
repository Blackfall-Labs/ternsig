//! Learning Extension (ExtID: 0x0004)
//!
//! All learning operations: mastery learning, CHL, babble, eligibility.
//! Learning is dopamine-gated — no learning without neuromodulator support.

use crate::vm::extension::{
    DomainOp, ExecutionContext, Extension, InstructionMeta, OperandPattern, StepResult,
};
use crate::vm::interpreter::{ColdBuffer, HotBuffer};
use crate::vm::register::Register;
use std::sync::Mutex;

/// CHL (Contrastive Hebbian Learning) phase state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ChlPhase {
    Idle,
    Free,
    Clamped,
}

/// Learning extension — weight update and exploration operations.
pub struct LearningExtension {
    instructions: Vec<InstructionMeta>,
    /// CHL phase tracking (free vs clamped).
    chl_phase: Mutex<ChlPhase>,
    /// CHL free-phase correlations per cold register.
    /// Key: cold register index. Value: correlation vector (outer product sums).
    chl_free_corr: Mutex<Vec<Option<Vec<i64>>>>,
    /// CHL clamped-phase correlations per cold register.
    chl_clamp_corr: Mutex<Vec<Option<Vec<i64>>>>,
    /// Weight checkpoints for rollback (per cold register index).
    checkpoints: Mutex<Vec<Option<ColdBuffer>>>,
}

impl LearningExtension {
    pub fn new() -> Self {
        Self {
            chl_phase: Mutex::new(ChlPhase::Idle),
            chl_free_corr: Mutex::new(vec![None; 64]),
            chl_clamp_corr: Mutex::new(vec![None; 64]),
            checkpoints: Mutex::new(vec![None; 64]),
            instructions: vec![
                InstructionMeta {
                    opcode: 0x0000,
                    mnemonic: "MASTERY_UPDATE",
                    operand_pattern: OperandPattern::Custom("[weights:1][activity:1][direction:1][flags:1]"),
                    description: "Accumulate learning pressure from error and activity",
                },
                InstructionMeta {
                    opcode: 0x0001,
                    mnemonic: "MASTERY_COMMIT",
                    operand_pattern: OperandPattern::Custom("[weights:1][_:1][_:1][flags:1]"),
                    description: "Commit accumulated pressure to weight changes",
                },
                InstructionMeta {
                    opcode: 0x0002,
                    mnemonic: "ADD_BABBLE",
                    operand_pattern: OperandPattern::Reg,
                    description: "Add exploration noise to activations",
                },
                InstructionMeta {
                    opcode: 0x0003,
                    mnemonic: "LOAD_TARGET",
                    operand_pattern: OperandPattern::Reg,
                    description: "Load target buffer into hot register",
                },
                InstructionMeta {
                    opcode: 0x0004,
                    mnemonic: "MARK_ELIGIBILITY",
                    operand_pattern: OperandPattern::RegReg,
                    description: "Mark weights eligible for update based on activity",
                },
                // CHL operations
                InstructionMeta {
                    opcode: 0x0005,
                    mnemonic: "CHL_FREE_START",
                    operand_pattern: OperandPattern::None,
                    description: "CHL: Start free phase",
                },
                InstructionMeta {
                    opcode: 0x0006,
                    mnemonic: "CHL_FREE_RECORD",
                    operand_pattern: OperandPattern::RegReg,
                    description: "CHL: Record free phase correlations",
                },
                InstructionMeta {
                    opcode: 0x0007,
                    mnemonic: "CHL_CLAMP_START",
                    operand_pattern: OperandPattern::None,
                    description: "CHL: Start clamped phase",
                },
                InstructionMeta {
                    opcode: 0x0008,
                    mnemonic: "CHL_CLAMP_RECORD",
                    operand_pattern: OperandPattern::RegReg,
                    description: "CHL: Record clamped phase correlations",
                },
                InstructionMeta {
                    opcode: 0x0009,
                    mnemonic: "CHL_UPDATE",
                    operand_pattern: OperandPattern::Reg,
                    description: "CHL: Compute weight updates",
                },
                InstructionMeta {
                    opcode: 0x000A,
                    mnemonic: "CHL_BACKPROP_CLAMP",
                    operand_pattern: OperandPattern::RegReg,
                    description: "CHL: Propagate clamped signal backward",
                },
                // Additional learning ops
                InstructionMeta {
                    opcode: 0x000B,
                    mnemonic: "DECAY_ELIGIBILITY",
                    operand_pattern: OperandPattern::Reg,
                    description: "Decay eligibility traces",
                },
                InstructionMeta {
                    opcode: 0x000C,
                    mnemonic: "COMPUTE_ERROR",
                    operand_pattern: OperandPattern::RegReg,
                    description: "Compute error between target and output",
                },
                InstructionMeta {
                    opcode: 0x000D,
                    mnemonic: "UPDATE_WEIGHTS",
                    operand_pattern: OperandPattern::Reg,
                    description: "Apply weight updates from eligibility and error",
                },
                InstructionMeta {
                    opcode: 0x000E,
                    mnemonic: "DECAY_BABBLE",
                    operand_pattern: OperandPattern::None,
                    description: "Decay babble exploration scale",
                },
                InstructionMeta {
                    opcode: 0x000F,
                    mnemonic: "COMPUTE_RPE",
                    operand_pattern: OperandPattern::RegReg,
                    description: "Compute Reward Prediction Error",
                },
                InstructionMeta {
                    opcode: 0x0010,
                    mnemonic: "GATE_ERROR",
                    operand_pattern: OperandPattern::RegImm8,
                    description: "Gate learning based on error threshold",
                },
                InstructionMeta {
                    opcode: 0x0011,
                    mnemonic: "CHECKPOINT_WEIGHTS",
                    operand_pattern: OperandPattern::Reg,
                    description: "Checkpoint weights for potential rollback",
                },
                InstructionMeta {
                    opcode: 0x0012,
                    mnemonic: "ROLLBACK_WEIGHTS",
                    operand_pattern: OperandPattern::Reg,
                    description: "Rollback to checkpointed weights",
                },
                InstructionMeta {
                    opcode: 0x0013,
                    mnemonic: "CONSOLIDATE",
                    operand_pattern: OperandPattern::None,
                    description: "Consolidate hot → cold (Thermogram)",
                },
            ],
        }
    }
}

impl Extension for LearningExtension {
    fn ext_id(&self) -> u16 {
        0x0004
    }

    fn name(&self) -> &str {
        "tvmr.learning"
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
            0x0000 => execute_mastery_update(operands, ctx),
            0x0001 => execute_mastery_commit(operands, ctx),
            0x0002 => execute_add_babble(operands, ctx),
            0x0003 => execute_load_target(operands, ctx),
            0x0004 => execute_mark_eligibility(operands, ctx),
            0x0005 => self.execute_chl_free_start(),
            0x0006 => self.execute_chl_free_record(operands, ctx),
            0x0007 => self.execute_chl_clamp_start(),
            0x0008 => self.execute_chl_clamp_record(operands, ctx),
            0x0009 => self.execute_chl_update(operands, ctx),
            0x000A => execute_chl_backprop_clamp(operands, ctx),
            0x000B => execute_decay_eligibility(operands, ctx),
            0x000C => execute_compute_error(operands, ctx),
            0x000D => execute_update_weights(operands, ctx),
            0x000E => execute_decay_babble(ctx),
            0x000F => execute_compute_rpe(operands, ctx),
            0x0010 => execute_gate_error(operands, ctx),
            0x0011 => self.execute_checkpoint_weights(operands, ctx),
            0x0012 => self.execute_rollback_weights(operands, ctx),
            0x0013 => StepResult::Yield(DomainOp::Consolidate),
            _ => StepResult::Error(format!("tvmr.learning: unknown opcode 0x{:04X}", opcode)),
        }
    }
}

// =============================================================================
// Standalone execute functions
// =============================================================================

fn execute_mastery_update(ops: [u8; 4], ctx: &mut ExecutionContext) -> StepResult {
    // Dopamine gating: no learning without surprise/reward
    if !ctx.chemical_state.learning_enabled() {
        return StepResult::Continue;
    }

    let weights_idx = Register(ops[0]).index();
    let activity_idx = Register(ops[1]).index();
    let direction_idx = ops[2] as usize & 0x0F;
    let base_scale = if ops[3] > 0 { ops[3] as i32 } else { 15 };

    let dopamine_scale = ctx.chemical_state.dopamine_scale();
    let scale = base_scale * dopamine_scale;

    let activity = match &ctx.hot_regs[activity_idx] {
        Some(buf) => buf.data.clone(),
        None => return StepResult::Error("Activity register not allocated".to_string()),
    };

    let direction = match &ctx.hot_regs[direction_idx] {
        Some(buf) => {
            if buf.data.is_empty() {
                0
            } else {
                buf.data[0].signum()
            }
        }
        None => return StepResult::Error("Direction register not allocated".to_string()),
    };

    let pressure = ctx.pressure_regs[weights_idx].get_or_insert_with(|| vec![0i32; activity.len()]);

    if pressure.len() != activity.len() {
        *pressure = vec![0i32; activity.len()];
    }

    let max_activity = activity.iter().cloned().max().unwrap_or(1).max(1);
    let threshold = max_activity / 4; // Default threshold_div

    for (i, &act) in activity.iter().enumerate() {
        if act > threshold {
            let activity_strength = (act - threshold) as i64 * 256 / max_activity as i64;
            let delta = (direction as i64 * activity_strength * scale as i64 / 256) as i32;
            pressure[i] = pressure[i].saturating_add(delta);
        }
    }

    StepResult::Continue
}

fn execute_mastery_commit(ops: [u8; 4], ctx: &mut ExecutionContext) -> StepResult {
    let weights_idx = Register(ops[0]).index();
    let base_threshold = if ops[3] > 0 { ops[3] as i32 } else { 50 };
    let mag_step = 5u8; // Default

    let pressure = match &mut ctx.pressure_regs[weights_idx] {
        Some(p) => p,
        None => return StepResult::Continue,
    };

    // Get temperatures first (before mutable borrow of signals)
    let temperatures: Vec<i32> = match &ctx.cold_regs[weights_idx] {
        Some(buf) => (0..buf.weights.len())
            .map(|i| buf.temperature(i).threshold_multiplier())
            .collect(),
        None => return StepResult::Error("Signal register not allocated".to_string()),
    };

    let signals = match &mut ctx.cold_regs[weights_idx] {
        Some(buf) => &mut buf.weights,
        None => return StepResult::Error("Signal register not allocated".to_string()),
    };

    for (i, p) in pressure.iter_mut().enumerate() {
        if i >= signals.len() {
            break;
        }

        let temp_multiplier = temperatures.get(i).copied().unwrap_or(1);
        let effective_threshold = if temp_multiplier == i32::MAX {
            i32::MAX
        } else {
            (base_threshold * temp_multiplier) / 2
        };

        if p.abs() >= effective_threshold {
            let needed_polarity = if *p > 0 { 1i8 } else { -1i8 };
            let s = &mut signals[i];

            if s.polarity == needed_polarity {
                s.magnitude = s.magnitude.saturating_add(mag_step);
            } else if s.polarity == 0 {
                s.polarity = needed_polarity;
                s.magnitude = mag_step;
            } else if s.magnitude > mag_step {
                s.magnitude -= mag_step;
            } else {
                s.polarity = needed_polarity;
                s.magnitude = mag_step;
            }

            *p = 0;
        }
    }

    StepResult::Continue
}

fn execute_add_babble(ops: [u8; 4], ctx: &mut ExecutionContext) -> StepResult {
    let idx = Register(ops[0]).index();

    if let Some(buf) = &mut ctx.hot_regs[idx] {
        let babble_base = *ctx.babble_scale * 50;

        for (i, val) in buf.data.iter_mut().enumerate() {
            let phase = *ctx.babble_phase;
            let magnitude_factor = 77 + ((phase * 7 + i * 13) % 179);
            let sign = if (i * 7 + phase / 10) % 5 < 3 {
                1i32
            } else {
                -1i32
            };
            let babble = sign * (babble_base * magnitude_factor as i32 / 255);
            let positive_bias = babble_base / 3;
            *val = val.saturating_add(babble + positive_bias);
        }

        *ctx.babble_phase += 1;
    }

    StepResult::Continue
}

fn execute_load_target(ops: [u8; 4], ctx: &mut ExecutionContext) -> StepResult {
    let idx = Register(ops[0]).index();

    let data = ctx.target_buffer.to_vec();
    let len = data.len();
    ctx.hot_regs[idx] = Some(HotBuffer {
        data,
        shape: vec![len],
    });

    StepResult::Continue
}

// =============================================================================
// MARK_ELIGIBILITY [weights:1][activity:1][_:2]
// Marks weights as eligible for update based on activity pattern.
// Neurons with activity above mean are eligible (eligibility = activity).
// =============================================================================
fn execute_mark_eligibility(ops: [u8; 4], ctx: &mut ExecutionContext) -> StepResult {
    let weights_idx = Register(ops[0]).index();
    let activity_idx = Register(ops[1]).index();

    let activity = match &ctx.hot_regs[activity_idx] {
        Some(buf) => buf.data.clone(),
        None => return StepResult::Error(format!("H{} not allocated", activity_idx)),
    };

    // Eligibility = activity values for active neurons, 0 for inactive
    let mean = if activity.is_empty() {
        0i64
    } else {
        activity.iter().map(|&v| v as i64).sum::<i64>() / activity.len() as i64
    };

    let eligibility: Vec<i32> = activity
        .iter()
        .map(|&v| if v as i64 > mean { v } else { 0 })
        .collect();

    // Store eligibility in pressure register (repurposed as eligibility trace)
    let pressure = ctx.pressure_regs[weights_idx].get_or_insert_with(|| vec![0i32; eligibility.len()]);
    if pressure.len() != eligibility.len() {
        *pressure = vec![0i32; eligibility.len()];
    }
    // Add eligibility to existing pressure (accumulative)
    for (i, &e) in eligibility.iter().enumerate() {
        if i < pressure.len() {
            pressure[i] = pressure[i].saturating_add(e);
        }
    }

    StepResult::Continue
}

// =============================================================================
// CHL (Contrastive Hebbian Learning) — methods on LearningExtension
// =============================================================================
impl LearningExtension {
    /// CHL_FREE_START: Enter free phase.
    fn execute_chl_free_start(&self) -> StepResult {
        let mut phase = self.chl_phase.lock().unwrap();
        *phase = ChlPhase::Free;
        // Clear all free-phase correlations
        let mut corr = self.chl_free_corr.lock().unwrap();
        for slot in corr.iter_mut() {
            *slot = None;
        }
        StepResult::Continue
    }

    /// CHL_FREE_RECORD [visible:1][hidden:1][_:2]
    /// Record free-phase correlations: corr[i*h+j] += visible[i] * hidden[j]
    fn execute_chl_free_record(&self, ops: [u8; 4], ctx: &mut ExecutionContext) -> StepResult {
        let phase = *self.chl_phase.lock().unwrap();
        if phase != ChlPhase::Free {
            return StepResult::Error("CHL_FREE_RECORD called outside free phase".to_string());
        }

        let vis_idx = Register(ops[0]).index();
        let hid_idx = Register(ops[1]).index();

        let visible = match &ctx.hot_regs[vis_idx] {
            Some(buf) => buf.data.clone(),
            None => return StepResult::Error(format!("H{} not allocated", vis_idx)),
        };
        let hidden = match &ctx.hot_regs[hid_idx] {
            Some(buf) => buf.data.clone(),
            None => return StepResult::Error(format!("H{} not allocated", hid_idx)),
        };

        let v_len = visible.len();
        let h_len = hidden.len();
        let corr_size = v_len * h_len;

        let mut corr = self.chl_free_corr.lock().unwrap();
        // Use vis_idx as the storage key
        let entry = corr[vis_idx].get_or_insert_with(|| vec![0i64; corr_size]);
        if entry.len() != corr_size {
            *entry = vec![0i64; corr_size];
        }

        for (i, &v) in visible.iter().enumerate() {
            for (j, &h) in hidden.iter().enumerate() {
                entry[i * h_len + j] += v as i64 * h as i64;
            }
        }

        StepResult::Continue
    }

    /// CHL_CLAMP_START: Enter clamped phase.
    fn execute_chl_clamp_start(&self) -> StepResult {
        let mut phase = self.chl_phase.lock().unwrap();
        *phase = ChlPhase::Clamped;
        // Clear all clamped-phase correlations
        let mut corr = self.chl_clamp_corr.lock().unwrap();
        for slot in corr.iter_mut() {
            *slot = None;
        }
        StepResult::Continue
    }

    /// CHL_CLAMP_RECORD [visible:1][hidden:1][_:2]
    /// Record clamped-phase correlations.
    fn execute_chl_clamp_record(&self, ops: [u8; 4], ctx: &mut ExecutionContext) -> StepResult {
        let phase = *self.chl_phase.lock().unwrap();
        if phase != ChlPhase::Clamped {
            return StepResult::Error("CHL_CLAMP_RECORD called outside clamped phase".to_string());
        }

        let vis_idx = Register(ops[0]).index();
        let hid_idx = Register(ops[1]).index();

        let visible = match &ctx.hot_regs[vis_idx] {
            Some(buf) => buf.data.clone(),
            None => return StepResult::Error(format!("H{} not allocated", vis_idx)),
        };
        let hidden = match &ctx.hot_regs[hid_idx] {
            Some(buf) => buf.data.clone(),
            None => return StepResult::Error(format!("H{} not allocated", hid_idx)),
        };

        let v_len = visible.len();
        let h_len = hidden.len();
        let corr_size = v_len * h_len;

        let mut corr = self.chl_clamp_corr.lock().unwrap();
        let entry = corr[vis_idx].get_or_insert_with(|| vec![0i64; corr_size]);
        if entry.len() != corr_size {
            *entry = vec![0i64; corr_size];
        }

        for (i, &v) in visible.iter().enumerate() {
            for (j, &h) in hidden.iter().enumerate() {
                entry[i * h_len + j] += v as i64 * h as i64;
            }
        }

        StepResult::Continue
    }

    /// CHL_UPDATE [weights:1][_:3]
    /// Compute delta_w = clamped_corr - free_corr, apply to weights.
    /// Weight update: polarity shifts based on sign of correlation difference.
    fn execute_chl_update(&self, ops: [u8; 4], ctx: &mut ExecutionContext) -> StepResult {
        let weights_idx = Register(ops[0]).index();

        // Dopamine gating
        if !ctx.chemical_state.learning_enabled() {
            return StepResult::Continue;
        }

        let free = self.chl_free_corr.lock().unwrap();
        let clamp = self.chl_clamp_corr.lock().unwrap();

        let free_corr = match &free[weights_idx] {
            Some(c) => c,
            None => return StepResult::Continue, // No free-phase recorded
        };
        let clamp_corr = match &clamp[weights_idx] {
            Some(c) => c,
            None => return StepResult::Continue, // No clamped-phase recorded
        };

        if free_corr.len() != clamp_corr.len() {
            return StepResult::Error("CHL correlation dimension mismatch".to_string());
        }

        // Compute delta and apply to pressure registers
        let deltas: Vec<i32> = free_corr
            .iter()
            .zip(clamp_corr.iter())
            .map(|(&f, &c)| {
                // delta = clamped - free (standard CHL rule)
                let d = c - f;
                // Scale down to i32 range
                (d >> 8).clamp(i32::MIN as i64, i32::MAX as i64) as i32
            })
            .collect();

        let pressure = ctx.pressure_regs[weights_idx]
            .get_or_insert_with(|| vec![0i32; deltas.len()]);
        if pressure.len() != deltas.len() {
            *pressure = vec![0i32; deltas.len()];
        }
        for (i, &d) in deltas.iter().enumerate() {
            pressure[i] = pressure[i].saturating_add(d);
        }

        // Reset CHL phase
        drop(free);
        drop(clamp);
        *self.chl_phase.lock().unwrap() = ChlPhase::Idle;

        StepResult::Continue
    }

    /// CHECKPOINT_WEIGHTS [cold_reg:1][_:3]
    /// Save a snapshot of the cold register for potential rollback.
    fn execute_checkpoint_weights(&self, ops: [u8; 4], ctx: &mut ExecutionContext) -> StepResult {
        let idx = Register(ops[0]).index();

        let cold = match &ctx.cold_regs[idx] {
            Some(buf) => buf.clone(),
            None => return StepResult::Error(format!("C{} not allocated", idx)),
        };

        let mut checkpoints = self.checkpoints.lock().unwrap();
        checkpoints[idx] = Some(cold);

        StepResult::Continue
    }

    /// ROLLBACK_WEIGHTS [cold_reg:1][_:3]
    /// Restore cold register from checkpoint.
    fn execute_rollback_weights(&self, ops: [u8; 4], ctx: &mut ExecutionContext) -> StepResult {
        let idx = Register(ops[0]).index();

        let mut checkpoints = self.checkpoints.lock().unwrap();
        match checkpoints[idx].take() {
            Some(saved) => {
                ctx.cold_regs[idx] = Some(saved);
                StepResult::Continue
            }
            None => StepResult::Error(format!("No checkpoint for C{}", idx)),
        }
    }
}

// =============================================================================
// CHL_BACKPROP_CLAMP [target:1][source:1][_:2]
// Propagate clamped signal backward through weights.
// target = transpose(weights) @ source (simplified backprop).
// =============================================================================
fn execute_chl_backprop_clamp(ops: [u8; 4], ctx: &mut ExecutionContext) -> StepResult {
    let target_idx = Register(ops[0]).index();
    let source_idx = Register(ops[1]).index();

    let source = match &ctx.hot_regs[source_idx] {
        Some(buf) => buf.data.clone(),
        None => return StepResult::Error(format!("H{} not allocated", source_idx)),
    };

    // Use the target register's corresponding cold weights for transpose matmul
    let cold = match &ctx.cold_regs[target_idx] {
        Some(buf) => buf,
        None => return StepResult::Error(format!("C{} not allocated", target_idx)),
    };

    if cold.shape.len() < 2 {
        return StepResult::Error("Weights must be 2D for backprop".to_string());
    }

    let out_dim = cold.shape[0]; // rows
    let in_dim = cold.shape[1];  // cols

    // Transpose matmul: result[j] = sum_i(weights[i][j] * source[i])
    let mut result = vec![0i64; in_dim];
    for i in 0..out_dim.min(source.len()) {
        for j in 0..in_dim {
            let w_idx = i * in_dim + j;
            if w_idx < cold.weights.len() {
                let w = &cold.weights[w_idx];
                let effective = w.polarity as i64 * w.magnitude as i64;
                result[j] += effective * source[i] as i64;
            }
        }
    }

    let result_i32: Vec<i32> = result
        .iter()
        .map(|&v| v.clamp(i32::MIN as i64, i32::MAX as i64) as i32)
        .collect();

    ctx.hot_regs[target_idx] = Some(HotBuffer {
        data: result_i32,
        shape: vec![in_dim],
    });

    StepResult::Continue
}

// =============================================================================
// DECAY_ELIGIBILITY [weights:1][_:3]
// Exponential decay of eligibility traces: pressure[i] = pressure[i] * 230 / 256
// (90% retention per tick — biological eligibility trace time constant).
// =============================================================================
fn execute_decay_eligibility(ops: [u8; 4], ctx: &mut ExecutionContext) -> StepResult {
    let idx = Register(ops[0]).index();

    if let Some(pressure) = &mut ctx.pressure_regs[idx] {
        for p in pressure.iter_mut() {
            // ~90% retention: 230/256 ≈ 0.898
            *p = ((*p as i64 * 230) / 256) as i32;
        }
    }

    StepResult::Continue
}

// =============================================================================
// COMPUTE_ERROR [target:1][output:1][_:2]
// Compute element-wise error: current_error = sum(|target - output|)
// Also yields DomainOp::ComputeError for host-level error tracking.
// =============================================================================
fn execute_compute_error(ops: [u8; 4], ctx: &mut ExecutionContext) -> StepResult {
    let target_idx = Register(ops[0]).index();
    let output_idx = Register(ops[1]).index();

    let target_data = match &ctx.hot_regs[target_idx] {
        Some(buf) => &buf.data,
        None => return StepResult::Error(format!("H{} not allocated", target_idx)),
    };

    let output_data = match &ctx.hot_regs[output_idx] {
        Some(buf) => &buf.data,
        None => return StepResult::Error(format!("H{} not allocated", output_idx)),
    };

    let len = target_data.len().min(output_data.len());
    let error: i64 = (0..len)
        .map(|i| (target_data[i] as i64 - output_data[i] as i64).abs())
        .sum();

    *ctx.current_error = error.clamp(0, i32::MAX as i64) as i32;

    StepResult::Continue
}

// =============================================================================
// UPDATE_WEIGHTS [weights:1][_:3]
// Apply accumulated pressure from eligibility traces to weights.
// Combines eligibility, error, and dopamine gating.
// =============================================================================
fn execute_update_weights(ops: [u8; 4], ctx: &mut ExecutionContext) -> StepResult {
    let idx = Register(ops[0]).index();

    if !ctx.chemical_state.learning_enabled() {
        return StepResult::Continue;
    }

    let error_scale = (*ctx.current_error as i64).min(1000).max(1);
    let dopamine = ctx.chemical_state.dopamine_scale() as i64;

    let pressure = match &ctx.pressure_regs[idx] {
        Some(p) => p.clone(),
        None => return StepResult::Continue,
    };

    let cold = match ctx.cold_regs[idx].as_mut() {
        Some(c) => c,
        None => return StepResult::Error(format!("C{} not allocated", idx)),
    };

    for (i, &p) in pressure.iter().enumerate() {
        if i >= cold.weights.len() {
            break;
        }
        // Scale pressure by error and dopamine
        let scaled = (p as i64 * error_scale * dopamine) / (256 * 256);
        if scaled.abs() > 30 {
            let s = &mut cold.weights[i];
            let direction = if scaled > 0 { 1i8 } else { -1i8 };
            if s.polarity == direction {
                s.magnitude = s.magnitude.saturating_add(3);
            } else if s.polarity == 0 {
                s.polarity = direction;
                s.magnitude = 3;
            } else if s.magnitude > 3 {
                s.magnitude -= 3;
            } else {
                s.polarity = direction;
                s.magnitude = 3;
            }
        }
    }

    // Clear pressure after applying
    if let Some(p) = &mut ctx.pressure_regs[idx] {
        p.iter_mut().for_each(|v| *v = 0);
    }

    StepResult::Continue
}

// =============================================================================
// DECAY_BABBLE
// Reduce babble scale: babble_scale = babble_scale * 245 / 256
// (~96% retention — gradual exploration reduction as learning stabilizes).
// =============================================================================
fn execute_decay_babble(ctx: &mut ExecutionContext) -> StepResult {
    let current = *ctx.babble_scale as i64;
    *ctx.babble_scale = ((current * 245) / 256) as i32;
    StepResult::Continue
}

// =============================================================================
// COMPUTE_RPE [predicted:1][actual:1][_:2]
// Reward Prediction Error: delta = actual_reward - predicted_reward.
// Result stored in current_error (signed — positive = better than expected).
// =============================================================================
fn execute_compute_rpe(ops: [u8; 4], ctx: &mut ExecutionContext) -> StepResult {
    let predicted_idx = Register(ops[0]).index();
    let actual_idx = Register(ops[1]).index();

    let predicted = match &ctx.hot_regs[predicted_idx] {
        Some(buf) => {
            if buf.data.is_empty() { 0i64 } else { buf.data[0] as i64 }
        }
        None => return StepResult::Error(format!("H{} not allocated", predicted_idx)),
    };

    let actual = match &ctx.hot_regs[actual_idx] {
        Some(buf) => {
            if buf.data.is_empty() { 0i64 } else { buf.data[0] as i64 }
        }
        None => return StepResult::Error(format!("H{} not allocated", actual_idx)),
    };

    // RPE = actual - predicted (positive = surprise reward, negative = surprise punishment)
    let rpe = actual - predicted;
    *ctx.current_error = rpe.clamp(i32::MIN as i64, i32::MAX as i64) as i32;

    StepResult::Continue
}

// =============================================================================
// GATE_ERROR [source:1][threshold:1][_:2]
// Gate learning based on error threshold.
// If |current_error| < threshold * 256, suppress learning (error too small).
// Writes gated error magnitude into source register H[reg][0].
// =============================================================================
fn execute_gate_error(ops: [u8; 4], ctx: &mut ExecutionContext) -> StepResult {
    let target_idx = Register(ops[0]).index();
    let threshold = ops[1] as i32 * 256;

    let error_abs = (*ctx.current_error).abs();
    let gated = if error_abs >= threshold { error_abs } else { 0 };

    ctx.hot_regs[target_idx] = Some(HotBuffer {
        data: vec![gated],
        shape: vec![1],
    });

    StepResult::Continue
}
