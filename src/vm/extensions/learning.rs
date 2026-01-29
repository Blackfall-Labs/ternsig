//! Learning Extension (ExtID: 0x0004)
//!
//! All learning operations: mastery learning, CHL, babble, eligibility.
//! Learning is dopamine-gated — no learning without neuromodulator support.

use crate::vm::extension::{
    ExecutionContext, Extension, InstructionMeta, OperandPattern, StepResult,
};
use crate::vm::interpreter::HotBuffer;
use crate::vm::register::Register;

/// Learning extension — weight update and exploration operations.
pub struct LearningExtension {
    instructions: Vec<InstructionMeta>,
}

impl LearningExtension {
    pub fn new() -> Self {
        Self {
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
            0x0004 => StepResult::Continue, // MARK_ELIGIBILITY (stub)
            // Unimplemented CHL/learning ops
            0x0005..=0x0013 => StepResult::Continue,
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
