//! Architecture Extension (ExtID: 0x0006)
//!
//! Runtime architecture modification: grow/prune neurons, wire connections,
//! allocate/free registers. This is the structural plasticity enabler.

use crate::vm::extension::{
    DomainOp, ExecutionContext, Extension, InstructionMeta, OperandPattern, StepResult,
};
use crate::vm::interpreter::{ColdBuffer, HotBuffer, SignalTemperature};
use crate::vm::register::Register;
use crate::Signal;

/// Architecture extension — structural plasticity operations.
pub struct ArchExtension {
    instructions: Vec<InstructionMeta>,
}

impl ArchExtension {
    pub fn new() -> Self {
        Self {
            instructions: vec![
                InstructionMeta {
                    opcode: 0x0000,
                    mnemonic: "ALLOC_TENSOR",
                    operand_pattern: OperandPattern::Custom("[reg:1][dim0:1][dim1_hi:1][dim1_lo:1]"),
                    description: "Allocate a register at runtime",
                },
                InstructionMeta {
                    opcode: 0x0001,
                    mnemonic: "FREE_TENSOR",
                    operand_pattern: OperandPattern::Reg,
                    description: "Free a register",
                },
                InstructionMeta {
                    opcode: 0x0002,
                    mnemonic: "WIRE_FORWARD",
                    operand_pattern: OperandPattern::RegRegReg,
                    description: "Dynamic forward connection: output = weights @ input",
                },
                InstructionMeta {
                    opcode: 0x0003,
                    mnemonic: "WIRE_SKIP",
                    operand_pattern: OperandPattern::RegRegReg,
                    description: "Dynamic skip connection: output = input1 + input2",
                },
                InstructionMeta {
                    opcode: 0x0004,
                    mnemonic: "GROW_NEURON",
                    operand_pattern: OperandPattern::Custom("[cold_reg:1][_:1][count:1][seed:1]"),
                    description: "Add neurons to a cold register",
                },
                InstructionMeta {
                    opcode: 0x0005,
                    mnemonic: "PRUNE_NEURON",
                    operand_pattern: OperandPattern::Custom("[cold_reg:1][_:1][neuron_idx:1][_:1]"),
                    description: "Remove a neuron from a cold register",
                },
                InstructionMeta {
                    opcode: 0x0006,
                    mnemonic: "INIT_RANDOM",
                    operand_pattern: OperandPattern::Custom("[cold_reg:1][_:1][_:1][seed:1]"),
                    description: "Initialize cold register with random weights",
                },
                InstructionMeta {
                    opcode: 0x0007,
                    mnemonic: "DEFINE_LAYER",
                    operand_pattern: OperandPattern::Custom("[layer:1][in_dim:1][out_hi:1][out_lo:1]"),
                    description: "Define layer dimensions",
                },
                InstructionMeta {
                    opcode: 0x0008,
                    mnemonic: "FREEZE_LAYER",
                    operand_pattern: OperandPattern::Reg,
                    description: "Mark layer as non-trainable",
                },
                InstructionMeta {
                    opcode: 0x0009,
                    mnemonic: "UNFREEZE_LAYER",
                    operand_pattern: OperandPattern::Reg,
                    description: "Unfreeze a layer for training",
                },
                InstructionMeta {
                    opcode: 0x000A,
                    mnemonic: "SET_ACTIVATION",
                    operand_pattern: OperandPattern::RegImm8,
                    description: "Set activation function for a layer",
                },
                InstructionMeta {
                    opcode: 0x000B,
                    mnemonic: "ALLOC_DYNAMIC",
                    operand_pattern: OperandPattern::Custom("[target:1][bank:1][size_hi:1][size_lo:1]"),
                    description: "Request dynamic register allocation from kernel. Yields AllocRegister.",
                },
                InstructionMeta {
                    opcode: 0x000C,
                    mnemonic: "FREE_DYNAMIC",
                    operand_pattern: OperandPattern::Reg,
                    description: "Release dynamically allocated register. Yields FreeRegister.",
                },
            ],
        }
    }
}

impl Extension for ArchExtension {
    fn ext_id(&self) -> u16 {
        0x0006
    }

    fn name(&self) -> &str {
        "tvmr.arch"
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
            0x0000 => execute_alloc_tensor(operands, ctx),
            0x0001 => execute_free_tensor(operands, ctx),
            0x0002 => execute_wire_forward(operands, ctx),
            0x0003 => execute_wire_skip(operands, ctx),
            0x0004 => execute_grow_neuron(operands, ctx),
            0x0005 => execute_prune_neuron(operands, ctx),
            0x0006 => execute_init_random(operands, ctx),
            0x0007 => execute_define_layer(operands, ctx),
            0x0008 => execute_freeze_layer(operands, ctx),
            0x0009 => execute_unfreeze_layer(operands, ctx),
            0x000A => execute_set_activation(operands, ctx),
            // ALLOC_DYNAMIC [target:1][bank:1][size_hi:1][size_lo:1]
            // Yields to kernel for managed register pool allocation.
            // bank: 0=Hot, 1=Cold, 2=Param, 3=Shape
            0x000B => {
                let target = Register(operands[0]);
                let bank = operands[1];
                let size = ((operands[2] as u16) << 8) | (operands[3] as u16);
                StepResult::Yield(DomainOp::AllocRegister { target, bank, size })
            }
            // FREE_DYNAMIC [reg:1][_:3]
            // Yields to kernel to release a dynamically allocated register.
            0x000C => {
                let register = Register(operands[0]);
                StepResult::Yield(DomainOp::FreeRegister { register })
            }
            _ => StepResult::Error(format!("tvmr.arch: unknown opcode 0x{:04X}", opcode)),
        }
    }
}

// =============================================================================
// Standalone execute functions
// =============================================================================

fn execute_alloc_tensor(ops: [u8; 4], ctx: &mut ExecutionContext) -> StepResult {
    let target = Register(ops[0]);
    let idx = target.index();
    let dim0 = ops[1] as usize;
    let dim1 = ((ops[2] as usize) << 8) | (ops[3] as usize);

    let shape = if dim1 == 0 {
        vec![dim0]
    } else {
        vec![dim0, dim1]
    };

    if target.is_hot() {
        if ctx.hot_regs[idx].is_some() {
            return StepResult::Error(format!("Hot register H{} already allocated", idx));
        }
        ctx.hot_regs[idx] = Some(HotBuffer::new(shape));
        StepResult::Continue
    } else if target.is_cold() {
        if ctx.cold_regs[idx].is_some() {
            return StepResult::Error(format!("Cold register C{} already allocated", idx));
        }
        ctx.cold_regs[idx] = Some(ColdBuffer::new(shape));
        StepResult::Continue
    } else {
        StepResult::Error(format!("Cannot allocate register {:?} - must be Hot or Cold", target))
    }
}

fn execute_free_tensor(ops: [u8; 4], ctx: &mut ExecutionContext) -> StepResult {
    let target = Register(ops[0]);
    let idx = target.index();

    if target.is_hot() {
        if ctx.hot_regs[idx].is_none() {
            return StepResult::Error(format!("Hot register H{} not allocated", idx));
        }
        ctx.hot_regs[idx] = None;
        StepResult::Continue
    } else if target.is_cold() {
        if ctx.cold_regs[idx].is_none() {
            return StepResult::Error(format!("Cold register C{} not allocated", idx));
        }
        ctx.cold_regs[idx] = None;
        StepResult::Continue
    } else {
        StepResult::Error(format!("Cannot free register {:?} - must be Hot or Cold", target))
    }
}

fn execute_wire_forward(ops: [u8; 4], ctx: &mut ExecutionContext) -> StepResult {
    let output_idx = Register(ops[0]).index();
    let weights_idx = Register(ops[1]).index();
    let input_idx = ops[2] as usize;

    let (out_dim, in_dim) = {
        let weights = match &ctx.cold_regs[weights_idx] {
            Some(buf) => buf,
            None => return StepResult::Error(format!("C{} not allocated", weights_idx)),
        };
        if weights.shape.len() < 2 {
            return StepResult::Error("Weights must be 2D".to_string());
        }
        (weights.shape[0], weights.shape[1])
    };

    let input_data = {
        let input = match &ctx.hot_regs[input_idx] {
            Some(buf) => buf,
            None => return StepResult::Error(format!("H{} not allocated", input_idx)),
        };
        input.data.clone()
    };

    let weights_data = {
        let weights = ctx.cold_regs[weights_idx].as_ref().unwrap();
        weights.weights.clone()
    };

    let mut output_data = vec![0i64; out_dim];
    for o in 0..out_dim {
        let mut sum = 0i64;
        for i in 0..in_dim.min(input_data.len()) {
            let w = &weights_data[o * in_dim + i];
            let effective = w.polarity as i64 * w.magnitude as i64;
            sum += effective * input_data[i] as i64;
        }
        output_data[o] = sum;
    }

    let output_i32: Vec<i32> = output_data
        .iter()
        .map(|&v| v.clamp(i32::MIN as i64, i32::MAX as i64) as i32)
        .collect();

    ctx.hot_regs[output_idx] = Some(HotBuffer {
        data: output_i32,
        shape: vec![out_dim],
    });

    StepResult::Continue
}

fn execute_wire_skip(ops: [u8; 4], ctx: &mut ExecutionContext) -> StepResult {
    let output_idx = Register(ops[0]).index();
    let input1_idx = Register(ops[1]).index();
    let input2_idx = ops[2] as usize;

    let (data1, shape1) = {
        let input1 = match &ctx.hot_regs[input1_idx] {
            Some(buf) => buf,
            None => return StepResult::Error(format!("H{} not allocated", input1_idx)),
        };
        (input1.data.clone(), input1.shape.clone())
    };

    let data2 = {
        let input2 = match &ctx.hot_regs[input2_idx] {
            Some(buf) => buf,
            None => return StepResult::Error(format!("H{} not allocated", input2_idx)),
        };
        input2.data.clone()
    };

    let len = data1.len().max(data2.len());
    let mut result = vec![0i32; len];
    for i in 0..len {
        let v1 = data1.get(i % data1.len()).copied().unwrap_or(0);
        let v2 = data2.get(i % data2.len()).copied().unwrap_or(0);
        result[i] = v1.saturating_add(v2);
    }

    ctx.hot_regs[output_idx] = Some(HotBuffer {
        data: result,
        shape: shape1,
    });

    StepResult::Continue
}

fn execute_grow_neuron(ops: [u8; 4], ctx: &mut ExecutionContext) -> StepResult {
    let idx = Register(ops[0]).index();
    let neurons_to_add = ops[2] as usize;

    if neurons_to_add == 0 {
        return StepResult::Error("Cannot grow by 0 neurons".to_string());
    }

    let cold = match ctx.cold_regs[idx].as_mut() {
        Some(c) => c,
        None => return StepResult::Error(format!("C{} not allocated", idx)),
    };

    let seed = ops[3] as u64;

    if cold.shape.is_empty() {
        return StepResult::Error("Cannot grow empty register".to_string());
    }

    if cold.shape.len() == 1 {
        let new_size = cold.shape[0] + neurons_to_add;
        let mut rng_state = seed.wrapping_add(cold.weights.len() as u64);
        for _ in 0..neurons_to_add {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let r = (rng_state >> 33) as u32;
            let polarity = match r % 3 {
                0 => -1,
                1 => 0,
                _ => 1,
            };
            let magnitude = ((r >> 2) % 200 + 30) as u8;
            cold.weights.push(Signal { polarity, magnitude });
        }
        cold.shape = vec![new_size];
    } else if cold.shape.len() == 2 {
        let in_dim = cold.shape[1];
        let new_out_dim = cold.shape[0] + neurons_to_add;
        let mut rng_state = seed.wrapping_add(cold.weights.len() as u64);
        for _ in 0..neurons_to_add {
            for _ in 0..in_dim {
                rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
                let r = (rng_state >> 33) as u32;
                let polarity = match r % 3 {
                    0 => -1,
                    1 => 0,
                    _ => 1,
                };
                let magnitude = ((r >> 2) % 200 + 30) as u8;
                cold.weights.push(Signal { polarity, magnitude });
            }
        }
        cold.shape = vec![new_out_dim, in_dim];
    } else {
        return StepResult::Error(format!(
            "Cannot grow {}D register, only 1D or 2D supported",
            cold.shape.len()
        ));
    }

    // Extend temperatures if tracking
    if let Some(temps) = &mut cold.temperatures {
        let new_count = cold.weights.len() - temps.len();
        temps.extend(std::iter::repeat(SignalTemperature::Hot).take(new_count));
    }

    StepResult::Continue
}

fn execute_prune_neuron(ops: [u8; 4], ctx: &mut ExecutionContext) -> StepResult {
    let idx = Register(ops[0]).index();
    let neuron_idx = ops[2] as usize;

    let cold = match ctx.cold_regs[idx].as_mut() {
        Some(c) => c,
        None => return StepResult::Error(format!("C{} not allocated", idx)),
    };

    if cold.shape.len() == 1 {
        if neuron_idx >= cold.shape[0] {
            return StepResult::Error(format!(
                "Neuron index {} out of bounds (size {})",
                neuron_idx, cold.shape[0]
            ));
        }
        if cold.shape[0] <= 1 {
            return StepResult::Error("Cannot prune last neuron".to_string());
        }
        cold.weights.remove(neuron_idx);
        cold.shape[0] -= 1;
        if let Some(temps) = &mut cold.temperatures {
            if neuron_idx < temps.len() {
                temps.remove(neuron_idx);
            }
        }
    } else if cold.shape.len() == 2 {
        let out_dim = cold.shape[0];
        let in_dim = cold.shape[1];
        if neuron_idx >= out_dim {
            return StepResult::Error(format!(
                "Neuron index {} out of bounds (out_dim {})",
                neuron_idx, out_dim
            ));
        }
        if out_dim <= 1 {
            return StepResult::Error("Cannot prune last neuron".to_string());
        }
        let start = neuron_idx * in_dim;
        let end = start + in_dim;
        let mut new_weights = Vec::with_capacity(cold.weights.len() - in_dim);
        new_weights.extend_from_slice(&cold.weights[..start]);
        new_weights.extend_from_slice(&cold.weights[end..]);
        cold.weights = new_weights;
        cold.shape = vec![out_dim - 1, in_dim];
        if let Some(temps) = &mut cold.temperatures {
            let mut new_temps = Vec::with_capacity(temps.len() - in_dim);
            new_temps.extend_from_slice(&temps[..start]);
            new_temps.extend_from_slice(&temps[end..]);
            *temps = new_temps;
        }
    } else {
        return StepResult::Error(format!(
            "Cannot prune {}D register, only 1D or 2D supported",
            cold.shape.len()
        ));
    }

    StepResult::Continue
}

// =============================================================================
// DEFINE_LAYER [layer:1][in_dim:1][out_hi:1][out_lo:1]
// Define layer dimensions by allocating a cold register with shape [out_dim, in_dim].
// out_dim = (out_hi << 8) | out_lo for dimensions > 255.
// Layer register index = layer operand (cold bank).
// =============================================================================
fn execute_define_layer(ops: [u8; 4], ctx: &mut ExecutionContext) -> StepResult {
    let layer_idx = ops[0] as usize & 0x3F; // cold register index
    let in_dim = ops[1] as usize;
    let out_dim = ((ops[2] as usize) << 8) | (ops[3] as usize);

    if in_dim == 0 || out_dim == 0 {
        return StepResult::Error("Layer dimensions must be > 0".to_string());
    }

    if ctx.cold_regs[layer_idx].is_some() {
        return StepResult::Error(format!("C{} already allocated", layer_idx));
    }

    ctx.cold_regs[layer_idx] = Some(ColdBuffer::new(vec![out_dim, in_dim]));
    StepResult::Continue
}

// =============================================================================
// FREEZE_LAYER [cold_reg:1][_:3]
// Mark all weights in a cold register as Cold temperature (non-trainable).
// Frozen weights have maximum threshold — learning pressure cannot flip them.
// =============================================================================
fn execute_freeze_layer(ops: [u8; 4], ctx: &mut ExecutionContext) -> StepResult {
    let idx = Register(ops[0]).index();

    let cold = match ctx.cold_regs[idx].as_mut() {
        Some(buf) => buf,
        None => return StepResult::Error(format!("C{} not allocated", idx)),
    };

    let count = cold.weights.len();
    cold.temperatures = Some(vec![SignalTemperature::Cold; count]);

    StepResult::Continue
}

// =============================================================================
// UNFREEZE_LAYER [cold_reg:1][_:3]
// Reset all weights to Hot temperature (fully trainable).
// =============================================================================
fn execute_unfreeze_layer(ops: [u8; 4], ctx: &mut ExecutionContext) -> StepResult {
    let idx = Register(ops[0]).index();

    let cold = match ctx.cold_regs[idx].as_mut() {
        Some(buf) => buf,
        None => return StepResult::Error(format!("C{} not allocated", idx)),
    };

    let count = cold.weights.len();
    cold.temperatures = Some(vec![SignalTemperature::Hot; count]);

    StepResult::Continue
}

// =============================================================================
// SET_ACTIVATION [cold_reg:1][activation_id:1][_:2]
// Store activation function ID in shape register for the cold register.
// The interpreter uses this when executing forward passes.
// IDs: 0=none, 1=relu, 2=sigmoid, 3=tanh, 4=softmax, 5=gelu.
// =============================================================================
fn execute_set_activation(ops: [u8; 4], ctx: &mut ExecutionContext) -> StepResult {
    let reg_idx = Register(ops[0]).index();
    let activation_id = ops[1] as usize;

    // Store activation ID in the shape register for this cold register
    // Convention: shape_regs[idx] = [activation_id] when used for activation metadata
    if reg_idx < ctx.shape_regs.len() {
        ctx.shape_regs[reg_idx] = vec![activation_id];
    }

    StepResult::Continue
}

fn execute_init_random(ops: [u8; 4], ctx: &mut ExecutionContext) -> StepResult {
    let idx = Register(ops[0]).index();
    let seed = ops[3] as u64;

    if let Some(cold) = ctx.cold_regs[idx].as_mut() {
        let mut rng_state = seed;
        for w in cold.weights.iter_mut() {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let r = (rng_state >> 33) as u32;
            w.polarity = match r % 3 {
                0 => -1,
                1 => 0,
                _ => 1,
            };
            w.magnitude = ((r >> 2) % 200 + 30) as u8;
        }
    }

    StepResult::Continue
}
