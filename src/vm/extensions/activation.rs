//! Activation Extension (ExtID: 0x0003)
//!
//! Nonlinear activation functions for neural network layers.
//! All implementations use integer-only arithmetic.

use crate::vm::extension::{
    ExecutionContext, Extension, InstructionMeta, OperandPattern, StepResult,
};
use crate::vm::interpreter::HotBuffer;
use crate::vm::register::Register;

/// Activation extension — nonlinear activation functions.
pub struct ActivationExtension {
    instructions: Vec<InstructionMeta>,
}

impl ActivationExtension {
    pub fn new() -> Self {
        Self {
            instructions: vec![
                InstructionMeta {
                    opcode: 0x0000,
                    mnemonic: "RELU",
                    operand_pattern: OperandPattern::RegReg,
                    description: "ReLU: target = max(0, source)",
                },
                InstructionMeta {
                    opcode: 0x0001,
                    mnemonic: "SIGMOID",
                    operand_pattern: OperandPattern::RegReg,
                    description: "Sigmoid: target = sigmoid(source) [integer approx]",
                },
                InstructionMeta {
                    opcode: 0x0002,
                    mnemonic: "TANH",
                    operand_pattern: OperandPattern::RegReg,
                    description: "Tanh: target = tanh(source) [integer approx]",
                },
                InstructionMeta {
                    opcode: 0x0003,
                    mnemonic: "SOFTMAX",
                    operand_pattern: OperandPattern::RegReg,
                    description: "Softmax: target = softmax(source) [integer approx, 0-255 range]",
                },
                InstructionMeta {
                    opcode: 0x0004,
                    mnemonic: "GELU",
                    operand_pattern: OperandPattern::RegReg,
                    description: "GELU: target = gelu(source) [integer approx]",
                },
            ],
        }
    }
}

impl Extension for ActivationExtension {
    fn ext_id(&self) -> u16 {
        0x0003
    }

    fn name(&self) -> &str {
        "tvmr.activation"
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
            0x0000 => execute_relu(operands, ctx),
            0x0001 => execute_sigmoid(operands, ctx),
            0x0002 => execute_tanh(operands, ctx),
            0x0003 => execute_softmax(operands, ctx),
            0x0004 => execute_gelu(operands, ctx),
            _ => StepResult::Error(format!(
                "tvmr.activation: unknown opcode 0x{:04X}",
                opcode
            )),
        }
    }
}

// =============================================================================
// Standalone execute functions
// =============================================================================

fn execute_relu(ops: [u8; 4], ctx: &mut ExecutionContext) -> StepResult {
    let dst = Register(ops[0]).index();
    let src = Register(ops[1]).index();

    let source = match &ctx.hot_regs[src] {
        Some(buf) => buf.clone(),
        None => return StepResult::Error(format!("H{} not allocated", src)),
    };

    let data: Vec<i32> = source.data.iter().map(|&v| v.max(0)).collect();
    ctx.hot_regs[dst] = Some(HotBuffer {
        data,
        shape: source.shape,
    });

    StepResult::Continue
}

fn execute_sigmoid(ops: [u8; 4], ctx: &mut ExecutionContext) -> StepResult {
    let dst = Register(ops[0]).index();
    let src = Register(ops[1]).index();

    let source = match &ctx.hot_regs[src] {
        Some(buf) => buf.clone(),
        None => return StepResult::Error(format!("H{} not allocated", src)),
    };

    // Integer sigmoid approximation: output in 0-255 range
    // sigmoid(x) ≈ 128 + x * 32 / (256 + |x|)
    let data: Vec<i32> = source
        .data
        .iter()
        .map(|&v| {
            let x = v.clamp(-2048, 2048) as i64;
            let result = 128 + (x * 32) / (256 + x.abs());
            result.clamp(0, 255) as i32
        })
        .collect();

    ctx.hot_regs[dst] = Some(HotBuffer {
        data,
        shape: source.shape,
    });

    StepResult::Continue
}

fn execute_tanh(ops: [u8; 4], ctx: &mut ExecutionContext) -> StepResult {
    let dst = Register(ops[0]).index();
    let src = Register(ops[1]).index();

    let source = match &ctx.hot_regs[src] {
        Some(buf) => buf.clone(),
        None => return StepResult::Error(format!("H{} not allocated", src)),
    };

    // Integer tanh approximation: output in -127..127 range
    let data: Vec<i32> = source
        .data
        .iter()
        .map(|&v| {
            let x = v.clamp(-2048, 2048) as i64;
            let result = (x * 127) / (128 + x.abs());
            result.clamp(-127, 127) as i32
        })
        .collect();

    ctx.hot_regs[dst] = Some(HotBuffer {
        data,
        shape: source.shape,
    });

    StepResult::Continue
}

fn execute_softmax(ops: [u8; 4], ctx: &mut ExecutionContext) -> StepResult {
    let dst = Register(ops[0]).index();
    let src = Register(ops[1]).index();

    let source = match &ctx.hot_regs[src] {
        Some(buf) => buf.clone(),
        None => return StepResult::Error(format!("H{} not allocated", src)),
    };

    if source.data.is_empty() {
        ctx.hot_regs[dst] = Some(HotBuffer {
            data: vec![],
            shape: source.shape,
        });
        return StepResult::Continue;
    }

    // Integer softmax approximation
    let max_val = *source.data.iter().max().unwrap_or(&0);
    let shifted: Vec<i64> = source.data.iter().map(|&v| (v - max_val).max(-1024) as i64).collect();

    // exp approximation: 256 * 2^(x/256) ≈ 256 + x for small x
    let exp_vals: Vec<i64> = shifted.iter().map(|&v| (256 + v).max(1)).collect();
    let total: i64 = exp_vals.iter().sum::<i64>().max(1);

    let data: Vec<i32> = exp_vals
        .iter()
        .map(|&e| ((e * 255) / total) as i32)
        .collect();

    ctx.hot_regs[dst] = Some(HotBuffer {
        data,
        shape: source.shape,
    });

    StepResult::Continue
}

fn execute_gelu(ops: [u8; 4], ctx: &mut ExecutionContext) -> StepResult {
    let dst = Register(ops[0]).index();
    let src = Register(ops[1]).index();

    let source = match &ctx.hot_regs[src] {
        Some(buf) => buf.clone(),
        None => return StepResult::Error(format!("H{} not allocated", src)),
    };

    // GELU ≈ x * sigmoid(1.702 * x)
    // Integer approx: x * (128 + clamp(x * 54 / 256, -128, 127)) / 256
    let data: Vec<i32> = source
        .data
        .iter()
        .map(|&v| {
            let x = v as i64;
            let gate = 128 + (x * 54 / 256).clamp(-128, 127);
            ((x * gate) / 256) as i32
        })
        .collect();

    ctx.hot_regs[dst] = Some(HotBuffer {
        data,
        shape: source.shape,
    });

    StepResult::Continue
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_activation_metadata() {
        let ext = ActivationExtension::new();
        assert_eq!(ext.ext_id(), 0x0003);
        assert_eq!(ext.name(), "tvmr.activation");
        assert_eq!(ext.instructions().len(), 5);
    }

    #[test]
    fn test_relu_extension() {
        let ext = ActivationExtension::new();

        let mut hot_regs: Vec<Option<HotBuffer>> = vec![None; 64];
        let mut cold_regs = vec![None; 64];
        let mut param_regs = vec![0i32; 64];
        let mut shape_regs: Vec<Vec<usize>> = vec![Vec::new(); 64];
        let mut pc = 0usize;
        let mut call_stack = Vec::new();
        let mut loop_stack = Vec::new();
        let input_buffer = vec![];
        let mut output_buffer = Vec::new();
        let target_buffer = vec![];
        let mut chemical_state = crate::vm::interpreter::ChemicalState::baseline();
        let mut current_error = 0i32;
        let mut babble_scale = 0i32;
        let mut babble_phase = 0usize;
        let mut pressure_regs: Vec<Option<Vec<i32>>> = vec![None; 16];

        // Set up source register H1 with mixed values
        hot_regs[1] = Some(HotBuffer {
            data: vec![100, -50, 200, -100],
            shape: vec![4],
        });

        let mut ctx = ExecutionContext {
            hot_regs: &mut hot_regs,
            cold_regs: &mut cold_regs,
            param_regs: &mut param_regs,
            shape_regs: &mut shape_regs,
            pc: &mut pc,
            call_stack: &mut call_stack,
            loop_stack: &mut loop_stack,
            input_buffer: &input_buffer,
            output_buffer: &mut output_buffer,
            target_buffer: &target_buffer,
            chemical_state: &mut chemical_state,
            current_error: &mut current_error,
            babble_scale: &mut babble_scale,
            babble_phase: &mut babble_phase,
            pressure_regs: &mut pressure_regs,
            bank_cache: None,
        };

        // RELU: H0 = relu(H1)
        // operands: [dst=H0, src=H1, _, _]
        let result = ext.execute(0x0000, [0x00, 0x01, 0x00, 0x00], &mut ctx);
        assert!(matches!(result, StepResult::Continue));

        let h0 = ctx.hot_regs[0].as_ref().unwrap();
        assert_eq!(h0.data, vec![100, 0, 200, 0]);
    }
}
