//! Assembler - Parses human-readable .ternsig assembly to binary
//!
//! ## Assembly Syntax
//!
//! ```text
//! ; Comments start with semicolon
//!
//! .registers
//!     C0: ternary[32, 12]  key="chip.audio.w1"
//!     C1: ternary[32]      key="chip.audio.b1"
//!     H0: i32[12]          ; input
//!     H1: i32[32]          ; layer1 output
//!
//! .program
//!     load_input    H0
//!     ternary_matmul H1, C0, H0
//!     add           H1, H1, C1
//!     shift         H1, H1, 8
//!     relu          H2, H1
//!     halt
//! ```

use super::{
    RegisterMeta, Action, Dtype, Instruction, Modifier, Register,
    INSTRUCTION_SIZE, TERNSIG_MAGIC, TERNSIG_VERSION,
};
use std::collections::HashMap;

/// Assembled Ternsig program
#[derive(Debug, Clone)]
pub struct AssembledProgram {
    /// Program name
    pub name: String,
    /// Register definitions
    pub registers: Vec<RegisterMeta>,
    /// Compiled instructions
    pub instructions: Vec<Instruction>,
    /// Labels to instruction indices
    pub labels: HashMap<String, usize>,
    /// Input shape
    pub input_shape: Vec<usize>,
    /// Output shape
    pub output_shape: Vec<usize>,
}

impl AssembledProgram {
    /// Convert to binary .ternsig format
    pub fn to_binary(&self) -> Vec<u8> {
        let mut bytes = Vec::new();

        // === HEADER (32 bytes) ===
        bytes.extend_from_slice(&TERNSIG_MAGIC);
        bytes.extend_from_slice(&TERNSIG_VERSION.to_le_bytes());
        bytes.extend_from_slice(&0u16.to_le_bytes()); // flags

        // Register counts by bank
        let hot_count = self.registers.iter().filter(|r| r.id.is_hot()).count() as u8;
        let cold_count = self.registers.iter().filter(|r| r.id.is_cold()).count() as u8;
        let param_count = self.registers.iter().filter(|r| r.id.is_param()).count() as u8;
        bytes.push(hot_count);
        bytes.push(cold_count);
        bytes.push(param_count);

        // Instruction count and init count
        bytes.extend_from_slice(&(self.instructions.len() as u32).to_le_bytes());
        bytes.extend_from_slice(&0u32.to_le_bytes()); // init_count (for now)

        // Reserved padding to 32 bytes
        bytes.resize(32, 0);

        // === REGISTER DEFINITIONS ===
        for reg in &self.registers {
            bytes.push(reg.id.0);
            bytes.push(reg.dtype as u8);
            bytes.push(reg.shape.len() as u8);
            for &dim in &reg.shape {
                bytes.extend_from_slice(&(dim as u16).to_le_bytes());
            }
            // Thermogram key (length-prefixed string)
            if let Some(key) = &reg.thermogram_key {
                let key_bytes = key.as_bytes();
                bytes.push(key_bytes.len() as u8);
                bytes.extend_from_slice(key_bytes);
            } else {
                bytes.push(0); // No key
            }
        }

        // === INSTRUCTIONS ===
        for instr in &self.instructions {
            bytes.extend_from_slice(&instr.to_bytes());
        }

        bytes
    }
}

/// Assembler for Ternsig VM
pub struct Assembler {
    /// Current line number (for error reporting)
    line_number: usize,
    /// Accumulated registers
    registers: Vec<RegisterMeta>,
    /// Accumulated instructions
    instructions: Vec<Instruction>,
    /// Labels to instruction indices
    labels: HashMap<String, usize>,
    /// Unresolved label references (instruction index, label name)
    unresolved_labels: Vec<(usize, String)>,
}

impl Assembler {
    pub fn new() -> Self {
        Self {
            line_number: 0,
            registers: Vec::new(),
            instructions: Vec::new(),
            labels: HashMap::new(),
            unresolved_labels: Vec::new(),
        }
    }

    /// Assemble source code into a program
    pub fn assemble(&mut self, source: &str) -> Result<AssembledProgram, AssemblerError> {
        self.registers.clear();
        self.instructions.clear();
        self.labels.clear();
        self.unresolved_labels.clear();

        let mut in_registers = false;
        let mut in_program = false;

        for (idx, line) in source.lines().enumerate() {
            self.line_number = idx + 1;
            let line = line.trim();

            // Skip empty lines and comments
            if line.is_empty() || line.starts_with(';') {
                continue;
            }

            // Remove inline comments
            let line = if let Some(pos) = line.find(';') {
                line[..pos].trim()
            } else {
                line
            };

            if line.is_empty() {
                continue;
            }

            // Section directives
            if line.starts_with('.') {
                match line.to_lowercase().as_str() {
                    ".registers" => {
                        in_registers = true;
                        in_program = false;
                        continue;
                    }
                    ".program" | ".code" => {
                        in_registers = false;
                        in_program = true;
                        continue;
                    }
                    _ => {
                        return Err(self.error(format!("Unknown directive: {}", line)));
                    }
                }
            }

            // Parse content based on section
            if in_registers {
                self.parse_register(line)?;
            } else if in_program {
                self.parse_instruction(line)?;
            } else {
                return Err(self.error("Code outside of section".to_string()));
            }
        }

        // Resolve labels
        self.resolve_labels()?;

        // Infer input/output shapes from load_input/store_output instructions
        let input_shape = self.infer_input_shape();
        let output_shape = self.infer_output_shape();

        Ok(AssembledProgram {
            name: String::new(),
            registers: self.registers.clone(),
            instructions: self.instructions.clone(),
            labels: self.labels.clone(),
            input_shape,
            output_shape,
        })
    }

    /// Parse a register definition line
    fn parse_register(&mut self, line: &str) -> Result<(), AssemblerError> {
        // Format: C0: ternary[32, 12]  key="chip.audio.w1"
        // Or:     H0: i32[12]

        let colon_pos = line
            .find(':')
            .ok_or_else(|| self.error("Missing ':' in register definition".to_string()))?;

        let reg_name = line[..colon_pos].trim();
        let rest = line[colon_pos + 1..].trim();

        // Parse register ID
        let reg = Register::parse(reg_name)
            .ok_or_else(|| self.error(format!("Invalid register: {}", reg_name)))?;

        // Parse type and shape: ternary[32, 12] or i32[12]
        let bracket_start = rest
            .find('[')
            .ok_or_else(|| self.error("Missing '[' in type definition".to_string()))?;
        let bracket_end = rest
            .find(']')
            .ok_or_else(|| self.error("Missing ']' in type definition".to_string()))?;

        let dtype_str = rest[..bracket_start].trim();
        let shape_str = &rest[bracket_start + 1..bracket_end];

        let dtype = Dtype::parse(dtype_str)
            .ok_or_else(|| self.error(format!("Unknown dtype: {}", dtype_str)))?;

        // Parse shape dimensions
        let shape: Vec<usize> = shape_str
            .split(',')
            .map(|s| {
                s.trim()
                    .parse()
                    .map_err(|_| self.error(format!("Invalid dimension: {}", s.trim())))
            })
            .collect::<Result<Vec<_>, _>>()?;

        // Parse optional key="..."
        let thermogram_key = if let Some(key_pos) = rest.find("key=") {
            let key_start = rest[key_pos..].find('"').map(|p| key_pos + p + 1);
            let key_end = key_start.and_then(|s| rest[s..].find('"').map(|e| s + e));

            if let (Some(start), Some(end)) = (key_start, key_end) {
                Some(rest[start..end].to_string())
            } else {
                None
            }
        } else {
            None
        };

        // Parse optional frozen=true
        let frozen = rest.contains("frozen=true");

        let mut meta = RegisterMeta::with_shape(reg, shape, dtype);
        meta.thermogram_key = thermogram_key;
        meta.frozen = frozen;

        self.registers.push(meta);
        Ok(())
    }

    /// Parse an instruction line
    fn parse_instruction(&mut self, line: &str) -> Result<(), AssemblerError> {
        // Check for label
        if line.ends_with(':') {
            let label = line[..line.len() - 1].trim();
            self.labels.insert(label.to_string(), self.instructions.len());
            return Ok(());
        }

        // Split into mnemonic and operands
        let parts: Vec<&str> = line.splitn(2, char::is_whitespace).collect();
        let mnemonic = parts[0].to_lowercase();
        let operands = parts.get(1).map(|s| s.trim()).unwrap_or("");

        // Parse operands
        let ops: Vec<&str> = if operands.is_empty() {
            Vec::new()
        } else {
            operands.split(',').map(|s| s.trim()).collect()
        };

        // Build instruction based on mnemonic
        let instr = self.build_instruction(&mnemonic, &ops)?;
        self.instructions.push(instr);

        Ok(())
    }

    /// Build an instruction from mnemonic and operands
    fn build_instruction(
        &mut self,
        mnemonic: &str,
        ops: &[&str],
    ) -> Result<Instruction, AssemblerError> {
        match mnemonic {
            // System
            "nop" => Ok(Instruction::nop()),
            "halt" => Ok(Instruction::halt()),

            // Register management
            "load_input" => {
                let target = self.parse_register_operand(ops.get(0))?;
                Ok(Instruction::load_input(target))
            }
            "store_output" => {
                let source = self.parse_register_operand(ops.get(0))?;
                Ok(Instruction::store_output(source))
            }
            "copy_reg" | "copy" | "mov" => {
                let target = self.parse_register_operand(ops.get(0))?;
                let source = self.parse_register_operand(ops.get(1))?;
                Ok(Instruction::new(
                    Action::COPY_REG,
                    target,
                    source,
                    0,
                    [0, 0, 0],
                ))
            }
            "zero_reg" | "zero" => {
                let target = self.parse_register_operand(ops.get(0))?;
                Ok(Instruction::new(
                    Action::ZERO_REG,
                    target,
                    Register::NULL,
                    0,
                    [0, 0, 0],
                ))
            }

            // Forward ops
            "ternary_matmul" => {
                let target = self.parse_register_operand(ops.get(0))?;
                let weights = self.parse_register_operand(ops.get(1))?;
                let input = self.parse_register_operand(ops.get(2))?;
                Ok(Instruction::ternary_matmul(target, weights, input))
            }
            "add" => {
                let target = self.parse_register_operand(ops.get(0))?;
                let source = self.parse_register_operand(ops.get(1))?;
                let other = self.parse_register_operand(ops.get(2))?;
                Ok(Instruction::add(target, source, other))
            }
            "sub" => {
                let target = self.parse_register_operand(ops.get(0))?;
                let source = self.parse_register_operand(ops.get(1))?;
                let other = self.parse_register_operand(ops.get(2))?;
                Ok(Instruction::new(
                    Action::SUB,
                    target,
                    source,
                    other.0,
                    [0, 0, 0],
                ))
            }
            "mul" => {
                let target = self.parse_register_operand(ops.get(0))?;
                let source = self.parse_register_operand(ops.get(1))?;
                let other = self.parse_register_operand(ops.get(2))?;
                Ok(Instruction::new(
                    Action::MUL,
                    target,
                    source,
                    other.0,
                    [0, 0, 0],
                ))
            }
            "relu" => {
                let target = self.parse_register_operand(ops.get(0))?;
                let source = self.parse_register_operand(ops.get(1))?;
                Ok(Instruction::relu(target, source))
            }
            "sigmoid" => {
                let target = self.parse_register_operand(ops.get(0))?;
                let source = self.parse_register_operand(ops.get(1))?;
                // Parse optional gain=X.X parameter
                let gain = self.parse_gain_param(ops.get(2)).unwrap_or(64); // 4.0 * 16 = 64
                Ok(Instruction::new(
                    Action::SIGMOID,
                    target,
                    source,
                    0,
                    [gain, 0, 0],
                ))
            }
            "shift" => {
                let target = self.parse_register_operand(ops.get(0))?;
                let source = self.parse_register_operand(ops.get(1))?;
                let amount = self.parse_immediate(ops.get(2))?;
                Ok(Instruction::shift(target, source, amount as u8))
            }
            "cmp_gt" => {
                let target = self.parse_register_operand(ops.get(0))?;
                let source = self.parse_register_operand(ops.get(1))?;
                let other = self.parse_register_operand(ops.get(2))?;
                Ok(Instruction::new(
                    Action::CMP_GT,
                    target,
                    source,
                    other.0,
                    [0, 0, 0],
                ))
            }
            "max_reduce" => {
                let target = self.parse_register_operand(ops.get(0))?;
                let source = self.parse_register_operand(ops.get(1))?;
                Ok(Instruction::new(
                    Action::MAX_REDUCE,
                    target,
                    source,
                    0,
                    [0, 0, 0],
                ))
            }

            // Learning ops
            "add_babble" => {
                let target = self.parse_register_operand(ops.get(0))?;
                let layer = self.parse_layer_param(ops.get(1))?;
                Ok(Instruction::add_babble(target, layer))
            }
            "mark_elig" | "mark_eligibility" => {
                let output = self.parse_register_operand(ops.get(0))?;
                let input = self.parse_register_operand(ops.get(1))?;
                let layer = self.parse_layer_param(ops.get(2))?;
                Ok(Instruction::mark_eligibility(output, input, layer))
            }
            "load_target" => {
                let target = self.parse_register_operand(ops.get(0))?;
                Ok(Instruction::new(
                    Action::LOAD_TARGET,
                    target,
                    Register::NULL,
                    0,
                    [0, 0, 0],
                ))
            }
            "mastery_update" => {
                // mastery_update weights, activity, direction [, scale=15, threshold_div=4]
                let weights = self.parse_register_operand(ops.get(0))?;
                let activity = self.parse_register_operand(ops.get(1))?;
                let direction = self.parse_register_operand(ops.get(2))?;
                let scale = self.parse_param_or_default(ops.get(3), "scale", 15)?;
                let threshold_div = self.parse_param_or_default(ops.get(4), "threshold_div", 4)?;
                Ok(Instruction::new(
                    Action::MASTERY_UPDATE,
                    weights,
                    activity,
                    direction.0,
                    [scale, threshold_div, 0],
                ))
            }
            "mastery_commit" => {
                // mastery_commit weights [, threshold=50, step=5]
                let weights = self.parse_register_operand(ops.get(0))?;
                let threshold = self.parse_param_or_default(ops.get(1), "threshold", 50)?;
                let step = self.parse_param_or_default(ops.get(2), "step", 5)?;
                Ok(Instruction::new(
                    Action::MASTERY_COMMIT,
                    weights,
                    Register::NULL,
                    0,
                    [threshold, step, 0],
                ))
            }

            // Ternary ops
            "dequantize" => {
                let target = self.parse_register_operand(ops.get(0))?;
                let source = if ops.len() > 1 {
                    self.parse_register_operand(ops.get(1))?
                } else {
                    target
                };
                let scale = self.parse_scale_param(ops.last())?;
                Ok(Instruction::dequantize(target, source, scale))
            }
            "ternary_add_bias" => {
                let target = self.parse_register_operand(ops.get(0))?;
                let source = self.parse_register_operand(ops.get(1))?;
                let bias = self.parse_register_operand(ops.get(2))?;
                Ok(Instruction::new(
                    Action::TERNARY_ADD_BIAS,
                    target,
                    source,
                    bias.0,
                    [0, 0, 0],
                ))
            }
            "embed_lookup" => {
                // embed_lookup target, table, indices
                let target = self.parse_register_operand(ops.get(0))?;
                let table = self.parse_register_operand(ops.get(1))?;
                let indices = self.parse_register_operand(ops.get(2))?;
                Ok(Instruction::embed_lookup(target, table, indices))
            }
            "reduce_avg" => {
                // reduce_avg target, source, start, count
                let target = self.parse_register_operand(ops.get(0))?;
                let source = self.parse_register_operand(ops.get(1))?;
                let start = self.parse_immediate(ops.get(2))? as u8;
                let count = self.parse_immediate(ops.get(3))? as u8;
                Ok(Instruction::reduce_avg(target, source, start, count))
            }
            "slice" | "narrow" => {
                // slice target, source, start, len
                let target = self.parse_register_operand(ops.get(0))?;
                let source = self.parse_register_operand(ops.get(1))?;
                let start = self.parse_immediate(ops.get(2))? as u8;
                let len = self.parse_immediate(ops.get(3))? as u8;
                Ok(Instruction::slice(target, source, start, len))
            }
            "argmax" => {
                // argmax target, source
                let target = self.parse_register_operand(ops.get(0))?;
                let source = self.parse_register_operand(ops.get(1))?;
                Ok(Instruction::argmax(target, source))
            }
            "concat" | "cat" => {
                // concat target, source, other
                let target = self.parse_register_operand(ops.get(0))?;
                let source = self.parse_register_operand(ops.get(1))?;
                let other = self.parse_register_operand(ops.get(2))?;
                Ok(Instruction::concat(target, source, other))
            }
            "squeeze" => {
                // squeeze target, source, dim
                let target = self.parse_register_operand(ops.get(0))?;
                let source = self.parse_register_operand(ops.get(1))?;
                let dim = self.parse_immediate(ops.get(2)).unwrap_or(0) as u8;
                Ok(Instruction::squeeze(target, source, dim))
            }
            "unsqueeze" => {
                // unsqueeze target, source, dim
                let target = self.parse_register_operand(ops.get(0))?;
                let source = self.parse_register_operand(ops.get(1))?;
                let dim = self.parse_immediate(ops.get(2)).unwrap_or(0) as u8;
                Ok(Instruction::unsqueeze(target, source, dim))
            }
            "transpose" => {
                // transpose target, source, dim1, dim2
                let target = self.parse_register_operand(ops.get(0))?;
                let source = self.parse_register_operand(ops.get(1))?;
                let dim1 = self.parse_immediate(ops.get(2)).unwrap_or(0) as u8;
                let dim2 = self.parse_immediate(ops.get(3)).unwrap_or(1) as u8;
                Ok(Instruction::transpose(target, source, dim1, dim2))
            }
            "gate_update" => {
                // gate_update target, gate, update, state
                let target = self.parse_register_operand(ops.get(0))?;
                let gate = self.parse_register_operand(ops.get(1))?;
                let update = self.parse_register_operand(ops.get(2))?;
                let state = self.parse_register_operand(ops.get(3))?;
                Ok(Instruction::gate_update(target, gate, update, state))
            }

            // Control flow
            "loop" => {
                let count = self.parse_immediate(ops.get(0))? as u16;
                Ok(Instruction::loop_n(count))
            }
            "end_loop" => Ok(Instruction::end_loop()),
            "break" => Ok(Instruction::break_loop()),

            _ => Err(self.error(format!("Unknown mnemonic: {}", mnemonic))),
        }
    }

    /// Parse a register operand
    fn parse_register_operand(
        &self,
        op: Option<&&str>,
    ) -> Result<Register, AssemblerError> {
        let s = op.ok_or_else(|| self.error("Missing register operand".to_string()))?;
        Register::parse(s).ok_or_else(|| self.error(format!("Invalid register: {}", s)))
    }

    /// Parse an immediate value
    fn parse_immediate(&self, op: Option<&&str>) -> Result<i32, AssemblerError> {
        let s = op.ok_or_else(|| self.error("Missing immediate operand".to_string()))?;

        // Handle hex prefix
        if s.starts_with("0x") || s.starts_with("0X") {
            i32::from_str_radix(&s[2..], 16)
                .map_err(|_| self.error(format!("Invalid hex: {}", s)))
        } else {
            s.parse()
                .map_err(|_| self.error(format!("Invalid number: {}", s)))
        }
    }

    /// Parse layer=N parameter
    fn parse_layer_param(&self, op: Option<&&str>) -> Result<u8, AssemblerError> {
        let s = op.ok_or_else(|| self.error("Missing layer parameter".to_string()))?;

        if let Some(eq_pos) = s.find('=') {
            let value = &s[eq_pos + 1..];
            value
                .parse()
                .map_err(|_| self.error(format!("Invalid layer number: {}", value)))
        } else {
            s.parse()
                .map_err(|_| self.error(format!("Invalid layer: {}", s)))
        }
    }

    /// Parse scale=N parameter
    fn parse_scale_param(&self, op: Option<&&str>) -> Result<u16, AssemblerError> {
        let s = op.ok_or_else(|| self.error("Missing scale parameter".to_string()))?;

        if let Some(eq_pos) = s.find('=') {
            let value = &s[eq_pos + 1..];
            value
                .parse()
                .map_err(|_| self.error(format!("Invalid scale: {}", value)))
        } else {
            s.parse()
                .map_err(|_| self.error(format!("Invalid scale: {}", s)))
        }
    }

    /// Parse gain=X.X parameter (returns gain * 16 as u8)
    fn parse_gain_param(&self, op: Option<&&str>) -> Result<u8, AssemblerError> {
        let s = match op {
            Some(s) => *s,
            None => return Err(self.error("Missing gain parameter".to_string())),
        };

        // Extract value after "gain=" if present
        let value_str = if let Some(eq_pos) = s.find('=') {
            &s[eq_pos + 1..]
        } else {
            s
        };

        // Parse as float and convert to u8 (gain * 16)
        let gain: f32 = value_str
            .parse()
            .map_err(|_| self.error(format!("Invalid gain: {}", value_str)))?;

        Ok((gain * 16.0).clamp(0.0, 255.0) as u8)
    }

    /// Parse param=value or use default
    fn parse_param_or_default(
        &self,
        op: Option<&&str>,
        _name: &str,
        default: u8,
    ) -> Result<u8, AssemblerError> {
        match op {
            Some(s) => {
                let value_str = if let Some(eq_pos) = s.find('=') {
                    &s[eq_pos + 1..]
                } else {
                    *s
                };
                value_str
                    .parse()
                    .map_err(|_| self.error(format!("Invalid parameter: {}", s)))
            }
            None => Ok(default),
        }
    }

    /// Infer input shape from load_input instruction's target register
    fn infer_input_shape(&self) -> Vec<usize> {
        for instr in &self.instructions {
            if instr.action == Action::LOAD_INPUT {
                // Find the target register's shape
                if let Some(reg_meta) = self.registers.iter().find(|r| r.id == instr.target) {
                    return reg_meta.shape.clone();
                }
            }
        }
        vec![1] // Default if no load_input found
    }

    /// Infer output shape from store_output instruction's source register
    fn infer_output_shape(&self) -> Vec<usize> {
        for instr in &self.instructions {
            if instr.action == Action::STORE_OUTPUT {
                // Find the source register's shape
                if let Some(reg_meta) = self.registers.iter().find(|r| r.id == instr.source) {
                    return reg_meta.shape.clone();
                }
            }
        }
        vec![1] // Default if no store_output found
    }

    /// Resolve label references
    fn resolve_labels(&mut self) -> Result<(), AssemblerError> {
        for (instr_idx, label) in &self.unresolved_labels {
            let target_idx = self
                .labels
                .get(label)
                .ok_or_else(|| self.error(format!("Undefined label: {}", label)))?;

            // Update instruction's modifier with target address
            let instr = &mut self.instructions[*instr_idx];
            let count = (*target_idx as u16).to_be_bytes();
            instr.modifier[0] = count[0];
            instr.modifier[1] = count[1];
        }
        Ok(())
    }

    /// Create an error at current line
    fn error(&self, message: String) -> AssemblerError {
        AssemblerError {
            line: self.line_number,
            message,
        }
    }
}

impl Default for Assembler {
    fn default() -> Self {
        Self::new()
    }
}

/// Assembler error
#[derive(Debug, Clone)]
pub struct AssemblerError {
    pub line: usize,
    pub message: String,
}

impl std::fmt::Display for AssemblerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Line {}: {}", self.line, self.message)
    }
}

impl std::error::Error for AssemblerError {}

/// Convenience function to assemble source
pub fn assemble(source: &str) -> Result<AssembledProgram, AssemblerError> {
    Assembler::new().assemble(source)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_assemble_simple() {
        let source = r#"
; Simple test program
.registers
    H0: i32[12]
    H1: i32[32]
    C0: ternary[32, 12]  key="test.w1"

.program
    load_input    H0
    ternary_matmul H1, C0, H0
    relu          H1, H1
    store_output  H1
    halt
"#;

        let program = assemble(source).expect("Assembly failed");

        assert_eq!(program.registers.len(), 3);
        assert_eq!(program.instructions.len(), 5);

        // Check first instruction
        assert_eq!(program.instructions[0].action, Action::LOAD_INPUT);

        // Check ternary_matmul
        assert_eq!(program.instructions[1].action, Action::TERNARY_MATMUL);

        // Check halt
        assert_eq!(program.instructions[4].action, Action::HALT);
    }

    #[test]
    fn test_register_parsing() {
        let source = r#"
.registers
    C0: ternary[32, 12]  key="chip.audio.w1"
    C1: ternary[32]      key="chip.audio.b1"
    H0: i32[12]
    H1: i32[32]

.program
    halt
"#;

        let program = assemble(source).expect("Assembly failed");

        assert_eq!(program.registers.len(), 4);

        // Check cold register
        let c0 = &program.registers[0];
        assert!(c0.id.is_cold());
        assert_eq!(c0.shape, vec![32, 12]);
        assert_eq!(c0.dtype, Dtype::Ternary);
        assert_eq!(c0.thermogram_key, Some("chip.audio.w1".to_string()));

        // Check hot register
        let h0 = &program.registers[2];
        assert!(h0.id.is_hot());
        assert_eq!(h0.shape, vec![12]);
        assert_eq!(h0.dtype, Dtype::I32);
    }

    #[test]
    fn test_binary_output() {
        let source = r#"
.registers
    H0: i32[12]

.program
    load_input H0
    halt
"#;

        let program = assemble(source).unwrap();
        let binary = program.to_binary();

        // Check magic
        assert_eq!(&binary[0..4], &TERNSIG_MAGIC);

        // Check version
        let version = u16::from_le_bytes([binary[4], binary[5]]);
        assert_eq!(version, TERNSIG_VERSION);
    }
}
