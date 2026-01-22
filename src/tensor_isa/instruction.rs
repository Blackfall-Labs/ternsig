//! TensorInstruction - 8-byte instruction format for TensorISA
//!
//! ## Format
//!
//! ```text
//! [ACTION:2][TARGET:1][SOURCE:1][AUX:1][MODIFIER:3]
//!     ↓         ↓         ↓        ↓       ↓
//!   opcode   dest_reg  src_reg  aux_val  params
//!
//! Total: 8 bytes
//! ```
//!
//! ## Examples
//!
//! ```text
//! TERNARY_MATMUL H1, C0, H0
//!   ACTION = 0x4000 (TERNARY_MATMUL)
//!   TARGET = 0x01 (H1)
//!   SOURCE = 0x10 (C0)
//!   AUX    = 0x00 (H0)
//!   MODIFIER = [0, 0, 0]
//!
//! SHIFT H1, H1, 8
//!   ACTION = 0x3009 (SHIFT)
//!   TARGET = 0x01 (H1)
//!   SOURCE = 0x01 (H1)
//!   AUX    = 8 (shift amount)
//!   MODIFIER = [0, 0, 0]
//! ```

use super::{TensorAction, TensorModifier, TensorRegister};
use std::fmt;

/// A TensorISA instruction (8 bytes)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TensorInstruction {
    /// Operation opcode
    pub action: TensorAction,
    /// Destination register
    pub target: TensorRegister,
    /// Source register
    pub source: TensorRegister,
    /// Auxiliary register or immediate value
    pub aux: u8,
    /// Operation-specific modifier (3 bytes)
    pub modifier: [u8; 3],
}

impl TensorInstruction {
    /// Instruction size in bytes
    pub const SIZE: usize = 8;

    /// Create a new instruction
    pub const fn new(
        action: TensorAction,
        target: TensorRegister,
        source: TensorRegister,
        aux: u8,
        modifier: [u8; 3],
    ) -> Self {
        Self {
            action,
            target,
            source,
            aux,
            modifier,
        }
    }

    /// Create a simple instruction with default modifier
    pub const fn simple(
        action: TensorAction,
        target: TensorRegister,
        source: TensorRegister,
    ) -> Self {
        Self::new(action, target, source, 0, [0, 0, 0])
    }

    /// Create a system instruction (no registers)
    pub const fn system(action: TensorAction) -> Self {
        Self::new(action, TensorRegister::NULL, TensorRegister::NULL, 0, [0, 0, 0])
    }

    // =========================================================================
    // Common Instruction Builders
    // =========================================================================

    /// NOP instruction
    pub const fn nop() -> Self {
        Self::system(TensorAction::NOP)
    }

    /// HALT instruction
    pub const fn halt() -> Self {
        Self::system(TensorAction::HALT)
    }

    /// Load from input buffer: target = input_buffer
    pub const fn load_input(target: TensorRegister) -> Self {
        Self::simple(TensorAction::LOAD_INPUT, target, TensorRegister::NULL)
    }

    /// Store to output buffer: output_buffer = source
    pub const fn store_output(source: TensorRegister) -> Self {
        Self::simple(TensorAction::STORE_OUTPUT, TensorRegister::NULL, source)
    }

    /// Ternary matrix multiply: target = weights @ input
    pub const fn ternary_matmul(
        target: TensorRegister,
        weights: TensorRegister,
        input: TensorRegister,
    ) -> Self {
        Self::new(TensorAction::TERNARY_MATMUL, target, weights, input.0, [0, 0, 0])
    }

    /// Add: target = source + aux_reg
    pub const fn add(
        target: TensorRegister,
        source: TensorRegister,
        other: TensorRegister,
    ) -> Self {
        Self::new(TensorAction::ADD, target, source, other.0, [0, 0, 0])
    }

    /// ReLU: target = max(0, source)
    pub const fn relu(target: TensorRegister, source: TensorRegister) -> Self {
        Self::simple(TensorAction::RELU, target, source)
    }

    /// Shift right: target = source >> shift_amount
    pub const fn shift(target: TensorRegister, source: TensorRegister, shift_amount: u8) -> Self {
        Self::new(TensorAction::SHIFT, target, source, shift_amount, [0, 0, 0])
    }

    /// Add babble noise
    pub fn add_babble(target: TensorRegister, layer_index: u8) -> Self {
        Self::new(TensorAction::ADD_BABBLE, target, TensorRegister::NULL, layer_index, [0, 0, 0])
    }

    /// Mark eligibility: mark weights based on input/output activity
    pub const fn mark_eligibility(
        output: TensorRegister,
        input: TensorRegister,
        layer_index: u8,
    ) -> Self {
        Self::new(TensorAction::MARK_ELIGIBILITY, output, input, layer_index, [0, 0, 0])
    }

    /// Dequantize: target = float(source) / scale
    pub fn dequantize(target: TensorRegister, source: TensorRegister, scale: u16) -> Self {
        let scale_bytes = scale.to_be_bytes();
        Self::new(
            TensorAction::DEQUANTIZE,
            target,
            source,
            0,
            [scale_bytes[0], scale_bytes[1], 0],
        )
    }

    /// Loop N times
    pub fn loop_n(count: u16) -> Self {
        let count_bytes = count.to_be_bytes();
        Self::new(
            TensorAction::LOOP,
            TensorRegister::NULL,
            TensorRegister::NULL,
            0,
            [count_bytes[0], count_bytes[1], 0],
        )
    }

    /// End of loop
    pub const fn end_loop() -> Self {
        Self::system(TensorAction::END_LOOP)
    }

    /// Break from loop
    pub const fn break_loop() -> Self {
        Self::system(TensorAction::BREAK)
    }

    // =========================================================================
    // Serialization
    // =========================================================================

    /// Parse from 8 bytes (big-endian)
    pub fn from_bytes(bytes: &[u8; 8]) -> Self {
        Self {
            action: TensorAction::from_u16(u16::from_be_bytes([bytes[0], bytes[1]])),
            target: TensorRegister(bytes[2]),
            source: TensorRegister(bytes[3]),
            aux: bytes[4],
            modifier: [bytes[5], bytes[6], bytes[7]],
        }
    }

    /// Serialize to 8 bytes (big-endian)
    pub fn to_bytes(&self) -> [u8; 8] {
        let action_bytes = self.action.as_u16().to_be_bytes();
        [
            action_bytes[0],
            action_bytes[1],
            self.target.0,
            self.source.0,
            self.aux,
            self.modifier[0],
            self.modifier[1],
            self.modifier[2],
        ]
    }

    /// Parse multiple instructions from byte slice
    pub fn parse_all(bytes: &[u8]) -> Result<Vec<Self>, &'static str> {
        if bytes.len() % Self::SIZE != 0 {
            return Err("byte length must be multiple of 8");
        }

        let mut result = Vec::with_capacity(bytes.len() / Self::SIZE);
        for chunk in bytes.chunks_exact(Self::SIZE) {
            let arr: [u8; 8] = chunk.try_into().unwrap();
            result.push(Self::from_bytes(&arr));
        }
        Ok(result)
    }

    /// Serialize multiple instructions to bytes
    pub fn to_bytes_all(instructions: &[Self]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(instructions.len() * Self::SIZE);
        for instr in instructions {
            bytes.extend_from_slice(&instr.to_bytes());
        }
        bytes
    }

    // =========================================================================
    // Query Methods
    // =========================================================================

    /// Convert to opcode string (hex format)
    pub fn to_opcode_string(&self) -> String {
        format!(
            "{:04X}:{:02X}:{:02X}:{:02X}:{:02X}{:02X}{:02X}",
            self.action.as_u16(),
            self.target.0,
            self.source.0,
            self.aux,
            self.modifier[0],
            self.modifier[1],
            self.modifier[2],
        )
    }

    /// Check if this instruction modifies program counter
    pub fn modifies_pc(&self) -> bool {
        self.action.modifies_pc()
    }

    /// Check if this is a domain operation requiring external execution
    pub fn is_domain_op(&self) -> bool {
        self.action.is_domain_op()
    }

    /// Get the modifier as a TensorModifier
    pub fn get_modifier(&self) -> TensorModifier {
        TensorModifier::from_bytes(self.modifier)
    }

    /// Extract shape from modifier (for DEFINE_LAYER)
    /// Returns (input_dim, output_dim)
    pub fn shape(&self) -> (usize, usize) {
        let combined = u32::from_be_bytes([0, self.modifier[0], self.modifier[1], self.modifier[2]]);
        let input = ((combined >> 12) & 0xFFF) as usize;
        let output = (combined & 0xFFF) as usize;
        (input, output)
    }

    /// Extract scale from modifier (for DEQUANTIZE, SCALE)
    pub fn scale(&self) -> u16 {
        u16::from_be_bytes([self.modifier[0], self.modifier[1]])
    }

    /// Extract count from modifier (for LOOP, SKIP)
    pub fn count(&self) -> u16 {
        u16::from_be_bytes([self.modifier[0], self.modifier[1]])
    }
}

impl fmt::Display for TensorInstruction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Format based on operation category
        match self.action.category() {
            0x00 => {
                // System
                write!(f, "{}", self.action)
            }
            0x10 => {
                // Register management
                if self.target.is_null() {
                    write!(f, "{} {}", self.action, self.source)
                } else if self.source.is_null() {
                    write!(f, "{} {}", self.action, self.target)
                } else {
                    write!(f, "{} {}, {}", self.action, self.target, self.source)
                }
            }
            0x30 | 0x40 => {
                // Forward/Ternary ops
                if self.aux != 0 && self.aux != 0xFF {
                    let aux_reg = TensorRegister(self.aux);
                    write!(f, "{} {}, {}, {}", self.action, self.target, self.source, aux_reg)
                } else if self.action == TensorAction::SHIFT {
                    write!(f, "{} {}, {}, {}", self.action, self.target, self.source, self.aux)
                } else {
                    write!(f, "{} {}, {}", self.action, self.target, self.source)
                }
            }
            0x50 => {
                // Learning ops
                write!(f, "{} {}, {}, layer={}", self.action, self.target, self.source, self.aux)
            }
            0x60 => {
                // Control flow
                if self.action == TensorAction::LOOP {
                    write!(f, "{} {}", self.action, self.count())
                } else {
                    write!(f, "{}", self.action)
                }
            }
            _ => write!(f, "{} {}, {}, {}", self.action, self.target, self.source, self.aux),
        }
    }
}

impl Default for TensorInstruction {
    fn default() -> Self {
        Self::nop()
    }
}

/// Builder for TensorInstruction
pub struct TensorInstructionBuilder {
    action: TensorAction,
    target: TensorRegister,
    source: TensorRegister,
    aux: u8,
    modifier: [u8; 3],
}

impl TensorInstructionBuilder {
    pub fn new(action: TensorAction) -> Self {
        Self {
            action,
            target: TensorRegister::NULL,
            source: TensorRegister::NULL,
            aux: 0,
            modifier: [0, 0, 0],
        }
    }

    pub fn target(mut self, reg: TensorRegister) -> Self {
        self.target = reg;
        self
    }

    pub fn source(mut self, reg: TensorRegister) -> Self {
        self.source = reg;
        self
    }

    pub fn aux(mut self, val: u8) -> Self {
        self.aux = val;
        self
    }

    pub fn aux_reg(mut self, reg: TensorRegister) -> Self {
        self.aux = reg.0;
        self
    }

    pub fn modifier(mut self, m: [u8; 3]) -> Self {
        self.modifier = m;
        self
    }

    pub fn shape(mut self, input_dim: usize, output_dim: usize) -> Self {
        let combined = ((input_dim & 0xFFF) << 12) | (output_dim & 0xFFF);
        self.modifier = [(combined >> 16) as u8, (combined >> 8) as u8, combined as u8];
        self
    }

    pub fn scale(mut self, scale: u16) -> Self {
        let bytes = scale.to_be_bytes();
        self.modifier[0] = bytes[0];
        self.modifier[1] = bytes[1];
        self
    }

    pub fn count(mut self, count: u16) -> Self {
        let bytes = count.to_be_bytes();
        self.modifier[0] = bytes[0];
        self.modifier[1] = bytes[1];
        self
    }

    pub fn build(self) -> TensorInstruction {
        TensorInstruction::new(self.action, self.target, self.source, self.aux, self.modifier)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_instruction_roundtrip() {
        let instr = TensorInstruction::ternary_matmul(
            TensorRegister::hot(1),
            TensorRegister::cold(0),
            TensorRegister::hot(0),
        );

        let bytes = instr.to_bytes();
        assert_eq!(bytes.len(), 8);

        let parsed = TensorInstruction::from_bytes(&bytes);
        assert_eq!(instr, parsed);
    }

    #[test]
    fn test_instruction_builders() {
        let instr = TensorInstruction::relu(TensorRegister::hot(2), TensorRegister::hot(1));
        assert_eq!(instr.action, TensorAction::RELU);
        assert_eq!(instr.target, TensorRegister::hot(2));
        assert_eq!(instr.source, TensorRegister::hot(1));
    }

    #[test]
    fn test_shape_encoding() {
        let instr = TensorInstructionBuilder::new(TensorAction::DEFINE_LAYER)
            .target(TensorRegister::cold(0))
            .shape(32, 12)
            .build();

        let (input_dim, output_dim) = instr.shape();
        assert_eq!(input_dim, 32);
        assert_eq!(output_dim, 12);
    }

    #[test]
    fn test_parse_all() {
        let instructions = vec![
            TensorInstruction::load_input(TensorRegister::hot(0)),
            TensorInstruction::ternary_matmul(
                TensorRegister::hot(1),
                TensorRegister::cold(0),
                TensorRegister::hot(0),
            ),
            TensorInstruction::relu(TensorRegister::hot(2), TensorRegister::hot(1)),
            TensorInstruction::halt(),
        ];

        let bytes = TensorInstruction::to_bytes_all(&instructions);
        assert_eq!(bytes.len(), 32); // 4 instructions * 8 bytes

        let parsed = TensorInstruction::parse_all(&bytes).unwrap();
        assert_eq!(parsed.len(), 4);
        assert_eq!(parsed[0].action, TensorAction::LOAD_INPUT);
        assert_eq!(parsed[1].action, TensorAction::TERNARY_MATMUL);
        assert_eq!(parsed[2].action, TensorAction::RELU);
        assert_eq!(parsed[3].action, TensorAction::HALT);
    }

    #[test]
    fn test_display() {
        let instr = TensorInstruction::ternary_matmul(
            TensorRegister::hot(1),
            TensorRegister::cold(0),
            TensorRegister::hot(0),
        );
        let display = format!("{}", instr);
        assert!(display.contains("TERNARY_MATMUL"));
        assert!(display.contains("H1"));
        assert!(display.contains("C0"));
        assert!(display.contains("H0"));
    }
}
