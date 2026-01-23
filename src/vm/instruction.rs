//! Instruction - 8-byte instruction format for Ternsig VM
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

use super::{Action, Modifier, Register};
use std::fmt;

/// A TensorISA instruction (8 bytes)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Instruction {
    /// Operation opcode
    pub action: Action,
    /// Destination register
    pub target: Register,
    /// Source register
    pub source: Register,
    /// Auxiliary register or immediate value
    pub aux: u8,
    /// Operation-specific modifier (3 bytes)
    pub modifier: [u8; 3],
}

impl Instruction {
    /// Instruction size in bytes
    pub const SIZE: usize = 8;

    /// Create a new instruction
    pub const fn new(
        action: Action,
        target: Register,
        source: Register,
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
        action: Action,
        target: Register,
        source: Register,
    ) -> Self {
        Self::new(action, target, source, 0, [0, 0, 0])
    }

    /// Create a system instruction (no registers)
    pub const fn system(action: Action) -> Self {
        Self::new(action, Register::NULL, Register::NULL, 0, [0, 0, 0])
    }

    // =========================================================================
    // Common Instruction Builders
    // =========================================================================

    /// NOP instruction
    pub const fn nop() -> Self {
        Self::system(Action::NOP)
    }

    /// HALT instruction
    pub const fn halt() -> Self {
        Self::system(Action::HALT)
    }

    /// Load from input buffer: target = input_buffer
    pub const fn load_input(target: Register) -> Self {
        Self::simple(Action::LOAD_INPUT, target, Register::NULL)
    }

    /// Store to output buffer: output_buffer = source
    pub const fn store_output(source: Register) -> Self {
        Self::simple(Action::STORE_OUTPUT, Register::NULL, source)
    }

    /// Ternary matrix multiply: target = weights @ input
    pub const fn ternary_matmul(
        target: Register,
        weights: Register,
        input: Register,
    ) -> Self {
        Self::new(Action::TERNARY_MATMUL, target, weights, input.0, [0, 0, 0])
    }

    /// Add: target = source + aux_reg
    pub const fn add(
        target: Register,
        source: Register,
        other: Register,
    ) -> Self {
        Self::new(Action::ADD, target, source, other.0, [0, 0, 0])
    }

    /// ReLU: target = max(0, source)
    pub const fn relu(target: Register, source: Register) -> Self {
        Self::simple(Action::RELU, target, source)
    }

    /// Shift right: target = source >> shift_amount
    pub const fn shift(target: Register, source: Register, shift_amount: u8) -> Self {
        Self::new(Action::SHIFT, target, source, shift_amount, [0, 0, 0])
    }

    /// Add babble noise
    pub fn add_babble(target: Register, layer_index: u8) -> Self {
        Self::new(Action::ADD_BABBLE, target, Register::NULL, layer_index, [0, 0, 0])
    }

    /// Mark eligibility: mark weights based on input/output activity
    pub const fn mark_eligibility(
        output: Register,
        input: Register,
        layer_index: u8,
    ) -> Self {
        Self::new(Action::MARK_ELIGIBILITY, output, input, layer_index, [0, 0, 0])
    }

    /// Embedding lookup: target[i] = table[indices[i]]
    /// table is a 2D cold register (num_embeddings x embedding_dim)
    /// indices is a hot register with integer indices
    /// output is a hot register receiving looked-up embeddings
    pub const fn embed_lookup(
        target: Register,
        table: Register,
        indices: Register,
    ) -> Self {
        Self::new(Action::EMBED_LOOKUP, target, table, indices.0, [0, 0, 0])
    }

    /// Reduce average: target[0] = mean(source[start..start+count])
    /// Useful for band pooling in audio, spatial pooling, etc.
    pub const fn reduce_avg(
        target: Register,
        source: Register,
        start: u8,
        count: u8,
    ) -> Self {
        Self::new(Action::REDUCE_AVG, target, source, start, [count, 0, 0])
    }

    /// Slice: target = source[start..start+len]
    pub const fn slice(
        target: Register,
        source: Register,
        start: u8,
        len: u8,
    ) -> Self {
        Self::new(Action::SLICE, target, source, start, [len, 0, 0])
    }

    /// Argmax: target[0] = index of max value in source
    pub const fn argmax(target: Register, source: Register) -> Self {
        Self::simple(Action::ARGMAX, target, source)
    }

    /// Concat: target = concat(source, other)
    pub const fn concat(
        target: Register,
        source: Register,
        other: Register,
    ) -> Self {
        Self::new(Action::CONCAT, target, source, other.0, [0, 0, 0])
    }

    /// Squeeze: target = source with dimension removed
    /// For 1D Signal vectors, this is effectively a copy
    pub const fn squeeze(target: Register, source: Register, dim: u8) -> Self {
        Self::new(Action::SQUEEZE, target, source, dim, [0, 0, 0])
    }

    /// Unsqueeze: target = source with dimension added
    /// For 1D Signal vectors, this is effectively a copy
    pub const fn unsqueeze(target: Register, source: Register, dim: u8) -> Self {
        Self::new(Action::UNSQUEEZE, target, source, dim, [0, 0, 0])
    }

    /// Transpose: target = source with dims swapped
    /// For 1D Signal vectors, this is effectively a copy
    pub const fn transpose(
        target: Register,
        source: Register,
        dim1: u8,
        dim2: u8,
    ) -> Self {
        Self::new(Action::TRANSPOSE, target, source, dim1, [dim2, 0, 0])
    }

    /// Gate update: target = gate * update + (1 - gate) * state
    /// Fused operation for gated recurrent updates (GRU-style)
    pub const fn gate_update(
        target: Register,
        gate: Register,
        update: Register,
        state: Register,
    ) -> Self {
        Self::new(Action::GATE_UPDATE, target, gate, update.0, [state.0, 0, 0])
    }

    /// Dequantize: target = float(source) / scale
    pub fn dequantize(target: Register, source: Register, scale: u16) -> Self {
        let scale_bytes = scale.to_be_bytes();
        Self::new(
            Action::DEQUANTIZE,
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
            Action::LOOP,
            Register::NULL,
            Register::NULL,
            0,
            [count_bytes[0], count_bytes[1], 0],
        )
    }

    /// End of loop
    pub const fn end_loop() -> Self {
        Self::system(Action::END_LOOP)
    }

    /// Break from loop
    pub const fn break_loop() -> Self {
        Self::system(Action::BREAK)
    }

    // =========================================================================
    // Serialization
    // =========================================================================

    /// Parse from 8 bytes (big-endian)
    pub fn from_bytes(bytes: &[u8; 8]) -> Self {
        Self {
            action: Action::from_u16(u16::from_be_bytes([bytes[0], bytes[1]])),
            target: Register(bytes[2]),
            source: Register(bytes[3]),
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

    /// Get the modifier as a Modifier
    pub fn get_modifier(&self) -> Modifier {
        Modifier::from_bytes(self.modifier)
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

impl fmt::Display for Instruction {
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
                if self.action == Action::TERNARY_MATMUL {
                    // ternary_matmul always has aux as input register
                    let aux_reg = Register(self.aux);
                    write!(f, "{} {}, {}, {}", self.action, self.target, self.source, aux_reg)
                } else if self.aux != 0 && self.aux != 0xFF {
                    let aux_reg = Register(self.aux);
                    write!(f, "{} {}, {}, {}", self.action, self.target, self.source, aux_reg)
                } else if self.action == Action::SHIFT {
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
                if self.action == Action::LOOP {
                    write!(f, "{} {}", self.action, self.count())
                } else {
                    write!(f, "{}", self.action)
                }
            }
            _ => write!(f, "{} {}, {}, {}", self.action, self.target, self.source, self.aux),
        }
    }
}

impl Default for Instruction {
    fn default() -> Self {
        Self::nop()
    }
}

/// Builder for Instruction
pub struct InstructionBuilder {
    action: Action,
    target: Register,
    source: Register,
    aux: u8,
    modifier: [u8; 3],
}

impl InstructionBuilder {
    pub fn new(action: Action) -> Self {
        Self {
            action,
            target: Register::NULL,
            source: Register::NULL,
            aux: 0,
            modifier: [0, 0, 0],
        }
    }

    pub fn target(mut self, reg: Register) -> Self {
        self.target = reg;
        self
    }

    pub fn source(mut self, reg: Register) -> Self {
        self.source = reg;
        self
    }

    pub fn aux(mut self, val: u8) -> Self {
        self.aux = val;
        self
    }

    pub fn aux_reg(mut self, reg: Register) -> Self {
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

    pub fn build(self) -> Instruction {
        Instruction::new(self.action, self.target, self.source, self.aux, self.modifier)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_instruction_roundtrip() {
        let instr = Instruction::ternary_matmul(
            Register::hot(1),
            Register::cold(0),
            Register::hot(0),
        );

        let bytes = instr.to_bytes();
        assert_eq!(bytes.len(), 8);

        let parsed = Instruction::from_bytes(&bytes);
        assert_eq!(instr, parsed);
    }

    #[test]
    fn test_instruction_builders() {
        let instr = Instruction::relu(Register::hot(2), Register::hot(1));
        assert_eq!(instr.action, Action::RELU);
        assert_eq!(instr.target, Register::hot(2));
        assert_eq!(instr.source, Register::hot(1));
    }

    #[test]
    fn test_shape_encoding() {
        let instr = InstructionBuilder::new(Action::DEFINE_LAYER)
            .target(Register::cold(0))
            .shape(32, 12)
            .build();

        let (input_dim, output_dim) = instr.shape();
        assert_eq!(input_dim, 32);
        assert_eq!(output_dim, 12);
    }

    #[test]
    fn test_parse_all() {
        let instructions = vec![
            Instruction::load_input(Register::hot(0)),
            Instruction::ternary_matmul(
                Register::hot(1),
                Register::cold(0),
                Register::hot(0),
            ),
            Instruction::relu(Register::hot(2), Register::hot(1)),
            Instruction::halt(),
        ];

        let bytes = Instruction::to_bytes_all(&instructions);
        assert_eq!(bytes.len(), 32); // 4 instructions * 8 bytes

        let parsed = Instruction::parse_all(&bytes).unwrap();
        assert_eq!(parsed.len(), 4);
        assert_eq!(parsed[0].action, Action::LOAD_INPUT);
        assert_eq!(parsed[1].action, Action::TERNARY_MATMUL);
        assert_eq!(parsed[2].action, Action::RELU);
        assert_eq!(parsed[3].action, Action::HALT);
    }

    #[test]
    fn test_display() {
        let instr = Instruction::ternary_matmul(
            Register::hot(1),
            Register::cold(0),
            Register::hot(0),
        );
        let display = format!("{}", instr);
        assert!(display.contains("TERNARY_MATMUL"));
        assert!(display.contains("H1"));
        assert!(display.contains("C0"));
        assert!(display.contains("H0"));
    }
}
