//! TVMR Instruction — 8-byte fixed-width instruction format
//!
//! ## Format
//!
//! ```text
//! [ExtID:2][OpCode:2][A:1][B:1][C:1][D:1]
//!   0-1       2-3     4    5    6    7
//! ```
//!
//! - **ExtID**: Extension identifier (0x0000 = core/legacy ISA)
//! - **OpCode**: Extension-local opcode (65,536 per extension)
//! - **A, B, C, D**: 4 operand bytes, interpretation per-instruction
//!
//! ## Common Operand Patterns
//!
//! ```text
//! [dst:1][src:1][aux:1][flags:1]  — register-register ops
//! [dst:1][src:1][imm16:2]         — register + 16-bit immediate
//! [imm32:4]                       — pure immediate (jump target)
//! [reg:1][_:3]                    — unary ops
//! ```

use super::register::Register;
use std::fmt;

/// A TVMR instruction (8 bytes, fixed width).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub struct Instruction {
    /// Extension identifier. 0x0000 = core/legacy ISA.
    pub ext_id: u16,
    /// Extension-local opcode.
    pub opcode: u16,
    /// 4 operand bytes (interpretation per-instruction).
    pub operands: [u8; 4],
}

impl Instruction {
    /// Instruction size in bytes.
    pub const SIZE: usize = 8;

    /// Create a new instruction.
    pub const fn new(ext_id: u16, opcode: u16, operands: [u8; 4]) -> Self {
        Self { ext_id, opcode, operands }
    }

    /// Create a core ISA instruction (ext_id = 0x0000).
    pub const fn core(opcode: u16, operands: [u8; 4]) -> Self {
        Self::new(0x0000, opcode, operands)
    }

    /// Create an extension instruction.
    pub const fn ext(ext_id: u16, opcode: u16, operands: [u8; 4]) -> Self {
        Self::new(ext_id, opcode, operands)
    }

    // =========================================================================
    // Operand Accessors (TVMR patterns)
    // =========================================================================

    /// Operand A as a register reference (byte 0).
    pub const fn reg_a(&self) -> Register { Register(self.operands[0]) }

    /// Operand B as a register reference (byte 1).
    pub const fn reg_b(&self) -> Register { Register(self.operands[1]) }

    /// Operand C as a register reference (byte 2).
    pub const fn reg_c(&self) -> Register { Register(self.operands[2]) }

    /// Operand D as raw byte (byte 3).
    pub const fn byte_d(&self) -> u8 { self.operands[3] }

    /// Operands C:D as u16 (big-endian).
    pub const fn imm16_cd(&self) -> u16 {
        ((self.operands[2] as u16) << 8) | (self.operands[3] as u16)
    }

    /// All 4 operand bytes as u32 (big-endian).
    pub const fn imm32(&self) -> u32 {
        ((self.operands[0] as u32) << 24)
            | ((self.operands[1] as u32) << 16)
            | ((self.operands[2] as u32) << 8)
            | (self.operands[3] as u32)
    }

    // =========================================================================
    // Legacy Accessors (bridge from old [Action:Target:Source:Aux:Modifier] format)
    // =========================================================================

    /// Legacy: target register (operands[0]).
    pub fn target(&self) -> Register { Register(self.operands[0]) }

    /// Legacy: source register (operands[1]).
    pub fn source(&self) -> Register { Register(self.operands[1]) }

    /// Legacy: aux byte (operands[2]).
    pub fn aux(&self) -> u8 { self.operands[2] }

    /// Legacy: 3-byte modifier.
    ///
    /// In the TVMR format, only operands[3] survives as modifier[0].
    /// modifier[1] and modifier[2] return 0, which triggers default values
    /// in instructions that use `if modifier[N] > 0 { modifier[N] } else { default }`.
    pub fn modifier(&self) -> [u8; 3] {
        [self.operands[3], 0, 0]
    }

    /// Legacy: count from modifier (for LOOP/SKIP).
    /// In TVMR format, count is stored in operands[2:3] for control flow.
    pub fn count(&self) -> u16 {
        self.imm16_cd()
    }

    /// Legacy: scale value (for DEQUANTIZE/SCALE).
    /// In TVMR format, scale is stored in operands[2:3].
    pub fn scale(&self) -> u16 {
        self.imm16_cd()
    }

    // =========================================================================
    // Legacy Bridge — Convert from old [Action:2][Target:1][Source:1][Aux:1][Mod:3]
    // =========================================================================

    /// Create from legacy instruction format.
    ///
    /// Maps the old 8-byte format to the new TVMR format:
    /// - ext_id = 0x0000 (all legacy instructions are "core" initially)
    /// - opcode = the legacy Action u16
    /// - operands = [target, source, aux, modifier[0]]
    ///
    /// Note: modifier bytes 1 and 2 are encoded into the opcode-specific
    /// interpretation. Instructions that used 3-byte modifiers (DEFINE_LAYER
    /// shape, SCALE u16) store the critical data in operands[2:3].
    pub fn from_legacy(
        action_u16: u16,
        target: u8,
        source: u8,
        aux: u8,
        modifier: [u8; 3],
    ) -> Self {
        // For most instructions: operands = [target, source, aux, modifier[0]]
        // Special cases for instructions that need modifier[1] or [2]:
        // - LOOP/SKIP: count is in modifier[0:1], we put count in operands[2:3]
        // - DEQUANTIZE/SCALE: scale in modifier[0:1], we put scale in operands[2:3]
        // - DEFINE_LAYER: shape in all 3 modifier bytes, needs special handling

        match action_u16 >> 8 {
            0x60 => {
                // Control flow — count in modifier[0:1], threshold in modifier[2]
                // Pack as: [target, source, count_hi, count_lo]
                // aux is unused for most control flow
                Self::new(0x0000, action_u16, [target, source, modifier[0], modifier[1]])
            }
            _ => {
                // Default: [target, source, aux, modifier[0]]
                Self::new(0x0000, action_u16, [target, source, aux, modifier[0]])
            }
        }
    }

    /// Convert back to legacy format (for backward compat with old interpreter).
    ///
    /// Returns (action_u16, target, source, aux, modifier).
    pub fn to_legacy(&self) -> (u16, u8, u8, u8, [u8; 3]) {
        if self.ext_id != 0x0000 {
            // Extension instructions have no legacy equivalent
            return (0xFFFF, 0, 0, 0, [0, 0, 0]);
        }

        match self.opcode >> 8 {
            0x60 => {
                // Control flow — unpack count from operands[2:3]
                (self.opcode, self.operands[0], self.operands[1], 0,
                 [self.operands[2], self.operands[3], 0])
            }
            _ => {
                // Default
                (self.opcode, self.operands[0], self.operands[1], self.operands[2],
                 [self.operands[3], 0, 0])
            }
        }
    }

    // =========================================================================
    // Serialization (TVMR binary format)
    // =========================================================================

    /// Parse from 8 bytes (big-endian for ext_id and opcode).
    pub fn from_bytes(bytes: &[u8; 8]) -> Self {
        Self {
            ext_id: u16::from_be_bytes([bytes[0], bytes[1]]),
            opcode: u16::from_be_bytes([bytes[2], bytes[3]]),
            operands: [bytes[4], bytes[5], bytes[6], bytes[7]],
        }
    }

    /// Serialize to 8 bytes (big-endian for ext_id and opcode).
    pub fn to_bytes(&self) -> [u8; 8] {
        let ext_bytes = self.ext_id.to_be_bytes();
        let op_bytes = self.opcode.to_be_bytes();
        [
            ext_bytes[0], ext_bytes[1],
            op_bytes[0], op_bytes[1],
            self.operands[0], self.operands[1],
            self.operands[2], self.operands[3],
        ]
    }

    /// Parse multiple instructions from byte slice.
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

    /// Serialize multiple instructions to bytes.
    pub fn to_bytes_all(instructions: &[Self]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(instructions.len() * Self::SIZE);
        for instr in instructions {
            bytes.extend_from_slice(&instr.to_bytes());
        }
        bytes
    }

    // =========================================================================
    // Convenience Builders (legacy-compatible)
    // =========================================================================

    /// NOP instruction.
    pub const fn nop() -> Self {
        Self::core(0x0000, [0xFF, 0xFF, 0, 0])
    }

    /// HALT instruction.
    pub const fn halt() -> Self {
        Self::core(0x0001, [0xFF, 0xFF, 0, 0])
    }

    /// Load from input buffer: target = input_buffer.
    pub const fn load_input(target: Register) -> Self {
        Self::core(0x1007, [target.0, 0xFF, 0, 0])
    }

    /// Store to output buffer: output_buffer = source.
    pub const fn store_output(source: Register) -> Self {
        Self::core(0x1008, [0xFF, source.0, 0, 0])
    }

    /// Ternary matrix multiply: target = weights @ input.
    pub const fn ternary_matmul(target: Register, weights: Register, input: Register) -> Self {
        Self::core(0x4000, [target.0, weights.0, input.0, 0])
    }

    /// Ternary batch matmul: target[i] = weights @ input[i].
    pub const fn ternary_batch_matmul(target: Register, weights: Register, input: Register) -> Self {
        Self::core(0x4013, [target.0, weights.0, input.0, 0])
    }

    /// Add: target = source + other.
    pub const fn add(target: Register, source: Register, other: Register) -> Self {
        Self::core(0x3001, [target.0, source.0, other.0, 0])
    }

    /// Sub: target = source - other.
    pub const fn sub(target: Register, source: Register, other: Register) -> Self {
        Self::core(0x300E, [target.0, source.0, other.0, 0])
    }

    /// ReLU: target = max(0, source).
    pub const fn relu(target: Register, source: Register) -> Self {
        Self::core(0x3003, [target.0, source.0, 0, 0])
    }

    /// Sigmoid: target = sigmoid(source).
    pub const fn sigmoid(target: Register, source: Register) -> Self {
        Self::core(0x3004, [target.0, source.0, 0, 0])
    }

    /// Shift right: target = source >> shift_amount.
    pub const fn shift(target: Register, source: Register, amount: u8) -> Self {
        Self::core(0x3009, [target.0, source.0, amount, 0])
    }

    /// Ternary add bias: target = source + bias.
    pub const fn ternary_add_bias(target: Register, source: Register, bias: Register) -> Self {
        Self::core(0x4009, [target.0, source.0, bias.0, 0])
    }

    /// Embedding lookup: target[i] = table[indices[i]].
    pub const fn embed_lookup(target: Register, table: Register, indices: Register) -> Self {
        Self::core(0x400A, [target.0, table.0, indices.0, 0])
    }

    /// Embed sequence: target[i] = table[i] for i in 0..count.
    pub const fn embed_sequence(target: Register, table: Register, count: u8) -> Self {
        Self::core(0x4014, [target.0, table.0, count, 0])
    }

    /// Reduce average: target[0] = mean(source[start..start+count]).
    pub const fn reduce_avg(target: Register, source: Register, start: u8, count: u8) -> Self {
        Self::core(0x400B, [target.0, source.0, start, count])
    }

    /// Reduce mean along dimension.
    pub const fn reduce_mean_dim(target: Register, source: Register, dim: u8) -> Self {
        Self::core(0x4015, [target.0, source.0, dim, 0])
    }

    /// Slice: target = source[start..start+len].
    pub const fn slice(target: Register, source: Register, start: u8, len: u8) -> Self {
        Self::core(0x400C, [target.0, source.0, start, len])
    }

    /// Argmax: target[0] = index of max value in source.
    pub const fn argmax(target: Register, source: Register) -> Self {
        Self::core(0x400D, [target.0, source.0, 0, 0])
    }

    /// Concat: target = concat(source, other).
    pub const fn concat(target: Register, source: Register, other: Register) -> Self {
        Self::core(0x400E, [target.0, source.0, other.0, 0])
    }

    /// Squeeze: remove dimension.
    pub const fn squeeze(target: Register, source: Register, dim: u8) -> Self {
        Self::core(0x400F, [target.0, source.0, dim, 0])
    }

    /// Unsqueeze: add dimension.
    pub const fn unsqueeze(target: Register, source: Register, dim: u8) -> Self {
        Self::core(0x4010, [target.0, source.0, dim, 0])
    }

    /// Transpose: swap dims.
    pub const fn transpose(target: Register, source: Register, dim1: u8, dim2: u8) -> Self {
        Self::core(0x4011, [target.0, source.0, dim1, dim2])
    }

    /// Gate update: target = gate * update + (1-gate) * state.
    pub const fn gate_update(target: Register, gate: Register, update: Register, state: Register) -> Self {
        Self::core(0x4012, [target.0, gate.0, update.0, state.0])
    }

    /// Add babble noise.
    pub const fn add_babble(target: Register, layer_index: u8) -> Self {
        Self::core(0x5004, [target.0, 0xFF, layer_index, 0])
    }

    /// Mark eligibility.
    pub const fn mark_eligibility(output: Register, input: Register, layer_index: u8) -> Self {
        Self::core(0x5000, [output.0, input.0, layer_index, 0])
    }

    /// Dequantize: target = float(source) / scale.
    pub fn dequantize(target: Register, source: Register, scale: u16) -> Self {
        let sb = scale.to_be_bytes();
        Self::core(0x4002, [target.0, source.0, sb[0], sb[1]])
    }

    /// Loop N times.
    pub fn loop_n(count: u16) -> Self {
        let cb = count.to_be_bytes();
        Self::core(0x6000, [0xFF, 0xFF, cb[0], cb[1]])
    }

    /// End of loop.
    pub const fn end_loop() -> Self {
        Self::core(0x6001, [0xFF, 0xFF, 0, 0])
    }

    /// Break from loop.
    pub const fn break_loop() -> Self {
        Self::core(0x6002, [0xFF, 0xFF, 0, 0])
    }

    // =========================================================================
    // Query Methods
    // =========================================================================

    /// Whether this is a core ISA instruction (ext_id == 0x0000).
    pub const fn is_core(&self) -> bool {
        self.ext_id == 0x0000
    }

    /// Whether this is an extension instruction.
    pub const fn is_extension(&self) -> bool {
        self.ext_id != 0x0000
    }

    /// Format as hex opcode string.
    pub fn to_opcode_string(&self) -> String {
        format!(
            "{:04X}:{:04X}:{:02X}{:02X}{:02X}{:02X}",
            self.ext_id, self.opcode,
            self.operands[0], self.operands[1],
            self.operands[2], self.operands[3],
        )
    }
}

impl Default for Instruction {
    fn default() -> Self {
        Self::nop()
    }
}

impl fmt::Display for Instruction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_core() {
            // Use legacy Action name lookup for display
            let action = super::action::Action::from_u16(self.opcode);
            let name = action.name();

            let a = Register(self.operands[0]);
            let b = Register(self.operands[1]);

            if a.is_null() && b.is_null() {
                write!(f, "{}", name)
            } else if b.is_null() {
                write!(f, "{} {}", name, a)
            } else if self.operands[2] != 0 && self.operands[2] != 0xFF {
                let c = Register(self.operands[2]);
                if c.is_null() {
                    write!(f, "{} {}, {}, {}", name, a, b, self.operands[2])
                } else {
                    write!(f, "{} {}, {}, {}", name, a, b, c)
                }
            } else {
                write!(f, "{} {}, {}", name, a, b)
            }
        } else {
            write!(f, "EXT[0x{:04X}]:0x{:04X} [{:02X},{:02X},{:02X},{:02X}]",
                self.ext_id, self.opcode,
                self.operands[0], self.operands[1],
                self.operands[2], self.operands[3],
            )
        }
    }
}

/// Builder for constructing instructions fluently.
pub struct InstructionBuilder {
    ext_id: u16,
    opcode: u16,
    operands: [u8; 4],
}

impl InstructionBuilder {
    /// Start building a core ISA instruction.
    pub fn core(opcode: u16) -> Self {
        Self { ext_id: 0x0000, opcode, operands: [0; 4] }
    }

    /// Start building an extension instruction.
    pub fn ext(ext_id: u16, opcode: u16) -> Self {
        Self { ext_id, opcode, operands: [0; 4] }
    }

    /// Set operand A (byte 0) as register.
    pub fn a(mut self, reg: Register) -> Self { self.operands[0] = reg.0; self }

    /// Set operand B (byte 1) as register.
    pub fn b(mut self, reg: Register) -> Self { self.operands[1] = reg.0; self }

    /// Set operand C (byte 2) as register or immediate.
    pub fn c(mut self, val: u8) -> Self { self.operands[2] = val; self }

    /// Set operand C as register.
    pub fn c_reg(mut self, reg: Register) -> Self { self.operands[2] = reg.0; self }

    /// Set operand D (byte 3) as immediate.
    pub fn d(mut self, val: u8) -> Self { self.operands[3] = val; self }

    /// Set operands C:D as u16 (big-endian).
    pub fn cd_u16(mut self, val: u16) -> Self {
        let bytes = val.to_be_bytes();
        self.operands[2] = bytes[0];
        self.operands[3] = bytes[1];
        self
    }

    /// Build the instruction.
    pub fn build(self) -> Instruction {
        Instruction::new(self.ext_id, self.opcode, self.operands)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_roundtrip() {
        let instr = Instruction::ternary_matmul(
            Register::hot(1), Register::cold(0), Register::hot(0),
        );
        let bytes = instr.to_bytes();
        assert_eq!(bytes.len(), 8);
        let parsed = Instruction::from_bytes(&bytes);
        assert_eq!(instr, parsed);
    }

    #[test]
    fn test_new_format_encoding() {
        let instr = Instruction::ext(0x0002, 0x0001, [0x01, 0x40, 0x00, 0x00]);
        let bytes = instr.to_bytes();
        // ExtID = 0x0002 → [0x00, 0x02]
        // OpCode = 0x0001 → [0x00, 0x01]
        // Operands = [0x01, 0x40, 0x00, 0x00]
        assert_eq!(bytes, [0x00, 0x02, 0x00, 0x01, 0x01, 0x40, 0x00, 0x00]);
    }

    #[test]
    fn test_core_instructions() {
        let nop = Instruction::nop();
        assert!(nop.is_core());
        assert_eq!(nop.ext_id, 0x0000);
        assert_eq!(nop.opcode, 0x0000);

        let halt = Instruction::halt();
        assert_eq!(halt.opcode, 0x0001);
    }

    #[test]
    fn test_builder() {
        let instr = InstructionBuilder::core(0x3001) // ADD
            .a(Register::hot(2))
            .b(Register::hot(0))
            .c_reg(Register::hot(1))
            .build();

        assert_eq!(instr.ext_id, 0x0000);
        assert_eq!(instr.opcode, 0x3001);
        assert_eq!(instr.reg_a(), Register::hot(2));
        assert_eq!(instr.reg_b(), Register::hot(0));
        assert_eq!(instr.reg_c(), Register::hot(1));
    }

    #[test]
    fn test_extension_instruction() {
        let instr = Instruction::ext(0x0005, 0x0004, [0x01, 0x00, 0x00, 0x00]);
        assert!(instr.is_extension());
        assert_eq!(instr.ext_id, 0x0005);
    }

    #[test]
    fn test_parse_all() {
        let instructions = vec![
            Instruction::load_input(Register::hot(0)),
            Instruction::ternary_matmul(Register::hot(1), Register::cold(0), Register::hot(0)),
            Instruction::relu(Register::hot(2), Register::hot(1)),
            Instruction::halt(),
        ];
        let bytes = Instruction::to_bytes_all(&instructions);
        assert_eq!(bytes.len(), 32);
        let parsed = Instruction::parse_all(&bytes).unwrap();
        assert_eq!(parsed.len(), 4);
    }

    #[test]
    fn test_imm16() {
        let instr = InstructionBuilder::core(0x6000) // LOOP
            .cd_u16(100)
            .build();
        assert_eq!(instr.imm16_cd(), 100);
    }
}
