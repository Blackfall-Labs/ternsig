//! Binary serialization for Ternsig programs
//!
//! Provides `.ternsig` binary format for compiled programs, enabling:
//! - Fast loading without parsing
//! - Cartridge storage
//! - Hot-reload via binary swap
//!
//! ## Binary Format
//!
//! ```text
//! HEADER (32 bytes)
//! ├── Magic:      "TERN" (4 bytes)
//! ├── Version:    u16 (format version)
//! ├── Flags:      u16 (compression, encryption flags)
//! ├── RegCounts:  u8 (hot), u8 (cold), u8 (param), u8 (shape)
//! ├── InstrCount: u32 (number of instructions)
//! ├── RegDefSize: u32 (size of register definitions section)
//! ├── Checksum:   u64 (xxhash of content)
//! └── Reserved:   8 bytes
//!
//! REGISTER_DEFS (variable)
//! ├── For each register:
//! │   ├── RegisterId: u8
//! │   ├── Dtype:      u8
//! │   ├── Ndims:      u8
//! │   ├── Dims:       [u16; ndims]
//! │   └── KeyLen:     u8
//! │   └── Key:        [u8; key_len] (thermogram key)
//!
//! INSTRUCTIONS (instr_count * 8 bytes)
//! └── Each instruction: 8 bytes raw
//!
//! METADATA (optional, JSON)
//! └── Length-prefixed JSON blob
//! ```

use super::{
    AssembledProgram, RegisterMeta, Dtype, Instruction, Register,
    INSTRUCTION_SIZE, TERNSIG_MAGIC, TERNSIG_VERSION,
};
use anyhow::{Context, Result};
use std::collections::HashMap;
use std::io::{Read, Write};

/// Header size in bytes
pub const HEADER_SIZE: usize = 32;

/// Binary format flags
#[derive(Clone, Copy, Debug, Default)]
pub struct BinaryFlags {
    /// Content is compressed
    pub compressed: bool,
    /// Content is encrypted
    pub encrypted: bool,
    /// Has metadata section
    pub has_metadata: bool,
}

impl BinaryFlags {
    fn to_u16(&self) -> u16 {
        let mut flags = 0u16;
        if self.compressed {
            flags |= 0x01;
        }
        if self.encrypted {
            flags |= 0x02;
        }
        if self.has_metadata {
            flags |= 0x04;
        }
        flags
    }

    fn from_u16(value: u16) -> Self {
        Self {
            compressed: value & 0x01 != 0,
            encrypted: value & 0x02 != 0,
            has_metadata: value & 0x04 != 0,
        }
    }
}

/// Binary header for .ternsig files
#[derive(Clone, Debug)]
pub struct Header {
    /// Magic bytes "TERN"
    pub magic: [u8; 4],
    /// Format version
    pub version: u16,
    /// Flags
    pub flags: BinaryFlags,
    /// Hot register count
    pub hot_count: u8,
    /// Cold register count
    pub cold_count: u8,
    /// Param register count
    pub param_count: u8,
    /// Shape register count
    pub shape_count: u8,
    /// Instruction count
    pub instr_count: u32,
    /// Register definitions section size
    pub reg_def_size: u32,
    /// Content checksum (xxhash64)
    pub checksum: u64,
}

impl Header {
    /// Create header from assembled program
    pub fn from_program(program: &AssembledProgram, reg_def_size: u32) -> Self {
        let mut hot_count = 0u8;
        let mut cold_count = 0u8;
        let mut param_count = 0u8;
        let mut shape_count = 0u8;

        for reg in &program.registers {
            match reg.id.bank() {
                super::RegisterBank::Hot => hot_count += 1,
                super::RegisterBank::Cold => cold_count += 1,
                super::RegisterBank::Param => param_count += 1,
                super::RegisterBank::Shape => shape_count += 1,
            }
        }

        Self {
            magic: TERNSIG_MAGIC,
            version: TERNSIG_VERSION,
            flags: BinaryFlags::default(),
            hot_count,
            cold_count,
            param_count,
            shape_count,
            instr_count: program.instructions.len() as u32,
            reg_def_size,
            checksum: 0, // Computed after serialization
        }
    }

    /// Write header to bytes
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = vec![0u8; HEADER_SIZE];

        // Magic (4 bytes)
        buf[0..4].copy_from_slice(&self.magic);

        // Version (2 bytes, little-endian)
        buf[4..6].copy_from_slice(&self.version.to_le_bytes());

        // Flags (2 bytes)
        buf[6..8].copy_from_slice(&self.flags.to_u16().to_le_bytes());

        // Register counts (4 bytes)
        buf[8] = self.hot_count;
        buf[9] = self.cold_count;
        buf[10] = self.param_count;
        buf[11] = self.shape_count;

        // Instruction count (4 bytes)
        buf[12..16].copy_from_slice(&self.instr_count.to_le_bytes());

        // Register definitions size (4 bytes)
        buf[16..20].copy_from_slice(&self.reg_def_size.to_le_bytes());

        // Checksum (8 bytes)
        buf[20..28].copy_from_slice(&self.checksum.to_le_bytes());

        // Reserved (4 bytes)
        // Already zeros

        buf
    }

    /// Read header from bytes
    pub fn from_bytes(data: &[u8]) -> Result<Self> {
        if data.len() < HEADER_SIZE {
            anyhow::bail!("Header too short: {} bytes", data.len());
        }

        let magic: [u8; 4] = data[0..4].try_into()?;
        if magic != TERNSIG_MAGIC {
            anyhow::bail!(
                "Invalid magic: expected TERN, got {:?}",
                std::str::from_utf8(&magic)
            );
        }

        let version = u16::from_le_bytes(data[4..6].try_into()?);
        let flags = BinaryFlags::from_u16(u16::from_le_bytes(data[6..8].try_into()?));
        let hot_count = data[8];
        let cold_count = data[9];
        let param_count = data[10];
        let shape_count = data[11];
        let instr_count = u32::from_le_bytes(data[12..16].try_into()?);
        let reg_def_size = u32::from_le_bytes(data[16..20].try_into()?);
        let checksum = u64::from_le_bytes(data[20..28].try_into()?);

        Ok(Self {
            magic,
            version,
            flags,
            hot_count,
            cold_count,
            param_count,
            shape_count,
            instr_count,
            reg_def_size,
            checksum,
        })
    }
}

/// Serialize a register definition
fn serialize_register(reg: &RegisterMeta) -> Vec<u8> {
    let mut buf = Vec::new();

    // Register ID (1 byte)
    buf.push(reg.id.0);

    // Dtype (1 byte)
    buf.push(match reg.dtype {
        Dtype::F32 => 0,
        Dtype::I32 => 1,
        Dtype::Ternary => 2,
        Dtype::PackedTernary => 3,
        Dtype::F16 => 4,
        Dtype::I16 => 5,
        Dtype::I8 => 6,
        Dtype::I64 => 7,
    });

    // Flags (1 byte): allocated, frozen
    let flags = (reg.allocated as u8) | ((reg.frozen as u8) << 1);
    buf.push(flags);

    // Number of dimensions (1 byte)
    buf.push(reg.shape.len() as u8);

    // Dimensions (2 bytes each)
    for &dim in &reg.shape {
        buf.extend_from_slice(&(dim as u16).to_le_bytes());
    }

    // Thermogram key
    if let Some(ref key) = reg.thermogram_key {
        let key_bytes = key.as_bytes();
        buf.push(key_bytes.len() as u8);
        buf.extend_from_slice(key_bytes);
    } else {
        buf.push(0);
    }

    buf
}

/// Deserialize a register definition
fn deserialize_register(data: &[u8], offset: &mut usize) -> Result<RegisterMeta> {
    if *offset >= data.len() {
        anyhow::bail!("Unexpected end of data in register definition");
    }

    // Register ID
    let id = Register(data[*offset]);
    *offset += 1;

    // Dtype
    let dtype = match data.get(*offset) {
        Some(0) => Dtype::F32,
        Some(1) => Dtype::I32,
        Some(2) => Dtype::Ternary,
        Some(3) => Dtype::PackedTernary,
        Some(4) => Dtype::F16,
        Some(5) => Dtype::I16,
        Some(6) => Dtype::I8,
        Some(7) => Dtype::I64,
        _ => anyhow::bail!("Invalid dtype at offset {}", *offset),
    };
    *offset += 1;

    // Flags: allocated, frozen
    let flags = data.get(*offset).copied().unwrap_or(0);
    let allocated = (flags & 0x01) != 0;
    let frozen = (flags & 0x02) != 0;
    *offset += 1;

    // Number of dimensions
    let ndims = data.get(*offset).copied().unwrap_or(0) as usize;
    *offset += 1;

    // Dimensions
    let mut shape = Vec::with_capacity(ndims);
    for _ in 0..ndims {
        if *offset + 2 > data.len() {
            anyhow::bail!("Unexpected end of data reading dimensions");
        }
        let dim = u16::from_le_bytes(data[*offset..*offset + 2].try_into()?);
        shape.push(dim as usize);
        *offset += 2;
    }

    // Thermogram key
    let key_len = data.get(*offset).copied().unwrap_or(0) as usize;
    *offset += 1;

    let thermogram_key = if key_len > 0 {
        if *offset + key_len > data.len() {
            anyhow::bail!("Unexpected end of data reading thermogram key");
        }
        let key = std::str::from_utf8(&data[*offset..*offset + key_len])?.to_string();
        *offset += key_len;
        Some(key)
    } else {
        None
    };

    Ok(RegisterMeta {
        id,
        dtype,
        shape,
        allocated,
        thermogram_key,
        frozen,
    })
}

/// Serialize an assembled program to binary
pub fn serialize(program: &AssembledProgram) -> Result<Vec<u8>> {
    // Serialize register definitions
    let mut reg_defs = Vec::new();
    for reg in &program.registers {
        reg_defs.extend(serialize_register(reg));
    }

    // Create header
    let mut header = Header::from_program(program, reg_defs.len() as u32);

    // Serialize instructions
    let mut instrs = Vec::with_capacity(program.instructions.len() * INSTRUCTION_SIZE);
    for instr in &program.instructions {
        instrs.extend_from_slice(&instr.to_bytes());
    }

    // Compute checksum over content (reg_defs + instrs)
    let mut content = Vec::new();
    content.extend(&reg_defs);
    content.extend(&instrs);
    header.checksum = xxhash(&content);

    // Assemble final binary
    let mut output = Vec::new();
    output.extend(header.to_bytes());
    output.extend(reg_defs);
    output.extend(instrs);

    Ok(output)
}

/// Deserialize a binary to assembled program
pub fn deserialize(data: &[u8]) -> Result<AssembledProgram> {
    // Parse header
    let header = Header::from_bytes(data)?;

    // Verify checksum
    let content_start = HEADER_SIZE;
    let content_end = content_start + header.reg_def_size as usize
        + header.instr_count as usize * INSTRUCTION_SIZE;

    if content_end > data.len() {
        anyhow::bail!(
            "Data too short: need {} bytes, have {}",
            content_end,
            data.len()
        );
    }

    let content = &data[content_start..content_end];
    let checksum = xxhash(content);

    if checksum != header.checksum {
        anyhow::bail!(
            "Checksum mismatch: expected {:016x}, got {:016x}",
            header.checksum,
            checksum
        );
    }

    // Parse register definitions
    let reg_def_end = content_start + header.reg_def_size as usize;
    let reg_def_data = &data[content_start..reg_def_end];
    let mut offset = 0;
    let mut registers = Vec::new();

    let total_regs =
        header.hot_count + header.cold_count + header.param_count + header.shape_count;
    for _ in 0..total_regs {
        registers.push(deserialize_register(reg_def_data, &mut offset)?);
    }

    // Parse instructions
    let instr_data = &data[reg_def_end..content_end];
    let mut instructions = Vec::with_capacity(header.instr_count as usize);

    for i in 0..header.instr_count as usize {
        let start = i * INSTRUCTION_SIZE;
        let end = start + INSTRUCTION_SIZE;
        if end > instr_data.len() {
            anyhow::bail!("Instruction {} out of bounds", i);
        }
        let bytes: [u8; 8] = instr_data[start..end]
            .try_into()
            .context("Failed to convert instruction bytes")?;
        instructions.push(Instruction::from_bytes(&bytes));
    }

    Ok(AssembledProgram {
        name: String::new(),          // Name not preserved in binary
        version: 1,                   // Version not preserved in binary
        domain: None,                 // Domain not preserved in binary
        registers,
        instructions,
        labels: HashMap::new(),       // Labels not preserved in binary
        input_shape: Vec::new(),      // Could be inferred from registers
        output_shape: Vec::new(),     // Could be inferred from registers
    })
}

/// Simple xxhash-like checksum (for demonstration)
/// In production, use actual xxhash crate
fn xxhash(data: &[u8]) -> u64 {
    let mut hash = 0x9E3779B97F4A7C15u64;
    for chunk in data.chunks(8) {
        let mut v = 0u64;
        for (i, &byte) in chunk.iter().enumerate() {
            v |= (byte as u64) << (i * 8);
        }
        hash = hash.wrapping_mul(0x85EBCA6B).wrapping_add(v);
        hash = (hash << 31) | (hash >> 33);
    }
    hash
}

/// Save program to binary file
pub fn save_to_file(program: &AssembledProgram, path: impl AsRef<std::path::Path>) -> Result<()> {
    let binary = serialize(program)?;
    let mut file = std::fs::File::create(path)?;
    file.write_all(&binary)?;
    Ok(())
}

/// Load program from binary file
pub fn load_from_file(path: impl AsRef<std::path::Path>) -> Result<AssembledProgram> {
    let mut file = std::fs::File::open(path)?;
    let mut data = Vec::new();
    file.read_to_end(&mut data)?;
    deserialize(&data)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vm::assemble;

    #[test]
    fn test_roundtrip() {
        let source = r#"
.registers
    C0: ternary[4, 2]  key="test.weights"
    H0: i32[2]
    H1: i32[4]

.program
    load_input H0
    ternary_matmul H1, C0, H0
    relu H1, H1
    store_output H1
    halt
"#;

        let program = assemble(source).unwrap();
        let binary = serialize(&program).unwrap();
        let recovered = deserialize(&binary).unwrap();

        assert_eq!(program.registers.len(), recovered.registers.len());
        assert_eq!(program.instructions.len(), recovered.instructions.len());

        // Verify register metadata
        for (orig, rec) in program.registers.iter().zip(recovered.registers.iter()) {
            assert_eq!(orig.id.0, rec.id.0);
            assert_eq!(orig.shape, rec.shape);
            assert_eq!(orig.thermogram_key, rec.thermogram_key);
        }
    }

    #[test]
    fn test_checksum_validation() {
        let source = r#"
.registers
    H0: i32[4]

.program
    halt
"#;

        let program = assemble(source).unwrap();
        let mut binary = serialize(&program).unwrap();

        // Corrupt the binary
        if binary.len() > 40 {
            binary[40] ^= 0xFF;
        }

        // Should fail checksum
        assert!(deserialize(&binary).is_err());
    }
}
