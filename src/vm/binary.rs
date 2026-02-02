//! TVMR Binary Format — Serialization for Ternsig Programs
//!
//! Two formats are supported:
//!
//! - **Legacy TERN** (32-byte header, magic "TERN"): backward-compatible read-only
//! - **TVMR v1** (48-byte header, magic "TVMR"): new format with extension table and TypeId
//!
//! `serialize()` always writes TVMR v1. `deserialize()` auto-detects the format.
//!
//! ## TVMR v1 Binary Layout
//!
//! ```text
//! HEADER (48 bytes)
//! Offset  Size  Field
//! 0x00    4     Magic: "TVMR" (0x54564D52)
//! 0x04    2     Format version: u16 LE (1)
//! 0x06    2     Flags: u16 LE
//! 0x08    4     Extension table offset: u32 LE
//! 0x0C    4     Extension table count: u32 LE
//! 0x10    4     Register defs offset: u32 LE
//! 0x14    4     Register defs count: u32 LE
//! 0x18    4     Instruction offset: u32 LE
//! 0x1C    4     Instruction count: u32 LE
//! 0x20    8     Checksum: xxhash64
//! 0x28    8     Reserved
//!
//! EXTENSION TABLE (8 bytes per entry)
//!   [ExtID:2 LE][VersionMajor:2 LE][VersionMinor:2 LE][VersionPatch:2 LE]
//!
//! REGISTER DEFINITIONS (variable per register)
//!   [RegId:1][TypeId:2 LE][Flags:1][Ndims:1][Dims:2*N LE][KeyLen:1][Key:K]
//!
//! INSTRUCTIONS (8 bytes each)
//!   [ExtID:2 BE][OpCode:2 BE][A:1][B:1][C:1][D:1]
//! ```

use super::{
    AssembledProgram, Dtype, Instruction, Register, RegisterMeta, RequiredExtension,
    INSTRUCTION_SIZE, TERNSIG_MAGIC, TERNSIG_VERSION, TVMR_MAGIC, TVMR_VERSION,
};
use super::types::TypeId;
use anyhow::{Context, Result};
use std::collections::HashMap;
use std::io::{Read, Write};

// =============================================================================
// Constants
// =============================================================================

/// Legacy TERN header size
pub const LEGACY_HEADER_SIZE: usize = 32;

/// TVMR v1 header size
pub const TVMR_HEADER_SIZE: usize = 48;

/// Re-export for backward compatibility
pub const HEADER_SIZE: usize = TVMR_HEADER_SIZE;

/// Extension table entry size (8 bytes)
const EXT_ENTRY_SIZE: usize = 8;

// =============================================================================
// Flags
// =============================================================================

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

// =============================================================================
// TVMR v1 Header
// =============================================================================

/// TVMR v1 binary header (48 bytes)
#[derive(Clone, Debug)]
pub struct Header {
    /// Magic bytes ("TVMR" or "TERN")
    pub magic: [u8; 4],
    /// Format version
    pub version: u16,
    /// Flags
    pub flags: BinaryFlags,
    /// Offset of extension table from start of file
    pub ext_table_offset: u32,
    /// Number of extension table entries
    pub ext_table_count: u32,
    /// Offset of register definitions from start of file
    pub reg_defs_offset: u32,
    /// Number of register definitions
    pub reg_defs_count: u32,
    /// Offset of instructions from start of file
    pub instr_offset: u32,
    /// Number of instructions
    pub instr_count: u32,
    /// xxhash64 checksum over content (ext_table + reg_defs + instructions)
    pub checksum: u64,
}

impl Header {
    /// Create TVMR v1 header from assembled program and section sizes.
    pub fn from_program(
        program: &AssembledProgram,
        ext_table_size: u32,
        reg_defs_size: u32,
    ) -> Self {
        let ext_table_offset = TVMR_HEADER_SIZE as u32;
        let reg_defs_offset = ext_table_offset + ext_table_size;
        let instr_offset = reg_defs_offset + reg_defs_size;

        Self {
            magic: TVMR_MAGIC,
            version: TVMR_VERSION,
            flags: BinaryFlags::default(),
            ext_table_offset,
            ext_table_count: program.required_extensions.len() as u32,
            reg_defs_offset,
            reg_defs_count: program.registers.len() as u32,
            instr_offset,
            instr_count: program.instructions.len() as u32,
            checksum: 0, // Computed after serialization
        }
    }

    /// Write TVMR v1 header to bytes (48 bytes)
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = vec![0u8; TVMR_HEADER_SIZE];

        // 0x00: Magic (4 bytes)
        buf[0..4].copy_from_slice(&self.magic);
        // 0x04: Version (2 bytes LE)
        buf[4..6].copy_from_slice(&self.version.to_le_bytes());
        // 0x06: Flags (2 bytes LE)
        buf[6..8].copy_from_slice(&self.flags.to_u16().to_le_bytes());
        // 0x08: Extension table offset (4 bytes LE)
        buf[8..12].copy_from_slice(&self.ext_table_offset.to_le_bytes());
        // 0x0C: Extension table count (4 bytes LE)
        buf[12..16].copy_from_slice(&self.ext_table_count.to_le_bytes());
        // 0x10: Register defs offset (4 bytes LE)
        buf[16..20].copy_from_slice(&self.reg_defs_offset.to_le_bytes());
        // 0x14: Register defs count (4 bytes LE)
        buf[20..24].copy_from_slice(&self.reg_defs_count.to_le_bytes());
        // 0x18: Instruction offset (4 bytes LE)
        buf[24..28].copy_from_slice(&self.instr_offset.to_le_bytes());
        // 0x1C: Instruction count (4 bytes LE)
        buf[28..32].copy_from_slice(&self.instr_count.to_le_bytes());
        // 0x20: Checksum (8 bytes LE)
        buf[32..40].copy_from_slice(&self.checksum.to_le_bytes());
        // 0x28: Reserved (8 bytes) — already zeros

        buf
    }

    /// Read TVMR v1 header from bytes
    pub fn from_bytes(data: &[u8]) -> Result<Self> {
        if data.len() < TVMR_HEADER_SIZE {
            anyhow::bail!("TVMR header too short: {} bytes (need {})", data.len(), TVMR_HEADER_SIZE);
        }

        let magic: [u8; 4] = data[0..4].try_into()?;
        if magic != TVMR_MAGIC {
            anyhow::bail!(
                "Invalid magic: expected TVMR, got {:?}",
                std::str::from_utf8(&magic)
            );
        }

        let version = u16::from_le_bytes(data[4..6].try_into()?);
        let flags = BinaryFlags::from_u16(u16::from_le_bytes(data[6..8].try_into()?));
        let ext_table_offset = u32::from_le_bytes(data[8..12].try_into()?);
        let ext_table_count = u32::from_le_bytes(data[12..16].try_into()?);
        let reg_defs_offset = u32::from_le_bytes(data[16..20].try_into()?);
        let reg_defs_count = u32::from_le_bytes(data[20..24].try_into()?);
        let instr_offset = u32::from_le_bytes(data[24..28].try_into()?);
        let instr_count = u32::from_le_bytes(data[28..32].try_into()?);
        let checksum = u64::from_le_bytes(data[32..40].try_into()?);

        Ok(Self {
            magic,
            version,
            flags,
            ext_table_offset,
            ext_table_count,
            reg_defs_offset,
            reg_defs_count,
            instr_offset,
            instr_count,
            checksum,
        })
    }
}

// =============================================================================
// Extension Table Serialization
// =============================================================================

/// Serialize extension dependencies to binary.
///
/// Each entry is 8 bytes: [ExtID:2 LE][Major:2 LE][Minor:2 LE][Patch:2 LE]
fn serialize_ext_table(extensions: &[RequiredExtension]) -> Vec<u8> {
    let mut buf = Vec::with_capacity(extensions.len() * EXT_ENTRY_SIZE);
    for ext in extensions {
        buf.extend_from_slice(&ext.ext_id.to_le_bytes());
        // Version defaults to 1.0.0 — RequiredExtension doesn't carry version yet
        buf.extend_from_slice(&1u16.to_le_bytes()); // major
        buf.extend_from_slice(&0u16.to_le_bytes()); // minor
        buf.extend_from_slice(&0u16.to_le_bytes()); // patch
    }
    buf
}

/// Deserialize extension dependencies from binary.
fn deserialize_ext_table(data: &[u8], count: u32) -> Result<Vec<RequiredExtension>> {
    let mut extensions = Vec::with_capacity(count as usize);
    let expected_size = count as usize * EXT_ENTRY_SIZE;
    if data.len() < expected_size {
        anyhow::bail!(
            "Extension table too short: need {} bytes, have {}",
            expected_size,
            data.len()
        );
    }

    for i in 0..count as usize {
        let off = i * EXT_ENTRY_SIZE;
        let ext_id = u16::from_le_bytes(data[off..off + 2].try_into()?);
        // major/minor/patch at off+2..off+8 — not stored in RequiredExtension yet
        let name = ext_id_to_name(ext_id).to_string();
        extensions.push(RequiredExtension { name, ext_id });
    }

    Ok(extensions)
}

/// Map known extension IDs back to names.
fn ext_id_to_name(ext_id: u16) -> &'static str {
    match ext_id {
        0x0001 => "tensor",
        0x0002 => "ternary",
        0x0003 => "activation",
        0x0004 => "learning",
        0x0005 => "neuro",
        0x0006 => "arch",
        0x0007 => "orchestration",
        0x0008 => "lifecycle",
        0x0009 => "ipc",
        0x000A => "test",
        _ => "unknown",
    }
}

// =============================================================================
// Register Definition Serialization (TVMR v1 — TypeId)
// =============================================================================

/// Convert legacy Dtype to TypeId for binary serialization.
fn dtype_to_type_id(dtype: Dtype) -> TypeId {
    match dtype {
        Dtype::F32 => TypeId::F32,
        Dtype::I32 => TypeId::I32,
        Dtype::Ternary => TypeId::SIGNAL,
        Dtype::PackedTernary => TypeId::PACKED_SIGNAL,
        Dtype::F16 => TypeId::F16,
        Dtype::I16 => TypeId::I16,
        Dtype::I8 => TypeId::I8,
        Dtype::I64 => TypeId::I64,
    }
}

/// Convert TypeId back to legacy Dtype.
fn type_id_to_dtype(type_id: TypeId) -> Dtype {
    match type_id.0 {
        0x000B => Dtype::F32,
        0x0007 => Dtype::I32,
        0x0100 => Dtype::Ternary,
        0x0101 => Dtype::PackedTernary,
        0x000A => Dtype::F16,
        0x0005 => Dtype::I16,
        0x0003 => Dtype::I8,
        0x0009 => Dtype::I64,
        _ => Dtype::I32, // fallback
    }
}

/// Serialize a register definition (TVMR v1 format with TypeId).
///
/// Format: [RegId:1][TypeId:2 LE][Flags:1][Ndims:1][Dims:2*N LE][KeyLen:1][Key:K]
fn serialize_register(reg: &RegisterMeta) -> Vec<u8> {
    let mut buf = Vec::new();

    // Register ID (1 byte)
    buf.push(reg.id.0);

    // TypeId (2 bytes LE) — replaces legacy 1-byte Dtype
    let type_id = dtype_to_type_id(reg.dtype);
    buf.extend_from_slice(&type_id.0.to_le_bytes());

    // Flags (1 byte): allocated, frozen
    let flags = (reg.allocated as u8) | ((reg.frozen as u8) << 1);
    buf.push(flags);

    // Number of dimensions (1 byte)
    buf.push(reg.shape.len() as u8);

    // Dimensions (2 bytes each, LE)
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

/// Deserialize a register definition (TVMR v1 format with TypeId).
fn deserialize_register(data: &[u8], offset: &mut usize) -> Result<RegisterMeta> {
    if *offset >= data.len() {
        anyhow::bail!("Unexpected end of data in register definition");
    }

    // Register ID (1 byte)
    let id = Register(data[*offset]);
    *offset += 1;

    // TypeId (2 bytes LE)
    if *offset + 2 > data.len() {
        anyhow::bail!("Unexpected end of data reading TypeId at offset {}", *offset);
    }
    let type_id = TypeId(u16::from_le_bytes(data[*offset..*offset + 2].try_into()?));
    *offset += 2;
    let dtype = type_id_to_dtype(type_id);

    // Flags (1 byte): allocated, frozen
    let flags = data.get(*offset).copied().unwrap_or(0);
    let allocated = (flags & 0x01) != 0;
    let frozen = (flags & 0x02) != 0;
    *offset += 1;

    // Number of dimensions (1 byte)
    let ndims = data.get(*offset).copied().unwrap_or(0) as usize;
    *offset += 1;

    // Dimensions (2 bytes each LE)
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

// =============================================================================
// TVMR v1 Serialize / Deserialize
// =============================================================================

/// Serialize an assembled program to TVMR v1 binary.
pub fn serialize(program: &AssembledProgram) -> Result<Vec<u8>> {
    // 1. Serialize extension table
    let ext_table = serialize_ext_table(&program.required_extensions);

    // 2. Serialize register definitions
    let mut reg_defs = Vec::new();
    for reg in &program.registers {
        reg_defs.extend(serialize_register(reg));
    }

    // 3. Serialize instructions
    let mut instrs = Vec::with_capacity(program.instructions.len() * INSTRUCTION_SIZE);
    for instr in &program.instructions {
        instrs.extend_from_slice(&instr.to_bytes());
    }

    // 4. Create header with correct offsets
    let mut header = Header::from_program(
        program,
        ext_table.len() as u32,
        reg_defs.len() as u32,
    );

    // 5. Compute checksum over content (ext_table + reg_defs + instrs)
    let mut content = Vec::new();
    content.extend(&ext_table);
    content.extend(&reg_defs);
    content.extend(&instrs);
    header.checksum = xxhash(&content);

    // 6. Assemble final binary
    let mut output = Vec::with_capacity(TVMR_HEADER_SIZE + content.len());
    output.extend(header.to_bytes());
    output.extend(content);

    Ok(output)
}

/// Deserialize a binary to assembled program.
///
/// Auto-detects format: TVMR v1 ("TVMR" magic) or legacy ("TERN" magic).
pub fn deserialize(data: &[u8]) -> Result<AssembledProgram> {
    if data.len() < 4 {
        anyhow::bail!("Data too short to contain magic bytes");
    }

    let magic: [u8; 4] = data[0..4].try_into()?;
    match magic {
        TVMR_MAGIC => deserialize_tvmr(data),
        TERNSIG_MAGIC => deserialize_legacy(data),
        _ => anyhow::bail!(
            "Unknown magic: {:?}",
            std::str::from_utf8(&magic).unwrap_or("???")
        ),
    }
}

/// Deserialize TVMR v1 format.
fn deserialize_tvmr(data: &[u8]) -> Result<AssembledProgram> {
    let header = Header::from_bytes(data)?;

    // Compute content bounds
    let content_start = header.ext_table_offset as usize;
    let content_end = header.instr_offset as usize
        + header.instr_count as usize * INSTRUCTION_SIZE;

    if content_end > data.len() {
        anyhow::bail!(
            "Data too short: need {} bytes, have {}",
            content_end,
            data.len()
        );
    }

    // Verify checksum
    let content = &data[content_start..content_end];
    let checksum = xxhash(content);
    if checksum != header.checksum {
        anyhow::bail!(
            "Checksum mismatch: expected {:016x}, got {:016x}",
            header.checksum,
            checksum
        );
    }

    // Parse extension table
    let ext_start = header.ext_table_offset as usize;
    let ext_end = ext_start + header.ext_table_count as usize * EXT_ENTRY_SIZE;
    if ext_end > data.len() {
        anyhow::bail!("Extension table exceeds data bounds");
    }
    let required_extensions = deserialize_ext_table(
        &data[ext_start..ext_end],
        header.ext_table_count,
    )?;

    // Parse register definitions
    let reg_start = header.reg_defs_offset as usize;
    let reg_data_end = header.instr_offset as usize; // reg defs end where instructions start
    if reg_data_end > data.len() {
        anyhow::bail!("Register definitions exceed data bounds");
    }
    let reg_data = &data[reg_start..reg_data_end];
    let mut offset = 0;
    let mut registers = Vec::with_capacity(header.reg_defs_count as usize);
    for _ in 0..header.reg_defs_count {
        registers.push(deserialize_register(reg_data, &mut offset)?);
    }

    // Parse instructions
    let instr_start = header.instr_offset as usize;
    let instr_data = &data[instr_start..content_end];
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
        name: String::new(),
        version: 1,
        domain: None,
        required_extensions,
        registers,
        instructions,
        labels: HashMap::new(),
        input_shape: Vec::new(),
        output_shape: Vec::new(),
        projections: Vec::new(),
    })
}

// =============================================================================
// Legacy TERN Format (read-only)
// =============================================================================

/// Deserialize legacy TERN format (32-byte header, 1-byte Dtype).
fn deserialize_legacy(data: &[u8]) -> Result<AssembledProgram> {
    if data.len() < LEGACY_HEADER_SIZE {
        anyhow::bail!("Legacy header too short: {} bytes", data.len());
    }

    // Parse legacy header
    let version = u16::from_le_bytes(data[4..6].try_into()?);
    let _flags = u16::from_le_bytes(data[6..8].try_into()?);
    let hot_count = data[8];
    let cold_count = data[9];
    let param_count = data[10];
    let shape_count = data[11];
    let instr_count = u32::from_le_bytes(data[12..16].try_into()?);
    let reg_def_size = u32::from_le_bytes(data[16..20].try_into()?);
    let checksum = u64::from_le_bytes(data[20..28].try_into()?);

    // Verify checksum
    let content_start = LEGACY_HEADER_SIZE;
    let content_end = content_start + reg_def_size as usize
        + instr_count as usize * INSTRUCTION_SIZE;

    if content_end > data.len() {
        anyhow::bail!(
            "Data too short: need {} bytes, have {}",
            content_end,
            data.len()
        );
    }

    let content = &data[content_start..content_end];
    let computed = xxhash(content);
    if computed != checksum {
        anyhow::bail!(
            "Legacy checksum mismatch: expected {:016x}, got {:016x}",
            checksum,
            computed
        );
    }

    // Parse legacy register definitions (1-byte Dtype)
    let reg_def_end = content_start + reg_def_size as usize;
    let reg_def_data = &data[content_start..reg_def_end];
    let mut offset = 0;
    let mut registers = Vec::new();
    let total_regs = hot_count as u32 + cold_count as u32 + param_count as u32 + shape_count as u32;
    for _ in 0..total_regs {
        registers.push(deserialize_register_legacy(reg_def_data, &mut offset)?);
    }

    // Parse instructions
    let instr_data = &data[reg_def_end..content_end];
    let mut instructions = Vec::with_capacity(instr_count as usize);
    for i in 0..instr_count as usize {
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
        name: String::new(),
        version: version as u32,
        domain: None,
        required_extensions: Vec::new(),
        registers,
        instructions,
        labels: HashMap::new(),
        input_shape: Vec::new(),
        output_shape: Vec::new(),
        projections: Vec::new(),
    })
}

/// Deserialize a legacy register definition (1-byte Dtype).
fn deserialize_register_legacy(data: &[u8], offset: &mut usize) -> Result<RegisterMeta> {
    if *offset >= data.len() {
        anyhow::bail!("Unexpected end of data in legacy register definition");
    }

    let id = Register(data[*offset]);
    *offset += 1;

    let dtype = match data.get(*offset) {
        Some(0) => Dtype::F32,
        Some(1) => Dtype::I32,
        Some(2) => Dtype::Ternary,
        Some(3) => Dtype::PackedTernary,
        Some(4) => Dtype::F16,
        Some(5) => Dtype::I16,
        Some(6) => Dtype::I8,
        Some(7) => Dtype::I64,
        _ => anyhow::bail!("Invalid legacy dtype at offset {}", *offset),
    };
    *offset += 1;

    let flags = data.get(*offset).copied().unwrap_or(0);
    let allocated = (flags & 0x01) != 0;
    let frozen = (flags & 0x02) != 0;
    *offset += 1;

    let ndims = data.get(*offset).copied().unwrap_or(0) as usize;
    *offset += 1;

    let mut shape = Vec::with_capacity(ndims);
    for _ in 0..ndims {
        if *offset + 2 > data.len() {
            anyhow::bail!("Unexpected end of data reading dimensions");
        }
        let dim = u16::from_le_bytes(data[*offset..*offset + 2].try_into()?);
        shape.push(dim as usize);
        *offset += 2;
    }

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

// =============================================================================
// Checksum
// =============================================================================

/// xxhash-like checksum.
///
/// Deterministic hash used for binary integrity verification.
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

// =============================================================================
// File I/O
// =============================================================================

/// Save program to binary file (TVMR v1 format).
pub fn save_to_file(program: &AssembledProgram, path: impl AsRef<std::path::Path>) -> Result<()> {
    let binary = serialize(program)?;
    let mut file = std::fs::File::create(path)?;
    file.write_all(&binary)?;
    Ok(())
}

/// Load program from binary file (auto-detects format).
pub fn load_from_file(path: impl AsRef<std::path::Path>) -> Result<AssembledProgram> {
    let mut file = std::fs::File::open(path)?;
    let mut data = Vec::new();
    file.read_to_end(&mut data)?;
    deserialize(&data)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vm::assemble;

    #[test]
    fn test_tvmr_roundtrip() {
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

        // Verify TVMR magic
        assert_eq!(&binary[0..4], b"TVMR");

        let recovered = deserialize(&binary).unwrap();

        assert_eq!(program.registers.len(), recovered.registers.len());
        assert_eq!(program.instructions.len(), recovered.instructions.len());

        // Verify register metadata roundtrips
        for (orig, rec) in program.registers.iter().zip(recovered.registers.iter()) {
            assert_eq!(orig.id.0, rec.id.0);
            assert_eq!(orig.shape, rec.shape);
            assert_eq!(orig.dtype, rec.dtype);
            assert_eq!(orig.thermogram_key, rec.thermogram_key);
            assert_eq!(orig.frozen, rec.frozen);
        }

        // Verify instructions roundtrip
        for (orig, rec) in program.instructions.iter().zip(recovered.instructions.iter()) {
            assert_eq!(orig, rec);
        }
    }

    #[test]
    fn test_tvmr_with_extensions() {
        let source = r#"
.requires
    ternary    0x0002
    activation 0x0003

.registers
    C0: ternary[4, 2]  key="test.w1"
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
        assert_eq!(program.required_extensions.len(), 2);

        let binary = serialize(&program).unwrap();
        let recovered = deserialize(&binary).unwrap();

        // Extension table roundtrips
        assert_eq!(recovered.required_extensions.len(), 2);
        assert_eq!(recovered.required_extensions[0].ext_id, 0x0002);
        assert_eq!(recovered.required_extensions[0].name, "ternary");
        assert_eq!(recovered.required_extensions[1].ext_id, 0x0003);
        assert_eq!(recovered.required_extensions[1].name, "activation");
    }

    #[test]
    fn test_tvmr_header_offsets() {
        let source = r#"
.requires
    tensor 0x0001

.registers
    H0: i32[4]

.program
    halt
"#;

        let program = assemble(source).unwrap();
        let binary = serialize(&program).unwrap();
        let header = Header::from_bytes(&binary).unwrap();

        // Extension table starts right after header
        assert_eq!(header.ext_table_offset, TVMR_HEADER_SIZE as u32);
        // 1 extension entry = 8 bytes
        assert_eq!(header.ext_table_count, 1);
        // Register defs start after ext table
        assert_eq!(header.reg_defs_offset, TVMR_HEADER_SIZE as u32 + EXT_ENTRY_SIZE as u32);
        assert_eq!(header.reg_defs_count, 1);
        // Instructions after reg defs
        assert!(header.instr_offset > header.reg_defs_offset);
        assert_eq!(header.instr_count, 1); // just HALT
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

        // Corrupt content after header
        let corrupt_offset = TVMR_HEADER_SIZE + 2;
        if corrupt_offset < binary.len() {
            binary[corrupt_offset] ^= 0xFF;
        }

        assert!(deserialize(&binary).is_err());
    }

    #[test]
    fn test_type_id_in_register_defs() {
        let source = r#"
.registers
    C0: ternary[8, 4]  key="test.signal"
    H0: i32[4]
    H1: f32[8]

.program
    halt
"#;

        let program = assemble(source).unwrap();
        let binary = serialize(&program).unwrap();
        let recovered = deserialize(&binary).unwrap();

        // Ternary register preserves its dtype
        assert_eq!(recovered.registers[0].dtype, Dtype::Ternary);
        assert_eq!(recovered.registers[0].shape, vec![8, 4]);
        assert_eq!(
            recovered.registers[0].thermogram_key,
            Some("test.signal".to_string())
        );

        // I32 and F32 roundtrip
        assert_eq!(recovered.registers[1].dtype, Dtype::I32);
        assert_eq!(recovered.registers[2].dtype, Dtype::F32);
    }

    #[test]
    fn test_empty_program() {
        let program = AssembledProgram {
            name: String::new(),
            version: 1,
            domain: None,
            required_extensions: Vec::new(),
            registers: Vec::new(),
            instructions: Vec::new(),
            labels: HashMap::new(),
            input_shape: Vec::new(),
            output_shape: Vec::new(),
            projections: Vec::new(),
        };

        let binary = serialize(&program).unwrap();
        assert_eq!(&binary[0..4], b"TVMR");
        assert_eq!(binary.len(), TVMR_HEADER_SIZE); // Just header, no content

        let recovered = deserialize(&binary).unwrap();
        assert_eq!(recovered.registers.len(), 0);
        assert_eq!(recovered.instructions.len(), 0);
        assert_eq!(recovered.required_extensions.len(), 0);
    }
}
