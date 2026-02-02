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
    RegisterMeta, Action, Dtype, Instruction, Register,
    INSTRUCTION_SIZE, TERNSIG_MAGIC, TERNSIG_VERSION,
};
use std::collections::HashMap;

/// Required extension declaration (from .requires section)
#[derive(Debug, Clone)]
pub struct RequiredExtension {
    /// Extension name (e.g. "ternary", "activation")
    pub name: String,
    /// Extension ID
    pub ext_id: u16,
}

/// Projection declaration (from .projection section).
///
/// Declares an outgoing spike projection from this program's domain
/// to a target domain. Multiple `.projection` sections per program
/// are supported for multiple outgoing bundles.
#[derive(Debug, Clone)]
pub struct ProjectionMeta {
    /// Target domain name (e.g. "spatial", "auditory").
    pub target: String,
    /// Named bundle identifier (e.g. "dorsal_stream").
    pub bundle: String,
    /// Base synaptic weight (-128..127).
    pub weight: i8,
    /// Axonal delay in ticks (1-15).
    pub delay: u8,
    /// Fraction of source neurons that project (0.0-1.0).
    pub density: f32,
    /// Spatial mapping strategy.
    pub mapping: String,
    /// Optional neuromodulator name (e.g. "dopamine", "acetylcholine").
    pub modulator: Option<String>,
}

/// Assembled Ternsig program
#[derive(Debug, Clone)]
pub struct AssembledProgram {
    /// Program name
    pub name: String,
    /// Program version (from .meta)
    pub version: u32,
    /// Domain category (from .meta, e.g. "fleet.classifier")
    pub domain: Option<String>,
    /// Required extensions (from .requires section)
    pub required_extensions: Vec<RequiredExtension>,
    /// Register definitions
    pub registers: Vec<RegisterMeta>,
    /// Compiled instructions
    pub instructions: Vec<Instruction>,
    /// Labels to instruction indices
    pub labels: HashMap<String, usize>,
    /// Input shape (inferred from load_input)
    pub input_shape: Vec<usize>,
    /// Output shape (inferred from store_output)
    pub output_shape: Vec<usize>,
    /// Projection declarations (from .projection sections)
    pub projections: Vec<ProjectionMeta>,
    /// Skull-space origin (from .meta skull_x/skull_y/skull_z)
    pub skull_origin: Option<(u16, u16, u16)>,
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
    /// Required extensions (from .requires section)
    required_extensions: Vec<RequiredExtension>,
    /// User-provided extension metadata for assembly-time mnemonic resolution.
    /// Allows user-defined extensions (0x0100+) to resolve at assembly time
    /// without being compiled into the standard extension set.
    extra_extension_meta: Vec<ExtraExtMeta>,
    /// Meta: program name
    meta_name: Option<String>,
    /// Meta: program version
    meta_version: u32,
    /// Meta: domain category
    meta_domain: Option<String>,
    /// Meta: declared input dimension (for validation)
    meta_input_dim: Option<usize>,
    /// Meta: declared output dimension (for validation)
    meta_output_dim: Option<usize>,
    /// Accumulated projection declarations
    projections: Vec<ProjectionMeta>,
    /// Current projection being built (Some while inside a .projection section)
    current_projection: Option<ProjectionMeta>,
    /// Meta: skull placement coordinates (x, y, z)
    meta_skull_x: Option<u16>,
    meta_skull_y: Option<u16>,
    meta_skull_z: Option<u16>,
}

/// Extension metadata provided externally for assembly-time resolution.
pub struct ExtraExtMeta {
    pub ext_id: u16,
    pub name: String,
    pub instructions: Vec<super::extension::InstructionMeta>,
}

impl Assembler {
    pub fn new() -> Self {
        Self {
            line_number: 0,
            registers: Vec::new(),
            instructions: Vec::new(),
            labels: HashMap::new(),
            unresolved_labels: Vec::new(),
            required_extensions: Vec::new(),
            extra_extension_meta: Vec::new(),
            meta_name: None,
            meta_version: 1,
            meta_domain: None,
            meta_input_dim: None,
            meta_output_dim: None,
            projections: Vec::new(),
            current_projection: None,
            meta_skull_x: None,
            meta_skull_y: None,
            meta_skull_z: None,
        }
    }

    /// Register external extension metadata for assembly-time resolution.
    ///
    /// User-defined extensions (ext_id >= 0x0100) aren't compiled into
    /// the standard extension set. This method lets the assembler resolve
    /// their mnemonics at assembly time.
    pub fn register_extension_meta(&mut self, ext_id: u16, name: &str, instructions: &[super::extension::InstructionMeta]) {
        self.extra_extension_meta.push(ExtraExtMeta {
            ext_id,
            name: name.to_string(),
            instructions: instructions.to_vec(),
        });
    }

    /// Assemble source code into a program
    pub fn assemble(&mut self, source: &str) -> Result<AssembledProgram, AssemblerError> {
        self.registers.clear();
        self.instructions.clear();
        self.labels.clear();
        self.unresolved_labels.clear();
        self.required_extensions.clear();
        self.meta_name = None;
        self.meta_version = 1;
        self.meta_domain = None;
        self.meta_input_dim = None;
        self.meta_output_dim = None;
        self.projections.clear();
        self.current_projection = None;
        self.meta_skull_x = None;
        self.meta_skull_y = None;
        self.meta_skull_z = None;

        let mut in_meta = false;
        let mut in_requires = false;
        let mut in_registers = false;
        let mut in_program = false;
        let mut in_projection = false;
        let mut in_skipped_section = false; // .aliases, .config - content ignored

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
                // Finalize any in-progress projection before switching sections
                if in_projection {
                    self.finalize_projection()?;
                }

                match line.to_lowercase().as_str() {
                    ".meta" => {
                        in_meta = true;
                        in_requires = false;
                        in_registers = false;
                        in_program = false;
                        in_projection = false;
                        in_skipped_section = false;
                        continue;
                    }
                    ".requires" => {
                        in_meta = false;
                        in_requires = true;
                        in_registers = false;
                        in_program = false;
                        in_projection = false;
                        in_skipped_section = false;
                        continue;
                    }
                    ".registers" => {
                        in_meta = false;
                        in_requires = false;
                        in_registers = true;
                        in_program = false;
                        in_projection = false;
                        in_skipped_section = false;
                        continue;
                    }
                    ".program" | ".code" => {
                        in_meta = false;
                        in_requires = false;
                        in_registers = false;
                        in_program = true;
                        in_projection = false;
                        in_skipped_section = false;
                        continue;
                    }
                    ".projection" => {
                        in_meta = false;
                        in_requires = false;
                        in_registers = false;
                        in_program = false;
                        in_projection = true;
                        in_skipped_section = false;
                        // Start a new projection with defaults
                        self.current_projection = Some(ProjectionMeta {
                            target: String::new(),
                            bundle: String::new(),
                            weight: 50,
                            delay: 1,
                            density: 0.3,
                            mapping: "topographic".to_string(),
                            modulator: None,
                        });
                        continue;
                    }
                    ".aliases" => {
                        in_meta = false;
                        in_requires = false;
                        in_registers = false;
                        in_program = false;
                        in_projection = false;
                        in_skipped_section = true;
                        continue;
                    }
                    ".config" => {
                        in_meta = false;
                        in_requires = false;
                        in_registers = false;
                        in_program = false;
                        in_projection = false;
                        in_skipped_section = true;
                        continue;
                    }
                    _ => {
                        return Err(self.error(format!("Unknown directive: {}", line)));
                    }
                }
            }

            // Parse content based on section
            if in_skipped_section {
                // Skip content in .aliases, .config sections
                continue;
            } else if in_meta {
                self.parse_meta(line)?;
            } else if in_requires {
                self.parse_requires(line)?;
            } else if in_registers {
                self.parse_register(line)?;
            } else if in_projection {
                self.parse_projection_field(line)?;
            } else if in_program {
                // Handle include directive - skip as we use embedded firmware
                if line.starts_with("include") {
                    continue;
                }
                self.parse_instruction(line)?;
            } else {
                return Err(self.error("Code outside of section".to_string()));
            }
        }

        // Finalize any trailing projection section
        if in_projection {
            self.finalize_projection()?;
        }

        // Resolve labels
        self.resolve_labels()?;

        // Infer input/output shapes from load_input/store_output instructions
        let input_shape = self.infer_input_shape();
        let output_shape = self.infer_output_shape();

        // Validate declared vs inferred dimensions (warn, don't error)
        if let Some(declared) = self.meta_input_dim {
            let inferred: usize = input_shape.iter().product();
            if declared != inferred {
                log::warn!(
                    "Input dim mismatch: .meta declares {}, inferred {} from registers",
                    declared, inferred
                );
            }
        }
        if let Some(declared) = self.meta_output_dim {
            let inferred: usize = output_shape.iter().product();
            if declared != inferred {
                log::warn!(
                    "Output dim mismatch: .meta declares {}, inferred {} from registers",
                    declared, inferred
                );
            }
        }

        // Validate skull coordinates: all three or none
        let skull_origin = match (self.meta_skull_x, self.meta_skull_y, self.meta_skull_z) {
            (Some(x), Some(y), Some(z)) => Some((x, y, z)),
            (None, None, None) => None,
            _ => return Err(self.error(
                "Partial skull placement: skull_x, skull_y, skull_z must all be present or all absent".to_string()
            )),
        };

        Ok(AssembledProgram {
            name: self.meta_name.clone().unwrap_or_default(),
            version: self.meta_version,
            domain: self.meta_domain.clone(),
            required_extensions: self.required_extensions.clone(),
            registers: self.registers.clone(),
            instructions: self.instructions.clone(),
            labels: self.labels.clone(),
            input_shape,
            output_shape,
            projections: self.projections.clone(),
            skull_origin,
        })
    }

    /// Parse a .meta section line
    fn parse_meta(&mut self, line: &str) -> Result<(), AssemblerError> {
        // Format: key  value  or  key  "string value"
        // Examples:
        //   name        "complexity_classifier"
        //   version     1
        //   input_dim   48
        //   output_dim  3
        //   domain      "fleet.classifier"

        let parts: Vec<&str> = line.splitn(2, char::is_whitespace).collect();
        if parts.is_empty() {
            return Ok(()); // Empty line
        }

        let key = parts[0].trim().to_lowercase();
        let value = parts.get(1).map(|s| s.trim()).unwrap_or("");

        match key.as_str() {
            "name" => {
                // Remove surrounding quotes if present
                let name = value.trim_matches('"').to_string();
                self.meta_name = Some(name);
            }
            "version" => {
                let version = value.parse::<u32>()
                    .map_err(|_| self.error(format!("Invalid version: {}", value)))?;
                self.meta_version = version;
            }
            "input_dim" => {
                let dim = value.parse::<usize>()
                    .map_err(|_| self.error(format!("Invalid input_dim: {}", value)))?;
                self.meta_input_dim = Some(dim);
            }
            "output_dim" => {
                let dim = value.parse::<usize>()
                    .map_err(|_| self.error(format!("Invalid output_dim: {}", value)))?;
                self.meta_output_dim = Some(dim);
            }
            "domain" => {
                let domain = value.trim_matches('"').to_string();
                self.meta_domain = Some(domain);
            }
            "skull_x" => {
                let v = value.parse::<u16>()
                    .map_err(|_| self.error(format!("Invalid skull_x: {}", value)))?;
                self.meta_skull_x = Some(v);
            }
            "skull_y" => {
                let v = value.parse::<u16>()
                    .map_err(|_| self.error(format!("Invalid skull_y: {}", value)))?;
                self.meta_skull_y = Some(v);
            }
            "skull_z" => {
                let v = value.parse::<u16>()
                    .map_err(|_| self.error(format!("Invalid skull_z: {}", value)))?;
                self.meta_skull_z = Some(v);
            }
            _ => {
                // Unknown meta key - just ignore for forward compatibility
                log::debug!("Ignoring unknown .meta key: {}", key);
            }
        }

        Ok(())
    }

    /// Parse a key-value field inside a .projection section.
    ///
    /// Format: `key  value` or `key  "string value"`
    fn parse_projection_field(&mut self, line: &str) -> Result<(), AssemblerError> {
        let parts: Vec<&str> = line.splitn(2, char::is_whitespace).collect();
        if parts.is_empty() {
            return Ok(());
        }

        let key = parts[0].trim().to_lowercase();
        let value = parts.get(1).map(|s| s.trim()).unwrap_or("");
        let line_number = self.line_number;

        let proj = match self.current_projection.as_mut() {
            Some(p) => p,
            None => return Err(AssemblerError {
                line: line_number,
                message: ".projection field outside projection section".to_string(),
            }),
        };

        match key.as_str() {
            "target" => proj.target = value.trim_matches('"').to_string(),
            "bundle" => proj.bundle = value.trim_matches('"').to_string(),
            "weight" => {
                proj.weight = value.parse::<i8>().map_err(|_| AssemblerError {
                    line: line_number,
                    message: format!("Invalid projection weight: {}", value),
                })?;
            }
            "delay" => {
                proj.delay = value.parse::<u8>().map_err(|_| AssemblerError {
                    line: line_number,
                    message: format!("Invalid projection delay: {}", value),
                })?;
            }
            "density" => {
                proj.density = value.parse::<f32>().map_err(|_| AssemblerError {
                    line: line_number,
                    message: format!("Invalid projection density: {}", value),
                })?;
            }
            "mapping" => proj.mapping = value.trim_matches('"').to_lowercase(),
            "modulator" => {
                let mod_name = value.trim_matches('"').to_lowercase();
                if mod_name == "none" || mod_name.is_empty() {
                    proj.modulator = None;
                } else {
                    proj.modulator = Some(mod_name);
                }
            }
            _ => {
                log::debug!("Ignoring unknown .projection key: {}", key);
            }
        }

        Ok(())
    }

    /// Finalize and store the current in-progress projection.
    fn finalize_projection(&mut self) -> Result<(), AssemblerError> {
        if let Some(proj) = self.current_projection.take() {
            if proj.target.is_empty() {
                return Err(self.error(".projection section missing 'target' field".to_string()));
            }
            if proj.bundle.is_empty() {
                return Err(self.error(".projection section missing 'bundle' field".to_string()));
            }
            self.projections.push(proj);
        }
        Ok(())
    }

    /// Parse a .requires section line
    ///
    /// Format: `extension_name  0xEXT_ID`
    /// Example: `ternary    0x0002`
    fn parse_requires(&mut self, line: &str) -> Result<(), AssemblerError> {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.is_empty() {
            return Ok(());
        }

        let name = parts[0].to_lowercase();

        // Parse explicit ext_id if provided, otherwise resolve from name
        let ext_id = if let Some(id_str) = parts.get(1) {
            if id_str.starts_with("0x") || id_str.starts_with("0X") {
                u16::from_str_radix(&id_str[2..], 16)
                    .map_err(|_| self.error(format!("Invalid extension ID: {}", id_str)))?
            } else {
                id_str
                    .parse::<u16>()
                    .map_err(|_| self.error(format!("Invalid extension ID: {}", id_str)))?
            }
        } else {
            // Resolve from standard extension names
            super::extensions::resolve_ext_name(&name)
                .ok_or_else(|| self.error(format!("Unknown extension: {}", name)))?
        };

        // Check for duplicate
        if self.required_extensions.iter().any(|r| r.ext_id == ext_id) {
            return Err(self.error(format!("Duplicate extension requirement: {}", name)));
        }

        self.required_extensions.push(RequiredExtension {
            name,
            ext_id,
        });

        Ok(())
    }

    /// Parse a register definition line
    fn parse_register(&mut self, line: &str) -> Result<(), AssemblerError> {
        // Format: C0: ternary[32, 12]  key="chip.audio.w1"
        // Or:     H0: i32[12]
        // Or:     T0: f32              (scalar - no brackets means shape [1])

        let colon_pos = line
            .find(':')
            .ok_or_else(|| self.error("Missing ':' in register definition".to_string()))?;

        let reg_name = line[..colon_pos].trim();
        let rest = line[colon_pos + 1..].trim();

        // Parse register ID
        let reg = Register::parse(reg_name)
            .ok_or_else(|| self.error(format!("Invalid register: {}", reg_name)))?;

        // Parse type and shape: ternary[32, 12] or i32[12] or f32 (scalar)
        let (dtype_str, shape) = if let Some(bracket_start) = rest.find('[') {
            // Has brackets - parse shape
            let bracket_end = rest
                .find(']')
                .ok_or_else(|| self.error("Missing ']' in type definition".to_string()))?;

            let dtype_str = rest[..bracket_start].trim();
            let shape_str = &rest[bracket_start + 1..bracket_end];

            // Parse shape dimensions
            let shape: Vec<usize> = shape_str
                .split(',')
                .map(|s| {
                    s.trim()
                        .parse()
                        .map_err(|_| self.error(format!("Invalid dimension: {}", s.trim())))
                })
                .collect::<Result<Vec<_>, _>>()?;

            (dtype_str, shape)
        } else {
            // No brackets - scalar type, shape is [1]
            // Extract dtype (first word before any whitespace or key=)
            let dtype_str = rest
                .split(|c: char| c.is_whitespace())
                .next()
                .unwrap_or(rest)
                .trim();
            (dtype_str, vec![1])
        };

        let dtype = Dtype::parse(dtype_str)
            .ok_or_else(|| self.error(format!("Unknown dtype: {}", dtype_str)))?;

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
                Ok(Instruction::core(
                    Action::COPY_REG.0,
                    [target.0, source.0, 0, 0],
                ))
            }
            "zero_reg" | "zero" => {
                let target = self.parse_register_operand(ops.get(0))?;
                Ok(Instruction::core(
                    Action::ZERO_REG.0,
                    [target.0, Register::NULL.0, 0, 0],
                ))
            }

            // Forward ops
            "ternary_matmul" => {
                let target = self.parse_register_operand(ops.get(0))?;
                let weights = self.parse_register_operand(ops.get(1))?;
                let input = self.parse_register_operand(ops.get(2))?;
                Ok(Instruction::ternary_matmul(target, weights, input))
            }
            "ternary_batch_matmul" => {
                let target = self.parse_register_operand(ops.get(0))?;
                let weights = self.parse_register_operand(ops.get(1))?;
                let input = self.parse_register_operand(ops.get(2))?;
                Ok(Instruction::ternary_batch_matmul(target, weights, input))
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
                Ok(Instruction::core(
                    Action::SUB.0,
                    [target.0, source.0, other.0, 0],
                ))
            }
            "mul" => {
                let target = self.parse_register_operand(ops.get(0))?;
                let source = self.parse_register_operand(ops.get(1))?;
                let other = self.parse_register_operand(ops.get(2))?;
                Ok(Instruction::core(
                    Action::MUL.0,
                    [target.0, source.0, other.0, 0],
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
                Ok(Instruction::core(
                    Action::SIGMOID.0,
                    [target.0, source.0, 0, gain],
                ))
            }
            "gelu" => {
                let target = self.parse_register_operand(ops.get(0))?;
                let source = self.parse_register_operand(ops.get(1))?;
                Ok(Instruction::core(
                    Action::GELU.0,
                    [target.0, source.0, 0, 0],
                ))
            }
            "tanh" => {
                let target = self.parse_register_operand(ops.get(0))?;
                let source = self.parse_register_operand(ops.get(1))?;
                Ok(Instruction::core(
                    Action::TANH.0,
                    [target.0, source.0, 0, 0],
                ))
            }
            "softmax" => {
                let target = self.parse_register_operand(ops.get(0))?;
                let source = self.parse_register_operand(ops.get(1))?;
                Ok(Instruction::core(
                    Action::SOFTMAX.0,
                    [target.0, source.0, 0, 0],
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
                Ok(Instruction::core(
                    Action::CMP_GT.0,
                    [target.0, source.0, other.0, 0],
                ))
            }
            "max_reduce" => {
                let target = self.parse_register_operand(ops.get(0))?;
                let source = self.parse_register_operand(ops.get(1))?;
                Ok(Instruction::core(
                    Action::MAX_REDUCE.0,
                    [target.0, source.0, 0, 0],
                ))
            }
            "set" => {
                // set target, value  — load scalar constant into register
                let target = self.parse_register_operand(ops.get(0))?;
                let value = self.parse_immediate(ops.get(1))? as u16;
                let hi = (value >> 8) as u8;
                let lo = (value & 0xFF) as u8;
                Ok(Instruction::core(
                    Action::SET_CONST.0,
                    [target.0, 0, hi, lo],
                ))
            }
            "clamp" => {
                // clamp target, source, min, max
                let target = self.parse_register_operand(ops.get(0))?;
                let source = self.parse_register_operand(ops.get(1))?;
                let min_val = self.parse_immediate(ops.get(2))? as u8;
                let max_val = self.parse_immediate(ops.get(3))? as u8;
                Ok(Instruction::core(
                    Action::CLAMP.0,
                    [target.0, source.0, min_val, max_val],
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
                Ok(Instruction::core(
                    Action::LOAD_TARGET.0,
                    [target.0, Register::NULL.0, 0, 0],
                ))
            }
            "mastery_update" => {
                // mastery_update weights, activity, direction [, scale=15, threshold_div=4]
                let weights = self.parse_register_operand(ops.get(0))?;
                let activity = self.parse_register_operand(ops.get(1))?;
                let direction = self.parse_register_operand(ops.get(2))?;
                let scale = self.parse_param_or_default(ops.get(3), "scale", 15)?;
                let _threshold_div = self.parse_param_or_default(ops.get(4), "threshold_div", 4)?;
                // Note: threshold_div encoded in aux position (operands[2] high bits)
                // but legacy bridge only preserves operands[3] as modifier[0].
                // For now, scale goes to operands[3] (modifier[0]), threshold_div
                // falls back to default (4) via the legacy bridge.
                Ok(Instruction::core(
                    Action::MASTERY_UPDATE.0,
                    [weights.0, activity.0, direction.0, scale],
                ))
            }
            "mastery_commit" => {
                // mastery_commit weights [, threshold=50, step=5]
                let weights = self.parse_register_operand(ops.get(0))?;
                let threshold = self.parse_param_or_default(ops.get(1), "threshold", 50)?;
                let _step = self.parse_param_or_default(ops.get(2), "step", 5)?;
                // Note: step falls back to default (5) via legacy bridge.
                Ok(Instruction::core(
                    Action::MASTERY_COMMIT.0,
                    [weights.0, Register::NULL.0, 0, threshold],
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
                Ok(Instruction::core(
                    Action::TERNARY_ADD_BIAS.0,
                    [target.0, source.0, bias.0, 0],
                ))
            }
            "embed_lookup" => {
                // embed_lookup target, table, indices
                let target = self.parse_register_operand(ops.get(0))?;
                let table = self.parse_register_operand(ops.get(1))?;
                let indices = self.parse_register_operand(ops.get(2))?;
                Ok(Instruction::embed_lookup(target, table, indices))
            }
            "embed_sequence" => {
                // embed_sequence target, table, count
                let target = self.parse_register_operand(ops.get(0))?;
                let table = self.parse_register_operand(ops.get(1))?;
                let count = self.parse_immediate(ops.get(2))? as u8;
                Ok(Instruction::embed_sequence(target, table, count))
            }
            "reduce_avg" => {
                // reduce_avg target, source, start, count
                let target = self.parse_register_operand(ops.get(0))?;
                let source = self.parse_register_operand(ops.get(1))?;
                let start = self.parse_immediate(ops.get(2))? as u8;
                let count = self.parse_immediate(ops.get(3))? as u8;
                Ok(Instruction::reduce_avg(target, source, start, count))
            }
            "reduce_mean" | "mean_dim" => {
                // reduce_mean target, source, dim
                let target = self.parse_register_operand(ops.get(0))?;
                let source = self.parse_register_operand(ops.get(1))?;
                let dim = self.parse_immediate(ops.get(2))? as u8;
                Ok(Instruction::reduce_mean_dim(target, source, dim))
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
            "jump" => {
                let label = ops.get(0)
                    .ok_or_else(|| self.error("jump requires a label".into()))?
                    .trim();
                let idx = self.instructions.len();
                self.unresolved_labels.push((idx, label.to_string()));
                Ok(Instruction::core(
                    Action::JUMP.0,
                    [0, 0, 0, 0], // operands[2:3] patched by resolve_labels
                ))
            }
            "if_zero" => {
                let reg = self.parse_register_operand(ops.get(0))?;
                let label = ops.get(1)
                    .ok_or_else(|| self.error("if_zero requires reg, label".into()))?
                    .trim();
                let idx = self.instructions.len();
                self.unresolved_labels.push((idx, label.to_string()));
                Ok(Instruction::core(
                    Action::IF_ZERO.0,
                    [reg.0, 0, 0, 0], // operands[2:3] patched by resolve_labels
                ))
            }
            "if_nonzero" => {
                let reg = self.parse_register_operand(ops.get(0))?;
                let label = ops.get(1)
                    .ok_or_else(|| self.error("if_nonzero requires reg, label".into()))?
                    .trim();
                let idx = self.instructions.len();
                self.unresolved_labels.push((idx, label.to_string()));
                Ok(Instruction::core(
                    Action::IF_NONZERO.0,
                    [reg.0, 0, 0, 0], // operands[2:3] patched by resolve_labels
                ))
            }
            "loop" => {
                let count = self.parse_immediate(ops.get(0))? as u16;
                Ok(Instruction::loop_n(count))
            }
            "end_loop" => Ok(Instruction::end_loop()),
            "break" => Ok(Instruction::break_loop()),

            _ => {
                // Try extension-qualified mnemonic: ext.MNEMONIC
                if let Some(dot_pos) = mnemonic.find('.') {
                    let ext_name = &mnemonic[..dot_pos];
                    let ext_mnemonic = &mnemonic[dot_pos + 1..];
                    self.build_ext_instruction(ext_name, ext_mnemonic, ops)
                } else {
                    Err(self.error(format!("Unknown mnemonic: {}", mnemonic)))
                }
            }
        }
    }

    /// Build an extension-qualified instruction (e.g. `activation.relu`)
    fn build_ext_instruction(
        &self,
        ext_name: &str,
        mnemonic: &str,
        ops: &[&str],
    ) -> Result<Instruction, AssemblerError> {
        let upper_mnemonic = mnemonic.to_uppercase();

        // --- Try standard extensions first ---
        if let Some((ext_id, opcode)) = super::extensions::resolve_qualified_mnemonic(ext_name, mnemonic) {
            // Standard extension resolved. Try extension-specific operand assembly.
            let exts = super::extensions::standard_extensions();
            for ext in &exts {
                if ext.ext_id() == ext_id {
                    if let Some(result) = ext.assemble_operands(&upper_mnemonic, ops) {
                        let operands = result.map_err(|e| self.error(e))?;
                        return Ok(Instruction::ext(ext_id, opcode, operands));
                    }
                    break;
                }
            }

            let operands = self.parse_generic_ext_operands(ops);
            return Ok(Instruction::ext(ext_id, opcode, operands));
        }

        // --- Try user-provided extension metadata ---
        // Check if this ext_name was declared in .requires or registered via register_extension_meta
        let ext_id = self.required_extensions.iter()
            .find(|req| req.name == ext_name)
            .map(|req| req.ext_id);

        if let Some(ext_id) = ext_id {
            // Declared extension — look up mnemonic in extra_extension_meta
            for extra in &self.extra_extension_meta {
                if extra.ext_id == ext_id {
                    for meta in &extra.instructions {
                        if meta.mnemonic == upper_mnemonic {
                            // Try extension-specific operand assembly from metadata
                            let operands = self.parse_ext_operands_from_pattern(&meta.operand_pattern, ops)?;
                            return Ok(Instruction::ext(ext_id, meta.opcode, operands));
                        }
                    }
                    // Extension found but mnemonic not recognized
                    return Err(self.error(format!(
                        "Unknown mnemonic '{}' in extension '{}' (0x{:04X})",
                        mnemonic, ext_name, ext_id
                    )));
                }
            }

            // Extension declared in .requires but no metadata registered.
            // Fall back to generic: opcode=0, generic operands.
            // Runtime will error if the opcode is wrong.
            let operands = self.parse_generic_ext_operands(ops);
            return Ok(Instruction::ext(ext_id, 0, operands));
        }

        Err(self.error(format!(
            "Unknown extension mnemonic: {}.{}",
            ext_name, mnemonic
        )))
    }

    /// Parse operands based on an OperandPattern from extension metadata.
    fn parse_ext_operands_from_pattern(
        &self,
        pattern: &super::extension::OperandPattern,
        ops: &[&str],
    ) -> Result<[u8; 4], AssemblerError> {
        use super::extension::OperandPattern;
        match pattern {
            OperandPattern::None => Ok([0u8; 4]),
            OperandPattern::Reg => {
                let mut operands = [0u8; 4];
                if let Some(op) = ops.first() {
                    operands[0] = self.parse_register_operand(Some(op))?.0;
                }
                Ok(operands)
            }
            OperandPattern::RegReg => {
                let mut operands = [0u8; 4];
                if let Some(op) = ops.get(0) {
                    operands[0] = self.parse_register_operand(Some(op))?.0;
                }
                if let Some(op) = ops.get(1) {
                    operands[1] = self.parse_register_operand(Some(op))?.0;
                }
                Ok(operands)
            }
            OperandPattern::RegImm8 => {
                let mut operands = [0u8; 4];
                if let Some(op) = ops.get(0) {
                    operands[0] = self.parse_register_operand(Some(op))?.0;
                }
                if let Some(op) = ops.get(1) {
                    operands[1] = self.parse_immediate(Some(op))? as u8;
                }
                Ok(operands)
            }
            OperandPattern::Imm8 => {
                let mut operands = [0u8; 4];
                if let Some(op) = ops.get(0) {
                    operands[0] = self.parse_immediate(Some(op))? as u8;
                }
                Ok(operands)
            }
            OperandPattern::Custom(_) | _ => {
                // Custom patterns: fall back to generic parsing
                Ok(self.parse_generic_ext_operands(ops))
            }
        }
    }

    /// Generic operand parsing for extension instructions.
    fn parse_generic_ext_operands(&self, ops: &[&str]) -> [u8; 4] {
        let mut operands = [0u8; 4];
        for (i, op) in ops.iter().take(4).enumerate() {
            if let Ok(reg) = self.parse_register_operand(Some(op)) {
                operands[i] = reg.0;
            } else if let Ok(imm) = self.parse_immediate(Some(op)) {
                operands[i] = imm as u8;
            }
        }
        operands
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
            if instr.opcode == Action::LOAD_INPUT.0 && instr.ext_id == 0 {
                // Find the target register's shape
                if let Some(reg_meta) = self.registers.iter().find(|r| r.id == instr.target()) {
                    return reg_meta.shape.clone();
                }
            }
        }
        vec![1] // Default if no load_input found
    }

    /// Infer output shape from store_output instruction's source register
    fn infer_output_shape(&self) -> Vec<usize> {
        for instr in &self.instructions {
            if instr.opcode == Action::STORE_OUTPUT.0 && instr.ext_id == 0 {
                // Find the source register's shape
                if let Some(reg_meta) = self.registers.iter().find(|r| r.id == instr.source()) {
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

            // Update instruction's operands with target address
            let instr = &mut self.instructions[*instr_idx];
            let count = (*target_idx as u16).to_be_bytes();
            instr.operands[2] = count[0];
            instr.operands[3] = count[1];
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
        assert_eq!(program.instructions[0].opcode, Action::LOAD_INPUT.0);

        // Check ternary_matmul
        assert_eq!(program.instructions[1].opcode, Action::TERNARY_MATMUL.0);

        // Check halt
        assert_eq!(program.instructions[4].opcode, Action::HALT.0);
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

    #[test]
    fn test_requires_section() {
        let source = r#"
.meta
    name "test_program"
    version 1

.requires
    ternary    0x0002
    activation 0x0003
    learning

.registers
    H0: i32[12]
    H1: i32[32]
    C0: ternary[32, 12]  key="test.w1"

.program
    load_input    H0
    ternary_matmul H1, C0, H0
    relu          H1, H1
    halt
"#;

        let program = assemble(source).expect("Assembly failed");

        // Verify required extensions
        assert_eq!(program.required_extensions.len(), 3);
        assert_eq!(program.required_extensions[0].name, "ternary");
        assert_eq!(program.required_extensions[0].ext_id, 0x0002);
        assert_eq!(program.required_extensions[1].name, "activation");
        assert_eq!(program.required_extensions[1].ext_id, 0x0003);
        assert_eq!(program.required_extensions[2].name, "learning");
        assert_eq!(program.required_extensions[2].ext_id, 0x0004);
    }

    #[test]
    fn test_ext_qualified_mnemonic() {
        let source = r#"
.requires
    activation

.registers
    H0: i32[4]
    H1: i32[4]

.program
    activation.relu H0, H1
    halt
"#;

        let program = assemble(source).expect("Assembly failed");

        // First instruction should be extension instruction
        let instr = &program.instructions[0];
        assert_eq!(instr.ext_id, 0x0003); // activation ext_id
        assert_eq!(instr.opcode, 0x0000); // RELU is opcode 0 in activation
    }

    #[test]
    fn test_requires_name_only() {
        let source = r#"
.requires
    tensor
    ternary
    activation
    learning
    neuro
    arch

.registers
    H0: i32[4]

.program
    halt
"#;

        let program = assemble(source).expect("Assembly failed");
        assert_eq!(program.required_extensions.len(), 6);
        assert_eq!(program.required_extensions[0].ext_id, 0x0001);
        assert_eq!(program.required_extensions[5].ext_id, 0x0006);
    }

    #[test]
    fn test_projection_section() {
        let source = r#"
.meta
    name "visual_encoder"
    domain "visual"

.projection
    target      "spatial"
    bundle      "dorsal_stream"
    weight      60
    delay       2
    density     0.3
    mapping     topographic
    modulator   dopamine

.projection
    target      "auditory"
    bundle      "ventral_stream"
    weight      50
    delay       3
    density     0.25
    mapping     convergent
    modulator   dopamine

.registers
    H0: i32[4]

.program
    halt
"#;

        let program = assemble(source).expect("Assembly failed");
        assert_eq!(program.projections.len(), 2);

        let p0 = &program.projections[0];
        assert_eq!(p0.target, "spatial");
        assert_eq!(p0.bundle, "dorsal_stream");
        assert_eq!(p0.weight, 60);
        assert_eq!(p0.delay, 2);
        assert!((p0.density - 0.3).abs() < 0.01);
        assert_eq!(p0.mapping, "topographic");
        assert_eq!(p0.modulator, Some("dopamine".to_string()));

        let p1 = &program.projections[1];
        assert_eq!(p1.target, "auditory");
        assert_eq!(p1.bundle, "ventral_stream");
        assert_eq!(p1.weight, 50);
        assert_eq!(p1.delay, 3);
        assert_eq!(p1.mapping, "convergent");
    }

    #[test]
    fn test_projection_missing_target_errors() {
        let source = r#"
.projection
    bundle  "test"

.program
    halt
"#;
        let result = assemble(source);
        assert!(result.is_err(), "should error on missing target");
    }

    #[test]
    fn test_projection_inhibitory_weight() {
        let source = r#"
.projection
    target      "thalamus"
    bundle      "pallido_thalamic"
    weight      -50
    delay       1
    density     0.3
    mapping     convergent
    modulator   dopamine

.program
    halt
"#;
        let program = assemble(source).expect("Assembly failed");
        assert_eq!(program.projections.len(), 1);
        assert_eq!(program.projections[0].weight, -50);
    }

    #[test]
    fn test_meta_skull_position_parsed() {
        let source = r#"
.meta
    name        "test_placement"
    domain      "spatial"
    skull_x     36
    skull_y     44
    skull_z     28

.program
    halt
"#;
        let program = assemble(source).expect("Assembly failed");
        assert_eq!(program.skull_origin, Some((36, 44, 28)));
    }

    #[test]
    fn test_meta_skull_position_absent_is_none() {
        let source = r#"
.meta
    name        "no_placement"
    domain      "auditory"

.program
    halt
"#;
        let program = assemble(source).expect("Assembly failed");
        assert_eq!(program.skull_origin, None);
    }

    #[test]
    fn test_meta_skull_partial_errors() {
        let source = r#"
.meta
    name        "partial"
    skull_x     10
    skull_y     20

.program
    halt
"#;
        let result = assemble(source);
        assert!(result.is_err(), "Partial skull placement should error");
    }
}
