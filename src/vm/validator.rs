//! TVMR Program Validator — Pre-Execution Validation Pipeline
//!
//! Validates programs before execution to catch errors early:
//! - Extension dependency checking
//! - Opcode validity within extensions
//! - Control flow integrity (matched loops, valid jumps)
//! - Thermogram key uniqueness
//! - Register bounds validation
//! - Operand pattern conformance
//! - Semantic checks (mastery ordering, halt termination)
//! - Shape checking (optional, expensive)
//! - Bank/orchestration plausibility

use super::action::Action;
use super::assembler::{AssembledProgram, RequiredExtension};
use super::extension::{InstructionMeta, OperandPattern};
use super::instruction::Instruction;
use super::register::{Register, RegisterMeta};
use super::registry::ExtensionRegistry;
use std::collections::{HashMap, HashSet};
use std::fmt;

// =========================================================================
// Configuration
// =========================================================================

/// Controls which validation passes run.
#[derive(Debug, Clone)]
pub struct ValidationConfig {
    /// Check extension dependencies and opcode existence.
    pub check_extensions: bool,
    /// Check register operand bounds (bank/index validity).
    pub check_register_bounds: bool,
    /// Check control flow integrity (loops, jumps, calls).
    pub check_control_flow: bool,
    /// Check operand patterns match instruction metadata.
    pub check_operand_patterns: bool,
    /// Check semantic ordering (mastery after store_output, halt termination).
    pub check_semantics: bool,
    /// Check tensor shapes for provable incompatibilities (expensive).
    pub check_shapes: bool,
    /// Check bank/orchestration/neuro plausibility.
    pub check_bank_ops: bool,
    /// Check thermogram key uniqueness.
    pub check_thermogram_keys: bool,
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            check_extensions: true,
            check_register_bounds: true,
            check_control_flow: true,
            check_operand_patterns: true,
            check_semantics: true,
            check_shapes: false, // off by default — expensive
            check_bank_ops: true,
            check_thermogram_keys: true,
        }
    }
}

impl ValidationConfig {
    /// All checks enabled, including shapes.
    pub fn full() -> Self {
        Self {
            check_shapes: true,
            ..Self::default()
        }
    }

    /// Assembly-only: extensions + control flow (fast).
    pub fn quick() -> Self {
        Self {
            check_extensions: true,
            check_register_bounds: false,
            check_control_flow: true,
            check_operand_patterns: false,
            check_semantics: false,
            check_shapes: false,
            check_bank_ops: false,
            check_thermogram_keys: true,
        }
    }
}

// =========================================================================
// Diagnostics
// =========================================================================

/// Diagnostic severity level.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DiagnosticLevel {
    /// Blocks execution — the program is invalid.
    Error,
    /// Allows execution but indicates potential issues.
    Warning,
    /// Informational — no action needed.
    Info,
}

/// A single validation diagnostic.
#[derive(Debug, Clone)]
pub struct Diagnostic {
    pub level: DiagnosticLevel,
    /// Instruction index where the issue was found (if applicable).
    pub instruction_idx: Option<usize>,
    /// Human-readable message.
    pub message: String,
}

impl fmt::Display for Diagnostic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let level_str = match self.level {
            DiagnosticLevel::Error => "ERROR",
            DiagnosticLevel::Warning => "WARN",
            DiagnosticLevel::Info => "INFO",
        };
        if let Some(idx) = self.instruction_idx {
            write!(f, "[{}] instruction {}: {}", level_str, idx, self.message)
        } else {
            write!(f, "[{}] {}", level_str, self.message)
        }
    }
}

// =========================================================================
// Validator
// =========================================================================

/// Validates TVMR programs before execution.
pub struct ProgramValidator<'a> {
    registry: &'a ExtensionRegistry,
}

impl<'a> ProgramValidator<'a> {
    /// Create a new validator with the given extension registry.
    pub fn new(registry: &'a ExtensionRegistry) -> Self {
        Self { registry }
    }

    /// Validate an assembled program with the given config.
    ///
    /// Convenience wrapper that extracts fields from `AssembledProgram`.
    pub fn validate_assembled(
        &self,
        program: &AssembledProgram,
        config: &ValidationConfig,
    ) -> Vec<Diagnostic> {
        self.validate(
            &program.instructions,
            &program.registers,
            &program.required_extensions,
            &program.labels,
            config,
        )
    }

    /// Validate a program. Returns all diagnostics found.
    ///
    /// A program with any `Error`-level diagnostics should not be executed.
    pub fn validate(
        &self,
        instructions: &[Instruction],
        registers: &[RegisterMeta],
        required_extensions: &[RequiredExtension],
        labels: &HashMap<String, usize>,
        config: &ValidationConfig,
    ) -> Vec<Diagnostic> {
        let mut diagnostics = Vec::new();

        if config.check_extensions {
            self.check_extensions(instructions, required_extensions, &mut diagnostics);
        }
        if config.check_control_flow {
            self.check_control_flow(instructions, labels, &mut diagnostics);
        }
        if config.check_thermogram_keys {
            self.check_thermogram_keys(registers, &mut diagnostics);
        }
        if config.check_register_bounds {
            self.check_register_bounds(instructions, registers, &mut diagnostics);
        }
        if config.check_operand_patterns {
            self.check_operand_patterns(instructions, &mut diagnostics);
        }
        if config.check_semantics {
            self.check_semantics(instructions, &mut diagnostics);
        }
        if config.check_shapes {
            self.check_shapes(instructions, registers, &mut diagnostics);
        }
        if config.check_bank_ops {
            self.check_bank_ops(instructions, &mut diagnostics);
        }

        diagnostics
    }

    // =====================================================================
    // Phase 1: Extension dependency checking
    // =====================================================================

    /// Check that all extension IDs referenced in instructions are registered.
    fn check_extensions(
        &self,
        instructions: &[Instruction],
        required_extensions: &[RequiredExtension],
        diags: &mut Vec<Diagnostic>,
    ) {
        let mut missing_exts: HashSet<u16> = HashSet::new();
        let mut used_ext_ids: HashSet<u16> = HashSet::new();

        for (idx, instr) in instructions.iter().enumerate() {
            if instr.ext_id == 0x0000 {
                continue; // Core ISA always available
            }

            used_ext_ids.insert(instr.ext_id);

            if !self.registry.has_extension(instr.ext_id) && !missing_exts.contains(&instr.ext_id) {
                missing_exts.insert(instr.ext_id);
                diags.push(Diagnostic {
                    level: DiagnosticLevel::Error,
                    instruction_idx: Some(idx),
                    message: format!(
                        "Extension 0x{:04X} not registered (first use at instruction {})",
                        instr.ext_id, idx,
                    ),
                });
            }

            // Check opcode exists within extension
            if self.registry.has_extension(instr.ext_id)
                && self.registry.instruction_meta(instr.ext_id, instr.opcode).is_none()
            {
                diags.push(Diagnostic {
                    level: DiagnosticLevel::Warning,
                    instruction_idx: Some(idx),
                    message: format!(
                        "Opcode 0x{:04X} not found in extension 0x{:04X}",
                        instr.opcode, instr.ext_id,
                    ),
                });
            }
        }

        // Check that used extensions are declared in .requires
        let declared: HashSet<u16> = required_extensions.iter().map(|r| r.ext_id).collect();
        for &ext_id in &used_ext_ids {
            if !declared.contains(&ext_id) {
                diags.push(Diagnostic {
                    level: DiagnosticLevel::Warning,
                    instruction_idx: None,
                    message: format!(
                        "Extension 0x{:04X} used but not declared in .requires section",
                        ext_id,
                    ),
                });
            }
        }
    }

    // =====================================================================
    // Phase 2: Register safety
    // =====================================================================

    /// Check that register operands reference declared registers.
    fn check_register_bounds(
        &self,
        instructions: &[Instruction],
        registers: &[RegisterMeta],
        diags: &mut Vec<Diagnostic>,
    ) {
        let declared_ids: HashSet<u8> = registers.iter().map(|r| r.id.id()).collect();

        for (idx, instr) in instructions.iter().enumerate() {
            let pattern = self.resolve_pattern(instr);
            let reg_positions = register_positions(&pattern);

            for &pos in &reg_positions {
                let raw = instr.operands[pos];
                if raw == 0xFF {
                    continue; // NULL register — valid
                }
                // Validate register ID encoding
                let reg = Register(raw);
                let bank_idx = reg.index();
                if bank_idx >= 64 {
                    diags.push(Diagnostic {
                        level: DiagnosticLevel::Error,
                        instruction_idx: Some(idx),
                        message: format!(
                            "Register index {} out of range (max 63) in operand byte {}",
                            bank_idx, pos,
                        ),
                    });
                    continue;
                }
                // Warn if register not declared
                if !declared_ids.contains(&raw) {
                    diags.push(Diagnostic {
                        level: DiagnosticLevel::Warning,
                        instruction_idx: Some(idx),
                        message: format!(
                            "Register {} used but not declared in .registers section",
                            reg,
                        ),
                    });
                }
            }
        }
    }

    // =====================================================================
    // Phase 3: Control flow safety
    // =====================================================================

    /// Check control flow integrity.
    fn check_control_flow(
        &self,
        instructions: &[Instruction],
        labels: &HashMap<String, usize>,
        diags: &mut Vec<Diagnostic>,
    ) {
        let program_len = instructions.len();
        let mut loop_depth: i32 = 0;
        let mut has_call = false;
        let mut has_return = false;

        // Collect label targets for dead code analysis
        let label_targets: HashSet<usize> = labels.values().copied().collect();

        for (idx, instr) in instructions.iter().enumerate() {
            if instr.ext_id != 0x0000 {
                continue; // Only core ISA has control flow
            }

            let action = Action(instr.opcode);

            match action {
                Action::LOOP => {
                    loop_depth += 1;
                }
                Action::END_LOOP => {
                    loop_depth -= 1;
                    if loop_depth < 0 {
                        diags.push(Diagnostic {
                            level: DiagnosticLevel::Error,
                            instruction_idx: Some(idx),
                            message: "ENDLOOP without matching LOOP".to_string(),
                        });
                        loop_depth = 0;
                    }
                }
                Action::BREAK => {
                    if loop_depth <= 0 {
                        diags.push(Diagnostic {
                            level: DiagnosticLevel::Warning,
                            instruction_idx: Some(idx),
                            message: "BREAK outside of loop".to_string(),
                        });
                    }
                }
                Action::JUMP => {
                    let target = instr.imm16_cd() as usize;
                    if target >= program_len {
                        diags.push(Diagnostic {
                            level: DiagnosticLevel::Error,
                            instruction_idx: Some(idx),
                            message: format!(
                                "JUMP target {} out of bounds (program has {} instructions)",
                                target, program_len,
                            ),
                        });
                    }
                    // Dead code after unconditional jump
                    if idx + 1 < program_len && !label_targets.contains(&(idx + 1)) {
                        diags.push(Diagnostic {
                            level: DiagnosticLevel::Info,
                            instruction_idx: Some(idx + 1),
                            message: "Unreachable instruction after unconditional JUMP".to_string(),
                        });
                    }
                }
                Action::CALL => {
                    has_call = true;
                    let target = instr.imm16_cd() as usize;
                    if target >= program_len {
                        diags.push(Diagnostic {
                            level: DiagnosticLevel::Error,
                            instruction_idx: Some(idx),
                            message: format!(
                                "CALL target {} out of bounds (program has {} instructions)",
                                target, program_len,
                            ),
                        });
                    }
                }
                Action::RETURN => {
                    has_return = true;
                }
                Action::SKIP => {
                    let skip_count = instr.operands[0] as usize;
                    if idx + 1 + skip_count > program_len {
                        diags.push(Diagnostic {
                            level: DiagnosticLevel::Error,
                            instruction_idx: Some(idx),
                            message: format!(
                                "SKIP {} would jump past end of program (at instruction {}, program has {})",
                                skip_count, idx, program_len,
                            ),
                        });
                    }
                }
                Action::HALT => {
                    // Dead code after unconditional halt
                    if idx + 1 < program_len && !label_targets.contains(&(idx + 1)) {
                        // Only flag if there's no loop or jump that could reach the next instruction
                        let next_reachable = instructions.iter().enumerate().any(|(j, jinstr)| {
                            if j == idx || jinstr.ext_id != 0x0000 { return false; }
                            let ja = Action(jinstr.opcode);
                            match ja {
                                Action::JUMP | Action::CALL => jinstr.imm16_cd() as usize == idx + 1,
                                _ => false,
                            }
                        });
                        if !next_reachable {
                            diags.push(Diagnostic {
                                level: DiagnosticLevel::Info,
                                instruction_idx: Some(idx + 1),
                                message: "Unreachable instruction after HALT".to_string(),
                            });
                        }
                    }
                }
                _ => {}
            }
        }

        if loop_depth > 0 {
            diags.push(Diagnostic {
                level: DiagnosticLevel::Error,
                instruction_idx: None,
                message: format!("{} unclosed LOOP(s) at end of program", loop_depth),
            });
        }

        if has_return && !has_call {
            diags.push(Diagnostic {
                level: DiagnosticLevel::Warning,
                instruction_idx: None,
                message: "RETURN found but no CALL in program (possible orphan subroutine)".to_string(),
            });
        }
    }

    // =====================================================================
    // Phase 1 (continued): Thermogram key uniqueness
    // =====================================================================

    /// Check for duplicate thermogram keys across registers.
    fn check_thermogram_keys(&self, registers: &[RegisterMeta], diags: &mut Vec<Diagnostic>) {
        let mut seen_keys: HashSet<&str> = HashSet::new();

        for reg_meta in registers {
            if let Some(ref key) = reg_meta.thermogram_key {
                if !seen_keys.insert(key.as_str()) {
                    diags.push(Diagnostic {
                        level: DiagnosticLevel::Warning,
                        instruction_idx: None,
                        message: format!(
                            "Duplicate thermogram key '{}' on register {}",
                            key, reg_meta.id,
                        ),
                    });
                }
            }
        }
    }

    // =====================================================================
    // Phase 4: Operand pattern conformance
    // =====================================================================

    /// Check that operand bytes conform to declared patterns.
    fn check_operand_patterns(&self, instructions: &[Instruction], diags: &mut Vec<Diagnostic>) {
        for (idx, instr) in instructions.iter().enumerate() {
            if instr.ext_id == 0x0000 {
                continue; // Core ISA doesn't have registered InstructionMeta via extensions
            }

            if let Some(meta) = self.registry.instruction_meta(instr.ext_id, instr.opcode) {
                self.check_single_pattern(idx, instr, meta, diags);
            }
        }
    }

    fn check_single_pattern(
        &self,
        idx: usize,
        instr: &Instruction,
        meta: &InstructionMeta,
        diags: &mut Vec<Diagnostic>,
    ) {
        let ops = instr.operands;
        match &meta.operand_pattern {
            OperandPattern::None => {
                if ops != [0, 0, 0, 0] && ops != [0xFF, 0xFF, 0, 0] {
                    diags.push(Diagnostic {
                        level: DiagnosticLevel::Warning,
                        instruction_idx: Some(idx),
                        message: format!(
                            "{}: pattern is None but operands are non-zero: {:?}",
                            meta.mnemonic, ops,
                        ),
                    });
                }
            }
            OperandPattern::Reg => {
                // byte 0 = register, bytes 1-3 should be 0
                if ops[1] != 0 || ops[2] != 0 || ops[3] != 0 {
                    diags.push(Diagnostic {
                        level: DiagnosticLevel::Info,
                        instruction_idx: Some(idx),
                        message: format!(
                            "{}: Reg pattern has non-zero padding bytes: [{}, {}, {}]",
                            meta.mnemonic, ops[1], ops[2], ops[3],
                        ),
                    });
                }
            }
            OperandPattern::RegReg => {
                if ops[2] != 0 || ops[3] != 0 {
                    diags.push(Diagnostic {
                        level: DiagnosticLevel::Info,
                        instruction_idx: Some(idx),
                        message: format!(
                            "{}: RegReg pattern has non-zero padding: [{}, {}]",
                            meta.mnemonic, ops[2], ops[3],
                        ),
                    });
                }
            }
            OperandPattern::RegRegReg => {
                if ops[3] != 0 {
                    diags.push(Diagnostic {
                        level: DiagnosticLevel::Info,
                        instruction_idx: Some(idx),
                        message: format!(
                            "{}: RegRegReg pattern has non-zero byte 3: {}",
                            meta.mnemonic, ops[3],
                        ),
                    });
                }
            }
            // RegRegRegFlags, RegRegImm16, RegImm16, RegImm8, Imm32, Imm16,
            // Imm8, RegCondRegReg, Custom — all bytes are meaningful, no padding check
            _ => {}
        }
    }

    // =====================================================================
    // Phase 5: Semantic checks
    // =====================================================================

    /// Check semantic ordering and conventions.
    fn check_semantics(&self, instructions: &[Instruction], diags: &mut Vec<Diagnostic>) {
        if instructions.is_empty() {
            return;
        }

        let mut has_store_output = false;
        let mut has_mastery = false;
        let mut has_compute = false;
        let mut has_load_input = false;

        for (idx, instr) in instructions.iter().enumerate() {
            if instr.ext_id != 0x0000 {
                // Extension instructions count as compute
                if !has_compute {
                    has_compute = true;
                    if !has_load_input {
                        diags.push(Diagnostic {
                            level: DiagnosticLevel::Info,
                            instruction_idx: Some(idx),
                            message: "Compute instruction before load_input".to_string(),
                        });
                    }
                }
                continue;
            }

            let action = Action(instr.opcode);

            match action {
                Action::LOAD_INPUT => {
                    has_load_input = true;
                }
                Action::STORE_OUTPUT => {
                    has_store_output = true;
                }
                Action::MASTERY_UPDATE | Action::MASTERY_COMMIT => {
                    has_mastery = true;
                    if !has_store_output {
                        diags.push(Diagnostic {
                            level: DiagnosticLevel::Warning,
                            instruction_idx: Some(idx),
                            message: "Mastery instruction without preceding store_output".to_string(),
                        });
                    }
                }
                _ => {
                    // Track if we've hit a compute instruction
                    if action.is_forward() || action.is_ternary() {
                        if !has_compute {
                            has_compute = true;
                            if !has_load_input {
                                diags.push(Diagnostic {
                                    level: DiagnosticLevel::Info,
                                    instruction_idx: Some(idx),
                                    message: "Compute instruction before load_input".to_string(),
                                });
                            }
                        }
                    }
                }
            }
        }

        // Check last instruction is HALT
        let last = &instructions[instructions.len() - 1];
        if last.ext_id == 0x0000 && Action(last.opcode) != Action::HALT {
            diags.push(Diagnostic {
                level: DiagnosticLevel::Warning,
                instruction_idx: Some(instructions.len() - 1),
                message: "Program does not end with HALT".to_string(),
            });
        }

        // Informational: mastery program detection
        if has_mastery && has_store_output {
            diags.push(Diagnostic {
                level: DiagnosticLevel::Info,
                instruction_idx: None,
                message: "Learning program detected (has mastery instructions)".to_string(),
            });
        }
    }

    // =====================================================================
    // Phase 6: Shape checking (optional)
    // =====================================================================

    /// Track shapes through the program and flag provable incompatibilities.
    fn check_shapes(
        &self,
        instructions: &[Instruction],
        registers: &[RegisterMeta],
        diags: &mut Vec<Diagnostic>,
    ) {
        // Build shape map from register declarations
        let mut shapes: HashMap<u8, Vec<usize>> = HashMap::new();
        for reg in registers {
            if !reg.shape.is_empty() {
                shapes.insert(reg.id.id(), reg.shape.clone());
            }
        }

        for (idx, instr) in instructions.iter().enumerate() {
            if instr.ext_id != 0x0000 {
                continue; // Only check core ISA shapes for now
            }

            let action = Action(instr.opcode);

            match action {
                Action::TERNARY_MATMUL => {
                    // target = weights @ input
                    // weights: [M, N], input: [N] → target: [M]
                    let weights_id = instr.operands[1];
                    let input_id = instr.operands[2];

                    if let (Some(w_shape), Some(i_shape)) = (shapes.get(&weights_id), shapes.get(&input_id)) {
                        if w_shape.len() == 2 && i_shape.len() == 1 {
                            if w_shape[1] != i_shape[0] {
                                diags.push(Diagnostic {
                                    level: DiagnosticLevel::Error,
                                    instruction_idx: Some(idx),
                                    message: format!(
                                        "Shape mismatch in TERNARY_MATMUL: weights [{}, {}] @ input [{}] — inner dims {} != {}",
                                        w_shape[0], w_shape[1], i_shape[0], w_shape[1], i_shape[0],
                                    ),
                                });
                            } else {
                                // Propagate result shape
                                let target_id = instr.operands[0];
                                shapes.insert(target_id, vec![w_shape[0]]);
                            }
                        }
                    }
                }
                Action::ADD | Action::SUB => {
                    // target = a + b — shapes must match
                    let a_id = instr.operands[1];
                    let b_id = instr.operands[2];

                    if let (Some(a_shape), Some(b_shape)) = (shapes.get(&a_id), shapes.get(&b_id)) {
                        if a_shape != b_shape {
                            // Allow broadcast: [N] + [N] is ok, [N] + [1] is ok
                            let broadcast_ok = a_shape.len() == b_shape.len()
                                && a_shape.iter().zip(b_shape.iter()).all(|(&a, &b)| a == b || a == 1 || b == 1);
                            if !broadcast_ok {
                                diags.push(Diagnostic {
                                    level: DiagnosticLevel::Error,
                                    instruction_idx: Some(idx),
                                    message: format!(
                                        "Shape mismatch in {}: {:?} vs {:?}",
                                        action.name(), a_shape, b_shape,
                                    ),
                                });
                            }
                        }
                        // Propagate shape
                        let target_id = instr.operands[0];
                        shapes.insert(target_id, a_shape.clone());
                    }
                }
                Action::RELU | Action::SIGMOID | Action::TANH | Action::GELU => {
                    // Unary — shape preserved
                    let src_id = instr.operands[1];
                    if let Some(shape) = shapes.get(&src_id).cloned() {
                        let target_id = instr.operands[0];
                        shapes.insert(target_id, shape);
                    }
                }
                Action::LOAD_INPUT => {
                    // Shape comes from register declaration
                    let target_id = instr.operands[0];
                    // Already in shapes from register metadata
                    let _ = target_id;
                }
                _ => {}
            }
        }
    }

    // =====================================================================
    // Phase 7: Bank/orchestration checks
    // =====================================================================

    /// Check bank/orchestration/neuro operand plausibility.
    fn check_bank_ops(&self, instructions: &[Instruction], diags: &mut Vec<Diagnostic>) {
        // Known extension IDs
        const EXT_NEURO: u16 = 0x0005;
        const EXT_ORCHESTRATION: u16 = 0x0007;
        const EXT_BANK: u16 = 0x000B;

        // Max plausible chemical ID (0-7: DA, 5HT, NE, GABA, cortisol, endorphin, ACh, fatigue)
        const MAX_CHEM_ID: u8 = 7;
        // Max plausible bank slot
        const MAX_BANK_SLOT: u8 = 31;
        // Max plausible region ID
        const MAX_REGION_ID: u8 = 63;

        for (idx, instr) in instructions.iter().enumerate() {
            match instr.ext_id {
                EXT_NEURO => {
                    // Opcodes 0x0000-0x0002: CHEM_READ, CHEM_SET, CHEM_INJECT
                    // operands[1] = chem_id
                    if instr.opcode <= 0x0002 {
                        let chem_id = instr.operands[1];
                        if chem_id > MAX_CHEM_ID {
                            diags.push(Diagnostic {
                                level: DiagnosticLevel::Warning,
                                instruction_idx: Some(idx),
                                message: format!(
                                    "Chemical ID {} exceeds known range (0-{})",
                                    chem_id, MAX_CHEM_ID,
                                ),
                            });
                        }
                    }
                }
                EXT_ORCHESTRATION => {
                    // Check region IDs in operands where applicable
                    // Many orchestration ops use operands[1] as region_id or slot
                    let slot = instr.operands[1];
                    if slot > MAX_REGION_ID {
                        diags.push(Diagnostic {
                            level: DiagnosticLevel::Warning,
                            instruction_idx: Some(idx),
                            message: format!(
                                "Region/slot ID {} exceeds plausible range (0-{})",
                                slot, MAX_REGION_ID,
                            ),
                        });
                    }
                }
                EXT_BANK => {
                    // Bank ops often encode bank_slot in operands
                    // The specific operand position varies per opcode, but commonly operands[2]
                    // For now, check any operand that might be a bank_slot
                    // Opcodes 0x0000-0x000B use varying layouts, we check conservatively
                    let possible_slot = instr.operands[2]; // most bank ops put slot here
                    if possible_slot > MAX_BANK_SLOT && possible_slot != 0xFF {
                        diags.push(Diagnostic {
                            level: DiagnosticLevel::Warning,
                            instruction_idx: Some(idx),
                            message: format!(
                                "Bank slot {} exceeds plausible range (0-{})",
                                possible_slot, MAX_BANK_SLOT,
                            ),
                        });
                    }
                }
                _ => {}
            }
        }
    }

    // =====================================================================
    // Helpers
    // =====================================================================

    /// Resolve the operand pattern for an instruction.
    fn resolve_pattern(&self, instr: &Instruction) -> OperandPattern {
        if instr.ext_id == 0x0000 {
            // Core ISA — infer from opcode category
            core_isa_pattern(instr.opcode)
        } else {
            self.registry
                .instruction_meta(instr.ext_id, instr.opcode)
                .map(|m| m.operand_pattern.clone())
                .unwrap_or(OperandPattern::Custom("unknown"))
        }
    }

    /// Check if validation passed (no errors).
    pub fn is_valid(diagnostics: &[Diagnostic]) -> bool {
        !diagnostics.iter().any(|d| d.level == DiagnosticLevel::Error)
    }

    /// Count errors in diagnostics.
    pub fn error_count(diagnostics: &[Diagnostic]) -> usize {
        diagnostics.iter().filter(|d| d.level == DiagnosticLevel::Error).count()
    }

    /// Count warnings in diagnostics.
    pub fn warning_count(diagnostics: &[Diagnostic]) -> usize {
        diagnostics.iter().filter(|d| d.level == DiagnosticLevel::Warning).count()
    }
}

// =========================================================================
// Pattern helpers
// =========================================================================

/// Determine which operand byte positions contain register references
/// for a given operand pattern.
fn register_positions(pattern: &OperandPattern) -> Vec<usize> {
    match pattern {
        OperandPattern::None => vec![],
        OperandPattern::Reg => vec![0],
        OperandPattern::RegReg => vec![0, 1],
        OperandPattern::RegRegReg => vec![0, 1, 2],
        OperandPattern::RegRegRegFlags => vec![0, 1, 2],
        OperandPattern::RegRegImm16 => vec![0, 1],
        OperandPattern::RegImm16 => vec![0],
        OperandPattern::RegImm8 => vec![0],
        OperandPattern::Imm32 | OperandPattern::Imm16 | OperandPattern::Imm8 => vec![],
        OperandPattern::RegCondRegReg => vec![0, 2, 3],
        OperandPattern::Custom(_) => vec![], // Can't know without custom metadata
    }
}

/// Infer operand pattern for core ISA opcodes.
fn core_isa_pattern(opcode: u16) -> OperandPattern {
    let action = Action(opcode);
    match action {
        // System (no operands)
        Action::NOP | Action::HALT | Action::RESET | Action::SYNC | Action::YIELD => OperandPattern::None,
        Action::CHECKPOINT => OperandPattern::None,

        // Register management
        Action::LOAD_INPUT | Action::STORE_OUTPUT | Action::LOAD_TARGET => OperandPattern::Reg,
        Action::ALLOC_TENSOR | Action::FREE_TENSOR => OperandPattern::Reg,
        Action::LOAD_WEIGHTS | Action::STORE_WEIGHTS => OperandPattern::Reg,
        Action::COPY_REG | Action::SWAP_REG => OperandPattern::RegReg,
        Action::ZERO_REG => OperandPattern::Reg,

        // Architecture
        Action::DEFINE_LAYER | Action::SET_ACTIVATION | Action::SET_BIAS |
        Action::SET_DTYPE | Action::FREEZE_LAYER | Action::UNFREEZE_LAYER => OperandPattern::RegImm8,
        Action::WIRE_FORWARD | Action::WIRE_SKIP => OperandPattern::RegReg,
        Action::GROW_NEURON | Action::PRUNE_NEURON => OperandPattern::RegImm8,
        Action::INIT_RANDOM => OperandPattern::Reg,

        // Forward ops (mostly RegRegReg: dst, a, b)
        Action::MATMUL | Action::ADD | Action::MUL | Action::SUB | Action::CMP_GT => OperandPattern::RegRegReg,
        Action::RELU | Action::SIGMOID | Action::TANH | Action::GELU |
        Action::NEGATE | Action::LAYER_NORM | Action::BATCH_NORM => OperandPattern::RegReg,
        Action::SOFTMAX => OperandPattern::RegReg,
        Action::SCALE | Action::SHIFT | Action::CLAMP => OperandPattern::RegRegReg,
        Action::MAX_REDUCE => OperandPattern::RegReg,
        Action::SET_CONST => OperandPattern::RegImm16,

        // Ternary ops
        Action::TERNARY_MATMUL | Action::TERNARY_BATCH_MATMUL => OperandPattern::RegRegReg,
        Action::QUANTIZE | Action::DEQUANTIZE => OperandPattern::RegRegImm16,
        Action::PACK_TERNARY | Action::UNPACK_TERNARY => OperandPattern::RegReg,
        Action::APPLY_POLARITY | Action::APPLY_MAGNITUDE | Action::THRESHOLD_POLARITY => OperandPattern::RegRegReg,
        Action::ACCUMULATE_PRESSURE => OperandPattern::RegRegReg,
        Action::TERNARY_ADD_BIAS | Action::EMBED_LOOKUP | Action::CONCAT => OperandPattern::RegRegReg,
        Action::REDUCE_AVG | Action::SLICE => OperandPattern::RegRegRegFlags,
        Action::ARGMAX | Action::SQUEEZE | Action::UNSQUEEZE => OperandPattern::RegReg,
        Action::TRANSPOSE => OperandPattern::RegRegRegFlags,
        Action::GATE_UPDATE => OperandPattern::RegRegRegFlags,
        Action::EMBED_SEQUENCE | Action::REDUCE_MEAN_DIM => OperandPattern::RegRegReg,

        // Learning
        Action::MARK_ELIGIBILITY | Action::DECAY_ELIGIBILITY |
        Action::COMPUTE_ERROR | Action::UPDATE_WEIGHTS => OperandPattern::RegRegReg,
        Action::ADD_BABBLE | Action::DECAY_BABBLE => OperandPattern::RegImm8,
        Action::COMPUTE_RPE | Action::GATE_ERROR => OperandPattern::RegReg,
        Action::CHECKPOINT_WEIGHTS | Action::ROLLBACK_WEIGHTS | Action::CONSOLIDATE => OperandPattern::Reg,
        Action::ACTIVITY_WEIGHT => OperandPattern::RegRegReg,
        Action::CHL_FREE_START | Action::CHL_FREE_RECORD |
        Action::CHL_CLAMP_START | Action::CHL_CLAMP_RECORD |
        Action::CHL_UPDATE | Action::CHL_BACKPROP_CLAMP => OperandPattern::RegRegReg,
        Action::MASTERY_UPDATE | Action::MASTERY_COMMIT => OperandPattern::RegRegReg,

        // Control flow
        Action::LOOP => OperandPattern::Imm16,
        Action::END_LOOP | Action::BREAK => OperandPattern::None,
        Action::IF_ERROR_GT | Action::IF_ERROR_LT => OperandPattern::RegImm8,
        Action::IF_LAYER_ACTIVE => OperandPattern::Reg,
        Action::CALL | Action::JUMP => OperandPattern::Imm16,
        Action::RETURN => OperandPattern::None,
        Action::IF_ZERO | Action::IF_NONZERO => OperandPattern::Reg,
        Action::SKIP => OperandPattern::Imm8,

        // Debug
        Action::TRACE | Action::BREAKPOINT | Action::PROFILE_START |
        Action::PROFILE_END | Action::ASSERT | Action::DUMP_REG => OperandPattern::Reg,

        // Unknown
        _ => OperandPattern::Custom("unknown"),
    }
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::register::Register;

    fn default_config() -> ValidationConfig {
        ValidationConfig::default()
    }

    fn make_validator() -> (ExtensionRegistry, Vec<RequiredExtension>, HashMap<String, usize>) {
        let registry = ExtensionRegistry::with_standard_extensions();
        let required = Vec::new();
        let labels = HashMap::new();
        (registry, required, labels)
    }

    #[test]
    fn test_empty_program_valid() {
        let (registry, req, labels) = make_validator();
        let validator = ProgramValidator::new(&registry);
        let diags = validator.validate(&[], &[], &req, &labels, &default_config());
        assert!(ProgramValidator::is_valid(&diags));
    }

    #[test]
    fn test_core_instructions_always_valid() {
        let (registry, req, labels) = make_validator();
        let validator = ProgramValidator::new(&registry);

        let instructions = vec![
            Instruction { ext_id: 0x0000, opcode: 0x0000, operands: [0; 4] }, // NOP
            Instruction { ext_id: 0x0000, opcode: 0x0001, operands: [0; 4] }, // HALT
        ];

        let diags = validator.validate(&instructions, &[], &req, &labels, &default_config());
        assert!(ProgramValidator::is_valid(&diags));
    }

    #[test]
    fn test_missing_extension_error() {
        let registry = ExtensionRegistry::new(); // No extensions registered
        let req = Vec::new();
        let labels = HashMap::new();
        let validator = ProgramValidator::new(&registry);

        let instructions = vec![
            Instruction { ext_id: 0x0099, opcode: 0x0000, operands: [0; 4] },
        ];

        let diags = validator.validate(&instructions, &[], &req, &labels, &default_config());
        assert!(!ProgramValidator::is_valid(&diags));
        assert!(diags[0].message.contains("0x0099"));
    }

    #[test]
    fn test_unmatched_endloop() {
        let (registry, req, labels) = make_validator();
        let validator = ProgramValidator::new(&registry);

        let instructions = vec![
            Instruction { ext_id: 0x0000, opcode: Action::END_LOOP.0, operands: [0xFF, 0xFF, 0, 0] },
        ];

        let diags = validator.validate(&instructions, &[], &req, &labels, &default_config());
        assert!(!ProgramValidator::is_valid(&diags));
        assert!(diags.iter().any(|d| d.message.contains("ENDLOOP")));
    }

    #[test]
    fn test_unclosed_loop() {
        let (registry, req, labels) = make_validator();
        let validator = ProgramValidator::new(&registry);

        let instructions = vec![
            Instruction { ext_id: 0x0000, opcode: Action::LOOP.0, operands: [0xFF, 0xFF, 0, 10] },
            Instruction { ext_id: 0x0000, opcode: Action::HALT.0, operands: [0; 4] },
        ];

        let diags = validator.validate(&instructions, &[], &req, &labels, &default_config());
        assert!(!ProgramValidator::is_valid(&diags));
        assert!(diags.iter().any(|d| d.message.contains("unclosed LOOP")));
    }

    #[test]
    fn test_matched_loop_valid() {
        let (registry, req, labels) = make_validator();
        let validator = ProgramValidator::new(&registry);

        let instructions = vec![
            Instruction { ext_id: 0x0000, opcode: Action::LOOP.0, operands: [0xFF, 0xFF, 0, 5] },
            Instruction { ext_id: 0x0000, opcode: Action::NOP.0, operands: [0; 4] },
            Instruction { ext_id: 0x0000, opcode: Action::END_LOOP.0, operands: [0xFF, 0xFF, 0, 0] },
            Instruction { ext_id: 0x0000, opcode: Action::HALT.0, operands: [0; 4] },
        ];

        let diags = validator.validate(&instructions, &[], &req, &labels, &default_config());
        assert!(ProgramValidator::is_valid(&diags));
    }

    #[test]
    fn test_duplicate_thermogram_key() {
        let (registry, req, labels) = make_validator();
        let validator = ProgramValidator::new(&registry);

        let registers = vec![
            RegisterMeta {
                id: Register::cold(0),
                shape: vec![32],
                dtype: super::super::register::Dtype::Ternary,
                allocated: true,
                thermogram_key: Some("same.key".to_string()),
                frozen: false,
            },
            RegisterMeta {
                id: Register::cold(1),
                shape: vec![16],
                dtype: super::super::register::Dtype::Ternary,
                allocated: true,
                thermogram_key: Some("same.key".to_string()),
                frozen: false,
            },
        ];

        let diags = validator.validate(&[], &registers, &req, &labels, &default_config());
        assert!(diags.iter().any(|d| d.message.contains("Duplicate thermogram key")));
    }

    #[test]
    fn test_jump_out_of_bounds() {
        let (registry, req, labels) = make_validator();
        let validator = ProgramValidator::new(&registry);

        let instructions = vec![
            Instruction { ext_id: 0x0000, opcode: Action::JUMP.0, operands: [0xFF, 0xFF, 0x01, 0x00] }, // jump to 256
            Instruction { ext_id: 0x0000, opcode: Action::HALT.0, operands: [0; 4] },
        ];

        let diags = validator.validate(&instructions, &[], &req, &labels, &default_config());
        assert!(!ProgramValidator::is_valid(&diags));
        assert!(diags.iter().any(|d| d.message.contains("JUMP target")));
    }

    #[test]
    fn test_call_out_of_bounds() {
        let (registry, req, labels) = make_validator();
        let validator = ProgramValidator::new(&registry);

        let instructions = vec![
            Instruction { ext_id: 0x0000, opcode: Action::CALL.0, operands: [0xFF, 0xFF, 0x00, 0xFF] }, // call to 255
            Instruction { ext_id: 0x0000, opcode: Action::HALT.0, operands: [0; 4] },
        ];

        let diags = validator.validate(&instructions, &[], &req, &labels, &default_config());
        assert!(!ProgramValidator::is_valid(&diags));
        assert!(diags.iter().any(|d| d.message.contains("CALL target")));
    }

    #[test]
    fn test_return_without_call() {
        let (registry, req, labels) = make_validator();
        let validator = ProgramValidator::new(&registry);

        let instructions = vec![
            Instruction { ext_id: 0x0000, opcode: Action::RETURN.0, operands: [0; 4] },
            Instruction { ext_id: 0x0000, opcode: Action::HALT.0, operands: [0; 4] },
        ];

        let diags = validator.validate(&instructions, &[], &req, &labels, &default_config());
        assert!(diags.iter().any(|d| d.message.contains("RETURN found but no CALL")));
    }

    #[test]
    fn test_break_outside_loop() {
        let (registry, req, labels) = make_validator();
        let validator = ProgramValidator::new(&registry);

        let instructions = vec![
            Instruction { ext_id: 0x0000, opcode: Action::BREAK.0, operands: [0xFF, 0xFF, 0, 0] },
            Instruction { ext_id: 0x0000, opcode: Action::HALT.0, operands: [0; 4] },
        ];

        let diags = validator.validate(&instructions, &[], &req, &labels, &default_config());
        assert!(diags.iter().any(|d| d.message.contains("BREAK outside")));
    }

    #[test]
    fn test_no_halt_warning() {
        let (registry, req, labels) = make_validator();
        let validator = ProgramValidator::new(&registry);

        let instructions = vec![
            Instruction { ext_id: 0x0000, opcode: Action::NOP.0, operands: [0; 4] },
        ];

        let diags = validator.validate(&instructions, &[], &req, &labels, &default_config());
        assert!(diags.iter().any(|d| d.message.contains("does not end with HALT")));
    }

    #[test]
    fn test_mastery_without_store_output() {
        let (registry, req, labels) = make_validator();
        let validator = ProgramValidator::new(&registry);

        let h0 = Register::hot(0);
        let c0 = Register::cold(0);
        let instructions = vec![
            Instruction { ext_id: 0x0000, opcode: Action::LOAD_INPUT.0, operands: [h0.0, 0xFF, 0, 0] },
            Instruction { ext_id: 0x0000, opcode: Action::MASTERY_UPDATE.0, operands: [h0.0, c0.0, h0.0, 0] },
            Instruction { ext_id: 0x0000, opcode: Action::HALT.0, operands: [0; 4] },
        ];

        let regs = vec![
            RegisterMeta { id: h0, shape: vec![8], dtype: super::super::register::Dtype::I32, allocated: true, thermogram_key: None, frozen: false },
            RegisterMeta { id: c0, shape: vec![8], dtype: super::super::register::Dtype::Ternary, allocated: true, thermogram_key: Some("test.w".to_string()), frozen: false },
        ];

        let diags = validator.validate(&instructions, &regs, &req, &labels, &default_config());
        assert!(diags.iter().any(|d| d.message.contains("Mastery instruction without preceding store_output")));
    }

    #[test]
    fn test_shape_mismatch_matmul() {
        let (registry, req, labels) = make_validator();
        let config = ValidationConfig::full(); // enable shapes
        let validator = ProgramValidator::new(&registry);

        let h0 = Register::hot(0);
        let h1 = Register::hot(1);
        let c0 = Register::cold(0);

        // C0: [32, 12], H0: [8] — inner dim mismatch (12 vs 8)
        let instructions = vec![
            Instruction::ternary_matmul(h1, c0, h0),
            Instruction::halt(),
        ];

        let regs = vec![
            RegisterMeta { id: h0, shape: vec![8], dtype: super::super::register::Dtype::I32, allocated: true, thermogram_key: None, frozen: false },
            RegisterMeta { id: h1, shape: vec![32], dtype: super::super::register::Dtype::I32, allocated: true, thermogram_key: None, frozen: false },
            RegisterMeta { id: c0, shape: vec![32, 12], dtype: super::super::register::Dtype::Ternary, allocated: true, thermogram_key: Some("w1".to_string()), frozen: false },
        ];

        let diags = validator.validate(&instructions, &regs, &req, &labels, &config);
        assert!(diags.iter().any(|d| d.level == DiagnosticLevel::Error && d.message.contains("Shape mismatch")));
    }

    #[test]
    fn test_shape_valid_matmul() {
        let (registry, req, labels) = make_validator();
        let config = ValidationConfig::full();
        let validator = ProgramValidator::new(&registry);

        let h0 = Register::hot(0);
        let h1 = Register::hot(1);
        let c0 = Register::cold(0);

        // C0: [32, 12], H0: [12] — valid
        let instructions = vec![
            Instruction::ternary_matmul(h1, c0, h0),
            Instruction::halt(),
        ];

        let regs = vec![
            RegisterMeta { id: h0, shape: vec![12], dtype: super::super::register::Dtype::I32, allocated: true, thermogram_key: None, frozen: false },
            RegisterMeta { id: h1, shape: vec![32], dtype: super::super::register::Dtype::I32, allocated: true, thermogram_key: None, frozen: false },
            RegisterMeta { id: c0, shape: vec![32, 12], dtype: super::super::register::Dtype::Ternary, allocated: true, thermogram_key: Some("w1".to_string()), frozen: false },
        ];

        let diags = validator.validate(&instructions, &regs, &req, &labels, &config);
        assert!(!diags.iter().any(|d| d.level == DiagnosticLevel::Error && d.message.contains("Shape mismatch")));
    }

    #[test]
    fn test_undeclared_extension_warning() {
        let (registry, labels) = (ExtensionRegistry::with_standard_extensions(), HashMap::new());
        let validator = ProgramValidator::new(&registry);

        // Use activation extension (0x0003) without declaring it
        let instructions = vec![
            Instruction { ext_id: 0x0003, opcode: 0x0000, operands: [0x01, 0x00, 0, 0] }, // activation.RELU
            Instruction::halt(),
        ];

        let req = Vec::new(); // No .requires
        let diags = validator.validate(&instructions, &[], &req, &labels, &default_config());
        assert!(diags.iter().any(|d| d.message.contains("not declared in .requires")));
    }

    #[test]
    fn test_skip_past_end() {
        let (registry, req, labels) = make_validator();
        let validator = ProgramValidator::new(&registry);

        let instructions = vec![
            Instruction { ext_id: 0x0000, opcode: Action::SKIP.0, operands: [10, 0, 0, 0] }, // skip 10
            Instruction::halt(),
        ];

        let diags = validator.validate(&instructions, &[], &req, &labels, &default_config());
        assert!(!ProgramValidator::is_valid(&diags));
        assert!(diags.iter().any(|d| d.message.contains("SKIP")));
    }

    #[test]
    fn test_neuro_chemical_bounds() {
        let (registry, req, labels) = make_validator();
        let config = ValidationConfig::default();
        let validator = ProgramValidator::new(&registry);

        let instructions = vec![
            // CHEM_READ with chem_id=99 (out of range)
            Instruction { ext_id: 0x0005, opcode: 0x0000, operands: [0x00, 99, 0, 0] },
            Instruction::halt(),
        ];

        let req_with_neuro = vec![RequiredExtension { name: "tvmr.neuro".to_string(), ext_id: 0x0005 }];
        let diags = validator.validate(&instructions, &[], &req_with_neuro, &labels, &config);
        assert!(diags.iter().any(|d| d.message.contains("Chemical ID")));
    }

    #[test]
    fn test_validate_assembled_convenience() {
        let (registry, _, _) = make_validator();
        let validator = ProgramValidator::new(&registry);

        let program = AssembledProgram {
            name: "test".to_string(),
            version: 1,
            domain: None,
            required_extensions: Vec::new(),
            registers: Vec::new(),
            instructions: vec![Instruction::halt()],
            labels: HashMap::new(),
            input_shape: vec![1],
            output_shape: vec![1],
            projections: Vec::new(),
            skull_origin: None,
        };

        let diags = validator.validate_assembled(&program, &ValidationConfig::default());
        assert!(ProgramValidator::is_valid(&diags));
    }

    #[test]
    fn test_config_quick() {
        let config = ValidationConfig::quick();
        assert!(config.check_extensions);
        assert!(config.check_control_flow);
        assert!(!config.check_register_bounds);
        assert!(!config.check_operand_patterns);
        assert!(!config.check_semantics);
        assert!(!config.check_shapes);
    }

    #[test]
    fn test_config_full() {
        let config = ValidationConfig::full();
        assert!(config.check_shapes);
    }

    #[test]
    fn test_error_and_warning_counts() {
        let diags = vec![
            Diagnostic { level: DiagnosticLevel::Error, instruction_idx: None, message: "e1".into() },
            Diagnostic { level: DiagnosticLevel::Warning, instruction_idx: None, message: "w1".into() },
            Diagnostic { level: DiagnosticLevel::Warning, instruction_idx: None, message: "w2".into() },
            Diagnostic { level: DiagnosticLevel::Info, instruction_idx: None, message: "i1".into() },
        ];
        assert_eq!(ProgramValidator::error_count(&diags), 1);
        assert_eq!(ProgramValidator::warning_count(&diags), 2);
        assert!(!ProgramValidator::is_valid(&diags));
    }
}
