//! TVMR Program Validator — Pre-Execution Validation Pipeline
//!
//! Validates programs before execution to catch errors early:
//! - Extension dependency checking
//! - Opcode validity within extensions
//! - Control flow integrity (matched loops, valid jumps)
//! - Thermogram key uniqueness

use super::instruction::Instruction;
use super::register::RegisterMeta;
use super::registry::ExtensionRegistry;
use std::collections::HashSet;
use std::fmt;

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

/// Validates TVMR programs before execution.
pub struct ProgramValidator<'a> {
    registry: &'a ExtensionRegistry,
}

impl<'a> ProgramValidator<'a> {
    /// Create a new validator with the given extension registry.
    pub fn new(registry: &'a ExtensionRegistry) -> Self {
        Self { registry }
    }

    /// Validate a program. Returns all diagnostics found.
    ///
    /// A program with any `Error`-level diagnostics should not be executed.
    pub fn validate(
        &self,
        instructions: &[Instruction],
        registers: &[RegisterMeta],
    ) -> Vec<Diagnostic> {
        let mut diagnostics = Vec::new();

        self.check_extensions(instructions, &mut diagnostics);
        self.check_control_flow(instructions, &mut diagnostics);
        self.check_thermogram_keys(registers, &mut diagnostics);

        diagnostics
    }

    /// Check that all extension IDs referenced in instructions are registered.
    fn check_extensions(&self, instructions: &[Instruction], diags: &mut Vec<Diagnostic>) {
        let mut missing_exts: HashSet<u16> = HashSet::new();

        for (idx, instr) in instructions.iter().enumerate() {
            if instr.ext_id == 0x0000 {
                continue; // Core ISA always available
            }

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
    }

    /// Check control flow integrity.
    fn check_control_flow(&self, instructions: &[Instruction], diags: &mut Vec<Diagnostic>) {
        // Track LOOP/ENDLOOP nesting depth
        let mut loop_depth: i32 = 0;

        for (idx, instr) in instructions.iter().enumerate() {
            if instr.ext_id != 0x0000 {
                continue; // Only core ISA has control flow
            }

            // Core ISA control flow opcodes (from our plan)
            match instr.opcode {
                0x0405 => {
                    // LOOP
                    loop_depth += 1;
                }
                0x0406 => {
                    // ENDLOOP
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
                0x0407 => {
                    // BREAK
                    if loop_depth <= 0 {
                        diags.push(Diagnostic {
                            level: DiagnosticLevel::Warning,
                            instruction_idx: Some(idx),
                            message: "BREAK outside of loop".to_string(),
                        });
                    }
                }
                0x0404 => {
                    // RETURN — valid if call_stack might be non-empty
                    // (can't fully validate without execution, so just info)
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
    }

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

    /// Check if validation passed (no errors).
    pub fn is_valid(diagnostics: &[Diagnostic]) -> bool {
        !diagnostics.iter().any(|d| d.level == DiagnosticLevel::Error)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::register::Register;

    #[test]
    fn test_empty_program_valid() {
        let registry = ExtensionRegistry::new();
        let validator = ProgramValidator::new(&registry);
        let diags = validator.validate(&[], &[]);
        assert!(ProgramValidator::is_valid(&diags));
    }

    #[test]
    fn test_core_instructions_always_valid() {
        let registry = ExtensionRegistry::new();
        let validator = ProgramValidator::new(&registry);

        let instructions = vec![
            Instruction { ext_id: 0x0000, opcode: 0x0000, operands: [0; 4] }, // NOP
            Instruction { ext_id: 0x0000, opcode: 0x0001, operands: [0; 4] }, // HALT
        ];

        let diags = validator.validate(&instructions, &[]);
        assert!(ProgramValidator::is_valid(&diags));
    }

    #[test]
    fn test_missing_extension_error() {
        let registry = ExtensionRegistry::new();
        let validator = ProgramValidator::new(&registry);

        let instructions = vec![
            Instruction { ext_id: 0x0099, opcode: 0x0000, operands: [0; 4] },
        ];

        let diags = validator.validate(&instructions, &[]);
        assert!(!ProgramValidator::is_valid(&diags));
        assert!(diags[0].message.contains("0x0099"));
    }

    #[test]
    fn test_unmatched_endloop() {
        let registry = ExtensionRegistry::new();
        let validator = ProgramValidator::new(&registry);

        let instructions = vec![
            Instruction { ext_id: 0x0000, opcode: 0x0406, operands: [0; 4] }, // ENDLOOP without LOOP
        ];

        let diags = validator.validate(&instructions, &[]);
        assert!(!ProgramValidator::is_valid(&diags));
    }

    #[test]
    fn test_duplicate_thermogram_key() {
        let registry = ExtensionRegistry::new();
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

        let diags = validator.validate(&[], &registers);
        assert!(diags.iter().any(|d| d.message.contains("Duplicate thermogram key")));
    }
}
