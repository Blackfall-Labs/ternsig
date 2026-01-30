//! Validation utilities for ternsig files
//!
//! Provides batch validation with detailed error reporting.
//! Two levels of validation:
//! - Assembly validation (always runs): syntax, structure, register parsing
//! - Deep validation (with `--deep`): extensions, control flow, semantics, shapes
//!
//! # Example
//!
//! ```ignore
//! use ternsig::validate::{validate_file, validate_directory, ValidationResult};
//!
//! let results = validate_directory("path/to/ternsig/files")?;
//! for result in &results {
//!     match result {
//!         ValidationResult::Ok { path, program, diagnostics } => {
//!             println!("✓ {}: {} instructions, {} diagnostics",
//!                 path.display(), program.instructions.len(), diagnostics.len());
//!         }
//!         ValidationResult::Err { path, error } => {
//!             eprintln!("✗ {}: {}", path.display(), error);
//!         }
//!     }
//! }
//! ```

use std::path::{Path, PathBuf};
use crate::vm::{AssembledProgram, assemble, AssemblerError};
use crate::vm::validator::{Diagnostic, ProgramValidator, ValidationConfig, DiagnosticLevel};
use crate::vm::registry::ExtensionRegistry;

/// Result of validating a single ternsig file
#[derive(Debug)]
pub enum ValidationResult {
    /// File assembled successfully (and optionally deep-validated)
    Ok {
        path: PathBuf,
        program: AssembledProgram,
        /// Diagnostics from ProgramValidator (empty if deep validation not run)
        diagnostics: Vec<Diagnostic>,
    },
    /// File failed to assemble
    Err {
        path: PathBuf,
        error: ValidationError,
    },
}

impl ValidationResult {
    /// Returns true if validation succeeded (assembly passed, no error-level diagnostics)
    pub fn is_ok(&self) -> bool {
        match self {
            Self::Ok { diagnostics, .. } => !diagnostics.iter().any(|d| d.level == DiagnosticLevel::Error),
            Self::Err { .. } => false,
        }
    }

    /// Returns true if validation failed
    pub fn is_err(&self) -> bool {
        !self.is_ok()
    }

    /// Get the path
    pub fn path(&self) -> &Path {
        match self {
            Self::Ok { path, .. } => path,
            Self::Err { path, .. } => path,
        }
    }

    /// Get diagnostics (empty for Err variant)
    pub fn diagnostics(&self) -> &[Diagnostic] {
        match self {
            Self::Ok { diagnostics, .. } => diagnostics,
            Self::Err { .. } => &[],
        }
    }

    /// Check if deep validation produced errors
    pub fn has_errors(&self) -> bool {
        match self {
            Self::Ok { diagnostics, .. } => diagnostics.iter().any(|d| d.level == DiagnosticLevel::Error),
            Self::Err { .. } => true,
        }
    }
}

/// Validation error with context
#[derive(Debug)]
pub struct ValidationError {
    /// Line number (if available)
    pub line: Option<usize>,
    /// Error message
    pub message: String,
    /// Source snippet around the error (if available)
    pub snippet: Option<String>,
}

impl std::fmt::Display for ValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(line) = self.line {
            write!(f, "line {}: {}", line, self.message)?;
        } else {
            write!(f, "{}", self.message)?;
        }
        if let Some(snippet) = &self.snippet {
            write!(f, "\n  | {}", snippet)?;
        }
        Ok(())
    }
}

impl From<AssemblerError> for ValidationError {
    fn from(e: AssemblerError) -> Self {
        Self {
            line: Some(e.line),
            message: e.message,
            snippet: None,
        }
    }
}

impl From<std::io::Error> for ValidationError {
    fn from(e: std::io::Error) -> Self {
        Self {
            line: None,
            message: e.to_string(),
            snippet: None,
        }
    }
}

/// Validate a single ternsig file (assembly only).
pub fn validate_file<P: AsRef<Path>>(path: P) -> ValidationResult {
    validate_file_with_config(path, None)
}

/// Validate a single ternsig file with optional deep validation.
///
/// When `config` is Some, runs ProgramValidator after assembly.
pub fn validate_file_with_config<P: AsRef<Path>>(
    path: P,
    config: Option<&ValidationConfig>,
) -> ValidationResult {
    let path = path.as_ref().to_path_buf();

    // Read file
    let source = match std::fs::read_to_string(&path) {
        Ok(s) => s,
        Err(e) => {
            return ValidationResult::Err {
                path,
                error: e.into(),
            };
        }
    };

    // Assemble
    match assemble(&source) {
        Ok(program) => {
            // Run deep validation if config provided
            let diagnostics = if let Some(config) = config {
                let registry = ExtensionRegistry::with_standard_extensions();
                let validator = ProgramValidator::new(&registry);
                validator.validate_assembled(&program, config)
            } else {
                Vec::new()
            };

            ValidationResult::Ok { path, program, diagnostics }
        }
        Err(e) => {
            // Extract snippet around error line
            let snippet = source
                .lines()
                .nth(e.line.saturating_sub(1))
                .map(|s| s.trim().to_string());

            ValidationResult::Err {
                path,
                error: ValidationError {
                    line: Some(e.line),
                    message: e.message,
                    snippet,
                },
            }
        }
    }
}

/// Validate all .ternsig files in a directory (recursive, assembly only).
pub fn validate_directory<P: AsRef<Path>>(dir: P) -> std::io::Result<Vec<ValidationResult>> {
    validate_directory_with_config(dir, None)
}

/// Validate all .ternsig files in a directory with optional deep validation.
pub fn validate_directory_with_config<P: AsRef<Path>>(
    dir: P,
    config: Option<&ValidationConfig>,
) -> std::io::Result<Vec<ValidationResult>> {
    let mut results = Vec::new();
    validate_directory_recursive(dir.as_ref(), config, &mut results)?;

    // Sort by path for consistent output
    results.sort_by(|a, b| a.path().cmp(b.path()));

    Ok(results)
}

fn validate_directory_recursive(
    dir: &Path,
    config: Option<&ValidationConfig>,
    results: &mut Vec<ValidationResult>,
) -> std::io::Result<()> {
    if !dir.is_dir() {
        return Ok(());
    }

    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();

        if path.is_dir() {
            validate_directory_recursive(&path, config, results)?;
        } else if path.extension().map_or(false, |e| e == "ternsig") {
            results.push(validate_file_with_config(&path, config));
        }
    }

    Ok(())
}

/// Summary of validation results
#[derive(Debug, Default)]
pub struct ValidationSummary {
    pub total: usize,
    pub passed: usize,
    pub failed: usize,
    pub errors: Vec<(PathBuf, ValidationError)>,
    /// Deep validation diagnostics (errors/warnings/info counts)
    pub diag_errors: usize,
    pub diag_warnings: usize,
    pub diag_info: usize,
}

impl ValidationSummary {
    /// Create summary from results
    pub fn from_results(results: &[ValidationResult]) -> Self {
        let mut summary = Self::default();
        summary.total = results.len();

        for result in results {
            match result {
                ValidationResult::Ok { diagnostics, .. } => {
                    let has_errors = diagnostics.iter().any(|d| d.level == DiagnosticLevel::Error);
                    if has_errors {
                        summary.failed += 1;
                    } else {
                        summary.passed += 1;
                    }
                    for d in diagnostics {
                        match d.level {
                            DiagnosticLevel::Error => summary.diag_errors += 1,
                            DiagnosticLevel::Warning => summary.diag_warnings += 1,
                            DiagnosticLevel::Info => summary.diag_info += 1,
                        }
                    }
                }
                ValidationResult::Err { path, error } => {
                    summary.failed += 1;
                    summary.errors.push((path.clone(), ValidationError {
                        line: error.line,
                        message: error.message.clone(),
                        snippet: error.snippet.clone(),
                    }));
                }
            }
        }

        summary
    }

    /// Print summary to stderr
    pub fn print_report(&self) {
        if !self.errors.is_empty() {
            eprintln!("\n{} ASSEMBLY ERRORS:", self.errors.len());
            for (path, error) in &self.errors {
                eprintln!("\n  {}", path.display());
                if let Some(line) = error.line {
                    eprintln!("    line {}: {}", line, error.message);
                } else {
                    eprintln!("    {}", error.message);
                }
                if let Some(snippet) = &error.snippet {
                    eprintln!("    | {}", snippet);
                }
            }
            eprintln!();
        }

        eprintln!(
            "Validated {} files: {} passed, {} failed",
            self.total, self.passed, self.failed
        );

        if self.diag_errors + self.diag_warnings + self.diag_info > 0 {
            eprintln!(
                "Diagnostics: {} errors, {} warnings, {} info",
                self.diag_errors, self.diag_warnings, self.diag_info
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_valid_source() {
        use tempfile::NamedTempFile;
        use std::io::Write;

        let mut file = NamedTempFile::new().unwrap();
        write!(file, r#"
.registers
    H0: i32[1]

.program
    halt
"#).unwrap();

        let result = validate_file(file.path());
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_invalid_source() {
        use tempfile::NamedTempFile;
        use std::io::Write;

        let mut file = NamedTempFile::new().unwrap();
        write!(file, r#"
.registers
    H0: i32[1]

.program
    invalid_instruction H0
    halt
"#).unwrap();

        let result = validate_file(file.path());
        assert!(result.is_err());

        if let ValidationResult::Err { error, .. } = result {
            assert!(error.line.is_some());
            assert!(error.message.contains("Unknown") || error.message.contains("invalid"));
        }
    }

    #[test]
    fn test_validate_with_deep_config() {
        use tempfile::NamedTempFile;
        use std::io::Write;

        let mut file = NamedTempFile::new().unwrap();
        write!(file, r#"
.registers
    H0: i32[1]

.program
    load_input H0
    store_output H0
    halt
"#).unwrap();

        let config = ValidationConfig::default();
        let result = validate_file_with_config(file.path(), Some(&config));
        assert!(result.is_ok());
        // Should have diagnostics (no errors expected)
        assert!(ProgramValidator::is_valid(result.diagnostics()));
    }

    #[test]
    fn test_summary_with_diagnostics() {
        use tempfile::NamedTempFile;
        use std::io::Write;

        let mut file = NamedTempFile::new().unwrap();
        // Program without halt — triggers warning
        write!(file, r#"
.registers
    H0: i32[1]

.program
    nop
"#).unwrap();

        let config = ValidationConfig::default();
        let result = validate_file_with_config(file.path(), Some(&config));

        let results = vec![result];
        let summary = ValidationSummary::from_results(&results);
        assert_eq!(summary.total, 1);
        assert_eq!(summary.passed, 1); // warnings don't fail
        assert!(summary.diag_warnings > 0); // "does not end with HALT"
    }
}
