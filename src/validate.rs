//! Validation utilities for ternsig files
//!
//! Provides batch validation with detailed error reporting.
//!
//! # Example
//!
//! ```ignore
//! use ternsig::validate::{validate_directory, ValidationResult};
//!
//! let results = validate_directory("path/to/ternsig/files")?;
//! for result in &results {
//!     match result {
//!         ValidationResult::Ok { path, program } => {
//!             println!("✓ {}: {} instructions", path.display(), program.instructions.len());
//!         }
//!         ValidationResult::Err { path, error } => {
//!             eprintln!("✗ {}: {}", path.display(), error);
//!         }
//!     }
//! }
//! ```

use std::path::{Path, PathBuf};
use crate::vm::{AssembledProgram, assemble, AssemblerError};

/// Result of validating a single ternsig file
#[derive(Debug)]
pub enum ValidationResult {
    /// File assembled successfully
    Ok {
        path: PathBuf,
        program: AssembledProgram,
    },
    /// File failed to assemble
    Err {
        path: PathBuf,
        error: ValidationError,
    },
}

impl ValidationResult {
    /// Returns true if validation succeeded
    pub fn is_ok(&self) -> bool {
        matches!(self, Self::Ok { .. })
    }

    /// Returns true if validation failed
    pub fn is_err(&self) -> bool {
        matches!(self, Self::Err { .. })
    }

    /// Get the path
    pub fn path(&self) -> &Path {
        match self {
            Self::Ok { path, .. } => path,
            Self::Err { path, .. } => path,
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

/// Validate a single ternsig file
pub fn validate_file<P: AsRef<Path>>(path: P) -> ValidationResult {
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
        Ok(program) => ValidationResult::Ok { path, program },
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

/// Validate all .ternsig files in a directory (recursive)
pub fn validate_directory<P: AsRef<Path>>(dir: P) -> std::io::Result<Vec<ValidationResult>> {
    let mut results = Vec::new();
    validate_directory_recursive(dir.as_ref(), &mut results)?;

    // Sort by path for consistent output
    results.sort_by(|a, b| a.path().cmp(b.path()));

    Ok(results)
}

fn validate_directory_recursive(dir: &Path, results: &mut Vec<ValidationResult>) -> std::io::Result<()> {
    if !dir.is_dir() {
        return Ok(());
    }

    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();

        if path.is_dir() {
            validate_directory_recursive(&path, results)?;
        } else if path.extension().map_or(false, |e| e == "ternsig") {
            results.push(validate_file(&path));
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
}

impl ValidationSummary {
    /// Create summary from results
    pub fn from_results(results: &[ValidationResult]) -> Self {
        let mut summary = Self::default();
        summary.total = results.len();

        for result in results {
            match result {
                ValidationResult::Ok { .. } => summary.passed += 1,
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
        if self.failed > 0 {
            eprintln!("\n{} ERRORS:", self.failed);
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
}
