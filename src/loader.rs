//! Program Loader - Clean API for loading Ternsig programs
//!
//! Uses datacard-rs for .card file handling with proper validation.
//!
//! # Usage
//!
//! ## Runtime loading
//!
//! ```ignore
//! use ternsig::loader::{ProgramLoader, load_path};
//!
//! // Configure data directory once at startup
//! ProgramLoader::set_data_dir("/path/to/ternsig");
//!
//! // Load .ternsig source file
//! let program = load_path("core/networks/rule_encoder", false)?;
//!
//! // Load .card binary file
//! let program = load_path("core/networks/rule_encoder", true)?;
//! ```
//!
//! ## Testing only
//!
//! ```ignore
//! use ternsig::loader::load_string;
//!
//! let program = load_string(source_code, "test_name")?;
//! ```
//!
//! # File Locations
//!
//! Both use the same path, extension determined by `is_card`:
//! - `load_path("foo/bar", false)` → `{data_dir}/foo/bar.ternsig`
//! - `load_path("foo/bar", true)` → `{data_dir}/foo/bar.card`

use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use crate::vm::{AssembledProgram, deserialize, assemble, serialize, TVMR_MAGIC, TERNSIG_MAGIC};
use datacard_rs::{CardFormat, GenericCard, CardError};

// =============================================================================
// Ternsig Card Format
// =============================================================================

/// Ternsig card format - defines TVMR magic and validation
pub struct TernsigFormat;

impl CardFormat for TernsigFormat {
    /// "TVMR" magic bytes (updated from legacy "TERN")
    const MAGIC: [u8; 4] = TVMR_MAGIC;

    /// Current major version (TVMR v1)
    const VERSION_MAJOR: u8 = 1;

    /// Minimum minor version
    const VERSION_MINOR: u8 = 0;

    /// Validate payload contains valid ternsig/TVMR program
    fn validate_payload(payload: &[u8]) -> datacard_rs::Result<()> {
        // Must have at least magic + version
        if payload.len() < 8 {
            return Err(CardError::InvalidFormat(
                "Ternsig payload too small".to_string()
            ));
        }

        // Accept both TVMR and legacy TERN magic
        let magic = &payload[0..4];
        if magic != &TVMR_MAGIC && magic != &TERNSIG_MAGIC {
            return Err(CardError::InvalidFormat(
                format!("Invalid magic in payload: {:?}", magic)
            ));
        }

        Ok(())
    }

    fn format_name() -> &'static str {
        "TVMR"
    }
}

/// Type alias for ternsig cards
pub type TernsigCard = GenericCard<TernsigFormat>;

// =============================================================================
// Program Loader
// =============================================================================

/// Global data directory for runtime .card files
static DATA_DIR: OnceLock<PathBuf> = OnceLock::new();

/// Program loader configuration and utilities
pub struct ProgramLoader;

impl ProgramLoader {
    /// Set the data directory for runtime .card file loading
    ///
    /// Call this once at application startup:
    /// ```ignore
    /// ProgramLoader::set_data_dir("data/ternsig");
    /// ```
    pub fn set_data_dir<P: AsRef<Path>>(path: P) {
        let _ = DATA_DIR.set(path.as_ref().to_path_buf());
    }

    /// Get the configured data directory, or default to "data/ternsig"
    pub fn data_dir() -> PathBuf {
        DATA_DIR.get().cloned().unwrap_or_else(|| PathBuf::from("data/ternsig"))
    }

    /// Load a compiled .card program from the data directory
    ///
    /// Uses datacard-rs for card validation, then deserializes the ternsig program.
    /// The .card extension is added automatically by datacard - do NOT include it.
    ///
    /// # Arguments
    /// * `name` - Program path without extension, e.g. "core/networks/rule_encoder"
    ///
    /// # Returns
    /// The assembled program ready for execution
    pub fn load_card(name: &str) -> Result<AssembledProgram, String> {
        let path = Self::data_dir().join(name);

        // Load and validate through datacard (.card extension added automatically)
        let card: TernsigCard = TernsigCard::load(&path)
            .map_err(|e| format!("Failed to load card {}: {}", path.display(), e))?;

        // Deserialize the ternsig program from payload
        deserialize(card.payload())
            .map_err(|e| format!("Failed to deserialize ternsig program {}: {}", name, e))
    }

    /// Create a .card file from a ternsig program
    ///
    /// # Arguments
    /// * `name` - Program identifier
    /// * `program` - The assembled program to save
    /// * `path` - Output file path
    pub fn save_card<P: AsRef<Path>>(
        name: &str,
        program: &AssembledProgram,
        path: P,
    ) -> Result<(), String> {
        // Serialize the ternsig program
        let payload = serialize(program)
            .map_err(|e| format!("Failed to serialize program: {}", e))?;

        // Create card with checksum
        let card: TernsigCard = TernsigCard::new_with_checksum(name, payload);

        // Save through datacard
        card.save(path.as_ref())
            .map_err(|e| format!("Failed to save card: {}", e))
    }

    /// Load and assemble a .ternsig source file
    ///
    /// # Arguments
    /// * `source` - The ternsig source code string
    /// * `name` - Program name for error messages
    pub fn assemble_source(source: &str, name: &str) -> Result<AssembledProgram, String> {
        assemble(source)
            .map_err(|e| format!("Failed to assemble {}: {:?}", name, e))
    }
}

/// Load a program from a path
///
/// # Arguments
/// * `path` - Path without extension, e.g. "core/networks/rule_encoder"
/// * `is_card` - If true, loads .card binary. If false, loads .ternsig source.
///
/// # Returns
/// The assembled program
pub fn load_path(path: &str, is_card: bool) -> Result<AssembledProgram, String> {
    let full_path = ProgramLoader::data_dir().join(path);

    if is_card {
        // Load .card binary
        let card: TernsigCard = TernsigCard::load(&full_path)
            .map_err(|e| format!("Failed to load card {}: {}", full_path.display(), e))?;

        deserialize(card.payload())
            .map_err(|e| format!("Failed to deserialize {}: {}", path, e))
    } else {
        // Load .ternsig source
        let source_path = full_path.with_extension("ternsig");
        let source = std::fs::read_to_string(&source_path)
            .map_err(|e| format!("Failed to read {}: {}", source_path.display(), e))?;

        assemble(&source)
            .map_err(|e| format!("Failed to assemble {}: {:?}", path, e))
    }
}

/// Load a program from a source string (testing only)
///
/// # Arguments
/// * `source` - The ternsig source code
/// * `name` - Name for error messages
pub fn load_string(source: &str, name: &str) -> Result<AssembledProgram, String> {
    assemble(source)
        .map_err(|e| format!("Failed to assemble {}: {:?}", name, e))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_data_dir_default() {
        let dir = ProgramLoader::data_dir();
        assert!(dir.to_string_lossy().contains("ternsig"));
    }

    #[test]
    fn test_load_string() {
        let source = r#"
.registers
    H0: i32[1]

.code
    halt
"#;
        let result = load_string(source, "test_program");
        assert!(result.is_ok(), "Should assemble minimal program");
    }

    #[test]
    fn test_load_path_missing_fails() {
        // Should fail when file doesn't exist
        let result = load_path("definitely/does/not/exist", false);
        assert!(result.is_err(), "Should fail for missing .ternsig");

        let result = load_path("definitely/does/not/exist", true);
        assert!(result.is_err(), "Should fail for missing .card");
    }

    #[test]
    fn test_card_roundtrip() {
        use tempfile::tempdir;

        let source = r#"
.registers
    H0: i32[1]

.code
    halt
"#;
        let program = load_string(source, "roundtrip_test").unwrap();

        let dir = tempdir().unwrap();
        let card_path = dir.path().join("test");

        // Save as card
        ProgramLoader::save_card("roundtrip_test", &program, &card_path).unwrap();

        // Load back
        let card: TernsigCard = TernsigCard::load(&card_path).unwrap();
        assert_eq!(card.id(), "roundtrip_test");
        assert!(card.has_checksum());

        let loaded = deserialize(card.payload()).unwrap();
        assert_eq!(loaded.instructions.len(), program.instructions.len());
    }

    #[test]
    fn test_card_extension_rejected() {
        use tempfile::tempdir;

        let source = r#"
.registers
    H0: i32[1]

.code
    halt
"#;
        let program = load_string(source, "test").unwrap();

        let dir = tempdir().unwrap();
        let bad_path = dir.path().join("test.card");

        let result = ProgramLoader::save_card("test", &program, &bad_path);
        assert!(result.is_err(), "Should reject path with .card extension");
    }
}
