//! TVMR Extension Registry — Registration, Dispatch, and Validation
//!
//! The registry manages all loaded extensions, resolves mnemonics for the
//! assembler, detects duplicate instruction shapes, and dispatches execution
//! to the correct extension at runtime.

use super::extension::{Extension, ExecutionContext, InstructionMeta, OperandPattern, StepResult};
use std::collections::HashMap;
use std::fmt;

/// Warning produced during extension registration.
#[derive(Debug, Clone)]
pub enum RegistrationWarning {
    /// Two extensions register the same mnemonic with the same operand pattern.
    DuplicateShape {
        mnemonic: String,
        new_ext: u16,
        existing_ext: u16,
    },
    /// Two extensions register the same mnemonic with different operand patterns.
    MnemonicConflict {
        mnemonic: String,
        new_ext: u16,
        existing_ext: u16,
    },
}

impl fmt::Display for RegistrationWarning {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DuplicateShape { mnemonic, new_ext, existing_ext } => {
                write!(
                    f,
                    "WARN: mnemonic '{}' in ext 0x{:04X} has same operand shape as ext 0x{:04X}",
                    mnemonic, new_ext, existing_ext,
                )
            }
            Self::MnemonicConflict { mnemonic, new_ext, existing_ext } => {
                write!(
                    f,
                    "WARN: mnemonic '{}' in ext 0x{:04X} shadows ext 0x{:04X}",
                    mnemonic, new_ext, existing_ext,
                )
            }
        }
    }
}

/// Error produced during extension registration.
#[derive(Debug, Clone)]
pub enum RegistrationError {
    /// An extension with this ID is already registered.
    DuplicateExtId(u16),
    /// Extension ID 0x0000 is reserved for the core ISA.
    ReservedExtId,
}

impl fmt::Display for RegistrationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DuplicateExtId(id) => write!(f, "Extension 0x{:04X} already registered", id),
            Self::ReservedExtId => write!(f, "Extension ID 0x0000 is reserved for core ISA"),
        }
    }
}

impl std::error::Error for RegistrationError {}

/// Mnemonic resolution entry — maps a mnemonic to its extension + opcode.
#[derive(Debug, Clone)]
struct MnemonicEntry {
    ext_id: u16,
    opcode: u16,
    pattern: OperandPattern,
}

/// The TVMR Extension Registry.
///
/// Manages extension lifecycle: registration, mnemonic resolution,
/// duplicate detection, and runtime dispatch.
pub struct ExtensionRegistry {
    /// Registered extensions, indexed by position.
    extensions: Vec<Box<dyn Extension>>,
    /// ExtID → index into `extensions`.
    id_map: HashMap<u16, usize>,
    /// Lowercase mnemonic → resolution entry.
    /// If multiple extensions register the same mnemonic, the last one wins.
    mnemonic_map: HashMap<String, MnemonicEntry>,
}

impl ExtensionRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self {
            extensions: Vec::new(),
            id_map: HashMap::new(),
            mnemonic_map: HashMap::new(),
        }
    }

    /// Register an extension. Returns warnings about mnemonic conflicts.
    ///
    /// # Errors
    /// Returns `RegistrationError` if the extension ID is already taken or reserved.
    pub fn register(
        &mut self,
        ext: Box<dyn Extension>,
    ) -> Result<Vec<RegistrationWarning>, RegistrationError> {
        let ext_id = ext.ext_id();

        // ExtID 0x0000 is core ISA — cannot be registered as extension
        if ext_id == 0x0000 {
            return Err(RegistrationError::ReservedExtId);
        }

        // Check for duplicate ExtID
        if self.id_map.contains_key(&ext_id) {
            return Err(RegistrationError::DuplicateExtId(ext_id));
        }

        let mut warnings = Vec::new();

        // Check each instruction for mnemonic conflicts
        for meta in ext.instructions() {
            let mnemonic_lower = meta.mnemonic.to_lowercase();

            if let Some(existing) = self.mnemonic_map.get(&mnemonic_lower) {
                if existing.pattern == meta.operand_pattern {
                    warnings.push(RegistrationWarning::DuplicateShape {
                        mnemonic: mnemonic_lower.clone(),
                        new_ext: ext_id,
                        existing_ext: existing.ext_id,
                    });
                } else {
                    warnings.push(RegistrationWarning::MnemonicConflict {
                        mnemonic: mnemonic_lower.clone(),
                        new_ext: ext_id,
                        existing_ext: existing.ext_id,
                    });
                }
            }

            // Last-registered wins for unqualified mnemonics
            self.mnemonic_map.insert(mnemonic_lower, MnemonicEntry {
                ext_id,
                opcode: meta.opcode,
                pattern: meta.operand_pattern.clone(),
            });
        }

        let idx = self.extensions.len();
        self.id_map.insert(ext_id, idx);
        self.extensions.push(ext);

        Ok(warnings)
    }

    /// Dispatch an instruction to the correct extension for execution.
    ///
    /// The caller is responsible for handling core ISA (ext_id == 0x0000)
    /// before calling this method.
    pub fn dispatch(
        &self,
        ext_id: u16,
        opcode: u16,
        operands: [u8; 4],
        ctx: &mut ExecutionContext,
    ) -> StepResult {
        match self.id_map.get(&ext_id) {
            Some(&idx) => self.extensions[idx].execute(opcode, operands, ctx),
            None => StepResult::Error(format!("Unknown extension: 0x{:04X}", ext_id)),
        }
    }

    /// Resolve an unqualified mnemonic to (ext_id, opcode).
    pub fn resolve_mnemonic(&self, mnemonic: &str) -> Option<(u16, u16)> {
        self.mnemonic_map
            .get(&mnemonic.to_lowercase())
            .map(|entry| (entry.ext_id, entry.opcode))
    }

    /// Resolve a qualified mnemonic like "tensor.matmul" to (ext_id, opcode).
    pub fn resolve_qualified(&self, qualified: &str) -> Option<(u16, u16)> {
        if let Some(dot_pos) = qualified.find('.') {
            let ext_name = &qualified[..dot_pos];
            let mnemonic = &qualified[dot_pos + 1..];

            // Find extension by name
            for ext in &self.extensions {
                if ext.name().eq_ignore_ascii_case(ext_name) {
                    let mnemonic_lower = mnemonic.to_lowercase();
                    for meta in ext.instructions() {
                        if meta.mnemonic.to_lowercase() == mnemonic_lower {
                            return Some((ext.ext_id(), meta.opcode));
                        }
                    }
                    return None;
                }
            }
            None
        } else {
            self.resolve_mnemonic(qualified)
        }
    }

    /// Check if an extension is registered.
    pub fn has_extension(&self, ext_id: u16) -> bool {
        self.id_map.contains_key(&ext_id)
    }

    /// Get an extension by ID.
    pub fn get_extension(&self, ext_id: u16) -> Option<&dyn Extension> {
        self.id_map.get(&ext_id).map(|&idx| self.extensions[idx].as_ref())
    }

    /// Get all registered extension IDs.
    pub fn extension_ids(&self) -> Vec<u16> {
        let mut ids: Vec<u16> = self.id_map.keys().copied().collect();
        ids.sort();
        ids
    }

    /// Get all registered mnemonics.
    pub fn mnemonics(&self) -> Vec<(&str, u16, u16)> {
        self.mnemonic_map
            .iter()
            .map(|(m, e)| (m.as_str(), e.ext_id, e.opcode))
            .collect()
    }

    /// Get instruction metadata for a specific extension + opcode.
    pub fn instruction_meta(&self, ext_id: u16, opcode: u16) -> Option<&InstructionMeta> {
        let idx = *self.id_map.get(&ext_id)?;
        self.extensions[idx]
            .instructions()
            .iter()
            .find(|m| m.opcode == opcode)
    }

    /// Total number of registered extensions.
    pub fn len(&self) -> usize {
        self.extensions.len()
    }

    /// Whether the registry is empty (no extensions registered).
    pub fn is_empty(&self) -> bool {
        self.extensions.is_empty()
    }
}

impl Default for ExtensionRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for ExtensionRegistry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ExtensionRegistry({} extensions, {} mnemonics)",
            self.extensions.len(),
            self.mnemonic_map.len(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal test extension for testing the registry.
    struct TestExtension {
        id: u16,
        name: String,
    }

    impl Extension for TestExtension {
        fn ext_id(&self) -> u16 { self.id }
        fn name(&self) -> &str { &self.name }
        fn version(&self) -> (u16, u16, u16) { (1, 0, 0) }
        fn instructions(&self) -> &[InstructionMeta] {
            &[
                InstructionMeta {
                    opcode: 0x0000,
                    mnemonic: "TEST_OP",
                    operand_pattern: OperandPattern::RegReg,
                    description: "Test operation",
                },
            ]
        }
        fn execute(&self, _opcode: u16, _operands: [u8; 4], _ctx: &mut ExecutionContext) -> StepResult {
            StepResult::Continue
        }
    }

    #[test]
    fn test_register_extension() {
        let mut registry = ExtensionRegistry::new();
        let ext = Box::new(TestExtension { id: 0x0001, name: "test".into() });
        let warnings = registry.register(ext).unwrap();
        assert!(warnings.is_empty());
        assert!(registry.has_extension(0x0001));
        assert_eq!(registry.len(), 1);
    }

    #[test]
    fn test_reject_duplicate_ext_id() {
        let mut registry = ExtensionRegistry::new();
        let ext1 = Box::new(TestExtension { id: 0x0001, name: "test1".into() });
        let ext2 = Box::new(TestExtension { id: 0x0001, name: "test2".into() });
        registry.register(ext1).unwrap();
        assert!(registry.register(ext2).is_err());
    }

    #[test]
    fn test_reject_core_ext_id() {
        let mut registry = ExtensionRegistry::new();
        let ext = Box::new(TestExtension { id: 0x0000, name: "bad".into() });
        assert!(registry.register(ext).is_err());
    }

    #[test]
    fn test_mnemonic_resolution() {
        let mut registry = ExtensionRegistry::new();
        let ext = Box::new(TestExtension { id: 0x0001, name: "test".into() });
        registry.register(ext).unwrap();

        assert_eq!(registry.resolve_mnemonic("test_op"), Some((0x0001, 0x0000)));
        assert_eq!(registry.resolve_mnemonic("TEST_OP"), Some((0x0001, 0x0000)));
        assert_eq!(registry.resolve_mnemonic("nonexistent"), None);
    }

    #[test]
    fn test_mnemonic_conflict_detection() {
        let mut registry = ExtensionRegistry::new();
        let ext1 = Box::new(TestExtension { id: 0x0001, name: "ext1".into() });
        let ext2 = Box::new(TestExtension { id: 0x0002, name: "ext2".into() });

        registry.register(ext1).unwrap();
        let warnings = registry.register(ext2).unwrap();

        // Should warn about duplicate mnemonic with same pattern
        assert_eq!(warnings.len(), 1);
        assert!(matches!(warnings[0], RegistrationWarning::DuplicateShape { .. }));

        // Last-registered wins
        assert_eq!(registry.resolve_mnemonic("test_op"), Some((0x0002, 0x0000)));
    }

    #[test]
    fn test_qualified_resolution() {
        let mut registry = ExtensionRegistry::new();
        let ext = Box::new(TestExtension { id: 0x0001, name: "test".into() });
        registry.register(ext).unwrap();

        assert_eq!(registry.resolve_qualified("test.test_op"), Some((0x0001, 0x0000)));
        assert_eq!(registry.resolve_qualified("test.nonexistent"), None);
        assert_eq!(registry.resolve_qualified("bad_ext.test_op"), None);
    }
}
