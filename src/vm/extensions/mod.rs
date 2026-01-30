//! TVMR Standard Extensions
//!
//! Each extension provides a domain-specific instruction set. Programs declare
//! which extensions they require, and the registry validates dependencies
//! before execution.
//!
//! ## Standard Extension Allocation
//!
//! | ExtID  | Name          | Instructions | Status     |
//! |--------|---------------|-------------|------------|
//! | 0x0001 | tensor        | 18          | Functional |
//! | 0x0002 | ternary       | 14          | Functional |
//! | 0x0003 | activation    | 5           | Functional |
//! | 0x0004 | learning      | 21          | Functional |
//! | 0x0005 | neuro         | 20          | Functional |
//! | 0x0006 | arch          | 13          | Functional |
//! | 0x0007 | orchestration | 14          | Functional |
//! | 0x0008 | lifecycle     | 13          | Functional |
//! | 0x0009 | ipc           | 8           | Functional |
//! | 0x000A | test          | 8           | Functional |
//! | 0x000B | bank          | 12          | Functional |

pub mod activation;
pub mod arch;
pub mod bank;
pub mod ipc;
pub mod learning;
pub mod lifecycle;
pub mod neuro;
pub mod orchestration;
pub mod tensor;
pub mod ternary;
pub mod test_ext;

pub use activation::ActivationExtension;
pub use arch::ArchExtension;
pub use bank::BankExtension;
pub use ipc::IpcExtension;
pub use learning::LearningExtension;
pub use lifecycle::LifecycleExtension;
pub use neuro::NeuroExtension;
pub use orchestration::OrchestrationExtension;
pub use tensor::TensorExtension;
pub use ternary::TernaryExtension;
pub use test_ext::TestExtension;

use crate::vm::extension::Extension;

/// Resolve extension name to its standard ExtID.
///
/// Returns `None` for unknown extension names.
pub fn resolve_ext_name(name: &str) -> Option<u16> {
    match name {
        "tensor" => Some(0x0001),
        "ternary" => Some(0x0002),
        "activation" => Some(0x0003),
        "learning" => Some(0x0004),
        "neuro" => Some(0x0005),
        "arch" => Some(0x0006),
        "orchestration" => Some(0x0007),
        "lifecycle" => Some(0x0008),
        "ipc" => Some(0x0009),
        "test" => Some(0x000A),
        "bank" => Some(0x000B),
        _ => None,
    }
}

/// Resolve a qualified mnemonic (`ext_name.MNEMONIC`) to `(ext_id, opcode)`.
///
/// Looks up the extension by name and then searches its instruction metadata
/// for a matching mnemonic. Returns `None` if not found.
pub fn resolve_qualified_mnemonic(ext_name: &str, mnemonic: &str) -> Option<(u16, u16)> {
    let ext_id = resolve_ext_name(ext_name)?;
    let upper_mnemonic = mnemonic.to_uppercase();

    // Create the extension to inspect its instructions
    let exts = standard_extensions();
    for ext in &exts {
        if ext.ext_id() == ext_id {
            for meta in ext.instructions() {
                if meta.mnemonic == upper_mnemonic {
                    return Some((ext_id, meta.opcode));
                }
            }
        }
    }
    None
}

/// Create all standard TVMR extensions.
///
/// Returns a Vec of boxed extensions ready for registration.
/// This is the default extension set for a full TVMR runtime.
pub fn standard_extensions() -> Vec<Box<dyn Extension>> {
    vec![
        Box::new(TensorExtension::new()),
        Box::new(TernaryExtension::new()),
        Box::new(ActivationExtension::new()),
        Box::new(LearningExtension::new()),
        Box::new(NeuroExtension::new()),
        Box::new(ArchExtension::new()),
        Box::new(OrchestrationExtension::new()),
        Box::new(LifecycleExtension::new()),
        Box::new(IpcExtension::new()),
        Box::new(TestExtension::new()),
        Box::new(BankExtension::new()),
    ]
}

/// Create only the compute-focused extensions (no host interaction needed).
///
/// Includes extensions that execute entirely within the interpreter without
/// yielding DomainOps: tensor, ternary, activation.
pub fn compute_only_extensions() -> Vec<Box<dyn Extension>> {
    vec![
        Box::new(TensorExtension::new()),
        Box::new(TernaryExtension::new()),
        Box::new(ActivationExtension::new()),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_standard_extensions_count() {
        let exts = standard_extensions();
        assert_eq!(exts.len(), 11);
    }

    #[test]
    fn test_compute_only_extensions_count() {
        let exts = compute_only_extensions();
        assert_eq!(exts.len(), 3);
    }

    #[test]
    fn test_extension_ids_unique() {
        let exts = standard_extensions();
        let ids: Vec<u16> = exts.iter().map(|e| e.ext_id()).collect();
        for (i, id) in ids.iter().enumerate() {
            for (j, other) in ids.iter().enumerate() {
                if i != j {
                    assert_ne!(id, other, "Duplicate extension ID: 0x{:04X}", id);
                }
            }
        }
    }

    #[test]
    fn test_extension_id_allocation() {
        let exts = standard_extensions();
        let mut found = std::collections::HashSet::new();
        for ext in &exts {
            found.insert(ext.ext_id());
        }
        // Verify expected IDs
        assert!(found.contains(&0x0001), "Missing tensor");
        assert!(found.contains(&0x0002), "Missing ternary");
        assert!(found.contains(&0x0003), "Missing activation");
        assert!(found.contains(&0x0004), "Missing learning");
        assert!(found.contains(&0x0005), "Missing neuro");
        assert!(found.contains(&0x0006), "Missing arch");
        assert!(found.contains(&0x0007), "Missing orchestration");
        assert!(found.contains(&0x0008), "Missing lifecycle");
        assert!(found.contains(&0x0009), "Missing ipc");
        assert!(found.contains(&0x000A), "Missing test");
        assert!(found.contains(&0x000B), "Missing bank");
    }

    #[test]
    fn test_total_instruction_count() {
        let exts = standard_extensions();
        let total: usize = exts.iter().map(|e| e.instructions().len()).sum();
        // Should have a healthy number of instructions across all extensions
        assert!(total > 80, "Expected 80+ instructions, got {}", total);
    }
}
