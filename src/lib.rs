//! # Ternsig - Signal Foundation
//!
//! The foundational crate that unlocked a new way of thinking about neural learning.
//! Signals (polarity + magnitude) replace floating-point weights entirely.
//!
//! ## Core Components
//!
//! - **Signal**: The fundamental unit (s = p × m, polarity ∈ {-1,0,+1}, magnitude ∈ 0-255)
//! - **Ternsig VM**: Hot-reloadable neural network programs (.ternsig files)
//! - **Adaptive Learning**: Mastery learning - 23ms updates, 90% accuracy threshold
//! - **Thermogram Integration**: Persistent signal storage with temperature lifecycle
//!
//! ## Three-Tier Learning System
//!
//! 1. **Tier 1 (Priors)**: Offline instinct creation via mastery learning
//! 2. **Tier 2 (Coordination)**: SNN + neuromodulators gate learning
//! 3. **Tier 3 (Runtime)**: Continuous 23ms adaptive refinement
//!
//! ## Design Principles
//!
//! - **No floats**: All signals are Signal (2 bytes each)
//! - **CPU-only**: Integer arithmetic, no GPU required
//! - **Persistent**: All signals use thermograms (survive crashes)
//! - **Hot-reloadable**: .ternsig files define network architecture
//!
//! ## Example
//!
//! ```ignore
//! use ternsig::{Signal, Interpreter, assemble};
//!
//! // Load chip definition
//! let program = assemble(include_str!("onset.ternsig"))?;
//! let mut vm = Interpreter::new(&program)?;
//!
//! // Forward pass with signals
//! vm.set_input(&input_signals);
//! vm.execute()?;
//! let output = vm.get_output();
//! ```

// Signal - The fundamental unit (owned by ternsig)
mod ternary;
#[allow(deprecated)]
pub use ternary::{Signal, TernarySignal, Polarity};

// Ternsig VM - Hot-reloadable neural network programs
pub mod vm;
pub use vm::{
    // Core types
    Interpreter, Instruction, Action,
    Register, HotBuffer, ColdBuffer, Dtype, Modifier,
    // Assembly
    assemble, AssembledProgram, AssemblerError,
    // Binary format
    serialize, deserialize, load_from_file, save_to_file,
    // Hot reload
    HotReloadManager, ReloadableInterpreter,
    // Runtime modification
    ArchStats, ModEvent, ShapeSpec, WireSpec, WireType,
};

// Legacy re-exports for backwards compatibility (deprecated)
#[deprecated(note = "Use vm module directly")]
pub mod tensor_isa {
    pub use crate::vm::*;
}

// Mastery learning - pure integer adaptive learning
pub mod learning;
pub use learning::{
    MasteryConfig, MasteryState, mastery_update,
    init_random_structure, init_positive_bias,
    compute_participation_mask, count_active, sparsity,
};

// Thermogram integration
pub mod thermo;
pub use thermo::{
    TensorThermogram, WeightContent, ThermalState,
    TensorContent, ProgramMetaContent, LearningStateContent,
};

// Error types
mod error;
pub use error::TernsigError;

// Program loader - clean API for loading .ternsig/.card files
pub mod loader;
pub use loader::{
    ProgramLoader, load_path, load_string,
    TernsigFormat, TernsigCard,
};

// Validation utilities
pub mod validate;
pub use validate::{
    validate_file, validate_directory,
    ValidationResult, ValidationError, ValidationSummary,
};
