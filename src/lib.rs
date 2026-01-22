//! # Ternsig - TernarySignal Foundation
//!
//! The foundational crate that unlocked a new way of thinking about neural learning.
//! TernarySignals (polarity + magnitude) replace floating-point weights entirely.
//!
//! ## Core Components
//!
//! - **TernarySignal**: The fundamental unit (polarity ∈ {-1,0,+1}, magnitude ∈ 0-255)
//! - **TensorISA**: Hot-reloadable neural network instruction set (.tisa.asm files)
//! - **Adaptive Learning**: Mastery learning - 23ms updates, 90% accuracy threshold
//! - **Thermogram Integration**: Persistent weight storage with temperature lifecycle
//!
//! ## Three-Tier Learning System
//!
//! 1. **Tier 1 (Priors)**: Offline instinct creation via mastery learning
//! 2. **Tier 2 (Coordination)**: SNN + neuromodulators gate learning
//! 3. **Tier 3 (Runtime)**: Continuous 23ms adaptive refinement
//!
//! ## Design Principles
//!
//! - **No floats**: All weights are TernarySignal (2 bytes each)
//! - **CPU-only**: Integer arithmetic, no GPU required
//! - **Persistent**: All weights use thermograms (survive crashes)
//! - **Hot-reloadable**: .tisa.asm files define network architecture
//!
//! ## Example
//!
//! ```ignore
//! use ternsig::{TernarySignal, TensorInterpreter, assemble};
//!
//! // Load chip definition
//! let program = assemble(include_str!("onset.tisa.asm"))?;
//! let mut interpreter = TensorInterpreter::new(&program)?;
//!
//! // Forward pass with ternary weights
//! interpreter.set_input(&input_signals);
//! interpreter.execute()?;
//! let output = interpreter.get_output();
//! ```

// TernarySignal - The fundamental unit (owned by ternsig)
mod ternary;
pub use ternary::TernarySignal;

// TensorISA - Hot-reloadable neural network definitions
pub mod tensor_isa;
pub use tensor_isa::{
    // Core types
    TensorInterpreter, TensorInstruction, TensorAction,
    TensorRegister, HotBuffer, ColdBuffer, TensorDtype, TensorModifier,
    // Assembly
    assemble, AssembledProgram, AssemblerError,
    // Binary format
    serialize_tisa, deserialize_tisa, load_tisa_file, save_tisa_file,
    // Hot reload
    HotReloadManager, ReloadableInterpreter,
    // Runtime modification
    ArchStats, ModEvent, ShapeSpec, WireSpec, WireType,
};

// Adaptive learning utilities
pub mod learning;
pub use learning::{
    PolarityState, SurpriseOptimizer, SurpriseOptimizerConfig,
    FloatingTernaryLayer, OptimizerStats,
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
