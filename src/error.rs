//! Error types for ternsig

use thiserror::Error;

/// Ternsig error type
#[derive(Debug, Error)]
pub enum TernsigError {
    /// TensorISA assembly error
    #[error("Assembly error: {0}")]
    Assembly(String),

    /// TensorISA execution error
    #[error("Execution error: {0}")]
    Execution(String),

    /// Thermogram error
    #[error("Thermogram error: {0}")]
    Thermogram(String),

    /// Shape mismatch
    #[error("Shape mismatch: expected {expected:?}, got {actual:?}")]
    ShapeMismatch {
        expected: Vec<usize>,
        actual: Vec<usize>,
    },

    /// Invalid register
    #[error("Invalid register: {0}")]
    InvalidRegister(u8),

    /// IO error
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// Binary format error
    #[error("Binary format error: {0}")]
    BinaryFormat(String),

    /// Learning error
    #[error("Learning error: {0}")]
    Learning(String),
}

pub type Result<T> = std::result::Result<T, TernsigError>;
