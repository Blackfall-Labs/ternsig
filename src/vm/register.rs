//! Register - Typed register file for Ternsig VM
//!
//! ## Register Banks
//!
//! ```text
//! Hot Bank   (0x00-0x0F): Activations/intermediates (volatile, cleared between runs)
//! Cold Bank  (0x10-0x1F): Weights (Signal, persistent via Thermogram)
//! Param Bank (0x20-0x2F): Scalars (learning_rate, babble_scale, etc.)
//! Shape Bank (0x30-0x3F): Dimension metadata
//! ```
//!
//! ## Encoding
//!
//! Register ID is 1 byte: `[BANK:4 bits][INDEX:4 bits]`
//!
//! - Banks 0-3 map to Hot/Cold/Param/Shape
//! - Index 0-15 within each bank

use std::fmt;

/// Register bank categories
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum RegisterBank {
    /// Hot registers: activations, gradients (volatile)
    /// Cleared between forward passes, used for intermediates
    Hot = 0x00,

    /// Cold registers: weights (persistent in Thermogram)
    /// Signal storage, survives restarts
    Cold = 0x10,

    /// Param registers: scalars (learning rate, etc.)
    /// Configuration values, may be modified during training
    Param = 0x20,

    /// Shape registers: dimension metadata
    /// Stores tensor shapes for runtime validation
    Shape = 0x30,
}

impl RegisterBank {
    /// Get bank from register ID
    pub const fn from_id(id: u8) -> Self {
        match id >> 4 {
            0x0 => Self::Hot,
            0x1 => Self::Cold,
            0x2 => Self::Param,
            0x3 => Self::Shape,
            _ => Self::Hot, // Default to hot for unknown
        }
    }

    /// Get base offset for this bank
    pub const fn base_offset(&self) -> u8 {
        *self as u8
    }

    /// Get human-readable name
    pub const fn name(&self) -> &'static str {
        match self {
            Self::Hot => "H",
            Self::Cold => "C",
            Self::Param => "P",
            Self::Shape => "S",
        }
    }

    /// Get full bank name
    pub const fn full_name(&self) -> &'static str {
        match self {
            Self::Hot => "Hot",
            Self::Cold => "Cold",
            Self::Param => "Param",
            Self::Shape => "Shape",
        }
    }
}

impl fmt::Display for RegisterBank {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// A typed tensor register reference
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Register(pub u8);

impl Register {
    /// Create a hot register (H0-HF)
    pub const fn hot(index: u8) -> Self {
        debug_assert!(index < 16, "Register index must be 0-15");
        Self(RegisterBank::Hot as u8 | (index & 0x0F))
    }

    /// Create a cold register (C0-CF)
    pub const fn cold(index: u8) -> Self {
        debug_assert!(index < 16, "Register index must be 0-15");
        Self(RegisterBank::Cold as u8 | (index & 0x0F))
    }

    /// Create a param register (P0-PF)
    pub const fn param(index: u8) -> Self {
        debug_assert!(index < 16, "Register index must be 0-15");
        Self(RegisterBank::Param as u8 | (index & 0x0F))
    }

    /// Create a shape register (S0-SF)
    pub const fn shape(index: u8) -> Self {
        debug_assert!(index < 16, "Register index must be 0-15");
        Self(RegisterBank::Shape as u8 | (index & 0x0F))
    }

    /// Get the register bank
    pub const fn bank(&self) -> RegisterBank {
        RegisterBank::from_id(self.0)
    }

    /// Get the index within the bank (0-15)
    pub const fn index(&self) -> usize {
        (self.0 & 0x0F) as usize
    }

    /// Get the raw register ID
    pub const fn id(&self) -> u8 {
        self.0
    }

    /// Check if this is a hot register
    pub const fn is_hot(&self) -> bool {
        matches!(self.bank(), RegisterBank::Hot)
    }

    /// Check if this is a cold register
    pub const fn is_cold(&self) -> bool {
        matches!(self.bank(), RegisterBank::Cold)
    }

    /// Check if this is a param register
    pub const fn is_param(&self) -> bool {
        matches!(self.bank(), RegisterBank::Param)
    }

    /// Check if this is a shape register
    pub const fn is_shape(&self) -> bool {
        matches!(self.bank(), RegisterBank::Shape)
    }

    /// Null register (no operation)
    pub const NULL: Self = Self(0xFF);

    /// Check if this is the null register
    pub const fn is_null(&self) -> bool {
        self.0 == 0xFF
    }

    /// Parse from string like "H0", "C5", "P3", "S2"
    pub fn parse(s: &str) -> Option<Self> {
        let s = s.trim().to_uppercase();
        if s.len() < 2 {
            return None;
        }

        let bank = match s.chars().next()? {
            'H' => RegisterBank::Hot,
            'C' => RegisterBank::Cold,
            'P' => RegisterBank::Param,
            'S' => RegisterBank::Shape,
            _ => return None,
        };

        let index_str = &s[1..];
        let index: u8 = if index_str.starts_with("0X") || index_str.starts_with("0x") {
            u8::from_str_radix(&index_str[2..], 16).ok()?
        } else {
            index_str.parse().ok()?
        };

        if index > 15 {
            return None;
        }

        Some(Self(bank.base_offset() | index))
    }
}

impl fmt::Display for Register {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_null() {
            write!(f, "_")
        } else {
            write!(f, "{}{}", self.bank(), self.index())
        }
    }
}

impl Default for Register {
    fn default() -> Self {
        Self::NULL
    }
}

/// Data type for tensor registers
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Dtype {
    /// 32-bit float (standard)
    F32 = 0,
    /// 32-bit signed integer (quantized activations)
    I32 = 1,
    /// Signal (polarity + magnitude)
    Ternary = 2,
    /// Packed ternary (2-bit per weight)
    PackedTernary = 3,
    /// 16-bit float (half precision)
    F16 = 4,
    /// 16-bit signed integer
    I16 = 5,
    /// 8-bit signed integer
    I8 = 6,
    /// 64-bit signed integer (accumulators)
    I64 = 7,
}

impl Dtype {
    /// Size in bytes per element
    pub const fn element_size(&self) -> usize {
        match self {
            Self::F32 | Self::I32 => 4,
            Self::F16 | Self::I16 => 2,
            Self::I8 => 1,
            Self::I64 => 8,
            Self::Ternary => 2,       // polarity (1) + magnitude (1)
            Self::PackedTernary => 1, // 4 weights per byte
        }
    }

    /// Get from u8
    pub const fn from_u8(val: u8) -> Self {
        match val {
            0 => Self::F32,
            1 => Self::I32,
            2 => Self::Ternary,
            3 => Self::PackedTernary,
            4 => Self::F16,
            5 => Self::I16,
            6 => Self::I8,
            7 => Self::I64,
            _ => Self::F32,
        }
    }

    /// Get name for assembly
    pub const fn name(&self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::I32 => "i32",
            Self::Ternary => "ternary",
            Self::PackedTernary => "packed",
            Self::F16 => "f16",
            Self::I16 => "i16",
            Self::I8 => "i8",
            Self::I64 => "i64",
        }
    }

    /// Parse from string
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "f32" | "float" | "float32" => Some(Self::F32),
            "i32" | "int" | "int32" => Some(Self::I32),
            "ternary" => Some(Self::Ternary),
            "packed" | "packedternary" => Some(Self::PackedTernary),
            "f16" | "half" | "float16" => Some(Self::F16),
            "i16" | "int16" => Some(Self::I16),
            "i8" | "int8" => Some(Self::I8),
            "i64" | "int64" => Some(Self::I64),
            _ => None,
        }
    }
}

impl fmt::Display for Dtype {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

impl Default for Dtype {
    fn default() -> Self {
        Self::F32
    }
}

/// Metadata for a tensor register
#[derive(Debug, Clone)]
pub struct RegisterMeta {
    /// Register ID
    pub id: Register,
    /// Dimensions [batch, ..., features]
    pub shape: Vec<usize>,
    /// Data type
    pub dtype: Dtype,
    /// Whether register is currently allocated
    pub allocated: bool,
    /// Thermogram key for cold registers (for persistence)
    pub thermogram_key: Option<String>,
    /// Whether this register is frozen (non-trainable)
    pub frozen: bool,
}

impl RegisterMeta {
    /// Create new unallocated register metadata
    pub fn new(id: Register) -> Self {
        Self {
            id,
            shape: Vec::new(),
            dtype: Dtype::default(),
            allocated: false,
            thermogram_key: None,
            frozen: false,
        }
    }

    /// Create with shape and dtype
    pub fn with_shape(id: Register, shape: Vec<usize>, dtype: Dtype) -> Self {
        Self {
            id,
            shape,
            dtype,
            allocated: true,
            thermogram_key: None,
            frozen: false,
        }
    }

    /// Total number of elements
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// Size in bytes
    pub fn size_bytes(&self) -> usize {
        self.numel() * self.dtype.element_size()
    }

    /// Number of dimensions
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Set thermogram key (for cold registers)
    pub fn with_thermogram_key(mut self, key: impl Into<String>) -> Self {
        self.thermogram_key = Some(key.into());
        self
    }

    /// Set frozen state
    pub fn with_frozen(mut self, frozen: bool) -> Self {
        self.frozen = frozen;
        self
    }
}

impl fmt::Display for RegisterMeta {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let shape_str = self
            .shape
            .iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join(", ");

        write!(f, "{}: {}[{}]", self.id, self.dtype, shape_str)?;

        if let Some(key) = &self.thermogram_key {
            write!(f, " key=\"{}\"", key)?;
        }

        if self.frozen {
            write!(f, " (frozen)")?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_register_banks() {
        let h0 = Register::hot(0);
        assert_eq!(h0.bank(), RegisterBank::Hot);
        assert_eq!(h0.index(), 0);
        assert!(h0.is_hot());

        let c5 = Register::cold(5);
        assert_eq!(c5.bank(), RegisterBank::Cold);
        assert_eq!(c5.index(), 5);
        assert!(c5.is_cold());

        let p15 = Register::param(15);
        assert_eq!(p15.bank(), RegisterBank::Param);
        assert_eq!(p15.index(), 15);
        assert!(p15.is_param());
    }

    #[test]
    fn test_register_display() {
        assert_eq!(Register::hot(0).to_string(), "H0");
        assert_eq!(Register::cold(10).to_string(), "C10");
        assert_eq!(Register::param(3).to_string(), "P3");
        assert_eq!(Register::NULL.to_string(), "_");
    }

    #[test]
    fn test_register_parse() {
        assert_eq!(Register::parse("H0"), Some(Register::hot(0)));
        assert_eq!(Register::parse("c5"), Some(Register::cold(5)));
        assert_eq!(Register::parse("P15"), Some(Register::param(15)));
        assert_eq!(Register::parse("S0"), Some(Register::shape(0)));
        assert_eq!(Register::parse("X0"), None);
        assert_eq!(Register::parse("H16"), None); // Out of range
    }

    #[test]
    fn test_dtype() {
        assert_eq!(Dtype::F32.element_size(), 4);
        assert_eq!(Dtype::I32.element_size(), 4);
        assert_eq!(Dtype::Ternary.element_size(), 2);
        assert_eq!(Dtype::PackedTernary.element_size(), 1);
    }

    #[test]
    fn test_register_meta() {
        let meta = RegisterMeta::with_shape(
            Register::cold(0),
            vec![32, 12],
            Dtype::Ternary,
        )
        .with_thermogram_key("chip.audio.w1");

        assert_eq!(meta.numel(), 384);
        assert_eq!(meta.size_bytes(), 768); // 384 * 2 bytes per ternary
        assert_eq!(meta.thermogram_key, Some("chip.audio.w1".to_string()));
    }
}
