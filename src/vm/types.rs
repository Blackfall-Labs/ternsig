//! TVMR Type System — Cross-Language Portable Type IDs
//!
//! Every type in the TVMR has a `TypeId` (u16). Primitive types (0x00xx) are
//! universal across all TVMR engine implementations. Domain types (0x01xx+)
//! are registered by extensions.
//!
//! ## Type Ranges
//!
//! ```text
//! 0x0000-0x00FF  Primitive types (universal, cross-language)
//! 0x0100-0x01FF  Standard domain types (Signal, Chemical, etc.)
//! 0x0200-0x02FF  Composite types (future: Array<T>, Struct)
//! 0x0300+        User-defined types
//! ```

use std::fmt;

/// A cross-language type identifier.
///
/// TypeId is a u16 that uniquely identifies a data type in the TVMR.
/// Primitive types are defined here; domain types are registered by extensions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TypeId(pub u16);

impl TypeId {
    // =========================================================================
    // Primitive Types (0x00xx) — Universal across all TVMR engines
    // =========================================================================

    /// No data
    pub const VOID: Self = Self(0x0000);
    /// Boolean (1 byte)
    pub const BOOL: Self = Self(0x0001);
    /// Unsigned 8-bit integer
    pub const U8: Self = Self(0x0002);
    /// Signed 8-bit integer
    pub const I8: Self = Self(0x0003);
    /// Unsigned 16-bit integer
    pub const U16: Self = Self(0x0004);
    /// Signed 16-bit integer
    pub const I16: Self = Self(0x0005);
    /// Unsigned 32-bit integer
    pub const U32: Self = Self(0x0006);
    /// Signed 32-bit integer
    pub const I32: Self = Self(0x0007);
    /// Unsigned 64-bit integer
    pub const U64: Self = Self(0x0008);
    /// Signed 64-bit integer
    pub const I64: Self = Self(0x0009);
    /// Half-precision float (2 bytes)
    pub const F16: Self = Self(0x000A);
    /// Single-precision float (4 bytes)
    pub const F32: Self = Self(0x000B);
    /// Double-precision float (8 bytes)
    pub const F64: Self = Self(0x000C);

    // =========================================================================
    // Standard Domain Types (0x01xx) — Registered by standard extensions
    // =========================================================================

    /// Ternary signal: `#[repr(C)] { polarity: i8, magnitude: u8 }` (2 bytes)
    pub const SIGNAL: Self = Self(0x0100);
    /// 4 packed ternary values per byte (2 bits each)
    pub const PACKED_SIGNAL: Self = Self(0x0101);
    /// Neuromodulator quad: `{ da: u8, serotonin: u8, ne: u8, gaba: u8 }` (4 bytes)
    pub const CHEMICAL: Self = Self(0x0102);

    /// Size in bytes for a single element of this type.
    /// Returns `None` for variable-size or unknown types.
    pub const fn size_bytes(&self) -> Option<usize> {
        match self.0 {
            0x0000 => Some(0),  // void
            0x0001 => Some(1),  // bool
            0x0002 => Some(1),  // u8
            0x0003 => Some(1),  // i8
            0x0004 => Some(2),  // u16
            0x0005 => Some(2),  // i16
            0x0006 => Some(4),  // u32
            0x0007 => Some(4),  // i32
            0x0008 => Some(8),  // u64
            0x0009 => Some(8),  // i64
            0x000A => Some(2),  // f16
            0x000B => Some(4),  // f32
            0x000C => Some(8),  // f64
            0x0100 => Some(2),  // signal
            0x0101 => Some(1),  // packed_signal (4 values per byte)
            0x0102 => Some(4),  // chemical
            _ => None,
        }
    }

    /// Whether this is a primitive type (universal across all TVMR engines).
    pub const fn is_primitive(&self) -> bool {
        self.0 < 0x0100
    }

    /// Whether this is a standard domain type.
    pub const fn is_domain(&self) -> bool {
        self.0 >= 0x0100 && self.0 < 0x0200
    }

    /// Whether this is a user-defined type.
    pub const fn is_user(&self) -> bool {
        self.0 >= 0x0300
    }

    /// Get the raw u16 value.
    pub const fn as_u16(&self) -> u16 {
        self.0
    }

    /// Create from raw u16.
    pub const fn from_u16(v: u16) -> Self {
        Self(v)
    }

    /// Human-readable name for known types.
    pub const fn name(&self) -> &'static str {
        match self.0 {
            0x0000 => "void",
            0x0001 => "bool",
            0x0002 => "u8",
            0x0003 => "i8",
            0x0004 => "u16",
            0x0005 => "i16",
            0x0006 => "u32",
            0x0007 => "i32",
            0x0008 => "u64",
            0x0009 => "i64",
            0x000A => "f16",
            0x000B => "f32",
            0x000C => "f64",
            0x0100 => "signal",
            0x0101 => "packed_signal",
            0x0102 => "chemical",
            _ => "unknown",
        }
    }

    /// Parse from string (assembly format).
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "void" => Some(Self::VOID),
            "bool" => Some(Self::BOOL),
            "u8" => Some(Self::U8),
            "i8" => Some(Self::I8),
            "u16" => Some(Self::U16),
            "i16" => Some(Self::I16),
            "u32" => Some(Self::U32),
            "i32" | "int" | "int32" => Some(Self::I32),
            "u64" => Some(Self::U64),
            "i64" | "int64" => Some(Self::I64),
            "f16" | "half" | "float16" => Some(Self::F16),
            "f32" | "float" | "float32" => Some(Self::F32),
            "f64" | "double" | "float64" => Some(Self::F64),
            "signal" | "ternary" => Some(Self::SIGNAL),
            "packed_signal" | "packed" | "packedternary" => Some(Self::PACKED_SIGNAL),
            "chemical" => Some(Self::CHEMICAL),
            _ => None,
        }
    }
}

impl fmt::Display for TypeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = self.name();
        if name == "unknown" {
            write!(f, "type(0x{:04X})", self.0)
        } else {
            write!(f, "{}", name)
        }
    }
}

impl Default for TypeId {
    fn default() -> Self {
        Self::I32
    }
}

// =========================================================================
// Legacy compatibility: convert between old Dtype and new TypeId
// =========================================================================

impl TypeId {
    /// Convert from legacy Dtype discriminant (u8).
    pub const fn from_legacy_dtype(dtype_u8: u8) -> Self {
        match dtype_u8 {
            0 => Self::F32,
            1 => Self::I32,
            2 => Self::SIGNAL,
            3 => Self::PACKED_SIGNAL,
            4 => Self::F16,
            5 => Self::I16,
            6 => Self::I8,
            7 => Self::I64,
            _ => Self::F32,
        }
    }

    /// Convert to legacy Dtype discriminant (u8) for backward compat.
    /// Returns None if the type has no legacy equivalent.
    pub const fn to_legacy_dtype(&self) -> Option<u8> {
        match self.0 {
            0x000B => Some(0), // f32
            0x0007 => Some(1), // i32
            0x0100 => Some(2), // signal (was ternary)
            0x0101 => Some(3), // packed_signal
            0x000A => Some(4), // f16
            0x0005 => Some(5), // i16
            0x0003 => Some(6), // i8
            0x0009 => Some(7), // i64
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_primitive_sizes() {
        assert_eq!(TypeId::VOID.size_bytes(), Some(0));
        assert_eq!(TypeId::BOOL.size_bytes(), Some(1));
        assert_eq!(TypeId::U8.size_bytes(), Some(1));
        assert_eq!(TypeId::I32.size_bytes(), Some(4));
        assert_eq!(TypeId::F64.size_bytes(), Some(8));
        assert_eq!(TypeId::SIGNAL.size_bytes(), Some(2));
        assert_eq!(TypeId::CHEMICAL.size_bytes(), Some(4));
    }

    #[test]
    fn test_type_categories() {
        assert!(TypeId::I32.is_primitive());
        assert!(!TypeId::I32.is_domain());
        assert!(TypeId::SIGNAL.is_domain());
        assert!(!TypeId::SIGNAL.is_primitive());
        assert!(TypeId::from_u16(0x0300).is_user());
    }

    #[test]
    fn test_parse() {
        assert_eq!(TypeId::parse("i32"), Some(TypeId::I32));
        assert_eq!(TypeId::parse("signal"), Some(TypeId::SIGNAL));
        assert_eq!(TypeId::parse("ternary"), Some(TypeId::SIGNAL));
        assert_eq!(TypeId::parse("packed"), Some(TypeId::PACKED_SIGNAL));
        assert_eq!(TypeId::parse("f32"), Some(TypeId::F32));
        assert_eq!(TypeId::parse("nonexistent"), None);
    }

    #[test]
    fn test_legacy_roundtrip() {
        for dtype_u8 in 0u8..8 {
            let type_id = TypeId::from_legacy_dtype(dtype_u8);
            assert_eq!(type_id.to_legacy_dtype(), Some(dtype_u8));
        }
    }

    #[test]
    fn test_display() {
        assert_eq!(format!("{}", TypeId::I32), "i32");
        assert_eq!(format!("{}", TypeId::SIGNAL), "signal");
        assert_eq!(format!("{}", TypeId::from_u16(0xFFFF)), "type(0xFFFF)");
    }
}
