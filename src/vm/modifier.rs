//! Modifier - 3-byte operation-specific data encoding
//!
//! The modifier field has different interpretations per operation category:
//!
//! ## Architecture Ops (0x20xx)
//! ```text
//! [INPUT_DIM:12 bits][OUTPUT_DIM:12 bits]
//! Max dimensions: 4096 x 4096
//! ```
//!
//! ## Forward Ops (0x30xx)
//! ```text
//! [SCALE:16 bits][FLAGS:8 bits]
//! Scale is fixed-point: value = raw / 65536.0
//! ```
//!
//! ## Learning Ops (0x50xx)
//! ```text
//! [VALUE:16 bits][FLAGS:8 bits]
//! FLAGS: [CLAMP:1][RELATIVE:1][LOG:1][NEGATE:1][RESERVED:4]
//! ```
//!
//! ## Control Flow (0x60xx)
//! ```text
//! [COUNT:16 bits][THRESHOLD:8 bits]
//! Threshold is fixed-point: value = raw / 256.0
//! ```

use std::fmt;

/// Modifier flags (8 bits)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ModifierFlags(pub u8);

impl ModifierFlags {
    /// Clamp result to valid range
    pub const CLAMP: u8 = 0b0000_0001;
    /// Value is relative (multiply) vs absolute (set)
    pub const RELATIVE: u8 = 0b0000_0010;
    /// Apply logarithm to value
    pub const LOG: u8 = 0b0000_0100;
    /// Negate value
    pub const NEGATE: u8 = 0b0000_1000;
    /// In-place operation (target == source allowed)
    pub const INPLACE: u8 = 0b0001_0000;
    /// Broadcast scalar to all elements
    pub const BROADCAST: u8 = 0b0010_0000;

    pub fn clamp(self) -> bool {
        (self.0 & Self::CLAMP) != 0
    }

    pub fn relative(self) -> bool {
        (self.0 & Self::RELATIVE) != 0
    }

    pub fn log(self) -> bool {
        (self.0 & Self::LOG) != 0
    }

    pub fn negate(self) -> bool {
        (self.0 & Self::NEGATE) != 0
    }

    pub fn inplace(self) -> bool {
        (self.0 & Self::INPLACE) != 0
    }

    pub fn broadcast(self) -> bool {
        (self.0 & Self::BROADCAST) != 0
    }

    pub fn with_clamp(mut self) -> Self {
        self.0 |= Self::CLAMP;
        self
    }

    pub fn with_relative(mut self) -> Self {
        self.0 |= Self::RELATIVE;
        self
    }

    pub fn with_negate(mut self) -> Self {
        self.0 |= Self::NEGATE;
        self
    }

    pub fn with_inplace(mut self) -> Self {
        self.0 |= Self::INPLACE;
        self
    }

    pub fn with_broadcast(mut self) -> Self {
        self.0 |= Self::BROADCAST;
        self
    }
}

/// 3-byte modifier with multiple interpretations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Modifier {
    pub bytes: [u8; 3],
}

impl Modifier {
    /// Create from raw bytes
    pub const fn from_bytes(bytes: [u8; 3]) -> Self {
        Self { bytes }
    }

    /// Create empty modifier
    pub const fn empty() -> Self {
        Self { bytes: [0, 0, 0] }
    }

    /// Create from shape (input_dim, output_dim)
    /// Each dimension is 12 bits (max 4095)
    pub fn from_shape(input_dim: usize, output_dim: usize) -> Self {
        let combined = ((input_dim & 0xFFF) << 12) | (output_dim & 0xFFF);
        Self {
            bytes: [(combined >> 16) as u8, (combined >> 8) as u8, combined as u8],
        }
    }

    /// Extract shape (input_dim, output_dim)
    pub fn to_shape(&self) -> (usize, usize) {
        let combined = u32::from_be_bytes([0, self.bytes[0], self.bytes[1], self.bytes[2]]);
        let input = ((combined >> 12) & 0xFFF) as usize;
        let output = (combined & 0xFFF) as usize;
        (input, output)
    }

    /// Create from count and threshold (for control flow)
    pub fn from_count_threshold(count: u16, threshold: u8) -> Self {
        let count_bytes = count.to_be_bytes();
        Self {
            bytes: [count_bytes[0], count_bytes[1], threshold],
        }
    }

    /// Extract count (for LOOP, SKIP)
    pub fn count(&self) -> u16 {
        u16::from_be_bytes([self.bytes[0], self.bytes[1]])
    }

    /// Extract threshold (for IF_* ops)
    /// Returns value in range [0.0, 1.0]
    pub fn threshold(&self) -> f32 {
        self.bytes[2] as f32 / 256.0
    }

    /// Create from scale and flags (for forward ops)
    pub fn from_scale_flags(scale: u16, flags: ModifierFlags) -> Self {
        let scale_bytes = scale.to_be_bytes();
        Self {
            bytes: [scale_bytes[0], scale_bytes[1], flags.0],
        }
    }

    /// Extract scale as u16
    pub fn scale_u16(&self) -> u16 {
        u16::from_be_bytes([self.bytes[0], self.bytes[1]])
    }

    /// Extract scale as f32 (fixed-point: value / 65536.0)
    pub fn scale_f32(&self) -> f32 {
        self.scale_u16() as f32 / 65536.0
    }

    /// Extract flags
    pub fn flags(&self) -> ModifierFlags {
        ModifierFlags(self.bytes[2])
    }

    /// Create from value and flags (for learning ops)
    /// Value is encoded as fixed-point: raw = value * 256
    pub fn from_value_flags(value: f32, flags: ModifierFlags) -> Self {
        let raw = (value.clamp(0.0, 255.0) * 256.0) as u16;
        let val_bytes = raw.to_be_bytes();
        Self {
            bytes: [val_bytes[0], val_bytes[1], flags.0],
        }
    }

    /// Extract value as f32 (fixed-point: raw / 256.0)
    pub fn value_f32(&self) -> f32 {
        self.scale_u16() as f32 / 256.0
    }

    /// Create from absolute value (for SET_PARAM style ops)
    pub fn absolute(value: f32) -> Self {
        Self::from_value_flags(value, ModifierFlags::default())
    }

    /// Create from relative multiplier (for MULTIPLY_PARAM style ops)
    pub fn relative(multiplier: f32) -> Self {
        Self::from_value_flags(multiplier, ModifierFlags::default().with_relative())
    }

    /// Create from just a count
    pub fn just_count(count: u16) -> Self {
        Self::from_count_threshold(count, 0)
    }

    /// Get raw bytes
    pub fn to_bytes(&self) -> [u8; 3] {
        self.bytes
    }
}

impl fmt::Display for Modifier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{:02X}{:02X}{:02X}",
            self.bytes[0], self.bytes[1], self.bytes[2]
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shape_encoding() {
        let m = Modifier::from_shape(32, 12);
        let (input, output) = m.to_shape();
        assert_eq!(input, 32);
        assert_eq!(output, 12);

        // Test max values
        let m_max = Modifier::from_shape(4095, 4095);
        let (i, o) = m_max.to_shape();
        assert_eq!(i, 4095);
        assert_eq!(o, 4095);
    }

    #[test]
    fn test_count_threshold() {
        let m = Modifier::from_count_threshold(100, 128);
        assert_eq!(m.count(), 100);
        assert!((m.threshold() - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_scale_flags() {
        let m = Modifier::from_scale_flags(32768, ModifierFlags::default().with_clamp());
        assert_eq!(m.scale_u16(), 32768);
        assert!((m.scale_f32() - 0.5).abs() < 0.0001);
        assert!(m.flags().clamp());
    }

    #[test]
    fn test_value_encoding() {
        let m = Modifier::absolute(0.9);
        assert!((m.value_f32() - 0.9).abs() < 0.01);

        let m_rel = Modifier::relative(1.5);
        assert!((m_rel.value_f32() - 1.5).abs() < 0.01);
        assert!(m_rel.flags().relative());
    }

    #[test]
    fn test_flags() {
        let flags = ModifierFlags::default()
            .with_clamp()
            .with_relative()
            .with_negate();

        assert!(flags.clamp());
        assert!(flags.relative());
        assert!(flags.negate());
        assert!(!flags.log());
        assert!(!flags.inplace());
    }
}
