//! Action - Opcode definitions for Ternsig VM
//!
//! Opcodes organized by category in 256-opcode ranges:
//!
//! | Range   | Category         | Purpose                                      |
//! |---------|------------------|----------------------------------------------|
//! | 0x00xx  | System           | NOP, HALT, CHECKPOINT, RESET                 |
//! | 0x10xx  | Register Mgmt    | ALLOC, FREE, LOAD_WEIGHTS, STORE_WEIGHTS     |
//! | 0x20xx  | Architecture     | DEFINE_LAYER, SET_ACTIVATION, WIRE           |
//! | 0x30xx  | Forward Ops      | MATMUL, ADD, RELU, SIGMOID, SCALE, SHIFT     |
//! | 0x40xx  | Ternary Ops      | TERNARY_MATMUL, QUANTIZE, PACK, UNPACK       |
//! | 0x50xx  | Learning Ops     | MARK_ELIG, DECAY_ELIG, UPDATE_WEIGHTS, BABBLE|
//! | 0x60xx  | Control Flow     | LOOP, BREAK, IF_ERROR_GT, CALL, RETURN       |
//! | 0x70xx  | Debug            | TRACE, BREAKPOINT, PROFILE                   |

use std::fmt;

/// Operation opcode (2 bytes)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Action(pub u16);

impl Action {
    // =========================================================================
    // System Operations (0x00xx)
    // =========================================================================

    /// No operation
    pub const NOP: Self = Self(0x0000);
    /// Halt execution (end of forward pass)
    pub const HALT: Self = Self(0x0001);
    /// Save current state to Thermogram
    pub const CHECKPOINT: Self = Self(0x0002);
    /// Reset interpreter state
    pub const RESET: Self = Self(0x0003);
    /// Synchronization barrier
    pub const SYNC: Self = Self(0x0004);
    /// Yield to Learning ISA interpreter
    pub const YIELD: Self = Self(0x0005);

    // =========================================================================
    // Register Management (0x10xx)
    // =========================================================================

    /// Allocate a tensor register with shape from modifier
    pub const ALLOC_TENSOR: Self = Self(0x1000);
    /// Free a tensor register
    pub const FREE_TENSOR: Self = Self(0x1001);
    /// Load weights from Thermogram into cold register
    pub const LOAD_WEIGHTS: Self = Self(0x1002);
    /// Store weights from cold register to Thermogram
    pub const STORE_WEIGHTS: Self = Self(0x1003);
    /// Copy: target = source
    pub const COPY_REG: Self = Self(0x1004);
    /// Swap two registers
    pub const SWAP_REG: Self = Self(0x1005);
    /// Zero out a register
    pub const ZERO_REG: Self = Self(0x1006);
    /// Load from external input buffer into hot register
    pub const LOAD_INPUT: Self = Self(0x1007);
    /// Store from hot register to external output buffer
    pub const STORE_OUTPUT: Self = Self(0x1008);
    /// Load target label for learning (from external target buffer)
    pub const LOAD_TARGET: Self = Self(0x1009);

    // =========================================================================
    // Architecture Definition (0x20xx)
    // =========================================================================

    /// Define layer dimensions (input_dim, output_dim in modifier)
    pub const DEFINE_LAYER: Self = Self(0x2000);
    /// Set activation function for a layer
    pub const SET_ACTIVATION: Self = Self(0x2001);
    /// Enable/disable bias for a layer
    pub const SET_BIAS: Self = Self(0x2002);
    /// Wire forward connection between layers
    pub const WIRE_FORWARD: Self = Self(0x2003);
    /// Add skip connection
    pub const WIRE_SKIP: Self = Self(0x2004);
    /// Set data type for a register
    pub const SET_DTYPE: Self = Self(0x2005);
    /// Mark layer as non-trainable
    pub const FREEZE_LAYER: Self = Self(0x2006);
    /// Unfreeze a layer for training
    pub const UNFREEZE_LAYER: Self = Self(0x2007);
    /// Grow layer: add neurons to a cold register (increase output dimension)
    /// target = cold register, aux = number of neurons to add
    pub const GROW_NEURON: Self = Self(0x2008);
    /// Prune neuron: remove a neuron from a cold register
    /// target = cold register, aux = neuron index to remove
    pub const PRUNE_NEURON: Self = Self(0x2009);
    /// Initialize cold register with random ternary weights
    /// target = cold register, modifier encodes seed
    pub const INIT_RANDOM: Self = Self(0x200A);

    // =========================================================================
    // Forward Operations (0x30xx) - Float/Int tensor ops
    // =========================================================================

    /// Matrix multiply: target = source @ aux
    pub const MATMUL: Self = Self(0x3000);
    /// Element-wise add: target = source + aux
    pub const ADD: Self = Self(0x3001);
    /// Element-wise multiply: target = source * aux
    pub const MUL: Self = Self(0x3002);
    /// ReLU activation: target = max(0, source)
    pub const RELU: Self = Self(0x3003);
    /// Sigmoid activation: target = 1/(1 + exp(-source))
    pub const SIGMOID: Self = Self(0x3004);
    /// Tanh activation: target = tanh(source)
    pub const TANH: Self = Self(0x3005);
    /// Softmax: target = softmax(source, dim=aux)
    pub const SOFTMAX: Self = Self(0x3006);
    /// GELU activation
    pub const GELU: Self = Self(0x3007);
    /// Scale by constant: target = source * scale (scale in modifier)
    pub const SCALE: Self = Self(0x3008);
    /// Right-shift (for integer scaling): target = source >> aux
    pub const SHIFT: Self = Self(0x3009);
    /// Clamp to range: target = clamp(source, min, max)
    pub const CLAMP: Self = Self(0x300A);
    /// Negate: target = -source
    pub const NEGATE: Self = Self(0x300B);
    /// Layer normalization
    pub const LAYER_NORM: Self = Self(0x300C);
    /// Batch normalization
    pub const BATCH_NORM: Self = Self(0x300D);
    /// Subtract: target = source - aux
    pub const SUB: Self = Self(0x300E);
    /// Compare greater than: target = source > aux ? 1 : 0
    pub const CMP_GT: Self = Self(0x300F);
    /// Max reduce: target[0] = max(source)
    pub const MAX_REDUCE: Self = Self(0x3010);
    /// Set constant: target = [value] (scalar i32 from imm16)
    pub const SET_CONST: Self = Self(0x3011);

    // =========================================================================
    // Ternary Operations (0x40xx) - Integer-only, CPU-only
    // =========================================================================

    /// Ternary matrix multiply (integer-only): target = W.ternary @ input
    pub const TERNARY_MATMUL: Self = Self(0x4000);
    /// Quantize f32 to Signal
    pub const QUANTIZE: Self = Self(0x4001);
    /// Dequantize Signal to f32
    pub const DEQUANTIZE: Self = Self(0x4002);
    /// Pack Signal to 2-bit representation
    pub const PACK_TERNARY: Self = Self(0x4003);
    /// Unpack 2-bit to Signal
    pub const UNPACK_TERNARY: Self = Self(0x4004);
    /// Apply polarity update to weight
    pub const APPLY_POLARITY: Self = Self(0x4005);
    /// Apply magnitude update to weight
    pub const APPLY_MAGNITUDE: Self = Self(0x4006);
    /// Check polarity flip threshold (hysteresis)
    pub const THRESHOLD_POLARITY: Self = Self(0x4007);
    /// Accumulate polarity pressure
    pub const ACCUMULATE_PRESSURE: Self = Self(0x4008);
    /// Ternary add with bias
    pub const TERNARY_ADD_BIAS: Self = Self(0x4009);
    /// Embedding lookup: target[i] = table[indices[i]]
    /// target = hot register for output, source = cold register (table), aux = hot register (indices)
    pub const EMBED_LOOKUP: Self = Self(0x400A);
    /// Reduce average: target[0] = mean(source[start..start+count])
    /// Useful for band pooling in audio, spatial pooling, etc.
    /// aux encodes start index, modifier[0] encodes count
    pub const REDUCE_AVG: Self = Self(0x400B);
    /// Slice: target = source[start..start+len]
    /// aux encodes start index, modifier[0] encodes length
    pub const SLICE: Self = Self(0x400C);
    /// Argmax: target[0] = index of max value in source
    pub const ARGMAX: Self = Self(0x400D);
    /// Concat: target = concat(source, aux_reg)
    /// Appends aux_reg values after source values
    pub const CONCAT: Self = Self(0x400E);
    /// Squeeze: target = source with dimension removed (aux = dim index)
    /// For 1D tensors, this is effectively a no-op copy
    pub const SQUEEZE: Self = Self(0x400F);
    /// Unsqueeze: target = source with new dimension added (aux = dim index)
    /// For 1D tensors, this is effectively a no-op copy
    pub const UNSQUEEZE: Self = Self(0x4010);
    /// Transpose: target = source with dims swapped (aux = dim1, modifier[0] = dim2)
    /// For 1D tensors, this is effectively a no-op copy
    pub const TRANSPOSE: Self = Self(0x4011);
    /// Gate update: target = gate * update + (1 - gate) * state
    /// source = gate values, aux = update register, modifier references state register
    pub const GATE_UPDATE: Self = Self(0x4012);
    /// Ternary batch matrix multiply: applies same weight matrix to each row of input batch
    /// target[i] = weights @ input[i] for each i
    pub const TERNARY_BATCH_MATMUL: Self = Self(0x4013);
    /// Embed sequence: target[i] = table[i] for i in 0..count
    /// Generates sequential position embeddings (0, 1, 2, ..., count-1)
    /// aux encodes count of positions to embed
    pub const EMBED_SEQUENCE: Self = Self(0x4014);
    /// Reduce mean along dimension: target = mean(source, dim)
    /// For 2D tensor (rows, cols), dim=0 gives (cols,), dim=1 gives (rows,)
    /// aux encodes dimension to reduce along
    pub const REDUCE_MEAN_DIM: Self = Self(0x4015);

    // =========================================================================
    // Learning Operations (0x50xx)
    // =========================================================================

    /// Mark weights as eligible for update based on activity
    pub const MARK_ELIGIBILITY: Self = Self(0x5000);
    /// Decay eligibility traces
    pub const DECAY_ELIGIBILITY: Self = Self(0x5001);
    /// Compute error for layer
    pub const COMPUTE_ERROR: Self = Self(0x5002);
    /// Apply weight updates based on eligibility and error
    pub const UPDATE_WEIGHTS: Self = Self(0x5003);
    /// Add babble noise for exploration
    pub const ADD_BABBLE: Self = Self(0x5004);
    /// Decay babble scale
    pub const DECAY_BABBLE: Self = Self(0x5005);
    /// Compute Reward Prediction Error
    pub const COMPUTE_RPE: Self = Self(0x5006);
    /// Gate learning based on error threshold
    pub const GATE_ERROR: Self = Self(0x5007);
    /// Checkpoint weights for potential rollback
    pub const CHECKPOINT_WEIGHTS: Self = Self(0x5008);
    /// Rollback to checkpointed weights
    pub const ROLLBACK_WEIGHTS: Self = Self(0x5009);
    /// Consolidate hot → cold (Thermogram)
    pub const CONSOLIDATE: Self = Self(0x500A);
    /// Activity-weighted error distribution
    pub const ACTIVITY_WEIGHT: Self = Self(0x500B);

    // --- Contrastive Hebbian Learning (CHL) ---

    /// CHL: Start free phase, clear correlation buffers
    pub const CHL_FREE_START: Self = Self(0x500C);
    /// CHL: Record free phase correlations (pre × post at each synapse)
    pub const CHL_FREE_RECORD: Self = Self(0x500D);
    /// CHL: Start clamped phase (store target)
    pub const CHL_CLAMP_START: Self = Self(0x500E);
    /// CHL: Record clamped phase correlations (with output forced to target)
    pub const CHL_CLAMP_RECORD: Self = Self(0x500F);
    /// CHL: Compute weight updates from Δ(clamped - free) correlations
    pub const CHL_UPDATE: Self = Self(0x5010);
    /// CHL: Propagate clamped signal backward through layer
    pub const CHL_BACKPROP_CLAMP: Self = Self(0x5011);

    // --- Mastery Learning ---

    /// Mastery: Update pressure based on error direction and activity
    /// pressure[i] += direction * activity[i] * scale (if activity > threshold)
    pub const MASTERY_UPDATE: Self = Self(0x5012);
    /// Mastery: Commit pressure to weight changes
    /// If |pressure| > threshold: adjust magnitude, potentially flip polarity
    pub const MASTERY_COMMIT: Self = Self(0x5013);

    // =========================================================================
    // Control Flow (0x60xx)
    // =========================================================================

    /// Start loop (count in modifier)
    pub const LOOP: Self = Self(0x6000);
    /// End of loop body (jump back to LOOP)
    pub const END_LOOP: Self = Self(0x6001);
    /// Break out of current loop
    pub const BREAK: Self = Self(0x6002);
    /// Conditional: if error > threshold
    pub const IF_ERROR_GT: Self = Self(0x6003);
    /// Conditional: if error < threshold
    pub const IF_ERROR_LT: Self = Self(0x6004);
    /// Conditional: if layer has activity
    pub const IF_LAYER_ACTIVE: Self = Self(0x6005);
    /// Call subroutine (address in modifier)
    pub const CALL: Self = Self(0x6006);
    /// Return from subroutine
    pub const RETURN: Self = Self(0x6007);
    /// Unconditional jump
    pub const JUMP: Self = Self(0x6008);
    /// Conditional: if register is zero
    pub const IF_ZERO: Self = Self(0x6009);
    /// Conditional: if register is non-zero
    pub const IF_NONZERO: Self = Self(0x600A);
    /// Skip next N instructions
    pub const SKIP: Self = Self(0x600B);

    // =========================================================================
    // Debug/Profiling (0x70xx)
    // =========================================================================

    /// Emit trace message (for debugging)
    pub const TRACE: Self = Self(0x7000);
    /// Set breakpoint
    pub const BREAKPOINT: Self = Self(0x7001);
    /// Start profiling
    pub const PROFILE_START: Self = Self(0x7002);
    /// Stop profiling and emit stats
    pub const PROFILE_END: Self = Self(0x7003);
    /// Assert condition (halt if false)
    pub const ASSERT: Self = Self(0x7004);
    /// Print register contents
    pub const DUMP_REG: Self = Self(0x7005);

    // =========================================================================
    // Methods
    // =========================================================================

    /// Create from u16 opcode
    pub const fn from_u16(val: u16) -> Self {
        Self(val)
    }

    /// Get u16 value
    pub const fn as_u16(&self) -> u16 {
        self.0
    }

    /// Get opcode category (high byte)
    pub const fn category(&self) -> u8 {
        (self.0 >> 8) as u8
    }

    /// Check if this is a system operation
    pub const fn is_system(&self) -> bool {
        self.category() == 0x00
    }

    /// Check if this is a register management operation
    pub const fn is_register(&self) -> bool {
        self.category() == 0x10
    }

    /// Check if this is an architecture definition operation
    pub const fn is_architecture(&self) -> bool {
        self.category() == 0x20
    }

    /// Check if this is a forward operation
    pub const fn is_forward(&self) -> bool {
        self.category() == 0x30
    }

    /// Check if this is a ternary operation
    pub const fn is_ternary(&self) -> bool {
        self.category() == 0x40
    }

    /// Check if this is a learning operation
    pub const fn is_learning(&self) -> bool {
        self.category() == 0x50
    }

    /// Check if this is a control flow operation
    pub const fn is_control_flow(&self) -> bool {
        self.category() == 0x60
    }

    /// Check if this is a debug operation
    pub const fn is_debug(&self) -> bool {
        self.category() == 0x70
    }

    /// Check if this operation modifies the program counter
    pub const fn modifies_pc(&self) -> bool {
        matches!(
            self.0,
            0x6000..=0x600B // All control flow
        )
    }

    /// Check if this operation requires external domain execution
    pub const fn is_domain_op(&self) -> bool {
        matches!(
            self.0,
            0x1002 | 0x1003 | 0x1007 | 0x1008 | // LOAD/STORE_WEIGHTS, LOAD/STORE_INPUT/OUTPUT
            0x5002 | 0x500A // COMPUTE_ERROR, CONSOLIDATE
        )
    }

    /// Get human-readable name
    pub fn name(&self) -> &'static str {
        match self.0 {
            // System
            0x0000 => "NOP",
            0x0001 => "HALT",
            0x0002 => "CHECKPOINT",
            0x0003 => "RESET",
            0x0004 => "SYNC",
            0x0005 => "YIELD",

            // Register
            0x1000 => "ALLOC_TENSOR",
            0x1001 => "FREE_TENSOR",
            0x1002 => "LOAD_WEIGHTS",
            0x1003 => "STORE_WEIGHTS",
            0x1004 => "COPY_REG",
            0x1005 => "SWAP_REG",
            0x1006 => "ZERO_REG",
            0x1007 => "LOAD_INPUT",
            0x1008 => "STORE_OUTPUT",
            0x1009 => "LOAD_TARGET",

            // Architecture
            0x2000 => "DEFINE_LAYER",
            0x2001 => "SET_ACTIVATION",
            0x2002 => "SET_BIAS",
            0x2003 => "WIRE_FORWARD",
            0x2004 => "WIRE_SKIP",
            0x2005 => "SET_DTYPE",
            0x2006 => "FREEZE_LAYER",
            0x2007 => "UNFREEZE_LAYER",
            0x2008 => "GROW_NEURON",
            0x2009 => "PRUNE_NEURON",
            0x200A => "INIT_RANDOM",

            // Forward
            0x3000 => "MATMUL",
            0x3001 => "ADD",
            0x3002 => "MUL",
            0x3003 => "RELU",
            0x3004 => "SIGMOID",
            0x3005 => "TANH",
            0x3006 => "SOFTMAX",
            0x3007 => "GELU",
            0x3008 => "SCALE",
            0x3009 => "SHIFT",
            0x300A => "CLAMP",
            0x300B => "NEGATE",
            0x300C => "LAYER_NORM",
            0x300D => "BATCH_NORM",
            0x300E => "SUB",
            0x300F => "CMP_GT",
            0x3010 => "MAX_REDUCE",
            0x3011 => "SET_CONST",

            // Ternary
            0x4000 => "TERNARY_MATMUL",
            0x4001 => "QUANTIZE",
            0x4002 => "DEQUANTIZE",
            0x4003 => "PACK_TERNARY",
            0x4004 => "UNPACK_TERNARY",
            0x4005 => "APPLY_POLARITY",
            0x4006 => "APPLY_MAGNITUDE",
            0x4007 => "THRESHOLD_POLARITY",
            0x4008 => "ACCUMULATE_PRESSURE",
            0x4009 => "TERNARY_ADD_BIAS",
            0x400A => "EMBED_LOOKUP",
            0x400B => "REDUCE_AVG",
            0x400C => "SLICE",
            0x400D => "ARGMAX",
            0x400E => "CONCAT",
            0x400F => "SQUEEZE",
            0x4010 => "UNSQUEEZE",
            0x4011 => "TRANSPOSE",
            0x4012 => "GATE_UPDATE",
            0x4013 => "TERNARY_BATCH_MATMUL",
            0x4014 => "EMBED_SEQUENCE",
            0x4015 => "REDUCE_MEAN_DIM",

            // Learning
            0x5000 => "MARK_ELIGIBILITY",
            0x5001 => "DECAY_ELIGIBILITY",
            0x5002 => "COMPUTE_ERROR",
            0x5003 => "UPDATE_WEIGHTS",
            0x5004 => "ADD_BABBLE",
            0x5005 => "DECAY_BABBLE",
            0x5006 => "COMPUTE_RPE",
            0x5007 => "GATE_ERROR",
            0x5008 => "CHECKPOINT_WEIGHTS",
            0x5009 => "ROLLBACK_WEIGHTS",
            0x500A => "CONSOLIDATE",
            0x500B => "ACTIVITY_WEIGHT",
            // CHL
            0x500C => "CHL_FREE_START",
            0x500D => "CHL_FREE_RECORD",
            0x500E => "CHL_CLAMP_START",
            0x500F => "CHL_CLAMP_RECORD",
            0x5010 => "CHL_UPDATE",
            0x5011 => "CHL_BACKPROP_CLAMP",
            // Mastery
            0x5012 => "MASTERY_UPDATE",
            0x5013 => "MASTERY_COMMIT",

            // Control Flow
            0x6000 => "LOOP",
            0x6001 => "END_LOOP",
            0x6002 => "BREAK",
            0x6003 => "IF_ERROR_GT",
            0x6004 => "IF_ERROR_LT",
            0x6005 => "IF_LAYER_ACTIVE",
            0x6006 => "CALL",
            0x6007 => "RETURN",
            0x6008 => "JUMP",
            0x6009 => "IF_ZERO",
            0x600A => "IF_NONZERO",
            0x600B => "SKIP",

            // Debug
            0x7000 => "TRACE",
            0x7001 => "BREAKPOINT",
            0x7002 => "PROFILE_START",
            0x7003 => "PROFILE_END",
            0x7004 => "ASSERT",
            0x7005 => "DUMP_REG",

            _ => "UNKNOWN",
        }
    }
}

impl fmt::Display for Action {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

impl Default for Action {
    fn default() -> Self {
        Self::NOP
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_opcode_categories() {
        assert!(Action::NOP.is_system());
        assert!(Action::HALT.is_system());

        assert!(Action::LOAD_WEIGHTS.is_register());
        assert!(Action::STORE_OUTPUT.is_register());

        assert!(Action::DEFINE_LAYER.is_architecture());

        assert!(Action::MATMUL.is_forward());
        assert!(Action::RELU.is_forward());

        assert!(Action::TERNARY_MATMUL.is_ternary());
        assert!(Action::QUANTIZE.is_ternary());

        assert!(Action::MARK_ELIGIBILITY.is_learning());
        assert!(Action::UPDATE_WEIGHTS.is_learning());

        assert!(Action::LOOP.is_control_flow());
        assert!(Action::RETURN.is_control_flow());

        assert!(Action::TRACE.is_debug());
    }

    #[test]
    fn test_pc_modifying() {
        assert!(Action::LOOP.modifies_pc());
        assert!(Action::JUMP.modifies_pc());
        assert!(Action::CALL.modifies_pc());

        assert!(!Action::MATMUL.modifies_pc());
        assert!(!Action::ADD.modifies_pc());
    }

    #[test]
    fn test_names() {
        assert_eq!(Action::TERNARY_MATMUL.name(), "TERNARY_MATMUL");
        assert_eq!(Action::RELU.name(), "RELU");
        assert_eq!(Action::MARK_ELIGIBILITY.name(), "MARK_ELIGIBILITY");
    }
}
