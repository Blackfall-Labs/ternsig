//! TVMR Extension System — Pluggable Instruction Set Extensions
//!
//! Extensions add domain-specific instructions to the TVMR without modifying
//! the core runtime. Each extension is assigned a 2-byte ID (u16) and gets
//! its own 65,536-opcode address space.
//!
//! ## Extension ID Ranges
//!
//! ```text
//! 0x0000          Core ISA (built-in, not an extension)
//! 0x0001-0x00FF   Standard extensions (tensor, ternary, learning, etc.)
//! 0x0100-0xFFFE   User-defined extensions
//! 0xFFFF          Reserved
//! ```

use super::interpreter::{ColdBuffer, HotBuffer, ChemicalState};
use super::register::Register;
use std::fmt;

/// Result of executing a single instruction.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub enum StepResult {
    /// Continue to next instruction.
    Continue,
    /// Program halted normally.
    Halt,
    /// Break from current loop.
    Break,
    /// Return from subroutine.
    Return,
    /// Yield control to the host with a domain operation.
    Yield(DomainOp),
    /// Reached end of program (PC past last instruction).
    Ended,
    /// Execution error.
    Error(String),
}

/// Domain operations yielded to the host for external handling.
///
/// The VM does not own external resources (substrate, fields, regions, files).
/// When an instruction needs external state, it yields a DomainOp.
/// The host fulfills the operation and resumes execution.
///
/// Convention: operations that READ expect the host to write results into
/// the register specified by `target`. Operations that WRITE read from
/// the register specified by `source`.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub enum DomainOp {
    // =========================================================================
    // Persistence (existing)
    // =========================================================================
    /// Load weights from persistent storage by key identifier.
    /// key_id is resolved to a thermogram key string by the host.
    LoadWeights { register: Register, key_id: i32 },
    /// Store weights to persistent storage by key identifier.
    /// key_id is resolved to a thermogram key string by the host.
    StoreWeights { register: Register, key_id: i32 },
    /// Trigger consolidation (hot → cold thermogram).
    Consolidate,
    /// Compute error between target and output.
    ComputeError { target: i32, output: i32 },

    // =========================================================================
    // Neuro: Chemical substrate (0x0005)
    // =========================================================================
    /// Read chemical level. Host writes value into target H[reg][0].
    /// chem_id: 0=DA, 1=5HT, 2=NE, 3=GABA, 4=cortisol, 5=endorphin, 6=ACh, 7=fatigue.
    ChemRead { target: Register, chem_id: u8 },
    /// SET chemical level (authoritative). Host reads value from source H[reg][0].
    ChemSet { source: Register, chem_id: u8 },
    /// ADDITIVE chemical injection (phasic event). Host reads delta from source H[reg][elem_idx].
    /// elem_idx defaults to 0 for backwards compatibility with single-element reads.
    /// When `signed` is true, the raw register value is centered around 128 before
    /// clamping: spike (255) → +127, no spike (0) → -128. This makes pool-output-based
    /// injection bidirectional — untrained pools (~50% spike rate) produce net-zero
    /// chemical change, while trained pools bias the direction.
    ChemInject { source: Register, chem_id: u8, elem_idx: u8, signed: bool },

    // =========================================================================
    // Neuro: Field substrate (0x0005)
    // =========================================================================
    /// Read field region slice into target register.
    /// field_id: 0=activity, 1=chemical, 2=convergence, 3=sensory, etc.
    FieldRead { target: Register, field_id: u8 },
    /// Write source register data to field region slice.
    FieldWrite { source: Register, field_id: u8 },
    /// Advance field by 1 tick (decay, age frames).
    FieldTick { field_id: u8 },
    /// Apply metabolic decay to field: retention factor + fatigue boost.
    FieldDecay { field_id: u8, retention: u8, fatigue_boost: u8 },

    // =========================================================================
    // Neuro: Sensory / convergence (0x0005)
    // =========================================================================
    /// Read all stimulation levels into target register.
    StimRead { target: Register },
    /// Read valence [reward, punish] into target register.
    ValenceRead { target: Register },
    /// Read convergence field state into target register.
    ConvRead { target: Register },
    /// Winner-take-some lateral inhibition on field.
    LateralInhibit { source: Register, strength: u8 },
    /// Apply exhaustion decay boost to sustained activity.
    ExhaustionBoost { source: Register, factor: u8 },
    /// Compute novelty z-scores from region energies.
    NoveltyScore { target: Register, source: Register },

    // =========================================================================
    // Orchestration: Model table (0x0007)
    // =========================================================================
    /// Load a model from firmware path into table slot.
    ModelLoad { target: Register, slot: u8 },
    /// Execute model in slot. Host runs interpreter, writes output.
    ModelExec { target: Register, slot: u8 },
    /// Hot-reload model in slot from disk.
    ModelReload { slot: u8 },

    // =========================================================================
    // Orchestration: Region routing (0x0007)
    // =========================================================================
    /// Route input data to a specific region.
    RouteInput { source: Register, region_id: u8 },
    /// Fire a region's SNN, output into target register.
    RegionFire { target: Register, region_id: u8 },
    /// Aggregate all pending region outputs into target register.
    CollectOutputs { target: Register },
    /// Enable a region for routing.
    RegionEnable { region_id: u8 },
    /// Disable a region.
    RegionDisable { region_id: u8 },

    // =========================================================================
    // Lifecycle (0x0008)
    // =========================================================================
    /// Read current boot phase. Host writes into target H[reg][0].
    PhaseRead { target: Register },
    /// Read current tick count. Host writes into target H[reg][0].
    TickRead { target: Register },
    /// Read current neuronal level. Host writes into target H[reg][0].
    LevelRead { target: Register },
    /// Initialize thermogram for cold register.
    InitThermo { register: Register },
    /// Save cold register to thermogram.
    SaveThermo { register: Register },
    /// Load cold register from thermogram.
    LoadThermo { register: Register },
    /// Log a lifecycle event.
    LogEvent { source: Register, event_type: u8 },
    /// Halt a specific brain region.
    HaltRegion { region_id: u8 },

    // =========================================================================
    // IPC (0x0009)
    // =========================================================================
    /// Send signal to another region.
    SendSignal { source: Register, region_id: u8 },
    /// Receive signal from another region. Host writes into target.
    RecvSignal { target: Register, region_id: u8 },
    /// Broadcast signal to all regions.
    Broadcast { source: Register, channel: u8 },
    /// Subscribe to signals from a region.
    Subscribe { target: Register, region_id: u8 },
    /// Peek at mailbox without consuming.
    MailboxPeek { target: Register, mailbox_id: u8 },
    /// Pop from mailbox (consume).
    MailboxPop { target: Register, mailbox_id: u8 },
    /// Wait at synchronization barrier.
    BarrierWait { barrier_id: u8 },
    /// Atomic compare-and-swap.
    AtomicCas { target: Register, expected: Register, desired: Register },

    // =========================================================================
    // Orchestration: Region status (0x0007)
    // =========================================================================
    /// Query region status. Host writes into target H[reg][0]: 0=idle, 1=active, 2=firing.
    RegionStatus { target: Register, region_id: u8 },

    // =========================================================================
    // Crash Resistance: Checkpoints (1.1)
    // =========================================================================
    /// Persist a checkpoint of cold register weights.
    /// Host writes to crash-safe storage (temp+rename or WAL).
    CheckpointSave { register: Register },
    /// Restore cold register from persisted checkpoint.
    /// Host reads from crash-safe storage. Returns error if no checkpoint exists.
    CheckpointRestore { register: Register },
    /// Discard a persisted checkpoint (cleanup after successful operation).
    CheckpointDiscard { register: Register },

    // =========================================================================
    // Crash Resistance: Transactions (1.1)
    // =========================================================================
    /// Begin a persistence transaction. All subsequent SaveThermo/CheckpointSave
    /// operations are buffered until TxnCommit or discarded on TxnRollback.
    TxnBegin { txn_id: u8 },
    /// Commit all buffered writes in the transaction atomically.
    TxnCommit { txn_id: u8 },
    /// Discard all buffered writes in the transaction.
    TxnRollback { txn_id: u8 },

    // =========================================================================
    // Bank: Distributed Representational Memory (0x000B)
    // =========================================================================
    /// Query bank by vector similarity. Host reads query vector from source
    /// register, runs sparse cosine similarity on the named bank, writes
    /// top_k results (EntryId + score pairs) into target register.
    ///
    /// Register layout for results:
    ///   H[target][0] = result count (0..top_k)
    ///   H[target][1] = entry_0 score (i32, scaled x256)
    ///   H[target][2] = entry_0 id_high (upper 32 bits of EntryId)
    ///   H[target][3] = entry_0 id_low (lower 32 bits of EntryId)
    ///   H[target][4] = entry_1 score ...
    BankQuery { target: Register, source: Register, bank_slot: u8, top_k: u8 },

    /// Write entry to bank. Host reads Signal vector from source register,
    /// creates a BankEntry, inserts into bank. Returns new EntryId in target:
    ///   H[target][0] = entry_id_high
    ///   H[target][1] = entry_id_low
    BankWrite { target: Register, source: Register, bank_slot: u8 },

    /// Load full entry vector into register for pattern completion.
    /// Host reads EntryId from source (H[src][0..1] = id_high, id_low),
    /// loads the entry's full Signal vector into target as i32 values.
    BankLoad { target: Register, source: Register, bank_slot: u8 },

    /// Add typed edge between two entries. Host reads source and destination
    /// BankRefs from register:
    ///   H[src][0] = from_entry_id_high
    ///   H[src][1] = from_entry_id_low
    ///   H[src][2] = to_bank_slot (u8 in i32)
    ///   H[src][3] = to_entry_id_high
    ///   H[src][4] = to_entry_id_low
    ///   H[src][5] = edge_weight (u8 in i32)
    BankLink { source: Register, edge_type: u8, bank_slot: u8 },

    /// Traverse edges from an entry. Host reads starting EntryId from source,
    /// follows edges of specified type up to depth, writes discovered BankRefs
    /// into target register:
    ///   H[target][0] = result count
    ///   H[target][1] = ref_0 bank_slot
    ///   H[target][2] = ref_0 entry_id_high
    ///   H[target][3] = ref_0 entry_id_low ...
    BankTraverse { target: Register, source: Register, bank_slot: u8, edge_type: u8, depth: u8 },

    /// Touch (access) an entry to update its last_accessed_tick and access_count.
    /// Host reads EntryId from source (H[src][0..1] = id_high, id_low).
    BankTouch { source: Register, bank_slot: u8 },

    /// Delete an entry from the bank.
    /// Host reads EntryId from source (H[src][0..1] = id_high, id_low).
    BankDelete { source: Register, bank_slot: u8 },

    /// Get entry count for a bank. Host writes count into H[target][0].
    BankCount { target: Register, bank_slot: u8 },

    // =========================================================================
    // Dynamic Register Allocation (1.2)
    // =========================================================================
    /// Request dynamic register allocation from kernel-managed pool.
    /// Host allocates a register of the specified bank and writes the
    /// register address into H[target][0].
    /// bank: 0=Hot, 1=Cold, 2=Param, 3=Shape
    /// size: requested capacity
    AllocRegister { target: Register, bank: u8, size: u16 },

    /// Release a dynamically allocated register back to the pool.
    FreeRegister { register: Register },

    // =========================================================================
    // Substrate Slice Addressing (1.3)
    // =========================================================================
    /// Read a specific slice of a field into target register.
    /// field_id identifies the field type; slice_index identifies the region's slice.
    FieldSliceRead { target: Register, field_id: u8, slice_index: u8 },
    /// Write source register data to a specific field slice.
    FieldSliceWrite { source: Register, field_id: u8, slice_index: u8 },

    // =========================================================================
    // Co-Activation Tracking (1.3)
    // =========================================================================
    /// Record co-activation between two neurons (registers).
    /// The host accumulates co-activation counts for sleep-cycle Hebbian rewiring.
    CoActivation { source_a: Register, source_b: Register },
    /// Read co-activation matrix for a set of neurons.
    /// Host writes accumulated co-activation counts into target register.
    CoActivationRead { target: Register, source: Register },
    /// Reset co-activation counters (called at start of sleep cycle).
    CoActivationReset,

    // =========================================================================
    // Lifecycle: Firmware → Kernel state requests (Async Brain)
    // =========================================================================
    /// Set neuronal level. Monotonic constraint — can only progress forward.
    /// Host reads requested level from source H[reg][0].
    LevelWrite { source: Register },
    /// Set boot phase. Valid transition constraint enforced by host.
    /// Host reads requested phase from source H[reg][0].
    PhaseWrite { source: Register },
    /// Set rest mode. Cortical gate — requires cortex-level to enter rest.
    /// Host reads 0=wake, 1=rest from source H[reg][0].
    RestWrite { source: Register },
    /// Request sleep consolidation (bank promote/demote/evict).
    /// Only valid during rest mode. Host reads urgency from source H[reg][0].
    ConsolidationRequest { source: Register },

    // =========================================================================
    // Consolidation Lifecycle (1.3)
    // =========================================================================
    /// Promote entry temperature (e.g., Hot → Warm, Warm → Cool).
    /// Used during SWR replay for proven patterns.
    BankPromote { source: Register, bank_slot: u8 },
    /// Demote entry temperature (e.g., Warm → Hot, Cool → Warm).
    /// Used for patterns that failed consolidation validation.
    BankDemote { source: Register, bank_slot: u8 },
    /// Evict cold/low-scoring entries from a bank (sleep pruning).
    /// count: max entries to evict.
    BankEvict { bank_slot: u8, count: u8 },
    /// Compact a bank (defragment after pruning).
    BankCompact { bank_slot: u8 },

    // =========================================================================
    // Coherence: Per-division coherence measurement (Async Brain)
    // =========================================================================
    /// Write singular coherence for a specific division.
    /// Host reads division index from source H[reg][0], coherence value from H[reg][1].
    CoherenceSingularWrite { source: Register },
    /// Write multi-modal coherence for a division pair.
    /// Host reads div_a from source H[reg][0], div_b from H[reg][1], value from H[reg][2].
    CoherenceMultimodalWrite { source: Register },
    /// Read singular coherence for all divisions into target register.
    /// Host writes 7 values (one per division) into target H[reg][0..6].
    CoherenceSingularRead { target: Register },

    // =========================================================================
    // Neuro: SNN substrate (0x0005)
    // =========================================================================
    /// Set SNN chemical axes target. Host reads 4 axis values from source
    /// H[reg][0..3] = [excitability, inhibition, persistence, stress] as i8.
    SNNSetAxes { source: Register },
    /// Step SNN axes toward target at bounded rate. Host reads delta from
    /// source H[reg][0] (max step per tick).
    SNNStepAxes { source: Register },
    /// Read SNN mean activation. Host writes signed i32 into target H[reg][0].
    SNNReadActivation { target: Register },
    /// Read SNN neuron count. Host writes count into target H[reg][0].
    SNNReadNeuronCount { target: Register },
    /// Read all SNN neuron output signals into target register.
    /// Host writes neuron_count i32 values into target H[reg][0..n].
    SNNReadOutputs { target: Register },
    /// Read SNN timestep counter. Host writes into target H[reg][0].
    SNNReadTimestep { target: Register },

    // =========================================================================
    // Neuro: SNN structural plasticity (evolution)
    // =========================================================================
    /// Grow SNN by N neurons. Host reads count from source H[reg][0].
    /// Writes new total neuron count into result H[reg][0].
    SNNGrowNeurons { source: Register, result: Register },
    /// Prune SNN neurons at indices in source register.
    /// Host reads N indices from source H[reg][0..N].
    /// Writes new total neuron count into result H[reg][0].
    SNNPruneNeurons { source: Register, result: Register },
    /// Read per-neuron spike counts (activity) into target register.
    /// Host writes neuron_count u32 values (as i32) into target H[reg][0..N].
    SNNReadActivities { target: Register },

    // =========================================================================
    // Neuro: Thermogram immune response (0x0005)
    // =========================================================================
    /// Prune weak HOT thermogram entries. Systemic immune response.
    /// Source register gates: prune only if value > 0.
    /// Result register receives count of pruned entries.
    ThermoPruneHot { source: Register, result: Register },

    /// Apply valence credit to HOT+WARM thermogram entries.
    /// DA > 140 reinforces, Cortisol > 60 weakens. Source register gates (>0 = fire).
    /// Result register receives count of affected entries.
    ThermoValenceCredit { source: Register, result: Register },

    // =========================================================================
    // Pool: Biological neuron pool substrate (0x000C)
    // =========================================================================
    /// Step pool dynamics. Host reads input currents from source register,
    /// calls pool.tick(), writes spike count into source H[reg][0].
    PoolTick { source: Register },
    /// Persist pool state to .pool file.
    PoolSave,
    /// Restore pool state from .pool file.
    PoolLoad,
    /// Inject signal into pool neuron range. Host reads signal from source register.
    /// range_start/range_end define the target neuron population.
    PoolInject { source: Register, range_start: u16, range_end: u16 },
    /// Read output spikes from pool neuron range into target register.
    /// Host writes binary spike vector as i32 values into target.
    PoolReadOutput { target: Register, range_start: u16, range_end: u16 },
    /// Read pool statistics into target register.
    /// H[target][0]=spike_count, [1]=synapse_count, [2]=tick_count.
    PoolReadStats { target: Register },
    /// Apply three-factor plasticity. Host reads DA/Cortisol/ACh from chemical
    /// substrate, calls pool.apply_modulation(). Result receives (reinforced, weakened).
    PoolModulate { result: Register },
    /// Prune dead synapses (HOT + counter=0). Result receives pruned count.
    PoolPruneDead { result: Register },
    /// Create new synapses between co-active neurons (ACh-gated).
    /// Result receives count of new synapses created.
    PoolSynaptogenesis { result: Register },
}

/// Describes how an instruction's 4 operand bytes are interpreted.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum OperandPattern {
    /// No operands used.
    None,
    /// `[reg:1][_:3]` — single register.
    Reg,
    /// `[dst:1][src:1][_:2]` — two registers.
    RegReg,
    /// `[dst:1][a:1][b:1][_:1]` — three registers.
    RegRegReg,
    /// `[dst:1][a:1][b:1][flags:1]` — three registers + flags byte.
    RegRegRegFlags,
    /// `[dst:1][src:1][imm16:2]` — register + 16-bit immediate.
    RegRegImm16,
    /// `[dst:1][_:1][imm16:2]` — register + 16-bit immediate.
    RegImm16,
    /// `[reg:1][imm8:1][_:2]` — register + 8-bit immediate.
    RegImm8,
    /// `[imm32:4]` — 32-bit immediate (e.g., jump target).
    Imm32,
    /// `[imm16:2][_:2]` — 16-bit immediate.
    Imm16,
    /// `[imm8:1][_:3]` — 8-bit immediate.
    Imm8,
    /// `[dst:1][cond:1][a:1][b:1]` — conditional select.
    RegCondRegReg,
    /// Custom pattern described by a string.
    Custom(&'static str),
}

/// Metadata for a single instruction within an extension.
#[derive(Debug, Clone)]
pub struct InstructionMeta {
    /// Extension-local opcode (0x0000-0xFFFF).
    pub opcode: u16,
    /// Assembly mnemonic (e.g., "TERNARY_MATMUL").
    pub mnemonic: &'static str,
    /// How the 4 operand bytes are interpreted.
    pub operand_pattern: OperandPattern,
    /// Human-readable description.
    pub description: &'static str,
}

/// Trait for local bank access within the interpreter.
///
/// The v3 kernel provides an implementation backed by BankCluster.
/// When present in ExecutionContext, bank extension ops execute locally
/// instead of yielding DomainOps for simple operations.
pub trait BankAccess: Send + Sync {
    /// Query bank by similarity. Returns vec of (entry_id_as_i64, score_i32) pairs.
    fn query(&self, bank_slot: u8, query: &[i32], top_k: usize) -> Option<Vec<(i64, i32)>>;
    /// Load entry vector by ID (high/low i32 pair). Returns i32 vector.
    fn load(&self, bank_slot: u8, entry_id_high: i32, entry_id_low: i32) -> Option<Vec<i32>>;
    /// Get entry count for a bank.
    fn count(&self, bank_slot: u8) -> Option<i32>;
    /// Write a new entry. Returns (entry_id_high, entry_id_low).
    fn write(&mut self, bank_slot: u8, vector: &[i32]) -> Option<(i32, i32)>;
    /// Touch an entry (update access tick/count).
    fn touch(&mut self, bank_slot: u8, entry_id_high: i32, entry_id_low: i32);
    /// Delete an entry. Returns true if found.
    fn delete(&mut self, bank_slot: u8, entry_id_high: i32, entry_id_low: i32) -> bool;
}

/// Execution context passed to extension `execute()` calls.
///
/// Provides controlled, mutable access to the VM's register file,
/// program counter, stacks, and I/O buffers.
pub struct ExecutionContext<'a> {
    // Register banks
    pub hot_regs: &'a mut Vec<Option<HotBuffer>>,
    pub cold_regs: &'a mut Vec<Option<ColdBuffer>>,
    pub param_regs: &'a mut Vec<i32>,
    pub shape_regs: &'a mut Vec<Vec<usize>>,

    // Program counter & stacks
    pub pc: &'a mut usize,
    pub call_stack: &'a mut Vec<usize>,
    pub loop_stack: &'a mut Vec<LoopState>,

    // I/O buffers
    pub input_buffer: &'a [i32],
    pub output_buffer: &'a mut Vec<i32>,
    pub target_buffer: &'a [i32],

    // Learning state
    pub chemical_state: &'a mut ChemicalState,
    pub current_error: &'a mut i32,
    pub babble_scale: &'a mut i32,
    pub babble_phase: &'a mut usize,

    // Pressure registers (for mastery learning)
    pub pressure_regs: &'a mut Vec<Option<Vec<i32>>>,

    /// Optional local bank cache for inline bank execution.
    /// When present, simple bank ops execute locally without yielding.
    /// When absent, all bank ops yield DomainOps to the host.
    pub bank_cache: Option<&'a mut (dyn BankAccess + 'static)>,
}

/// Loop state for LOOP/ENDLOOP tracking.
#[derive(Debug, Clone)]
pub struct LoopState {
    /// PC of the instruction after LOOP (loop body start).
    pub start_pc: usize,
    /// Remaining iterations.
    pub remaining: u32,
}

/// Extension trait — implement this to add instructions to the TVMR.
///
/// Each extension gets a unique 2-byte ID and its own 65k opcode space.
/// Extensions declare their instruction metadata at registration and
/// receive dispatch calls during execution.
pub trait Extension: Send + Sync {
    /// Unique extension identifier (u16).
    fn ext_id(&self) -> u16;

    /// Human-readable name (e.g., "tvmr.tensor").
    fn name(&self) -> &str;

    /// Semantic version (major, minor, patch).
    fn version(&self) -> (u16, u16, u16);

    /// List all instructions this extension provides.
    fn instructions(&self) -> &[InstructionMeta];

    /// Execute an instruction.
    ///
    /// Called by the interpreter when an instruction with this extension's
    /// ID is encountered. The `opcode` and `operands` are the extension-local
    /// opcode and the 4 operand bytes from the instruction.
    fn execute(
        &self,
        opcode: u16,
        operands: [u8; 4],
        ctx: &mut ExecutionContext,
    ) -> StepResult;

    /// Custom assembly for extension-specific operand parsing.
    /// Called by the assembler when parsing operands for this extension's instructions.
    /// Returns assembled operand bytes or None to fall back to default parsing.
    fn assemble_operands(&self, _mnemonic: &str, _tokens: &[&str]) -> Option<Result<[u8; 4], String>> {
        None
    }

    /// Custom disassembly for extension-specific output formatting.
    /// Returns a formatted string or None to fall back to default formatting.
    fn disassemble(&self, _opcode: u16, _operands: [u8; 4]) -> Option<String> {
        None
    }
}

impl fmt::Debug for dyn Extension {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (maj, min, pat) = self.version();
        write!(f, "Extension(0x{:04X} \"{}\" v{}.{}.{})", self.ext_id(), self.name(), maj, min, pat)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_operand_pattern_eq() {
        assert_eq!(OperandPattern::None, OperandPattern::None);
        assert_eq!(OperandPattern::RegRegReg, OperandPattern::RegRegReg);
        assert_ne!(OperandPattern::Reg, OperandPattern::RegReg);
    }

    #[test]
    fn test_instruction_meta() {
        let meta = InstructionMeta {
            opcode: 0x0000,
            mnemonic: "TEST_OP",
            operand_pattern: OperandPattern::RegReg,
            description: "A test operation",
        };
        assert_eq!(meta.mnemonic, "TEST_OP");
        assert_eq!(meta.operand_pattern, OperandPattern::RegReg);
    }
}
