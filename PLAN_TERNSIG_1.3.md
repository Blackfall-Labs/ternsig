# PLAN: Ternsig 1.3 — Inline Bank Execution & Substrate Addressing

**Date:** 2026-01-29
**Crate:** `ternsig` (`E:\repos\blackfall-labs\ternsig`)
**Branch:** main
**Status:** Draft — awaiting operator approval
**Depends on:** Ternsig 1.2 (bank DomainOps), databank-rs 0.2 (fulfillment helpers)
**Enables:** databank-rs 0.3 (consolidation lifecycle), v3 Phase 5 (TVMR restructure)

---

## Summary

Ternsig 1.3 is the **inline execution release**. Simple bank operations (BANK_QUERY, BANK_LOAD, BANK_COUNT) can now execute within the extension itself when the interpreter has a local bank cache, avoiding the DomainOp yield round-trip. Complex operations (cross-bank BANK_TRAVERSE, BANK_LINK) still yield to the host.

This release also adds substrate slice addressing (field IDs as compound (type, slice) pairs), co-activation tracking DomainOps, and the consolidation lifecycle DomainOps needed for sleep firmware.

---

## Scope

### 1.3.1 — Inline Bank Cache

**File:** `src/vm/extension.rs`

Add optional bank cache to `ExecutionContext`:

```rust
pub struct ExecutionContext<'a> {
    // ... existing fields ...

    /// Optional local bank cache for inline bank operations.
    /// When present, simple bank ops execute locally without yielding.
    /// When absent, all bank ops yield DomainOps to the host.
    pub bank_cache: Option<&'a mut dyn BankAccess>,
}

/// Trait for local bank access within the interpreter.
/// The v3 kernel provides an implementation backed by BankCluster.
/// Extensions call this instead of yielding DomainOps for simple ops.
pub trait BankAccess: Send {
    fn query(&self, bank_slot: u8, query: &[i32], top_k: usize) -> Option<Vec<(i64, i32)>>;
    fn load(&self, bank_slot: u8, entry_id_high: i32, entry_id_low: i32) -> Option<Vec<i32>>;
    fn count(&self, bank_slot: u8) -> Option<i32>;
    fn write(&mut self, bank_slot: u8, vector: &[i32]) -> Option<(i32, i32)>;
    fn touch(&mut self, bank_slot: u8, entry_id_high: i32, entry_id_low: i32);
    fn delete(&mut self, bank_slot: u8, entry_id_high: i32, entry_id_low: i32) -> bool;
}
```

**Behavioral change to BankExtension:**

When `ctx.bank_cache` is `Some`, these opcodes execute inline:
- `BANK_QUERY` → calls `cache.query()`, packs results into target register, returns `StepResult::Continue`
- `BANK_LOAD` → calls `cache.load()`, writes vector into target register, returns `StepResult::Continue`
- `BANK_COUNT` → calls `cache.count()`, writes into target register, returns `StepResult::Continue`
- `BANK_WRITE` → calls `cache.write()`, writes EntryId into target register, returns `StepResult::Continue`
- `BANK_TOUCH` → calls `cache.touch()`, returns `StepResult::Continue`
- `BANK_DELETE` → calls `cache.delete()`, returns `StepResult::Continue`

When `ctx.bank_cache` is `None`, all opcodes yield DomainOps as in 1.2.

These opcodes ALWAYS yield (no inline path):
- `BANK_LINK` — requires cross-bank edge management that the extension cannot do locally
- `BANK_TRAVERSE` — requires multi-bank BFS which may cross bank boundaries

**Backward compatibility:** `bank_cache: None` is the default. All 1.2 behavior preserved unchanged.

### 1.3.2 — Substrate Slice Addressing

**File:** `src/vm/extension.rs`

Add compound field addressing DomainOps:

```rust
/// Read a specific slice of a field into target register.
/// field_id identifies the field type; slice_index identifies the region's slice.
FieldSliceRead { target: Register, field_id: u8, slice_index: u8 },

/// Write source register data to a specific field slice.
FieldSliceWrite { source: Register, field_id: u8, slice_index: u8 },
```

**File:** `src/vm/extensions/neuro.rs`

Add 2 new opcodes:

| Opcode | Mnemonic | Operands | DomainOp |
|--------|----------|----------|----------|
| 0x000F | `FIELD_SLICE_READ` | `[target:1][field_id:1][slice:1][_:1]` | `FieldSliceRead` |
| 0x0010 | `FIELD_SLICE_WRITE` | `[source:1][field_id:1][slice:1][_:1]` | `FieldSliceWrite` |

**Why this matters:** In v2, `FieldRead`/`FieldWrite` address a field by type only. The kernel implicitly uses the calling region's slice. In v3, firmware needs to read OTHER regions' slices (e.g., reticular formation reads ALL activity field slices to assess whole-brain stability). Explicit slice addressing enables this.

### 1.3.3 — Co-Activation Tracking DomainOps

**File:** `src/vm/extension.rs`

Add DomainOps for Hebbian co-activation tracking:

```rust
/// Record co-activation between two neurons (registers).
/// The host accumulates co-activation counts for sleep-cycle Hebbian rewiring.
CoActivation { source_a: Register, source_b: Register },

/// Read co-activation matrix for a set of neurons.
/// Host writes accumulated co-activation counts into target register.
CoActivationRead { target: Register, source: Register },

/// Reset co-activation counters (called at start of sleep cycle).
CoActivationReset,
```

**File:** `src/vm/extensions/neuro.rs`

Add 3 new opcodes:

| Opcode | Mnemonic | Operands | DomainOp |
|--------|----------|----------|----------|
| 0x0011 | `COACT_RECORD` | `[src_a:1][src_b:1][_:2]` | `CoActivation` |
| 0x0012 | `COACT_READ` | `[target:1][source:1][_:2]` | `CoActivationRead` |
| 0x0013 | `COACT_RESET` | `[_:4]` | `CoActivationReset` |

### 1.3.4 — Consolidation Lifecycle DomainOps

**File:** `src/vm/extension.rs`

Add DomainOps for sleep-cycle consolidation:

```rust
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
```

**File:** `src/vm/extensions/bank.rs`

Add 4 new opcodes:

| Opcode | Mnemonic | Operands | DomainOp |
|--------|----------|----------|----------|
| 0x0008 | `BANK_PROMOTE` | `[source:1][bank_slot:1][_:2]` | `BankPromote` |
| 0x0009 | `BANK_DEMOTE` | `[source:1][bank_slot:1][_:2]` | `BankDemote` |
| 0x000A | `BANK_EVICT` | `[bank_slot:1][count:1][_:2]` | `BankEvict` |
| 0x000B | `BANK_COMPACT` | `[bank_slot:1][_:3]` | `BankCompact` |

These always yield — temperature transitions and eviction are host-managed operations that participate in transactions.

### 1.3.5 — ExecutionContext Backward Compatibility

Adding `bank_cache` to `ExecutionContext` is a breaking change for all code that constructs an `ExecutionContext` manually (tests, interpreter's step() method).

**Mitigation:** Add `bank_cache: None` to all existing construction sites. The interpreter's inline construction in `step()` sets `bank_cache: None` by default. Hosts that want inline execution call `interpreter.set_bank_cache(cache)` before `run()`.

**File:** `src/vm/interpreter/mod.rs`

Add:
```rust
impl Interpreter {
    /// Set the bank cache for inline bank execution.
    /// When set, bank extension ops execute locally instead of yielding.
    pub fn set_bank_cache(&mut self, cache: Box<dyn BankAccess>);

    /// Remove the bank cache. Bank ops will yield DomainOps.
    pub fn clear_bank_cache(&mut self);
}
```

Store as `Option<Box<dyn BankAccess>>` in the Interpreter struct. When building `ExecutionContext` in `step()`, pass `self.bank_cache.as_deref_mut()`.

### 1.3.6 — Version Bump & Changelog

**File:** `Cargo.toml`

Version: `1.2.0` → `1.3.0`

**Changelog entries:**
- NEW: Inline bank execution via BankAccess trait — BANK_QUERY, BANK_LOAD, BANK_COUNT, BANK_WRITE, BANK_TOUCH, BANK_DELETE execute locally when cache present
- NEW: Substrate slice addressing — FIELD_SLICE_READ, FIELD_SLICE_WRITE opcodes with explicit slice index
- NEW: Co-activation tracking DomainOps — COACT_RECORD, COACT_READ, COACT_RESET
- NEW: Consolidation lifecycle — BANK_PROMOTE, BANK_DEMOTE, BANK_EVICT, BANK_COMPACT DomainOps
- NEW: BankAccess trait for pluggable inline bank execution
- NEW: Interpreter::set_bank_cache() / clear_bank_cache()
- CHANGED: ExecutionContext gains bank_cache field (None by default, backward compatible)

---

## Implementation Order

### Step 1: ExecutionContext changes (1.3.5)
Add `bank_cache: Option<&mut dyn BankAccess>` field. Update all construction sites. Compile check.

### Step 2: BankAccess trait (1.3.1)
Define the trait in extension.rs. No implementors yet.

### Step 3: Inline bank execution (1.3.1)
Modify bank extension to check `ctx.bank_cache` before yielding. Write tests with a mock BankAccess.

### Step 4: Substrate slice addressing (1.3.2)
Add DomainOps + neuro opcodes. Tests.

### Step 5: Co-activation tracking (1.3.3)
Add DomainOps + neuro opcodes. Tests.

### Step 6: Consolidation lifecycle (1.3.4)
Add DomainOps + bank opcodes. Tests.

### Step 7: Interpreter integration (1.3.5)
Add set_bank_cache/clear_bank_cache to Interpreter. Wire into step().

### Step 8: Version bump + full test (1.3.6)
Bump version. Full test suite. Verify astromind-v2 compiles clean.

---

## Files Modified / Created

| File | Change |
|------|--------|
| `src/vm/extension.rs` | +BankAccess trait, +bank_cache field, +8 DomainOp variants |
| `src/vm/extensions/bank.rs` | Inline execution paths, +4 consolidation opcodes |
| `src/vm/extensions/neuro.rs` | +5 opcodes (slice addressing, co-activation) |
| `src/vm/extensions/mod.rs` | Update instruction counts |
| `src/vm/interpreter/mod.rs` | +bank_cache storage, set/clear methods, update step() |
| `Cargo.toml` | Version 1.2.0 → 1.3.0 |

---

## Testing Strategy

- **Inline bank tests:** Mock BankAccess impl, verify BANK_QUERY/LOAD/COUNT execute locally and return Continue (not Yield)
- **Fallback tests:** With bank_cache=None, verify same opcodes yield DomainOps (1.2 behavior preserved)
- **Slice addressing tests:** FIELD_SLICE_READ/WRITE yield correct DomainOps with slice_index
- **Co-activation tests:** COACT_RECORD/READ/RESET yield correct DomainOps
- **Consolidation tests:** BANK_PROMOTE/DEMOTE/EVICT/COMPACT yield correct DomainOps
- **Regression:** All previous tests pass with bank_cache=None
- **Integration:** astromind-v2 compiles clean

---

## Deferred to 1.4+

| Feature | Rationale |
|---------|-----------|
| Cross-language spec (TVMR_SPEC.md) | Documentation, not code |
| Neuron state bank convention | Convention doc for SNN-in-firmware layout |
| Architecture thermogram format | Thermogram crate scope |
| Meta-learning feedback loop | Needs consumer implementation first |
| Bank cache write-back policy | 1.3 writes through to cluster. Deferred: lazy write-back for perf. |
