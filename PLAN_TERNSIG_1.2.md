# PLAN: Ternsig 1.2 — Bank DomainOps & Deferred 1.1 Items

**Date:** 2026-01-29
**Crate:** `ternsig` (`E:\repos\blackfall-labs\ternsig`)
**Branch:** main
**Status:** Draft — awaiting operator approval
**Depends on:** Ternsig 1.1 (complete), databank-rs 0.1 (complete)
**Enables:** databank-rs 0.2 (DomainOp fulfillment helpers)

---

## Summary

Ternsig 1.2 is the **bank integration release**. It adds the DomainOp variants and bank extension (0x000B) that let firmware explicitly query, write, and traverse distributed representational memory (databank-rs). It also delivers deferred 1.1 items: LoadWeights/StoreWeights lifecycle wiring, dynamic register allocation DomainOp, nested transactions, and per-extension assemble/disassemble hooks.

After 1.2, firmware can yield `BankQuery`, `BankWrite`, `BankLoad`, `BankLink`, and `BankTraverse` DomainOps. The v3 kernel fulfills these via databank-rs 0.2.

---

## Scope

### 1.2.1 — Bank DomainOp Variants

**File:** `src/vm/extension.rs`

Add 5 new DomainOp variants to the `#[non_exhaustive]` enum:

```rust
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
///   H[target][4] = entry_1 score
///   ...
BankQuery { target: Register, source: Register, bank_slot: u8, top_k: u8 },

/// Write entry to bank. Host reads Signal vector from source register,
/// creates a BankEntry with current temperature, inserts into bank.
/// Returns new EntryId in target register:
///   H[target][0] = entry_id_high
///   H[target][1] = entry_id_low
BankWrite { target: Register, source: Register, bank_slot: u8 },

/// Load full entry vector into register for pattern completion.
/// Host reads EntryId from source register (H[src][0..1] = id_high, id_low),
/// loads the entry's full Signal vector into target register as i32 values.
BankLoad { target: Register, source: Register, bank_slot: u8 },

/// Add typed edge between two entries. Host reads source and destination
/// BankRefs from registers:
///   H[src][0] = from_entry_id_high
///   H[src][1] = from_entry_id_low
///   H[src][2] = to_bank_slot (u8 in i32)
///   H[src][3] = to_entry_id_high
///   H[src][4] = to_entry_id_low
///   H[src][5] = edge_weight (u8 in i32)
BankLink { source: Register, edge_type: u8, bank_slot: u8 },

/// Traverse edges from an entry. Host reads starting EntryId from source,
/// follows edges of specified type up to depth, writes discovered BankRefs
/// into target register.
///   H[target][0] = result count
///   H[target][1] = ref_0 bank_slot
///   H[target][2] = ref_0 entry_id_high
///   H[target][3] = ref_0 entry_id_low
///   ...
BankTraverse { target: Register, source: Register, bank_slot: u8, edge_type: u8, depth: u8 },
```

**Design decisions:**

- `bank_slot: u8` is a per-interpreter slot index (0-255), NOT a global BankId. The kernel maps slots to actual BankIds via a `RegionBankMap`. This keeps firmware decoupled from global identity.
- Signal vectors are packed as i32 in hot registers (signed value = polarity * magnitude). The kernel converts between `Signal` and `i32` during fulfillment.
- EntryId (u64) is split into two i32 values (high/low) since hot registers hold i32.
- `BankTraverse` takes `depth: u8` (max 255 hops). The `edge_type` operand uses the same encoding as databank-rs `EdgeType::to_u8()`.

### 1.2.2 — Bank Extension (0x000B)

**New file:** `src/vm/extensions/bank.rs`

New extension: `BankExtension` with ext_id `0x000B`, name `"tvmr.bank"`.

**Instructions (8 opcodes):**

| Opcode | Mnemonic | Operands | DomainOp Yielded |
|--------|----------|----------|-----------------|
| 0x0000 | `BANK_QUERY` | `[target:1][source:1][bank_slot:1][top_k:1]` | `BankQuery` |
| 0x0001 | `BANK_WRITE` | `[target:1][source:1][bank_slot:1][_:1]` | `BankWrite` |
| 0x0002 | `BANK_LOAD` | `[target:1][source:1][bank_slot:1][_:1]` | `BankLoad` |
| 0x0003 | `BANK_LINK` | `[source:1][edge_type:1][bank_slot:1][_:1]` | `BankLink` |
| 0x0004 | `BANK_TRAVERSE` | `[target:1][source:1][bank_slot:1][packed:1]` | `BankTraverse` |
| 0x0005 | `BANK_TOUCH` | `[source:1][bank_slot:1][_:2]` | `BankTouch` |
| 0x0006 | `BANK_DELETE` | `[source:1][bank_slot:1][_:2]` | `BankDelete` |
| 0x0007 | `BANK_COUNT` | `[target:1][bank_slot:1][_:2]` | `BankCount` |

**Notes on BANK_TRAVERSE operand packing:**
- `packed` byte: `[edge_type:4bits][depth:4bits]` — supports 16 edge types, depth 0-15.
- For deeper traversals, the separate `edge_type` and `depth` fields in the DomainOp allow the full u8 range. The packed encoding is a convenience for common cases.

**Additional DomainOps for utility operations:**

```rust
/// Touch (access) an entry to update its last_accessed_tick and access_count.
BankTouch { source: Register, bank_slot: u8 },

/// Delete an entry from the bank.
BankDelete { source: Register, bank_slot: u8 },

/// Get entry count for a bank. Host writes count into H[target][0].
BankCount { target: Register, bank_slot: u8 },
```

**Total: 8 new DomainOp variants (5 core + 3 utility).**

### 1.2.3 — Register Extension in extensions/mod.rs

**File:** `src/vm/extensions/mod.rs`

- Add `pub mod bank;` module declaration
- Add `pub use bank::BankExtension;`
- Add `"bank" => Some(0x000B)` to `resolve_ext_name()`
- Add `Box::new(BankExtension::new())` to `standard_extensions()`
- Update extension table comment: 11 extensions
- Update `test_standard_extensions_count` assertion: 10 → 11
- Update `test_extension_id_allocation` to include `0x000B`

### 1.2.4 — LoadWeights/StoreWeights Lifecycle Wiring

**File:** `src/vm/extensions/lifecycle.rs`

Add 2 new opcodes to LifecycleExtension that yield the existing `LoadWeights` and `StoreWeights` DomainOps:

| Opcode | Mnemonic | Operands | DomainOp |
|--------|----------|----------|----------|
| 0x000B | `LOAD_WEIGHTS` | `[reg:1][key_src:1][_:2]` | `LoadWeights { register, key }` |
| 0x000C | `STORE_WEIGHTS` | `[reg:1][key_src:1][_:2]` | `StoreWeights { register, key }` |

**Key encoding:** The `key_src` operand references a hot register containing a key hash (i32). The kernel maps this hash to a thermogram key string. This avoids embedding variable-length strings in the instruction format. The convention:
- P[key_src] contains a pre-loaded i32 key identifier (set by the host during interpreter setup)
- The host maps key identifiers → thermogram key strings via a per-interpreter key table

Alternative: The key could be a param register index where the host pre-loads a key ID at interpreter setup. This is simpler — firmware references `P3` and the host knows P3 means `"frontal.dlpfc.w1"`.

**Decision:** Use param register for key ID. `key_src` is a param register index. The DomainOp carries `key: String` — the lifecycle extension reads P[key_src] as an i32 key_id, and the DomainOp carries that id in a fixed format. The kernel resolves the id to a string.

Updated DomainOp variants (modify existing):

```rust
/// Load weights from persistent storage by key identifier.
LoadWeights { register: Register, key_id: i32 },
/// Store weights to persistent storage by key identifier.
StoreWeights { register: Register, key_id: i32 },
```

**Breaking change note:** `LoadWeights` and `StoreWeights` currently carry `key: String`. Changing to `key_id: i32` is a variant field change. Since `DomainOp` is `#[non_exhaustive]`, this is a breaking change on the variant structure. However, no consumer exists yet (these variants were forward-declared in 1.0, never matched). Safe to change.

### 1.2.5 — Dynamic Register Allocation DomainOp

**File:** `src/vm/extension.rs`

Add DomainOp variant:

```rust
/// Request dynamic register allocation from kernel-managed pool.
/// Host allocates a register of the specified bank and writes the
/// register address into target H[reg][0].
/// bank: 0=Hot, 1=Cold, 2=Param, 3=Shape
/// size: requested capacity (interpretation depends on bank type)
AllocRegister { target: Register, bank: u8, size: u16 },

/// Release a dynamically allocated register back to the pool.
FreeRegister { register: Register },
```

**File:** `src/vm/extensions/arch.rs`

Wire `arch.ALLOC` (opcode 0x0000) to yield `AllocRegister` instead of local allocation. Currently ALLOC does local allocation within the interpreter's existing register array. In 1.2, it yields to the kernel for managed pool allocation.

Add `arch.FREE` opcode:

| Opcode | Mnemonic | Operands | DomainOp |
|--------|----------|----------|----------|
| 0x000B | `FREE` | `[reg:1][_:3]` | `FreeRegister` |

**Behavioral change:** `arch.ALLOC` goes from local to yield. Programs that use ALLOC will now require a host that fulfills AllocRegister. The interpreter's local fallback (for programs running without a host) returns an error. This is acceptable — ALLOC was a stub in 1.0 anyway.

### 1.2.6 — Nested Transactions

**File:** `src/vm/extension.rs`

No DomainOp changes needed — `TxnBegin { txn_id }` already supports multiple txn_ids. Nested transactions are a **host-side concern**: the kernel tracks a stack of transaction buffers.

**File:** `src/vm/extensions/lifecycle.rs`

Document that `TXN_BEGIN` with a different `txn_id` while another transaction is active constitutes nesting. The lifecycle extension does not enforce single-transaction-at-a-time — that was a 1.1 documentation note, not a code restriction.

**Actual change:** Add a note in lifecycle.rs documentation. No code change needed in ternsig — the host decides nesting policy.

### 1.2.7 — Extension Assemble/Disassemble Hooks

**File:** `src/vm/extension.rs`

Add optional methods to `Extension` trait with default implementations:

```rust
pub trait Extension: Send + Sync {
    // ... existing methods ...

    /// Custom assembly for extension-specific syntax.
    /// Called by the assembler when it encounters an unrecognized token
    /// in an extension-qualified context (e.g., `bank.QUERY ...custom_syntax...`).
    /// Returns assembled operand bytes or None to fall back to default parsing.
    fn assemble_operands(&self, _mnemonic: &str, _tokens: &[&str]) -> Option<Result<[u8; 4], String>> {
        None  // Default: use standard operand parsing
    }

    /// Custom disassembly for extension-specific output formatting.
    /// Returns a formatted string or None to fall back to default formatting.
    fn disassemble(&self, _opcode: u16, _operands: [u8; 4]) -> Option<String> {
        None  // Default: use standard disassembly
    }
}
```

These are default-impl trait methods — no breaking change. Extensions that want custom syntax opt in by overriding.

**File:** `src/vm/assembler.rs`

When parsing an extension-qualified instruction (e.g., `bank.BANK_QUERY H0, H1, 0, 4`), the assembler:
1. Resolves the extension by name
2. Finds the instruction meta by mnemonic
3. Calls `ext.assemble_operands(mnemonic, remaining_tokens)`
4. If `Some(Ok(bytes))`, uses those bytes
5. If `None`, falls back to standard operand parsing based on `OperandPattern`

This allows the bank extension to support syntax like:
```
bank.BANK_QUERY H0, H1, slot=0, top_k=4
```
Without requiring the assembler to know about bank-specific semantics.

### 1.2.8 — Version Bump & Changelog

**File:** `Cargo.toml`

Version: `1.1.0` → `1.2.0`

**Changelog entries:**
- NEW: Bank extension (0x000B) with 8 opcodes for distributed memory operations
- NEW: 8 DomainOp variants: BankQuery, BankWrite, BankLoad, BankLink, BankTraverse, BankTouch, BankDelete, BankCount
- NEW: LoadWeights/StoreWeights lifecycle opcodes (0x000B, 0x000C) with key_id encoding
- NEW: AllocRegister/FreeRegister DomainOps for kernel-managed register pools
- NEW: Extension::assemble_operands() and Extension::disassemble() hooks
- CHANGED: LoadWeights/StoreWeights DomainOp key field from String to i32 key_id
- CHANGED: arch.ALLOC yields AllocRegister instead of local allocation
- DOC: Nested transaction support clarified (host-side policy)

---

## Implementation Order

### Step 1: Bank DomainOp variants (1.2.1)
Add 8 new DomainOp variants to `extension.rs`. Compile check.

### Step 2: Bank extension (1.2.2 + 1.2.3)
Create `extensions/bank.rs`. Register in `extensions/mod.rs`. Write tests for metadata and all 8 opcodes yielding correct DomainOps. Compile check + test.

### Step 3: LoadWeights/StoreWeights changes (1.2.4)
Modify existing DomainOp variants (key → key_id). Add 2 lifecycle opcodes. Update lifecycle tests. Compile check + test.

### Step 4: Dynamic register allocation (1.2.5)
Add AllocRegister/FreeRegister DomainOps. Modify arch.ALLOC to yield. Add arch.FREE opcode. Update arch tests. Compile check + test.

### Step 5: Extension hooks (1.2.7)
Add default-impl methods to Extension trait. Modify assembler to call hooks. No breaking change — all existing extensions get default impls. Compile check + test.

### Step 6: Version bump + full test (1.2.8)
Bump version. Run full test suite. Verify astromind-v2 still compiles clean.

---

## Files Modified

| File | Change |
|------|--------|
| `src/vm/extension.rs` | +8 DomainOp variants, modify LoadWeights/StoreWeights, +AllocRegister/FreeRegister, +Extension trait methods |
| `src/vm/extensions/bank.rs` | **NEW** — BankExtension (0x000B), 8 opcodes |
| `src/vm/extensions/mod.rs` | Register bank extension, update counts |
| `src/vm/extensions/lifecycle.rs` | +2 opcodes (LOAD_WEIGHTS, STORE_WEIGHTS) |
| `src/vm/extensions/arch.rs` | Modify ALLOC to yield, add FREE opcode |
| `src/vm/assembler.rs` | Call extension assemble_operands() hook |
| `Cargo.toml` | Version 1.1.0 → 1.2.0 |

---

## Testing Strategy

- **Bank extension tests:** All 8 opcodes yield correct DomainOp variants with correct register/operand mapping
- **Lifecycle tests:** LOAD_WEIGHTS and STORE_WEIGHTS yield LoadWeights/StoreWeights with correct key_id
- **Arch tests:** ALLOC yields AllocRegister, FREE yields FreeRegister
- **Assembler tests:** Extension-qualified mnemonics resolve for bank extension (`bank.BANK_QUERY`)
- **Extension hook test:** Custom assemble_operands returns Some → assembler uses it; returns None → falls back
- **Regression:** All existing 167 tests pass unchanged
- **Integration:** astromind-v2 compiles clean with ternsig 1.2

---

## Deferred to 1.3+

| Feature | Rationale |
|---------|-----------|
| Substrate slice addressing | Requires field_id encoding changes. Evaluate during v3 Phase 5. |
| Co-activation tracking | New field type or DomainOp. Deferred until sleep firmware exists. |
| Architecture thermogram format | Thermogram crate change, not ternsig. |
| Neuron state bank convention | Convention doc, not code. Write when first consumer implements SNN-in-firmware. |
| Cross-language spec (TVMR_SPEC.md) | Documentation deliverable, not code. |
| Meta-learning feedback loop | Needs consumer to implement feedback measurement. |
| Bank extension inline execution | 1.3 scope — simple queries execute locally instead of yielding. |
