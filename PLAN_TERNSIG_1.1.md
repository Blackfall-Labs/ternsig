# PLAN: Ternsig 1.1 — TVMR Runtime Integration

**Date:** 2026-01-29
**Crate:** `ternsig` (`E:\repos\blackfall-labs\ternsig`)
**Branch:** v2
**Base:** v1.0.0 (published on crates.io)
**Status:** Draft — awaiting operator approval

---

## What Ternsig Actually Is

Ternsig is not a brain library. It is a **general-purpose virtual mainframe runtime** that happens to have neuromorphic extensions. The core ISA is pure computation: arithmetic, control flow, register management, data structures. The extensions add domain capabilities: tensor math, ternary operations, learning, architecture manipulation, inter-process communication.

This means Ternsig runs on anything:

| Domain | What Ternsig Does | Evolution? | Learning? |
|--------|-------------------|-----------|-----------|
| **Neuromorphic brains** (Astromind) | Full cognitive runtime with self-modifying neural architecture | Yes — grow/prune/rewire during sleep | Yes — mastery, CHL, RPE |
| **Pacemakers** | Cardiac rhythm firmware with adaptive timing | No | Yes — learns patient-specific rhythm patterns |
| **Security cameras** | Object recognition firmware with hot-reload | No | Yes — learns site-specific patterns |
| **Robotics** | Motor control with adaptive gait | Optional — prune inefficient motor paths | Yes — learns terrain, load, wear |
| **Satellites** | Signal processing with resource-constrained inference | No | Yes — adapts to degrading sensors |
| **Consumer electronics** | Adaptive UI, power management, predictive behavior | No | Yes — learns user patterns |
| **Toys** | Personality firmware that evolves with play | Yes — grows new response patterns | Yes — learns owner preferences |
| **Appliances** | Predictive maintenance, usage optimization | No | Yes — learns usage cycles |

**Evolution is a switch.** Learning is always sound. Resources are always minimum. A pacemaker doesn't need `arch.GROW` — it needs `learning.MASTERY_UPDATE` and `activation.SIGMOID` and a 200-instruction program that fits in 1.6KB of firmware. A brain needs everything. Ternsig serves both because the extension system is modular — you register what you need.

### The Meta-Learning Insight

The evolutionary process itself is Ternsig firmware. That means it is subject to mastery learning.

Consider: a brain's sleep consolidation firmware decides which neurons to prune, which connections to rewire, how aggressively to grow. That decision-making is itself a Ternsig program with learnable weights. Over many sleep cycles, the consolidation firmware gets BETTER at deciding what to keep and what to cut — because its pruning decisions produce measurable outcomes (next-day performance), and mastery learning adjusts the consolidation weights based on those outcomes.

**The brain learns how to learn.** Not as a metaphor — as a literal firmware feedback loop:

```
Wake cycle:
  brain_firmware runs → produces performance metrics

Sleep cycle:
  consolidation_firmware runs → decides prune/grow/rewire
  consolidation produces structural changes

Next wake cycle:
  brain_firmware runs on new structure → produces new metrics

Next sleep cycle:
  consolidation_firmware receives: "last restructure improved/worsened performance"
  mastery learning adjusts consolidation weights
  consolidation makes BETTER structural decisions this time

Repeat:
  consolidation firmware evolves toward optimal restructuring strategy
  the evolutionary process itself gets smarter
```

This is not something to build into 1.1. But 1.1 must not preclude it. The `run_with_host()` contract, the `DomainOp` yield pattern, and the extension registry must be designed knowing that downstream consumers will run firmware that runs other firmware that learns about its own learning.

---

## Context

Ternsig 1.0.0 shipped the TVMR extension system: 10 extensions, 121 instructions, cross-language type system, program validator, TVMR binary format, `.requires` assembler support. All 121 instructions are implemented (120 fully, 1 intentional delegation — `tensor.MATMUL` → `ternary.TERNARY_MATMUL`).

However, 1.0 built the **pieces** without wiring them together. The interpreter cannot dispatch extension instructions. DomainOp yields are swallowed. The registry exists but the interpreter doesn't own one. These are the integration gaps that block any consumer — Astromind, embedded, or otherwise.

Ternsig 1.1 closes every gap between "extensions exist" and "extensions execute."

---

## 1.0 Audit Summary

### What Works (Ship-Quality)

| Component | File | Status |
|-----------|------|--------|
| Extension trait | `vm/extension.rs` | Complete — 5 methods, OperandPattern, InstructionMeta, ExecutionContext, LoopState, StepResult, DomainOp (41 variants) |
| Extension registry | `vm/registry.rs` | Complete — register(), dispatch(), resolve_mnemonic(), resolve_qualified(), duplicate detection |
| Program validator | `vm/validator.rs` | Complete — extension deps, opcode validity, control flow, thermogram keys |
| Type system | `vm/types.rs` | Complete — TypeId(u16), 13 primitives, 3 domain types, legacy bridge |
| Instruction format | `vm/instruction.rs` | Complete — 8-byte [ExtID:2][OpCode:2][Operands:4], legacy bridge, builders |
| Binary format | `vm/binary.rs` | Complete — TVMR v1 48-byte header, extension table, auto-detect legacy/TVMR |
| Assembler | `vm/assembler.rs` | Complete — .requires section, registry-based mnemonic resolution |
| Hot reload | `vm/hot_reload.rs` | Complete — file watching, cold register preservation, debounce |
| 10 extensions | `vm/extensions/*.rs` | 121/121 instructions implemented |
| Core ISA | `vm/interpreter/mod.rs` | ~56 instructions dispatched inline via Action enum |

### What's Broken

| Issue | Severity | File | Details |
|-------|----------|------|---------|
| **Interpreter ignores extensions** | CRITICAL | `interpreter/mod.rs` | `is_extension()` → `StepResult::Error("not yet supported")`. All 121 extension instructions are unreachable at runtime. |
| **DomainOp yields swallowed** | CRITICAL | `interpreter/mod.rs` | `run()` matches `Yield(_) => continue` — yields are consumed, never returned to host. The host-interpreter contract is broken. |
| **Registry not owned by interpreter** | CRITICAL | `interpreter/mod.rs` | `ExtensionRegistry` exists in `registry.rs` but interpreter has no field for it. No dispatch path. |
| **REGION_STATUS yields wrong DomainOp** | BUG | `extensions/orchestration.rs:294` | Opcode 0x000B yields `DomainOp::RegionFire` instead of a status query. |
| **3 DomainOp variants unreferenced** | MINOR | `extension.rs` | `LoadWeights`, `StoreWeights`, `ComputeError` — defined but never yielded by any extension. |
| **DomainOp/StepResult not `#[non_exhaustive]`** | DESIGN DEBT | `extension.rs` | Downstream consumers doing exhaustive match will break on any new variant. |

### What's Missing for Any Consumer

| Feature | Why ANY host needs it |
|---------|----------------------|
| **Extension dispatch in interpreter** | Extensions exist but can't execute — the VM is core-ISA-only |
| **DomainOp yield-to-host contract** | Host needs to fulfill external operations (I/O, persistence, hardware access) |
| **`run_until_yield()` or callback interface** | Host needs to step firmware, handle yields, resume — not just fire-and-forget `run()` |
| **Registry injection into interpreter** | Host chooses which extensions to load (embedded device loads 2, brain loads 10) |

---

## Ternsig 1.1 Scope

### Tier 1: Critical Path (Blocks All Consumers)

#### 1.1.1 — Interpreter Extension Dispatch

**File:** `src/vm/interpreter/mod.rs`

Replace the stub:
```rust
if instr.is_extension() {
    return StepResult::Error(format!("Extension 0x{:04X} not yet supported", ...));
}
```

With registry dispatch:
```rust
if instr.is_extension() {
    let mut ctx = self.build_execution_context();
    return self.registry.dispatch(instr.ext_id, instr.opcode, instr.operands, &mut ctx);
}
```

**Changes:**
- Add `registry: ExtensionRegistry` field to `Interpreter`
- Add `Interpreter::with_registry(registry: ExtensionRegistry) -> Self`
- Add `Interpreter::from_program_with_registry(program, registry) -> Self`
- Implement `build_execution_context(&mut self) -> ExecutionContext<'_>`
- Core ISA (ext_id == 0x0000) remains inline — zero overhead, no regression

**Backward compatibility:** `Interpreter::new()` and `from_program()` create an empty registry. Core-only programs work identically. Extension instructions that hit an empty registry return `StepResult::Error("extension not registered")`.

#### 1.1.2 — DomainOp Yield Contract

**File:** `src/vm/interpreter/mod.rs`

The `run()` method swallows yields. Every host needs yields returned.

**New API:**
```rust
/// Step until a yield, halt, end, or error.
/// Returns the StepResult so the host can handle DomainOp yields.
pub fn run_until_yield(&mut self) -> StepResult {
    loop {
        match self.step() {
            StepResult::Continue => continue,
            StepResult::Break => continue,    // handled by loop stack
            StepResult::Return => continue,   // handled by call stack
            other => return other,            // Yield, Halt, Ended, Error
        }
    }
}

/// Run with a host callback for DomainOp fulfillment.
/// The callback receives the DomainOp, fulfills it (reading/writing registers),
/// and returns true to continue or false to halt.
pub fn run_with_host<F>(&mut self, mut handler: F) -> Result<(), String>
where
    F: FnMut(DomainOp, &mut ExecutionContext) -> Result<bool, String>,
{
    loop {
        match self.step() {
            StepResult::Continue | StepResult::Break | StepResult::Return => continue,
            StepResult::Halt | StepResult::Ended => return Ok(()),
            StepResult::Error(e) => return Err(e),
            StepResult::Yield(op) => {
                let mut ctx = self.build_execution_context();
                match handler(op, &mut ctx) {
                    Ok(true) => continue,
                    Ok(false) => return Ok(()),
                    Err(e) => return Err(e),
                }
            }
        }
    }
}
```

**`run()` unchanged** — keeps swallowing yields for backward compat (existing programs that don't yield still work).

#### 1.1.3 — ExecutionContext Construction

**File:** `src/vm/interpreter/mod.rs`

The interpreter owns all the state that ExecutionContext borrows. Add:

```rust
impl Interpreter {
    fn build_execution_context(&mut self) -> ExecutionContext<'_> {
        ExecutionContext {
            hot_regs: &mut self.hot_regs,
            cold_regs: &mut self.cold_regs,
            param_regs: &mut self.param_regs,
            shape_regs: &mut self.shape_regs,
            pc: &mut self.pc,
            call_stack: &mut self.call_stack,
            loop_stack: &mut self.loop_stack,
            input_buffer: &self.input_buffer,
            output_buffer: &mut self.output_buffer,
            target_buffer: &self.target_buffer,
            chemical_state: &mut self.chemical_state,
            current_error: &mut self.current_error,
            babble_scale: &mut self.babble_scale,
            babble_phase: &mut self.babble_phase,
            pressure_regs: &mut self.pressure_regs,
        }
    }
}
```

**Borrow checker:** Already solved. The interpreter's `step()` does `let instr = self.program[self.pc].clone()` at line 533 BEFORE any mutable work. The instruction is an owned `Instruction` (Copy type, 8 bytes). After cloning, `self.program` is no longer borrowed, so `build_execution_context()` can take `&mut self` freely. No borrow splitting needed, no unsafe, no `InterpreterState` extraction. The existing code pattern handles this cleanly.

### Tier 2: Bug Fixes + Hygiene

#### 1.1.4 — Fix REGION_STATUS DomainOp

**File:** `src/vm/extensions/orchestration.rs:294`

**Current (wrong):**
```rust
0x000B => {
    let reg = Register(operands[0]);
    let region_id = operands[1];
    StepResult::Yield(DomainOp::RegionFire { target: reg, region_id })
}
```

**Fix:** Add `DomainOp::RegionStatus` variant and yield it:

**File:** `src/vm/extension.rs`
```rust
/// Query region status. Host writes into target H[reg][0]: 0=idle, 1=active, 2=firing.
RegionStatus { target: Register, region_id: u8 },
```

**File:** `src/vm/extensions/orchestration.rs`
```rust
0x000B => {
    let reg = Register(operands[0]);
    let region_id = operands[1];
    StepResult::Yield(DomainOp::RegionStatus { target: reg, region_id })
}
```

#### 1.1.5 — `#[non_exhaustive]` on Public Enums

**File:** `src/vm/extension.rs`

Neither `DomainOp` nor `StepResult` is marked `#[non_exhaustive]`. Confirmed by audit — no such attribute exists on either enum.

Adding `#[non_exhaustive]` to both is technically a semver break: any downstream crate doing exhaustive `match` on `DomainOp` or `StepResult` will fail to compile. However:

- **Ternsig 1.0 has been published for less than 24 hours.** No external consumers exist yet.
- **Both enums WILL grow.** `DomainOp` already needs `RegionStatus` in this release. Future releases will add more DomainOps for new extensions, substrate evolution, etc.
- **One break now prevents infinite breaks later.** Every future variant addition would otherwise require a major version bump.

**Decision: Add `#[non_exhaustive]` to both `DomainOp` and `StepResult` in 1.1.** Document in CHANGELOG as the one intentional semver policy change. All downstream `match` statements must add `_ => {}` wildcard arms. This is correct engineering — hosts should handle unknown operations gracefully anyway.

#### 1.1.6 — Clean Up Orphan DomainOp Variants

Three DomainOp variants are defined but never yielded by any extension:

**`LoadWeights { register, key }` and `StoreWeights { register, key }`:**

These are persistence operations. The question is: does firmware need to explicitly request weight loading/saving, or does the host pre-load registers before execution and save after?

**Analysis:** Both patterns exist in real systems:
- **Pre-load pattern** (host-driven): Host loads thermogram weights into cold registers before `run()`. Firmware never knows about persistence. Simple, works for Astromind v3's boot sequence.
- **Yield pattern** (firmware-driven): Firmware explicitly requests `LoadWeights` mid-execution, host fulfills from disk. Works for hot-swap scenarios where a running program needs to pull in weights it didn't start with (e.g., a consolidation program loading another region's weights during sleep replay).

Both patterns are valid. The yield variants should STAY — they enable the firmware-driven pattern. But they don't need new opcodes in 1.1. The lifecycle extension already has `LOAD_THERMO` and `SAVE_THERMO` which yield `LoadThermo`/`SaveThermo`. `LoadWeights`/`StoreWeights` are a separate concept (named key-based access vs. register-based thermogram access). They can be wired to lifecycle opcodes in 1.2 when the semantics are fully defined by a real consumer.

**Decision: Keep `LoadWeights` and `StoreWeights` as-is. No new opcodes. They're forward-declared variants waiting for a consumer to define their semantics.**

**`ComputeError { target, output }`:**

The learning extension's `COMPUTE_ERROR` (opcode 0x000C) computes error locally using `current_error` and `output_buffer`. The DomainOp variant was defined for a host-computed error signal — where the host measures real-world error (e.g., a robot's actual vs. intended position) and writes it back.

**Analysis:** This is a valid pattern for embedded systems. A pacemaker's firmware might yield `ComputeError` to ask the host: "how far off was my timing prediction from the actual heartbeat?" The host has the sensor data, the firmware doesn't.

**Decision: Keep `ComputeError` as-is. It's a valid host-mediated error signal. No changes needed — just document its intended use case.**

### Tier 3: Crash Resistance (VM-Level Persistence Primitives)

The VM runtime owns persistence contracts. Hosts fulfill them, but the APIs, atomicity guarantees, and crash-safe patterns are Ternsig's responsibility. A pacemaker and a brain both need crash resistance — Ternsig provides it once, every consumer benefits.

#### Crash Risk Surface

Every TVMR host has in-memory state at risk:

| State Category | What It Is | Criticality | On Crash |
|---------------|-----------|-------------|----------|
| **Cold registers (weights)** | Learned knowledge, the "genome" | CRITICAL — loss = amnesia | Lost unless persisted via SaveThermo DomainOp |
| **Hot registers (computation)** | In-flight neuron dynamics, working memory | HIGH — loss = restart from program start | Acceptable IF cold registers are current |
| **Chemical state** | Neuromodulator levels (DA, 5HT, NE, etc.) | MEDIUM — regulation re-establishes baselines | Recoverable from thermograms + baseline model |
| **Interpreter state** | PC, call stack, loop stack, I/O buffers | LOW — programs are short-lived per tick | Re-run from program start |
| **In-memory checkpoints** | `CHECKPOINT_WEIGHTS` Mutex storage | **VIOLATION** — lost on crash, violates ASTRO_010 | **Completely lost** — this is the bug |
| **Structural changes** | Mid-sleep prune/grow/rewire operations | CRITICAL — partial changes = inconsistent brain | **Most dangerous** — half-pruned state |

**The two critical gaps:**
1. `CHECKPOINT_WEIGHTS` stores checkpoints in a volatile `Mutex<Vec<Option<ColdBuffer>>>`. Process crash = all checkpoints gone. Firmware that checkpointed before a risky learning operation has no safety net.
2. Multi-step structural operations (sleep consolidation: prune → grow → rewire → save) have no atomicity. Crash mid-sequence leaves the system in an inconsistent state that no single thermogram represents.

#### 1.1.9 — Persistent Checkpoint DomainOps

**Files:** `src/vm/extension.rs`, `src/vm/extensions/learning.rs`

`CHECKPOINT_WEIGHTS` currently stores in a volatile Mutex. Fix: yield a DomainOp so the host persists the checkpoint to disk.

**New DomainOp variants:**
```rust
/// Persist a checkpoint of cold register weights.
/// Host writes to crash-safe storage (temp+rename or WAL).
/// key format: "{thermogram_key}.checkpoint"
CheckpointSave { register: Register },

/// Restore cold register from persisted checkpoint.
/// Host reads from crash-safe storage. Returns error if no checkpoint exists.
CheckpointRestore { register: Register },

/// Discard a persisted checkpoint (cleanup after successful operation).
CheckpointDiscard { register: Register },
```

**Changes to learning.rs:**
- `CHECKPOINT_WEIGHTS` (0x0011): Keep in-memory Mutex for fast access, AND yield `DomainOp::CheckpointSave` for host persistence. The Mutex is a cache; the DomainOp write is the source of truth.
- `ROLLBACK_WEIGHTS` (0x0012): Try in-memory Mutex first (fast path). If empty (post-crash), yield `DomainOp::CheckpointRestore` (host reads from disk).
- New opcode `DISCARD_CHECKPOINT` (0x0013): Yield `DomainOp::CheckpointDiscard`. Called after a successful learning operation completes — tells the host the safety net is no longer needed.

**Host contract:**
- `CheckpointSave` MUST be atomic (temp file + rename, or WAL entry)
- `CheckpointRestore` MUST detect corruption (CRC32 or equivalent)
- `CheckpointDiscard` SHOULD delete the checkpoint file/entry

**Backward compatibility:** Programs that never call `CHECKPOINT_WEIGHTS` are unaffected. Programs that do now get crash-safe checkpoints instead of volatile ones. Existing `run()` swallows the yield — core-only hosts that don't handle DomainOps lose nothing (they already lost checkpoints on crash anyway).

#### 1.1.10 — Transaction DomainOps

**File:** `src/vm/extension.rs`

Sleep consolidation involves multiple structural changes that MUST be atomic: prune neurons, grow replacements, rewire connections, save thermograms. If the host crashes mid-sequence, the brain wakes with a half-pruned, half-rewired architecture.

**New DomainOp variants:**
```rust
/// Begin a persistence transaction. All subsequent SaveThermo/CheckpointSave
/// operations are buffered until TxnCommit or discarded on TxnRollback.
/// The host MUST NOT write any buffered data to disk until TxnCommit.
TxnBegin { txn_id: u8 },

/// Commit all buffered writes in the transaction atomically.
/// The host writes ALL buffered data, then marks the transaction complete.
/// If the host crashes during commit, it MUST either complete ALL writes
/// or roll back ALL writes on next boot (write-ahead log pattern).
TxnCommit { txn_id: u8 },

/// Discard all buffered writes in the transaction.
/// Called when firmware decides a structural change failed or is unsafe.
TxnRollback { txn_id: u8 },
```

**How firmware uses it:**
```ternsig
; Sleep consolidation — atomic structural change
lifecycle.PHASE_READ  H0          ; check we're in sleep mode
neuro.CHEM_READ       H1, 3      ; check GABA (sleep chemical) is high

; Begin atomic transaction
lifecycle.TXN_BEGIN   0           ; txn_id = 0

; Pruning pass
arch.PRUNE   C0, H2              ; remove low-activity neurons
arch.PRUNE   C1, H2
lifecycle.SAVE_THERMO C0          ; buffered, not yet on disk
lifecycle.SAVE_THERMO C1          ; buffered, not yet on disk

; Rewiring pass
arch.WIRE_FORWARD C0, C1         ; reconnect pruned topology
lifecycle.SAVE_THERMO C0          ; buffered
lifecycle.SAVE_THERMO C1          ; buffered

; Commit everything atomically
lifecycle.TXN_COMMIT  0           ; ALL writes go to disk together
```

**If the host crashes between TXN_BEGIN and TXN_COMMIT:** Nothing was written to disk. The brain boots from pre-consolidation thermograms. Consolidation retries next sleep cycle. No inconsistent state.

**If the host crashes DURING TXN_COMMIT:** The host's WAL contains the full transaction. On next boot, the host replays the WAL to completion (write-ahead log guarantee) or rolls back entirely. Either way: consistent state.

**New lifecycle extension opcodes:**
- `TXN_BEGIN` (0x0008): Yield `DomainOp::TxnBegin { txn_id }`
- `TXN_COMMIT` (0x0009): Yield `DomainOp::TxnCommit { txn_id }`
- `TXN_ROLLBACK` (0x000A): Yield `DomainOp::TxnRollback { txn_id }`

**Host contract:**
- Transactions are identified by `txn_id` (0-255). Nesting is NOT supported in 1.1. One transaction at a time per txn_id.
- Between `TxnBegin` and `TxnCommit`, all `SaveThermo` and `CheckpointSave` DomainOps for that txn_id are buffered in memory.
- `TxnCommit` writes ALL buffered data atomically (WAL-based or temp-dir-rename).
- `TxnRollback` discards all buffered data.
- If the host process restarts and finds an incomplete transaction in the WAL, it MUST roll back (discard buffered writes). Only complete TxnCommit entries are replayed.

#### 1.1.11 — Interpreter State Snapshot API

**File:** `src/vm/interpreter/mod.rs`

The host needs to periodically snapshot interpreter state for crash recovery. Ternsig provides the serialization, the host decides when and where to persist.

**New API:**
```rust
/// Complete snapshot of interpreter state, serializable for persistence.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct InterpreterSnapshot {
    /// All hot register contents.
    pub hot_regs: Vec<Option<HotBuffer>>,
    /// All cold register contents (weights + metadata).
    pub cold_regs: Vec<Option<ColdBuffer>>,
    /// Parameter registers.
    pub param_regs: Vec<i32>,
    /// Shape registers.
    pub shape_regs: Vec<Vec<usize>>,
    /// Chemical state.
    pub chemical_state: ChemicalState,
    /// Learning state.
    pub current_error: i32,
    pub babble_scale: i32,
    pub babble_phase: usize,
    /// Output buffer (last output).
    pub output_buffer: Vec<i32>,
    /// Pressure registers (mastery learning).
    pub pressure_regs: Vec<Option<Vec<i32>>>,
    /// Snapshot metadata.
    pub program_name: Option<String>,
    pub checksum: u32,
}

impl Interpreter {
    /// Capture a snapshot of all interpreter state.
    /// Does NOT include program code (the host knows which program was loaded).
    /// Does NOT include PC/call stack/loop stack (ephemeral execution state).
    pub fn snapshot(&self) -> InterpreterSnapshot { ... }

    /// Restore interpreter state from a snapshot.
    /// Program must already be loaded. Execution restarts from program start
    /// with the restored register state.
    pub fn restore(&mut self, snapshot: &InterpreterSnapshot) -> Result<(), String> { ... }
}
```

**What's captured vs. not:**

| Captured | Why |
|----------|-----|
| Hot registers | Neuron dynamics state (membrane potentials, recovery variables) |
| Cold registers | Learned weights (most critical) |
| Param/shape registers | Network configuration |
| Chemical state | Neuromodulator levels |
| Learning state | Error tracking, babble phase |
| Pressure registers | Mastery learning pressure |
| Output buffer | Last output for continuity |

| NOT captured | Why |
|-------------|-----|
| Program counter | Ephemeral — re-run from start |
| Call/loop stacks | Ephemeral — re-run from start |
| Input buffer | External — host provides on next tick |
| Program code | Host knows which program is loaded |

**Checksum:** CRC32 over serialized content. Verified on `restore()`.

**Host usage pattern:**
```rust
// Periodic snapshot (e.g., every 1000 ticks)
let snap = interpreter.snapshot();
let bytes = serde_json::to_vec(&snap)?;  // or bincode, or custom
atomic_write("state/region_X.snapshot", &bytes)?;

// On crash recovery
let bytes = std::fs::read("state/region_X.snapshot")?;
let snap: InterpreterSnapshot = serde_json::from_slice(&bytes)?;
interpreter.restore(&snap)?;
// interpreter.run_with_host(...) — resumes with restored state
```

### Tier 4: API Completeness

#### 1.1.7 — Default Registry with Standard Extensions

**File:** `src/vm/registry.rs`

Add a constructor that pre-registers all 10 standard extensions:

```rust
impl ExtensionRegistry {
    /// Create a registry pre-loaded with all standard TVMR extensions.
    pub fn with_standard_extensions() -> Self {
        let mut reg = Self::new();
        reg.register(Box::new(TensorExtension)).unwrap();
        reg.register(Box::new(TernaryExtension)).unwrap();
        reg.register(Box::new(ActivationExtension)).unwrap();
        reg.register(Box::new(LearningExtension)).unwrap();
        reg.register(Box::new(NeuroExtension)).unwrap();
        reg.register(Box::new(ArchExtension)).unwrap();
        reg.register(Box::new(OrchestrationExtension)).unwrap();
        reg.register(Box::new(LifecycleExtension)).unwrap();
        reg.register(Box::new(IpcExtension)).unwrap();
        reg.register(Box::new(TestExtension)).unwrap();
        reg
    }
}
```

A brain loads all 10. A pacemaker loads 3 (activation, learning, lifecycle). The registry is the extension selector.

#### 1.1.8 — ReloadableInterpreter Registry Support

**File:** `src/vm/hot_reload.rs`

`ReloadableInterpreter` creates an `Interpreter` internally. It needs to pass through the registry:

```rust
impl ReloadableInterpreter {
    pub fn from_file_with_registry(path: impl AsRef<Path>, registry: ExtensionRegistry) -> Result<Self> { ... }
}
```

The existing `from_file()` continues to work (empty registry, core-only programs).

### Tier 5: Deferred (1.2+)

These are NOT in 1.1 scope. Documented here for traceability.

| Feature | Target | Rationale |
|---------|--------|-----------|
| `Extension::assemble()` | 1.2 | Per-extension custom assembly syntax. All firmware currently uses the global assembler. |
| `Extension::disassemble()` | 1.2 | Per-extension disassembly for debugging. Nice-to-have, not blocking. |
| Dynamic register allocation DomainOp | 1.2 | `arch.ALLOC` yielding for kernel-managed register pools. Currently ALLOC is local. |
| Substrate slice addressing | 1.2 | Field IDs as (field_type, slice_index) pairs. Requires DomainOp variant changes. |
| Co-activation tracking field | 1.2 | Hebbian rewiring during sleep. New DomainOp or field type. |
| Architecture thermogram format | 1.2 | Extended thermogram payloads for topology metadata. Thermogram crate change, not ternsig. |
| Neuron state bank convention | 1.2 | SNN dynamics bank layout in hot registers. Convention doc, not code change. |
| `LoadWeights`/`StoreWeights` lifecycle opcodes | 1.2 | Wire persistence DomainOps to lifecycle extension when a consumer defines semantics. |
| Meta-learning feedback loop | 1.2+ | Consolidation firmware that learns from its own restructuring outcomes. Architecture is ready — just needs a consumer to implement the feedback measurement. |
| Cross-language spec (`TVMR_SPEC.md`) | 1.2+ | Language-agnostic contract for C/Python/JS implementations. |
| Nested transactions | 1.2 | `TxnBegin` within `TxnBegin` for sub-operations. 1.1 enforces one txn per txn_id. |
| Atomic thermogram-rs writes | 1.1 (upstream) | `thermogram-rs` uses bare `fs::write()` — needs temp+rename+fsync. Separate crate fix, not ternsig. Ternsig's DomainOp contracts ASSUME the host writes atomically; this fix ensures the thermogram crate delivers on that. |

---

## File Change Summary

### Modified Files

| File | Changes |
|------|---------|
| `src/vm/interpreter/mod.rs` | Add `registry` field, `with_registry()`, `from_program_with_registry()`, `build_execution_context()`, `run_until_yield()`, `run_with_host()`, `snapshot()`, `restore()`. Wire `is_extension()` to `registry.dispatch()`. Add `InterpreterSnapshot` struct. |
| `src/vm/extension.rs` | Add `#[non_exhaustive]` to `DomainOp` and `StepResult`. Add `DomainOp::RegionStatus`, `CheckpointSave`, `CheckpointRestore`, `CheckpointDiscard`, `TxnBegin`, `TxnCommit`, `TxnRollback` variants (+7 total). |
| `src/vm/extensions/orchestration.rs` | Fix REGION_STATUS (opcode 0x000B) to yield `RegionStatus` instead of `RegionFire`. |
| `src/vm/extensions/learning.rs` | `CHECKPOINT_WEIGHTS` (0x0011) now also yields `CheckpointSave`. `ROLLBACK_WEIGHTS` (0x0012) falls back to `CheckpointRestore` yield. Add `DISCARD_CHECKPOINT` (0x0013) yielding `CheckpointDiscard`. |
| `src/vm/extensions/lifecycle.rs` | Add `TXN_BEGIN` (0x0008), `TXN_COMMIT` (0x0009), `TXN_ROLLBACK` (0x000A) opcodes. |
| `src/vm/registry.rs` | Add `with_standard_extensions()`. |
| `src/vm/hot_reload.rs` | Add `from_file_with_registry()` to `ReloadableInterpreter`. |
| `Cargo.toml` | Version bump 1.0.0 → 1.1.0. |

### New Files

None.

### Deleted Files

None.

---

## Breaking Changes

**Semver analysis: 1.1.0 is technically a MINOR bump with one intentional policy break.**

| Change | Breaking? | Mitigation |
|--------|-----------|------------|
| New methods on Interpreter | No | Additive |
| New DomainOp variants (+7) | No (after `#[non_exhaustive]`) | Wildcard match arms |
| `#[non_exhaustive]` on DomainOp | **Yes** | One-time break. Downstream adds `_ => {}`. Prevents all future breaks. |
| `#[non_exhaustive]` on StepResult | **Yes** | Same as above. |
| New learning opcode (DISCARD_CHECKPOINT) | No | Additive opcode 0x0013 |
| New lifecycle opcodes (TXN_*) | No | Additive opcodes 0x0008-0x000A |
| InterpreterSnapshot struct | No | New public type |
| snapshot()/restore() on Interpreter | No | Additive methods |
| New registry constructors | No | Additive |
| Existing run()/step()/new() | No | Behavior unchanged |
| CHECKPOINT_WEIGHTS now yields DomainOp | **Behavioral** | Previously in-memory only. Now also yields `CheckpointSave`. Hosts using `run()` (swallows yields) see no change. Hosts using `run_with_host()` receive the new yield. |

**Published <24 hours ago. No known external consumers. This is the window to make the `#[non_exhaustive]` change cleanly.**

---

## Validation Plan

### Unit Tests

| Test | What It Validates |
|------|-------------------|
| Extension dispatch smoke test | Interpreter with registry executes `activation.RELU` correctly |
| DomainOp yield test | `run_until_yield()` returns `Yield(ChemRead)` for neuro instruction |
| `run_with_host()` callback test | Host handler receives DomainOp, writes result, execution resumes |
| Empty registry fallback | Extension instruction with no registry returns clear error |
| Standard extensions constructor | `with_standard_extensions()` registers all 10, no warnings |
| REGION_STATUS fix | Opcode 0x000B yields `RegionStatus`, not `RegionFire` |
| Hot reload with registry | `ReloadableInterpreter` preserves registry across reloads |
| Backward compat: core-only | Existing `Interpreter::new()` + core ISA programs work identically |

### Crash Resistance Tests

| Test | What It Validates |
|------|-------------------|
| Checkpoint save yield | `CHECKPOINT_WEIGHTS` yields `CheckpointSave` DomainOp to host |
| Checkpoint restore yield | `ROLLBACK_WEIGHTS` with empty Mutex yields `CheckpointRestore` DomainOp |
| Discard checkpoint | `DISCARD_CHECKPOINT` yields `CheckpointDiscard` DomainOp |
| Transaction begin/commit | `TXN_BEGIN` + `SAVE_THERMO` + `TXN_COMMIT` yields ops in order |
| Transaction rollback | `TXN_BEGIN` + `SAVE_THERMO` + `TXN_ROLLBACK` yields rollback |
| Snapshot round-trip | `snapshot()` → serialize → deserialize → `restore()` → identical register state |
| Snapshot checksum | Tampered snapshot bytes → `restore()` returns checksum error |
| Snapshot preserves learning state | Chemical state, pressure regs, babble phase survive snapshot round-trip |

### Integration Tests

| Test | What It Validates |
|------|-------------------|
| Full extension program | Assemble a program with `.requires`, execute through interpreter with registry, verify outputs |
| Yield round-trip | Program yields ChemRead → host writes value → program reads it → correct output |
| Mixed core + extension | Program uses both core ISA and extension instructions in sequence |
| Validator + executor | Validate program, then execute — no runtime errors for valid programs |
| Transaction + yield | Program uses TXN_BEGIN, multiple SaveThermo yields, TXN_COMMIT — host receives all ops in order |

### Regression

```bash
cargo test -p ternsig           # All existing 124 tests pass
cargo check -p ternsig          # Clean compile
cargo check -p astromind-v2     # v2 consumer still compiles
```

---

## Implementation Order

1. **1.1.5** — `#[non_exhaustive]` on DomainOp + StepResult (do first — everything downstream depends on match arms)
2. **1.1.3** — `build_execution_context()` (prerequisite for dispatch and yield)
3. **1.1.1** — Interpreter extension dispatch (the critical path)
4. **1.1.2** — `run_until_yield()` + `run_with_host()` (DomainOp contract)
5. **1.1.4** — Fix REGION_STATUS bug + add DomainOp::RegionStatus
6. **1.1.9** — Persistent checkpoint DomainOps (checkpoint/restore/discard variants + learning.rs changes)
7. **1.1.10** — Transaction DomainOps (TxnBegin/Commit/Rollback variants + lifecycle.rs opcodes)
8. **1.1.11** — InterpreterSnapshot API (snapshot/restore on interpreter)
9. **1.1.7** — `with_standard_extensions()` convenience
10. **1.1.8** — ReloadableInterpreter registry pass-through
11. Tests, version bump, CHANGELOG, `cargo test`, tag v1.1.0

---

## TVMR Phase Status

The original TVMR plan defined 6 phases:

| Phase | Status in 1.0 | Status After 1.1 |
|-------|---------------|-------------------|
| Phase 1: Core Infrastructure | COMPLETE | No changes |
| Phase 2: Standard Extensions | COMPLETE | Bug fix (REGION_STATUS) |
| Phase 3: Assembler & Validation | COMPLETE | No changes |
| Phase 4: Binary Format & Loader | COMPLETE | No changes |
| Phase 5: Consumer Integration | **INCOMPLETE** | **1.1 DELIVERS THIS** |
| Phase 6: Cross-Language Spec | NOT STARTED | Deferred (1.2+) |

**1.1 is Phase 5.** The interpreter becomes a real TVMR runtime that any host can embed.

### Instruction Count

| Category | 1.0 Count | 1.1 Delta | 1.1 Total |
|----------|-----------|-----------|-----------|
| Core ISA | ~56 | 0 | ~56 |
| Standard extensions | 121 | +4 (DISCARD_CHECKPOINT, TXN_BEGIN, TXN_COMMIT, TXN_ROLLBACK) | 125 |
| DomainOp variants | 41 | +7 (RegionStatus, CheckpointSave/Restore/Discard, TxnBegin/Commit/Rollback) | 48 |
| **Total instructions** | **~177** | **+4** | **~181** |

### Crash Resistance Coverage

| Risk | Ternsig 1.1 Mitigation | Host Responsibility |
|------|------------------------|---------------------|
| Volatile checkpoints | `CheckpointSave` DomainOp → host persists to disk | Atomic write (temp+rename+fsync) |
| Post-crash rollback | `CheckpointRestore` DomainOp → host reads from disk | CRC32 verification on load |
| Partial structural change | `TxnBegin`/`TxnCommit` → host buffers writes | WAL-based atomic commit |
| Interpreter state loss | `snapshot()`/`restore()` API | Periodic snapshot persistence |
| Thermogram corruption | DomainOp contract specifies atomicity | thermogram-rs atomic write fix (upstream) |
