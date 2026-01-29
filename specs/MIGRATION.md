# TVMR Migration Plan: Rust to Programs

## Principle

The Rust code is the **kernel**. Everything else becomes a `.ternsig` program.

The goal is not abstraction for its own sake. The goal is:
- **Hot-reload** instead of recompile (dev cycle: seconds, not minutes)
- **Self-descriptive programs** (the `.ternsig` file IS the documentation)
- **Cheaper iteration** (change behavior without touching Rust)
- **Faster testing** (load program, run, assert — no cargo test overhead)
- **Cheaper tokens** (Claude edits `.ternsig` files, not Rust modules)

## What Stays Rust (The Kernel)

These are the **irreducible** Rust components. They cannot be programs because they ARE the machine:

| Component | Why It Stays |
|-----------|-------------|
| `vm/instruction.rs` | IS the instruction format |
| `vm/interpreter/` | IS the execution engine |
| `vm/extension.rs` | IS the extension interface |
| `vm/registry.rs` | IS the dispatch table |
| `vm/assembler.rs` | IS the compiler |
| `vm/binary.rs` | IS the serializer |
| `vm/hot_reload.rs` | IS the reload engine |
| `ternary.rs` (Signal type) | IS the data type |
| `thermo.rs` (Thermogram bridge) | IS the persistence layer |
| `learning.rs` (mastery core) | IS the learning algorithm kernel |
| Brain boot sequence | IS the startup (calls kernel APIs) |
| Substrate bridges | IS the I/O layer (field reads, chemical reads) |
| Spool/telemetry | IS the observation layer |

Everything else — forward passes, learning rules, gating logic, routing, wiring, region behavior — becomes programs.

## What Becomes Programs

### Tier 1: Already Programs (140+ firmware files exist)

These are the existing `.ternsig` firmware files in astromind-v2. They already run as programs. No migration needed — just add `.requires` headers when reassembling.

- Region forward passes
- Weight layer definitions
- Basic learning sequences

### Tier 2: Region Behavior (the big win)

Every brain region in `astromind-v2/src/regions/` currently has a `region.rs` with a Rust `step()` method. Most of that logic is:

1. Read input from hot registers
2. Run forward pass through firmware
3. Apply gating/routing decisions
4. Write output

Steps 1-3 can be expressed as `.ternsig` programs using existing + new instructions.

**What needs to happen:**
- Implement **orchestration** extension (MODEL_LOAD, MODEL_EXEC, MODEL_CHAIN) — lets programs call other programs
- Implement **neuro** extension (CHEM_READ, FIELD_READ) — lets programs read substrate state
- Implement **lifecycle** extension (PHASE_READ, LEVEL_READ) — lets programs know boot context
- Implement **ipc** extension (SEND_SIGNAL, RECV_SIGNAL) — lets programs talk to each other

**Per-region migration pattern:**

```
BEFORE (Rust):
  fn step(&mut self, substrate: &mut Substrate) {
      let da = substrate.chemicals.dopamine;
      let input = self.read_hot(0);
      let output = self.model.forward(input);
      if da > 30 { self.learn(output, target); }
      self.write_hot(1, output);
  }

AFTER (.ternsig):
  .requires
      ternary 0x0002
      activation 0x0003
      learning 0x0004
      neuro 0x0005

  .registers
      C0: ternary[32, 12] key="region.w1"
      H0: i32[12]
      H1: i32[32]
      H2: i32[1]

  .program
      load_input H0
      ternary_matmul H1, C0, H0
      activation.relu H1, H1
      neuro.chem_read H2, 0         ; read dopamine
      learning.mastery_update C0, H1, H0
      learning.mastery_commit C0
      store_output H1
      halt
```

The Rust `region.rs` shrinks to:
```rust
fn step(&mut self, substrate: &mut Substrate) {
    self.interpreter.set_substrate(substrate);
    self.interpreter.run();
}
```

**Regions to migrate (ordered by complexity):**

| Region | Complexity | Models | Dependencies |
|--------|-----------|--------|-------------|
| Representation | Low | 1-2 | ternary, activation |
| Spatial | Low | 1-2 | ternary, activation |
| Personality | Medium | 3-4 | ternary, activation, learning, neuro |
| Language | Medium | 3-5 | ternary, activation, learning |
| Planning | High | 4-6 | ternary, activation, learning, neuro, ipc |
| Reasoning | High | 4-6 | ternary, activation, learning, orchestration |
| Regulation | High | 2-3 | neuro (chemical authority) |
| Dialog | High | 5+ | ternary, activation, learning, ipc, orchestration |
| Hippocampus | Very High | 6+ | All extensions |
| Orchestrator | Very High | N/A | orchestration, ipc, lifecycle |

### Tier 3: Boot Sequences

The boot sequence (`boot.rs`) creates regions, registers thermograms, validates preflight. Parts of this can become a "boot program":

```asm
; boot_topology.ternsig
.requires
    arch 0x0006
    lifecycle 0x0008

.program
    ; Define region topology as program
    arch.alloc_tensor C0, 32, 12    ; representation weights
    arch.alloc_tensor C1, 64, 32    ; language weights
    arch.wire_forward H1, C0, H0   ; representation -> language
    lifecycle.init_thermo C0
    lifecycle.init_thermo C1
    halt
```

**Prerequisite:** Implement `arch` and `lifecycle` stubs.

### Tier 4: Test Harnesses

Currently tests boot the brain and observe. With the `test` extension:

```asm
; test_representation.ternsig
.requires
    ternary 0x0002
    activation 0x0003
    test 0x000A

.program
    test.test_begin 1
    load_input H0
    ternary_matmul H1, C0, H0
    activation.relu H1, H1
    test.assert_active H1
    test.assert_range H1, 0, 255
    test.test_end
    halt
```

**Prerequisite:** Implement `test` extension stubs.

### Tier 5: Evolutionary Background

Self-modification programs that run on slow timescales:

```asm
; evolve_prune.ternsig — runs every N ticks
.requires
    arch 0x0006
    neuro 0x0005
    lifecycle 0x0008

.program
    neuro.temp_read H0, C0         ; read temperature field
    ; find coldest neuron
    tensor.argmax H1, H0           ; index of coldest
    ; prune if below threshold
    arch.prune_neuron C0, H1
    halt
```

## Implementation Order

Each step is a complete, shippable unit. No step depends on future steps.

### Step 1: Implement All Extension Stubs (71 instructions)

ASTRO_011 demands it. No stubs in production code. Priority order:

1. **test** (8 ops) — enables program-level testing immediately
2. **neuro** (8 ops) — requires SubstrateHandle trait; unlocks substrate I/O
3. **lifecycle** (8 ops) — boot phase awareness
4. **orchestration** (8 ops) — model table; unlocks multi-model programs
5. **ipc** (8 ops) — inter-region signaling
6. **learning** remaining (16 ops) — CHL, eligibility, consolidation
7. **ternary** remaining (7 ops) — quantize, pack/unpack, pressure
8. **arch** remaining (4 ops) — define_layer, freeze/unfreeze
9. **tensor** remaining (4 ops) — squeeze, unsqueeze, transpose, matmul

### Step 2: Wire Extension Dispatch in Interpreter

Currently the interpreter dispatches core ISA inline via `step()`. Extension instructions with `ext_id != 0` need to route through `ExtensionRegistry::dispatch()`. This is the Phase 5 consumer update.

### Step 3: SubstrateHandle Trait

Bridge between TVMR programs and the astromind substrate:

```rust
pub trait SubstrateHandle: Send + Sync {
    fn read_chemical(&self, chem_id: u8) -> i32;
    fn write_chemical(&mut self, chem_id: u8, value: i32);
    fn read_field(&self, field_id: u8, index: usize) -> i32;
    fn write_field(&mut self, field_id: u8, index: usize, value: i32);
    fn read_convergence(&self) -> i32;
    fn read_stimulation(&self, stim_id: u8) -> i32;
    fn read_temperature(&self, cold_reg: u8, index: usize) -> i32;
    fn boot_phase(&self) -> u8;
    fn tick_count(&self) -> u64;
    fn neuronal_level(&self) -> u8;
}
```

Implemented by astromind-v2 for each region. Passed to neuro/lifecycle extensions via ExecutionContext.

### Step 4: Model Table

Orchestration extension needs a model table — a runtime registry of loaded programs that can be invoked by slot index:

```rust
pub struct ModelTable {
    slots: Vec<Option<LoadedModel>>,
}

pub struct LoadedModel {
    program: AssembledProgram,
    interpreter: Interpreter,
}
```

This lets programs call other programs. A region's "top-level" program becomes an orchestrator:

```asm
orchestration.model_load H0, 0     ; load forward model into slot 0
orchestration.model_load H0, 1     ; load learning model into slot 1
orchestration.model_input H0, 0    ; set input for slot 0
orchestration.model_exec H0, 0     ; run forward pass
orchestration.model_output H1, 0   ; get output
orchestration.model_chain 0, 1     ; chain forward -> learning
```

### Step 5: Migrate Regions Incrementally

Start with the simplest regions. Each migration:
1. Write `.ternsig` program that replaces the Rust `step()` logic
2. Reduce Rust `step()` to interpreter dispatch
3. Add `.requires` headers
4. Verify with observation test (brain still boots, region still activates)

### Step 6: Hot-Reload in Dev Loop

Once regions are programs, the dev loop becomes:
1. Edit `.ternsig` file
2. File watcher detects change
3. Hot-reload engine reassembles and swaps
4. Brain behavior changes without restart

## Metrics

Track these to prove the migration is working:

| Metric | Before | After |
|--------|--------|-------|
| Rust LoC in regions/ | ~5000 | ~500 (just interpreter dispatch) |
| .ternsig program count | ~140 | ~300+ |
| Time to change region behavior | cargo build (~30s) | hot-reload (~100ms) |
| Test cycle | cargo test (~10s) | load + run program (~1ms) |

## Non-Goals

- **Not replacing the kernel.** Rust stays for the machine itself.
- **Not a general-purpose language.** TVMR is a neural architecture VM, not Python.
- **Not removing all Rust.** Substrate bridges, persistence, telemetry stay Rust.
- **Not breaking existing firmware.** All 140+ programs continue working.
