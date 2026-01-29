# TVMR Layered Safety Architecture

Five layers. Each layer has a privilege boundary. Higher layers cannot reach into lower layers. Lower layers are unaware of higher layers.

This is biological: your neocortex cannot rewrite your brainstem.

---

## Layer 0: TVMR Kernel (The Machine)

**Privilege:** Unrestricted. IS the execution engine.

**What lives here:**
- Instruction dispatch (`vm/interpreter/`)
- Extension registry (`vm/registry.rs`)
- Instruction format (`vm/instruction.rs`)
- Binary serializer (`vm/binary.rs`)
- Assembler (`vm/assembler.rs`)
- Hot-reload engine (`vm/hot_reload.rs`)
- Type system (`vm/types.rs`)
- Signal type (`ternary.rs`)
- Thermogram bridge (`thermo.rs`)
- Mastery learning kernel (`learning.rs`)

**Lifecycle:** Compiled once. Never modified at runtime. Never modified by programs.

**Invariant:** If the kernel is correct, programs cannot corrupt the machine. Programs execute within the kernel's rules — they cannot escape them.

**Analogy:** The CPU. Fixed instruction set etched in silicon. Programs run ON it, not IN it.

---

## Layer 1: Boot (The Topology)

**Privilege:** Creates regions, allocates thermograms, defines wiring. Runs once. Frozen after boot.

**What lives here:**
- Region topology definition (which regions exist)
- Thermogram registration (which weight matrices persist)
- Inter-region wiring (who connects to whom)
- Extension registration (which extensions are available)
- Preflight validation (all paths exist, all thermograms valid)

**Implemented by:**
- `boot.rs` in astromind-v2 (currently Rust, future: boot programs via `arch` + `lifecycle` extensions)
- Programs using: `arch.alloc_tensor`, `arch.wire_forward`, `lifecycle.init_thermo`

**Lifecycle:** Executes during brain startup. Once boot completes, the topology is frozen. No new regions can be created. No new thermograms can be registered.

**Invariant:** The set of regions and their connections is fixed after boot. Higher layers cannot add or remove regions.

**Safety boundary:** Layer 2 (Evolutionary) can modify neurons WITHIN existing regions but cannot create new regions or destroy existing ones.

---

## Layer 2: Evolutionary Background (Structural Plasticity)

**Privilege:** Grow/prune neurons within existing regions. Modify architecture on slow timescales. Bounded by Layer 1 topology.

**What lives here:**
- Neurogenesis (adding neurons to existing cold registers)
- Pruning (removing inactive neurons)
- Temperature-based lifecycle (HOT → WARM → COOL → COLD)
- Structural optimization (slow-timescale architecture search)

**Implemented by:**
- Programs using: `arch.grow_neuron`, `arch.prune_neuron`, `neuro.temp_read`
- Runs every N ticks (not every tick — evolutionary timescale)

**Lifecycle:** Active during runtime but on slow timescales. Changes are journaled. Convergence failure triggers rollback.

**Invariants:**
1. **Cannot exceed Layer 1 bounds.** If boot defined region X with max capacity M, evolutionary layer cannot grow beyond M.
2. **Cannot modify Layer 1 topology.** Cannot add regions, remove regions, or rewire inter-region connections.
3. **Journaled + rollback.** Every structural change is logged. If convergence field coherence drops below threshold after a structural change, the change is rolled back automatically.
4. **Temperature governs plasticity.** Only HOT neurons can be pruned or modified. COLD neurons are frozen — they represent proven structure.

**Safety boundary:** Layer 3 (Runtime) reads structure but cannot modify it. Forward passes use whatever architecture exists; they don't change it.

**Analogy:** Developmental plasticity in the brain. The brain can grow new synapses and prune unused ones, but it cannot grow a new lobe.

---

## Layer 3: Runtime (Forward Passes)

**Privilege:** Read weights. Compute activations. Route signals. Read-only access to structure.

**What lives here:**
- Forward passes (ternary matmul, activation functions)
- Gating logic (chemical-modulated routing)
- Sensory input processing
- Inter-region signal routing
- Output generation

**Implemented by:**
- Region firmware (`.ternsig` programs)
- Programs using: `ternary.*`, `activation.*`, `tensor.*`, `neuro.chem_read`, `neuro.field_read`

**Lifecycle:** Runs every tick. Deterministic given inputs + weights + chemicals. Produces activations and outputs.

**Invariants:**
1. **Read-only structure.** Cannot grow, prune, or rewire.
2. **Read-only weights.** Forward pass does not modify cold registers. Weight reads are pure.
3. **Chemical reads are observational.** Runtime reads dopamine level to modulate gating but does not set dopamine.
4. **Output is activations.** Hot registers hold activations. They are volatile (reset each tick or overwritten).

**Safety boundary:** Layer 4 (Learning) is the only layer that can modify weights based on runtime activations.

**Analogy:** Neural firing. Signals propagate through existing connections. The connections don't change during propagation.

---

## Layer 4: Learning (Weight Updates)

**Privilege:** Modify weights in cold registers. Gated by chemicals and temperature. Cannot modify structure.

**What lives here:**
- Mastery learning (pressure accumulation → weight update)
- Contrastive Hebbian Learning (CHL)
- Eligibility traces
- Babble (exploration noise)
- Consolidation (hot → cold thermogram persistence)
- Reward prediction error

**Implemented by:**
- Learning phases within region firmware
- Programs using: `learning.*`, `neuro.chem_read` (for dopamine gating)

**Lifecycle:** Runs after forward pass (Layer 3). Compares activations to targets/expectations. Updates weights if chemical gates are open.

**Invariants:**
1. **Chemical gating.** No learning without dopamine above threshold. The regulation region controls dopamine — learning cannot override it.
2. **Temperature gating.** Only HOT weights accept updates. COOL and COLD weights are read-only to this layer.
3. **Pressure-based.** Changes require sustained evidence. A single error signal does not flip a weight — pressure must accumulate across multiple ticks.
4. **Cannot modify structure.** Learning changes weight VALUES (polarity, magnitude) but cannot grow neurons, prune neurons, or rewire connections.
5. **Bounded magnitude.** Weight updates are bounded by the ternary range (-255 to +255). No runaway growth.

**Safety boundary:** Weight changes persist through thermograms (managed by Layer 0 kernel). Learning writes; the kernel persists.

**Analogy:** Synaptic plasticity. Long-term potentiation and depression change synapse strength, but they don't grow new axons.

---

## Privilege Matrix

| Action | L0 Kernel | L1 Boot | L2 Evolutionary | L3 Runtime | L4 Learning |
|--------|-----------|---------|-----------------|------------|-------------|
| Execute instructions | YES | — | — | — | — |
| Create regions | — | YES | NO | NO | NO |
| Register thermograms | — | YES | NO | NO | NO |
| Wire inter-region | — | YES | NO | NO | NO |
| Grow neurons | — | — | YES | NO | NO |
| Prune neurons | — | — | YES | NO | NO |
| Read weights | — | — | YES | YES | YES |
| Write weights | — | — | NO | NO | YES |
| Read chemicals | — | — | YES | YES | YES |
| Write chemicals | — | — | NO | NO | NO |
| Read activations | — | — | — | YES | YES |
| Write activations | — | — | — | YES | NO |
| Persist to thermogram | YES | — | — | — | — |

**Chemical authority** is a special case: the Regulation region writes chemicals, but it does so through Layer 3 runtime programs. The regulation region IS a runtime region — it just happens to be the one whose output IS chemical state. It doesn't break the layering because it writes to the chemical substrate, not to weights or structure.

---

## Anti-Brick Safety

Evolutionary changes (Layer 2) are the most dangerous — they modify the architecture that everything else depends on. Safety mechanisms:

1. **Journal all structural changes.** Before a grow/prune, log the operation and the pre-change state.
2. **Convergence watchdog.** After structural change, monitor convergence field coherence for N ticks. If coherence drops below threshold, rollback.
3. **Temperature lifecycle prevents regression.** Proven neurons cool to COLD and become immune to pruning. Only HOT (new, unproven) neurons can be pruned.
4. **Boot topology is sacred.** Even if every neuron in a region is pruned, the region itself persists. It can regrow. Regions are permanent; neurons are expendable.

## Anti-Devolve Safety

Learning (Layer 4) could theoretically destroy knowledge by overwriting good weights. Safety mechanisms:

1. **Pressure hysteresis.** A single bad learning signal doesn't flip weights. Pressure must accumulate consistently.
2. **Temperature protection.** Proven weights (COOL/COLD) are immune to learning. Only HOT weights accept updates.
3. **Dopamine gating.** Learning only occurs in reward-positive contexts. The regulation region controls dopamine — it's a biological circuit breaker.
4. **Thermogram snapshots.** Periodic persistence means weights can be recovered from the last known-good state.
5. **Consolidation is one-way.** Hot→Cold consolidation is explicit (via `learning.consolidate`). Random learning cannot accidentally freeze bad weights.

---

## Extension ↔ Layer Mapping

| Extension | Primary Layer | Secondary Layer | Notes |
|-----------|--------------|-----------------|-------|
| core (0x0000) | L0 | — | IS the kernel |
| tensor (0x0001) | L3 | — | Forward computation |
| ternary (0x0002) | L3 | — | Signal forward passes |
| activation (0x0003) | L3 | — | Activation functions |
| learning (0x0004) | L4 | — | Weight modification |
| neuro (0x0005) | L3 | L2, L4 | Substrate reads (L3), temp reads (L2) |
| arch (0x0006) | L2 | L1 | Structural changes (L2), topology (L1) |
| orchestration (0x0007) | L3 | — | Model execution routing |
| lifecycle (0x0008) | L1 | L3 | Boot (L1), phase reads (L3) |
| ipc (0x0009) | L3 | — | Inter-region signals |
| test (0x000A) | — | — | Development only |

---

## Enforcement

Layers are enforced by **program structure**, not runtime checks. A forward-pass program simply does not contain `arch.grow_neuron` instructions. A learning program does not contain `arch.prune_neuron` instructions. The assembler can validate layer compliance by checking which extension instructions appear in a program against its declared role.

Future: `ProgramValidator` can enforce layer policy — a program tagged `role=forward` that contains learning instructions is a validation error.
