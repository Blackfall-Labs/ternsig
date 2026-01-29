# TVMR Standard Extensions

10 extensions. 121 instructions total. Extension IDs 0x0001-0x000A. **All implemented.**

Extensions 0x000B-0x00FF reserved for future standards. 0x0100+ for user extensions.

---

## 0x0001 — tensor (18 instructions)

Matrix and tensor operations. General-purpose numeric computation.

| Op   | Mnemonic        | Pattern         | Description | Status |
|------|-----------------|-----------------|-------------|--------|
| 0000 | MATMUL          | RegRegReg       | target = source @ aux (errors, use ternary.TERNARY_MATMUL) | IMPL |
| 0001 | ADD             | RegRegReg       | target = source + aux (hot+hot or hot+cold) | IMPL |
| 0002 | SUB             | RegRegReg       | target = source - aux | IMPL |
| 0003 | MUL             | RegRegReg       | target = source * aux | IMPL |
| 0004 | SCALE           | RegRegImm16     | target = source * scale | IMPL |
| 0005 | SHIFT           | RegRegReg       | target = source >> amount | IMPL |
| 0006 | CLAMP           | RegRegRegFlags  | target = clamp(source, min, max) | IMPL |
| 0007 | CMP_GT          | RegRegReg       | target[i] = source[i] > aux[i] ? 1 : 0 | IMPL |
| 0008 | MAX_REDUCE      | RegReg          | target[0] = max(source) | IMPL |
| 0009 | NEGATE          | RegReg          | target = -source | IMPL |
| 000A | REDUCE_AVG      | Custom          | target[0] = mean(source[start..start+count]) | IMPL |
| 000B | REDUCE_MEAN_DIM | RegRegReg       | Reduce mean along dimension | IMPL |
| 000C | SLICE           | Custom          | target = source[start..start+len] | IMPL |
| 000D | ARGMAX          | RegReg          | target[0] = argmax(source) | IMPL |
| 000E | CONCAT          | RegRegReg       | target = concat(source, aux) | IMPL |
| 000F | SQUEEZE         | RegReg          | Remove shape dimension | IMPL |
| 0010 | UNSQUEEZE       | RegReg          | Add shape dimension | IMPL |
| 0011 | TRANSPOSE       | RegReg          | Swap shape dimensions | IMPL |

---

## 0x0002 — ternary (14 instructions)

Signal-specific operations. Temperature-gated ternary arithmetic.

| Op   | Mnemonic             | Pattern    | Description | Status |
|------|----------------------|------------|-------------|--------|
| 0000 | TERNARY_MATMUL       | RegRegReg  | target = cold @ hot (temperature-gated conductance) | IMPL |
| 0001 | TERNARY_BATCH_MATMUL | RegRegReg  | target[i] = cold @ hot[i] for each row | IMPL |
| 0002 | TERNARY_ADD_BIAS     | RegRegReg  | target = hot + cold_bias (signal dequant) | IMPL |
| 0003 | DEQUANTIZE           | RegRegImm16 | target = source_signal / scale | IMPL |
| 0004 | EMBED_LOOKUP         | RegRegReg  | target[i] = table[indices[i]] | IMPL |
| 0005 | EMBED_SEQUENCE       | Custom     | target[i] = table[i] for i in 0..count | IMPL |
| 0006 | GATE_UPDATE          | Custom     | GRU: target = gate*update + (1-gate)*state | IMPL |
| 0007 | QUANTIZE             | RegReg     | i32 to Signal quantization (polarity+magnitude) | IMPL |
| 0008 | PACK_TERNARY         | RegReg     | Pack cold signals to 2-bit (4 per byte) | IMPL |
| 0009 | UNPACK_TERNARY       | RegReg     | Unpack 2-bit packed to cold Signal register | IMPL |
| 000A | APPLY_POLARITY       | RegReg     | Apply polarity from hot to cold weights | IMPL |
| 000B | APPLY_MAGNITUDE      | RegReg     | Apply magnitude from hot to cold weights | IMPL |
| 000C | THRESHOLD_POLARITY   | RegReg     | Hysteresis polarity flip (pressure > mag*2) | IMPL |
| 000D | ACCUMULATE_PRESSURE  | RegReg     | Accumulate polarity pressure: dst += src | IMPL |

---

## 0x0003 — activation (5 instructions)

Activation functions. All integer-approximated, no floats.

| Op   | Mnemonic | Pattern | Description | Status |
|------|----------|---------|-------------|--------|
| 0000 | RELU     | RegReg  | max(0, x) | IMPL |
| 0001 | SIGMOID  | RegReg  | Integer sigmoid, output 0-255 | IMPL |
| 0002 | TANH     | RegReg  | Integer tanh, output -127..127 | IMPL |
| 0003 | SOFTMAX  | RegReg  | Integer softmax, output 0-255 | IMPL |
| 0004 | GELU     | RegReg  | Integer GELU approximation | IMPL |

---

## 0x0004 — learning (20 instructions)

Learning algorithms. Dopamine-gated, temperature-aware, pressure-based.

| Op   | Mnemonic           | Pattern  | Description | Status |
|------|--------------------|----------|-------------|--------|
| 0000 | MASTERY_UPDATE     | Custom   | Accumulate pressure from error+activity (DA-gated) | IMPL |
| 0001 | MASTERY_COMMIT     | Custom   | Commit pressure to weight changes (temperature-aware) | IMPL |
| 0002 | ADD_BABBLE         | Reg      | Add exploration noise to activations | IMPL |
| 0003 | LOAD_TARGET        | Reg      | Load target buffer into hot register | IMPL |
| 0004 | MARK_ELIGIBILITY   | RegReg   | Mark weights eligible by activity (above-mean threshold) | IMPL |
| 0005 | CHL_FREE_START     | None     | Start Contrastive Hebbian free phase | IMPL |
| 0006 | CHL_FREE_RECORD    | RegReg   | Record free phase correlations (outer product) | IMPL |
| 0007 | CHL_CLAMP_START    | None     | Start CHL clamped phase | IMPL |
| 0008 | CHL_CLAMP_RECORD   | RegReg   | Record clamped phase correlations | IMPL |
| 0009 | CHL_UPDATE         | Reg      | Apply CHL delta (clamped - free) to pressure (DA-gated) | IMPL |
| 000A | CHL_BACKPROP_CLAMP | RegReg   | Transpose matmul: backward propagation through weights | IMPL |
| 000B | DECAY_ELIGIBILITY  | Reg      | Exponential decay (230/256 ≈ 90% retention) | IMPL |
| 000C | COMPUTE_ERROR      | RegReg   | current_error = sum(\|target - output\|) | IMPL |
| 000D | UPDATE_WEIGHTS     | Reg      | Apply pressure × error × dopamine to weights | IMPL |
| 000E | DECAY_BABBLE       | None     | Decay babble scale (245/256 ≈ 96% retention) | IMPL |
| 000F | COMPUTE_RPE        | RegReg   | RPE = actual[0] - predicted[0] (signed) | IMPL |
| 0010 | GATE_ERROR         | RegImm8  | Suppress learning below threshold × 256 | IMPL |
| 0011 | CHECKPOINT_WEIGHTS | Reg      | Snapshot cold register for rollback | IMPL |
| 0012 | ROLLBACK_WEIGHTS   | Reg      | Restore cold register from checkpoint | IMPL |
| 0013 | CONSOLIDATE        | None     | Yields DomainOp::Consolidate | IMPL |

---

## 0x0005 — neuro (15 instructions)

Substrate I/O. All operations yield `DomainOp` back to the host for external handling.
The VM does not own the substrate — the host does. Programs request substrate operations;
the host fulfills them and resumes execution.

Chemical IDs: 0=dopamine, 1=serotonin, 2=norepinephrine, 3=gaba, 4=cortisol, 5=endorphin, 6=acetylcholine, 7=fatigue.

Field IDs: 0=activity, 1=chemical, 2=convergence, 3=sensory, 4=stimulation, 5=valence,
6=hippocampal, 7=priming, 8=attention, 9=articulation, 10=efference, 11=expression.

| Op   | Mnemonic          | Pattern  | Description | Status |
|------|-------------------|----------|-------------|--------|
| 0000 | CHEM_READ         | RegImm8  | Read chemical level → H[reg][0]. Yields ChemRead. | IMPL |
| 0001 | CHEM_SET          | RegImm8  | SET chemical level from H[reg][0] (authoritative). Yields ChemSet. | IMPL |
| 0002 | CHEM_INJECT       | RegImm8  | ADDITIVE injection (phasic events). Yields ChemInject. | IMPL |
| 0003 | FIELD_READ        | RegImm8  | Read field region slice → H[reg]. Yields FieldRead. | IMPL |
| 0004 | FIELD_WRITE       | RegImm8  | Write H[reg] to field region slice. Yields FieldWrite. | IMPL |
| 0005 | FIELD_TICK        | Imm8     | Advance field by 1 tick (decay, age frames). Yields FieldTick. | IMPL |
| 0006 | STIM_READ         | Reg      | Read all stimulation levels → H[reg]. Yields StimRead. | IMPL |
| 0007 | VALENCE_READ      | Reg      | Read valence [reward, punish] → H[reg]. Yields ValenceRead. | IMPL |
| 0008 | CONV_READ         | Reg      | Read convergence field state → H[reg]. Yields ConvRead. | IMPL |
| 0009 | TEMP_READ         | RegReg   | Read cold register temperatures → H[dst]. | IMPL |
| 000A | TEMP_WRITE        | RegReg   | Write temperatures from H[src] to cold register. | IMPL |
| 000B | FIELD_DECAY       | Custom   | Apply metabolic decay: retention factor + fatigue. Yields FieldDecay. | IMPL |
| 000C | LATERAL_INHIBIT   | RegImm8  | Winner-take-some: dominant suppresses others. Yields LateralInhibit. | IMPL |
| 000D | EXHAUSTION_BOOST  | RegImm8  | Apply exhaustion decay boost to sustained activity. Yields ExhaustionBoost. | IMPL |
| 000E | NOVELTY_SCORE     | RegReg   | Compute novelty z-scores from region energies → H[dst]. Yields NoveltyScore. | IMPL |

---

## 0x0006 — arch (11 instructions)

Structural plasticity. Runtime architecture modification.

| Op   | Mnemonic       | Pattern  | Description | Status |
|------|----------------|----------|-------------|--------|
| 0000 | ALLOC_TENSOR   | Custom   | Allocate register at runtime | IMPL |
| 0001 | FREE_TENSOR    | Reg      | Free register | IMPL |
| 0002 | WIRE_FORWARD   | RegRegReg | output = weights @ input | IMPL |
| 0003 | WIRE_SKIP      | RegRegReg | output = input1 + input2 | IMPL |
| 0004 | GROW_NEURON    | Custom   | Add neurons to cold register (LCG PRNG) | IMPL |
| 0005 | PRUNE_NEURON   | Custom   | Remove neuron from cold register | IMPL |
| 0006 | INIT_RANDOM    | Custom   | Initialize cold register random (LCG PRNG) | IMPL |
| 0007 | DEFINE_LAYER   | Custom   | Allocate cold register with [out_dim, in_dim] shape | IMPL |
| 0008 | FREEZE_LAYER   | Reg      | Set all temperatures to Cold (non-trainable) | IMPL |
| 0009 | UNFREEZE_LAYER | Reg      | Set all temperatures to Hot (trainable) | IMPL |
| 000A | SET_ACTIVATION | RegImm8  | Store activation ID in shape register | IMPL |

---

## 0x0007 — orchestration (14 instructions)

Model table management and region routing. Two halves:
- **Model table** (0x0000-0x0007): Load, execute, chain Ternsig models by slot index.
- **Region routing** (0x0008-0x000D): Route signals between brain regions. Yields DomainOps.

| Op   | Mnemonic          | Pattern  | Description | Status |
|------|-------------------|----------|-------------|--------|
| 0000 | MODEL_LOAD        | RegImm8  | Load model into table slot. Yields ModelLoad. | IMPL |
| 0001 | MODEL_EXEC        | RegImm8  | Execute model from table slot. Yields ModelExec. | IMPL |
| 0002 | MODEL_INPUT       | RegImm8  | Set input register for table slot. | IMPL |
| 0003 | MODEL_OUTPUT      | RegImm8  | Get output from table slot into register. | IMPL |
| 0004 | MODEL_UNLOAD      | Imm8     | Unload model from table slot. | IMPL |
| 0005 | MODEL_STATUS      | RegImm8  | Get model status (0=empty, 1=loaded, 2=running). | IMPL |
| 0006 | MODEL_RELOAD      | Imm8     | Hot-reload model in slot. Yields ModelReload. | IMPL |
| 0007 | MODEL_CHAIN       | Custom   | Chain slot1 output → slot2 input. | IMPL |
| 0008 | ROUTE_INPUT       | RegImm8  | Route H[reg] to region ID. Yields RouteInput. | IMPL |
| 0009 | REGION_FIRE       | RegImm8  | Fire region, output → H[reg]. Yields RegionFire. | IMPL |
| 000A | COLLECT_OUTPUTS   | Reg      | Aggregate region outputs → H[reg]. Yields CollectOutputs. | IMPL |
| 000B | REGION_STATUS     | RegImm8  | Read region status (0=idle, 1=active, 2=firing). | IMPL |
| 000C | REGION_ENABLE     | Imm8     | Enable region for routing. Yields RegionEnable. | IMPL |
| 000D | REGION_DISABLE    | Imm8     | Disable region. Yields RegionDisable. | IMPL |

---

## 0x0008 — lifecycle (8 instructions)

Boot phases, lifecycle events, state management.

| Op   | Mnemonic    | Pattern  | Description | Status |
|------|-------------|----------|-------------|--------|
| 0000 | PHASE_READ  | Reg      | Yields PhaseRead. Host writes boot phase ordinal. | IMPL |
| 0001 | TICK_READ   | Reg      | Yields TickRead. Host writes tick count. | IMPL |
| 0002 | LEVEL_READ  | Reg      | Yields LevelRead. Host writes neuronal level. | IMPL |
| 0003 | INIT_THERMO | Reg      | Yields InitThermo. Host initializes thermogram storage. | IMPL |
| 0004 | SAVE_THERMO | Reg      | Yields SaveThermo. Host persists weights. | IMPL |
| 0005 | LOAD_THERMO | Reg      | Yields LoadThermo. Host loads persisted weights. | IMPL |
| 0006 | LOG_EVENT   | RegImm8  | Yields LogEvent. event_type: 0-7 (boot/phase/error/warn/metric/checkpoint/recovery/shutdown). | IMPL |
| 0007 | HALT_REGION | Imm8     | Yields HaltRegion. Host halts specified region. | IMPL |

---

## 0x0009 — ipc (8 instructions)

Inter-region communication. Signals, barriers, atomics.

| Op   | Mnemonic      | Pattern    | Description | Status |
|------|---------------|------------|-------------|--------|
| 0000 | SEND_SIGNAL   | RegImm8    | Yields SendSignal. Host delivers to target region. | IMPL |
| 0001 | RECV_SIGNAL   | RegImm8    | Yields RecvSignal. Host writes received data. | IMPL |
| 0002 | BROADCAST     | RegImm8    | Yields Broadcast. Host fans out to all subscribers. | IMPL |
| 0003 | SUBSCRIBE     | RegImm8    | Yields Subscribe. Host registers subscription. | IMPL |
| 0004 | MAILBOX_PEEK  | RegImm8    | Yields MailboxPeek. Host reads without consuming. | IMPL |
| 0005 | MAILBOX_POP   | RegImm8    | Yields MailboxPop. Host reads and consumes. | IMPL |
| 0006 | BARRIER_WAIT  | Imm8       | Yields BarrierWait. Host blocks until all arrive. | IMPL |
| 0007 | ATOMIC_CAS    | RegRegReg  | Yields AtomicCas. Host does atomic compare-and-swap. | IMPL |

---

## 0x000A — test (8 instructions)

Testing and assertions. Development and CI.

| Op   | Mnemonic      | Pattern  | Description | Status |
|------|---------------|----------|-------------|--------|
| 0000 | ASSERT_EQ     | RegReg   | Assert registers equal | IMPL |
| 0001 | ASSERT_GT     | RegReg   | Assert first > second | IMPL |
| 0002 | ASSERT_ACTIVE | Reg      | Assert nonzero values | IMPL |
| 0003 | ASSERT_RANGE  | Custom   | Assert all values in [min, max] | IMPL |
| 0004 | TEST_BEGIN    | Imm8     | Mark test case start | IMPL |
| 0005 | TEST_END      | None     | Mark test case end | IMPL |
| 0006 | EXPECT_CHEM   | Custom   | Assert chemical level in range | IMPL |
| 0007 | SNAPSHOT      | Reg      | Snapshot register for comparison | IMPL |

---

## Summary

| ExtID  | Name          | Total | Status |
|--------|---------------|-------|--------|
| 0x0001 | tensor        | 18    | ALL IMPL |
| 0x0002 | ternary       | 14    | ALL IMPL |
| 0x0003 | activation    | 5     | ALL IMPL |
| 0x0004 | learning      | 20    | ALL IMPL |
| 0x0005 | neuro         | 15    | ALL IMPL |
| 0x0006 | arch          | 11    | ALL IMPL |
| 0x0007 | orchestration | 14    | ALL IMPL |
| 0x0008 | lifecycle     | 8     | ALL IMPL |
| 0x0009 | ipc           | 8     | ALL IMPL |
| 0x000A | test          | 8     | ALL IMPL |
| **Total** |            | **121** | **121/121** |

**All 121 instructions implemented. 0 stubs remaining. 167 tests passing.**
