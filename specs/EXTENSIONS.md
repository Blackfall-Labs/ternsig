# TVMR Standard Extensions

10 extensions. 108 instructions total. Extension IDs 0x0001-0x000A.

Extensions 0x000B-0x00FF reserved for future standards. 0x0100+ for user extensions.

---

## 0x0001 — tensor (18 instructions)

Matrix and tensor operations. General-purpose numeric computation.

| Op   | Mnemonic        | Pattern         | Description | Status |
|------|-----------------|-----------------|-------------|--------|
| 0000 | MATMUL          | RegRegReg       | target = source @ aux | STUB |
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
| 000F | SQUEEZE         | RegReg          | Remove shape dimension | STUB |
| 0010 | UNSQUEEZE       | RegReg          | Add shape dimension | STUB |
| 0011 | TRANSPOSE       | RegReg          | Swap shape dimensions | STUB |

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
| 0007 | QUANTIZE             | RegReg     | Float to signal quantization | STUB |
| 0008 | PACK_TERNARY         | RegReg     | Pack signals to 2-bit per weight | STUB |
| 0009 | UNPACK_TERNARY       | RegReg     | Unpack 2-bit to Signal | STUB |
| 000A | APPLY_POLARITY       | RegReg     | Apply polarity update to cold weight | STUB |
| 000B | APPLY_MAGNITUDE      | RegReg     | Apply magnitude update to cold weight | STUB |
| 000C | THRESHOLD_POLARITY   | RegReg     | Check polarity flip threshold (hysteresis) | STUB |
| 000D | ACCUMULATE_PRESSURE  | RegReg     | Accumulate polarity pressure register | STUB |

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
| 0004 | MARK_ELIGIBILITY   | RegReg   | Mark weights eligible by activity | STUB |
| 0005 | CHL_FREE_START     | None     | Start Contrastive Hebbian free phase | STUB |
| 0006 | CHL_FREE_RECORD    | RegReg   | Record free phase correlations | STUB |
| 0007 | CHL_CLAMP_START    | None     | Start CHL clamped phase | STUB |
| 0008 | CHL_CLAMP_RECORD   | RegReg   | Record clamped phase correlations | STUB |
| 0009 | CHL_UPDATE         | Reg      | Compute CHL weight delta | STUB |
| 000A | CHL_BACKPROP_CLAMP | RegReg   | Propagate clamped signal backward | STUB |
| 000B | DECAY_ELIGIBILITY  | Reg      | Decay eligibility traces by factor | STUB |
| 000C | COMPUTE_ERROR      | RegReg   | Error = target - output | STUB |
| 000D | UPDATE_WEIGHTS     | Reg      | Apply eligibility * error to weights | STUB |
| 000E | DECAY_BABBLE       | None     | Decay babble exploration scale | STUB |
| 000F | COMPUTE_RPE        | RegReg   | Reward Prediction Error | STUB |
| 0010 | GATE_ERROR         | RegImm8  | Gate learning by error threshold | STUB |
| 0011 | CHECKPOINT_WEIGHTS | Reg      | Checkpoint weights for rollback | STUB |
| 0012 | ROLLBACK_WEIGHTS   | Reg      | Rollback to checkpoint | STUB |
| 0013 | CONSOLIDATE        | None     | Consolidate hot->cold (Thermogram) | STUB |

---

## 0x0005 — neuro (8 instructions)

Substrate I/O. Requires SubstrateHandle trait implementation from host.

| Op   | Mnemonic   | Pattern  | Description | Status |
|------|------------|----------|-------------|--------|
| 0000 | CHEM_READ  | RegImm8  | Read neuromodulator level (DA/5HT/NE/GABA) | STUB |
| 0001 | CHEM_WRITE | RegImm8  | Write neuromodulator level | STUB |
| 0002 | FIELD_READ | RegImm8  | Read from temporal field | STUB |
| 0003 | FIELD_WRITE | RegImm8 | Write to temporal field | STUB |
| 0004 | STIM_READ  | RegImm8  | Read stimulation level | STUB |
| 0005 | CONV_READ  | Reg      | Read convergence field state | STUB |
| 0006 | TEMP_READ  | RegReg   | Read cold register temperatures | STUB |
| 0007 | TEMP_WRITE | RegReg   | Write cold register temperatures | STUB |

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
| 0007 | DEFINE_LAYER   | Custom   | Define layer dimensions | STUB |
| 0008 | FREEZE_LAYER   | Reg      | Mark layer non-trainable | STUB |
| 0009 | UNFREEZE_LAYER | Reg      | Unfreeze layer for training | STUB |
| 000A | SET_ACTIVATION | RegImm8  | Set activation function for layer | STUB |

---

## 0x0007 — orchestration (8 instructions)

Model table management. Multi-model execution and chaining.

| Op   | Mnemonic     | Pattern  | Description | Status |
|------|-------------|----------|-------------|--------|
| 0000 | MODEL_LOAD   | RegImm8  | Load model into table slot | STUB |
| 0001 | MODEL_EXEC   | RegImm8  | Execute model from table slot | STUB |
| 0002 | MODEL_INPUT  | RegImm8  | Set input for table slot | STUB |
| 0003 | MODEL_OUTPUT | RegImm8  | Get output from table slot | STUB |
| 0004 | MODEL_UNLOAD | Imm8     | Unload model from table slot | STUB |
| 0005 | MODEL_STATUS | RegImm8  | Get model status | STUB |
| 0006 | MODEL_RELOAD | Imm8     | Hot-reload model in slot | STUB |
| 0007 | MODEL_CHAIN  | Custom   | Chain slot1 output -> slot2 input | STUB |

---

## 0x0008 — lifecycle (8 instructions)

Boot phases, lifecycle events, state management.

| Op   | Mnemonic    | Pattern  | Description | Status |
|------|-------------|----------|-------------|--------|
| 0000 | PHASE_READ  | Reg      | Read current boot phase | STUB |
| 0001 | TICK_READ   | Reg      | Read current tick count | STUB |
| 0002 | LEVEL_READ  | Reg      | Read neuronal level | STUB |
| 0003 | INIT_THERMO | Reg      | Initialize thermogram for cold reg | STUB |
| 0004 | SAVE_THERMO | Reg      | Save cold reg to thermogram | STUB |
| 0005 | LOAD_THERMO | Reg      | Load cold reg from thermogram | STUB |
| 0006 | LOG_EVENT   | RegImm8  | Log lifecycle event | STUB |
| 0007 | HALT_REGION | Imm8     | Halt specific brain region | STUB |

---

## 0x0009 — ipc (8 instructions)

Inter-region communication. Signals, barriers, atomics.

| Op   | Mnemonic      | Pattern    | Description | Status |
|------|---------------|------------|-------------|--------|
| 0000 | SEND_SIGNAL   | RegImm8    | Send signal to region | STUB |
| 0001 | RECV_SIGNAL   | RegImm8    | Receive signal from region | STUB |
| 0002 | BROADCAST     | RegImm8    | Broadcast to all regions | STUB |
| 0003 | SUBSCRIBE     | RegImm8    | Subscribe to region signals | STUB |
| 0004 | MAILBOX_PEEK  | RegImm8    | Peek at mailbox (no consume) | STUB |
| 0005 | MAILBOX_POP   | RegImm8    | Pop from mailbox (consume) | STUB |
| 0006 | BARRIER_WAIT  | Imm8       | Wait at synchronization barrier | STUB |
| 0007 | ATOMIC_CAS    | RegRegReg  | Compare-and-swap | STUB |

---

## 0x000A — test (8 instructions)

Testing and assertions. Development and CI.

| Op   | Mnemonic      | Pattern  | Description | Status |
|------|---------------|----------|-------------|--------|
| 0000 | ASSERT_EQ     | RegReg   | Assert registers equal | STUB |
| 0001 | ASSERT_GT     | RegReg   | Assert first > second | STUB |
| 0002 | ASSERT_ACTIVE | Reg      | Assert nonzero values | STUB |
| 0003 | ASSERT_RANGE  | Custom   | Assert all values in [min, max] | STUB |
| 0004 | TEST_BEGIN    | Imm8     | Mark test case start | STUB |
| 0005 | TEST_END      | None     | Mark test case end | STUB |
| 0006 | EXPECT_CHEM   | Custom   | Assert chemical level in range | STUB |
| 0007 | SNAPSHOT      | Reg      | Snapshot register for comparison | STUB |

---

## Summary

| ExtID  | Name          | IMPL | STUB | Total |
|--------|---------------|------|------|-------|
| 0x0001 | tensor        | 14   | 4    | 18    |
| 0x0002 | ternary       | 7    | 7    | 14    |
| 0x0003 | activation    | 5    | 0    | 5     |
| 0x0004 | learning      | 4    | 16   | 20    |
| 0x0005 | neuro         | 0    | 8    | 8     |
| 0x0006 | arch          | 7    | 4    | 11    |
| 0x0007 | orchestration | 0    | 8    | 8     |
| 0x0008 | lifecycle     | 0    | 8    | 8     |
| 0x0009 | ipc           | 0    | 8    | 8     |
| 0x000A | test          | 0    | 8    | 8     |
| **Total** |            | **37** | **71** | **108** |

**71 stubs to implement before v3.**
