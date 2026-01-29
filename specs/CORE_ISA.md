# Core ISA (ExtID 0x0000)

104 instructions. Built into the interpreter. Zero dispatch overhead.

All core instructions use the legacy opcode layout: `[0x0000][opcode:2][operands:4]`.

## 0x00xx — System (6)

| Opcode | Mnemonic   | Operands        | Description |
|--------|------------|-----------------|-------------|
| 0x0000 | NOP        | None            | No operation |
| 0x0001 | HALT       | None            | Stop execution |
| 0x0002 | CHECKPOINT | None            | Save execution state |
| 0x0003 | RESET      | None            | Reset interpreter state |
| 0x0004 | SYNC       | None            | Synchronization barrier |
| 0x0005 | YIELD      | None            | Yield execution to scheduler |

## 0x10xx — Register Management (10)

| Opcode | Mnemonic       | Operands           | Description |
|--------|----------------|--------------------|-------------|
| 0x1000 | ALLOC          | `[reg:1][dtype:1]` | Allocate register with type |
| 0x1001 | FREE           | `[reg:1]`          | Free register |
| 0x1002 | LOAD_WEIGHTS   | `[cold:1]`         | Load weights from thermogram |
| 0x1003 | SAVE_WEIGHTS   | `[cold:1]`         | Save weights to thermogram |
| 0x1004 | COPY           | `[dst:1][src:1]`   | Copy register contents |
| 0x1005 | ZERO           | `[reg:1]`          | Zero register contents |
| 0x1006 | SWAP           | `[a:1][b:1]`       | Swap two registers |
| 0x1007 | LOAD_INPUT     | `[dst:1]`          | Load from input buffer |
| 0x1008 | STORE_OUTPUT   | `[src:1]`          | Store to output buffer |
| 0x1009 | LOAD_CONST     | `[dst:1][val:1]`   | Load immediate constant |

## 0x20xx — Architecture (11)

| Opcode | Mnemonic       | Operands                    | Description |
|--------|----------------|-----------------------------|-------------|
| 0x2000 | DEFINE_LAYER   | `[cold:1][in:1][out:1]`     | Define layer shape |
| 0x2001 | WIRE_FORWARD   | `[out:1][cold:1][in:1]`     | Wire forward connection |
| 0x2002 | WIRE_SKIP      | `[out:1][a:1][b:1]`         | Wire skip/residual connection |
| 0x2003 | WIRE_LATERAL   | `[out:1][cold:1][in:1]`     | Wire lateral connection |
| 0x2004 | WIRE_RECURRENT | `[out:1][cold:1][in:1]`     | Wire recurrent connection |
| 0x2005 | FREEZE         | `[cold:1]`                  | Freeze layer (no learning) |
| 0x2006 | UNFREEZE       | `[cold:1]`                  | Unfreeze layer |
| 0x2007 | GROW           | `[cold:1][count:1]`         | Add neurons |
| 0x2008 | PRUNE          | `[cold:1][index:1]`         | Remove neuron |
| 0x2009 | INIT_RANDOM    | `[cold:1][seed_lo:1][seed_hi:1]` | Initialize random weights |
| 0x200A | RESHAPE        | `[reg:1][dim0:1][dim1:1]`   | Reshape register |

## 0x30xx — Forward Operations (17)

| Opcode | Mnemonic        | Operands              | Description |
|--------|-----------------|------------------------|-------------|
| 0x3000 | MATMUL          | `[dst:1][src:1][aux:1]` | Matrix multiply |
| 0x3001 | ADD             | `[dst:1][src:1][aux:1]` | Element-wise add |
| 0x3002 | SCALE           | `[dst:1][src:1][scale_hi:1][scale_lo:1]` | Scale by constant |
| 0x3003 | RELU            | `[dst:1][src:1]`       | ReLU activation |
| 0x3004 | SIGMOID         | `[dst:1][src:1]`       | Sigmoid activation |
| 0x3005 | TANH            | `[dst:1][src:1]`       | Tanh activation |
| 0x3006 | SOFTMAX         | `[dst:1][src:1]`       | Softmax activation |
| 0x3007 | LAYER_NORM      | `[dst:1][src:1]`       | Layer normalization |
| 0x3008 | BATCH_NORM      | `[dst:1][src:1]`       | Batch normalization |
| 0x3009 | SHIFT           | `[dst:1][src:1][amount:1]` | Right shift |
| 0x300A | CLAMP           | `[dst:1][src:1][min:1][max:1]` | Clamp to range |
| 0x300B | CMP_GT          | `[dst:1][src:1][aux:1]` | Compare greater |
| 0x300C | MAX_REDUCE      | `[dst:1][src:1]`       | Max reduction |
| 0x300D | NEGATE          | `[dst:1][src:1]`       | Negate |
| 0x300E | SUB             | `[dst:1][src:1][aux:1]` | Element-wise subtract |
| 0x300F | MUL             | `[dst:1][src:1][aux:1]` | Element-wise multiply |
| 0x3010 | GELU            | `[dst:1][src:1]`       | GELU activation |

## 0x40xx — Ternary Operations (22)

| Opcode | Mnemonic             | Operands                  | Description |
|--------|----------------------|---------------------------|-------------|
| 0x4000 | TERNARY_MATMUL       | `[dst:1][cold:1][src:1]`  | Signal matmul (temperature-gated) |
| 0x4001 | QUANTIZE             | `[dst:1][src:1]`          | Float to signal |
| 0x4002 | DEQUANTIZE           | `[dst:1][src:1][scale_hi:1][scale_lo:1]` | Signal to int |
| 0x4003 | PACK                 | `[dst:1][src:1]`          | Pack signals (4 per byte) |
| 0x4004 | UNPACK               | `[dst:1][src:1]`          | Unpack signals |
| 0x4005 | APPLY_POLARITY       | `[cold:1][src:1]`         | Update polarities |
| 0x4006 | APPLY_MAGNITUDE      | `[cold:1][src:1]`         | Update magnitudes |
| 0x4007 | THRESHOLD_POLARITY   | `[cold:1][threshold:1]`   | Polarity flip check |
| 0x4008 | ACCUMULATE_PRESSURE  | `[cold:1][src:1]`         | Accumulate pressure |
| 0x4009 | TERNARY_ADD_BIAS     | `[dst:1][src:1][cold:1]`  | Add signal bias |
| 0x400A | EMBED_LOOKUP         | `[dst:1][cold:1][idx:1]`  | Embedding lookup |
| 0x400B | REDUCE_AVG           | `[dst:1][src:1][start:1][count:1]` | Average reduction |
| 0x400C | SLICE                | `[dst:1][src:1][start:1][len:1]` | Slice buffer |
| 0x400D | ARGMAX               | `[dst:1][src:1]`          | Index of max |
| 0x400E | CONCAT               | `[dst:1][src:1][aux:1]`   | Concatenate |
| 0x400F | SQUEEZE              | `[dst:1][src:1][dim:1]`   | Remove dimension |
| 0x4010 | UNSQUEEZE            | `[dst:1][src:1][dim:1]`   | Add dimension |
| 0x4011 | TRANSPOSE            | `[dst:1][src:1][d0:1][d1:1]` | Swap dimensions |
| 0x4012 | GATE_UPDATE          | `[dst:1][gate:1][upd:1][st:1]` | Gated update (GRU-style) |
| 0x4013 | TERNARY_BATCH_MATMUL | `[dst:1][cold:1][src:1]`  | Batch signal matmul |
| 0x4014 | EMBED_SEQUENCE       | `[dst:1][cold:1][count:1]` | Sequential embed |
| 0x4015 | REDUCE_MEAN_DIM      | `[dst:1][src:1][dim:1]`   | Mean along dimension |

## 0x50xx — Learning Operations (20)

| Opcode | Mnemonic          | Operands                | Description |
|--------|-------------------|-------------------------|-------------|
| 0x5000 | MARK_ELIGIBILITY  | `[out:1][in:1][layer:1]` | Mark eligible weights |
| 0x5001 | MASTERY_UPDATE    | `[cold:1][hot:1][dir:1]` | Accumulate learning pressure |
| 0x5002 | MASTERY_COMMIT    | `[cold:1]`              | Commit pressure to weights |
| 0x5003 | LOAD_TARGET       | `[dst:1]`               | Load target buffer |
| 0x5004 | ADD_BABBLE        | `[dst:1][_:1][layer:1]` | Add exploration noise |
| 0x5005 | CHL_FREE_START    | None                    | Start CHL free phase |
| 0x5006 | CHL_FREE_RECORD   | `[dst:1][src:1]`        | Record free correlations |
| 0x5007 | CHL_CLAMP_START   | None                    | Start CHL clamped phase |
| 0x5008 | CHL_CLAMP_RECORD  | `[dst:1][src:1]`        | Record clamped correlations |
| 0x5009 | CHL_UPDATE        | `[cold:1]`              | Compute CHL weight update |
| 0x500A | CHL_BACKPROP      | `[dst:1][src:1]`        | Backprop clamped signal |
| 0x500B | DECAY_ELIGIBILITY | `[cold:1]`              | Decay eligibility traces |
| 0x500C | COMPUTE_ERROR     | `[dst:1][src:1]`        | Compute error signal |
| 0x500D | UPDATE_WEIGHTS    | `[cold:1]`              | Apply weight updates |
| 0x500E | DECAY_BABBLE      | None                    | Decay babble scale |
| 0x500F | COMPUTE_RPE       | `[dst:1][src:1]`        | Reward prediction error |
| 0x5010 | GATE_ERROR        | `[cold:1][thresh:1]`    | Gate learning by error |
| 0x5011 | CHECKPOINT_WEIGHTS | `[cold:1]`             | Checkpoint for rollback |
| 0x5012 | ROLLBACK_WEIGHTS  | `[cold:1]`              | Rollback to checkpoint |
| 0x5013 | CONSOLIDATE       | None                    | Hot->cold thermogram |

## 0x60xx — Control Flow (12)

| Opcode | Mnemonic    | Operands                     | Description |
|--------|-------------|------------------------------|-------------|
| 0x6000 | LOOP        | `[_:2][count_hi:1][count_lo:1]` | Loop N times |
| 0x6001 | ENDLOOP     | None                         | End of loop body |
| 0x6002 | BREAK       | None                         | Break from loop |
| 0x6003 | SKIP        | `[_:2][count_hi:1][count_lo:1]` | Skip N instructions |
| 0x6004 | JUMP        | `[addr_3:1][addr_2:1][addr_1:1][addr_0:1]` | Jump to address |
| 0x6005 | JUMP_IF     | `[reg:1][_:1][addr_hi:1][addr_lo:1]` | Jump if reg nonzero |
| 0x6006 | JUMP_IFN    | `[reg:1][_:1][addr_hi:1][addr_lo:1]` | Jump if reg zero |
| 0x6007 | CALL        | `[addr_3:1][addr_2:1][addr_1:1][addr_0:1]` | Call subroutine |
| 0x6008 | RETURN      | None                         | Return from subroutine |
| 0x6009 | SELECT      | `[dst:1][cond:1][t:1][f:1]`  | Conditional select |
| 0x600A | CMP_BRANCH  | `[a:1][b:1][addr_hi:1][addr_lo:1]` | Compare and branch |
| 0x600B | LOOP_WHILE  | `[reg:1][_:1][thresh_hi:1][thresh_lo:1]` | Loop while condition |

## 0x70xx — Debug (6)

| Opcode | Mnemonic       | Operands        | Description |
|--------|----------------|-----------------|-------------|
| 0x7000 | TRACE          | `[reg:1][id:1]` | Trace register to log |
| 0x7001 | BREAKPOINT     | None            | Debugger breakpoint |
| 0x7002 | PROFILE_START  | `[id:1]`        | Start profiling section |
| 0x7003 | PROFILE_END    | `[id:1]`        | End profiling section |
| 0x7004 | ASSERT         | `[reg:1][val:1]` | Assert register value |
| 0x7005 | DUMP_REG       | `[reg:1]`       | Dump register to stderr |
