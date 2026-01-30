# Core ISA (0x0000)

**Instructions:** 52  
**ExtID:** 0x0000 (built-in, not an extension)

Auto-generated from the live extension registry by `ternsig-spec`.

## System

| Opcode | Mnemonic | Operands | Description |
|--------|----------|----------|-------------|
| 0x0000 | `NOP` | `—` | No operation |
| 0x0001 | `HALT` | `—` | Halt execution |
| 0x0003 | `RESET` | `—` | Reset PC and loop stack |

## Register Management

| Opcode | Mnemonic | Operands | Description |
|--------|----------|----------|-------------|
| 0x1000 | `ALLOC_TENSOR` | `[reg:1][dim0:1][dim1_hi:1][dim1_lo:1]` | Allocate tensor register with shape |
| 0x1001 | `FREE_TENSOR` | `[reg:1][_:3]` | Free tensor register |
| 0x1004 | `COPY_REG` | `[dst:1][src:1][_:2]` | Copy between hot registers |
| 0x1006 | `ZERO_REG` | `[reg:1][_:3]` | Zero out register |
| 0x1007 | `LOAD_INPUT` | `[dst:1][_:3]` | Load input buffer into hot register |
| 0x1008 | `STORE_OUTPUT` | `[src:1][_:3]` | Store hot register to output buffer |
| 0x1009 | `LOAD_TARGET` | `[dst:1][_:3]` | Load target buffer for learning |

## Forward Operations

| Opcode | Mnemonic | Operands | Description |
|--------|----------|----------|-------------|
| 0x3001 | `ADD` | `[dst:1][a:1][b:1][_:1]` | Element-wise addition with saturation |
| 0x300E | `SUB` | `[dst:1][a:1][b:1][_:1]` | Element-wise subtraction |
| 0x3002 | `MUL` | `[dst:1][a:1][b:1][_:1]` | Element-wise multiply (>>8 fixed-point) |
| 0x3003 | `RELU` | `[dst:1][src:1][_:2]` | ReLU activation |
| 0x3004 | `SIGMOID` | `[dst:1][src:1][_:2]` | Integer sigmoid approximation |
| 0x3005 | `TANH` | `[dst:1][src:1][_:2]` | Integer tanh approximation |
| 0x3006 | `SOFTMAX` | `[dst:1][src:1][_:2]` | Integer softmax |
| 0x3007 | `GELU` | `[dst:1][src:1][_:2]` | Integer GELU approximation |
| 0x3009 | `SHIFT` | `[dst:1][src:1][amt:1][_:1]` | Right-shift (integer scaling) |
| 0x300A | `CLAMP` | `[dst:1][src:1][min:1][max:1]` | Clamp to range [min*256, max*256] |
| 0x300F | `CMP_GT` | `[dst:1][a:1][b:1][_:1]` | Compare greater-than (outputs 1/0) |
| 0x3010 | `MAX_REDUCE` | `[dst:1][src:1][_:2]` | Reduce to maximum value |

## Ternary Operations

| Opcode | Mnemonic | Operands | Description |
|--------|----------|----------|-------------|
| 0x4000 | `TERNARY_MATMUL` | `[dst:1][weights:1][input:1][_:1]` | Temperature-gated ternary matmul |
| 0x4002 | `DEQUANTIZE` | `[dst:1][src:1][shift:2]` | Dequantize signals (right-shift) |
| 0x4009 | `TERNARY_ADD_BIAS` | `[dst:1][src:1][bias:1][_:1]` | Add signal bias to activations |
| 0x400A | `EMBED_LOOKUP` | `[dst:1][table:1][indices:1][_:1]` | Embedding table lookup |
| 0x400B | `REDUCE_AVG` | `[dst:1][src:1][start:1][count:1]` | Slice average |
| 0x400C | `SLICE` | `[dst:1][src:1][start:1][len:1]` | Slice tensor |
| 0x400D | `ARGMAX` | `[dst:1][src:1][_:2]` | Index of maximum value |
| 0x400E | `CONCAT` | `[dst:1][a:1][b:1][_:1]` | Concatenate two registers |
| 0x400F | `SQUEEZE` | `[dst:1][src:1][_:2]` | Copy (shape semantics) |
| 0x4010 | `UNSQUEEZE` | `[dst:1][src:1][_:2]` | Copy (shape semantics) |
| 0x4011 | `TRANSPOSE` | `[dst:1][src:1][_:2]` | Copy (shape semantics) |
| 0x4012 | `GATE_UPDATE` | `[dst:1][gate:1][upd:1][state:1]` | Gated update: gate*upd + (1-gate)*state |
| 0x4013 | `TERNARY_BATCH_MATMUL` | `[dst:1][weights:1][input:1][_:1]` | Batch ternary matmul |
| 0x4014 | `EMBED_SEQUENCE` | `[dst:1][table:1][count:1][_:1]` | Sequential position embeddings |
| 0x4015 | `REDUCE_MEAN_DIM` | `[dst:1][src:1][dim:1][_:1]` | Reduce mean along dimension |

## Learning Operations

| Opcode | Mnemonic | Operands | Description |
|--------|----------|----------|-------------|
| 0x5000 | `MARK_ELIGIBILITY` | `[reg:1][_:3]` | Mark weights eligible by activity |
| 0x5004 | `ADD_BABBLE` | `[dst:1][src:1][_:2]` | Add exploration noise |
| 0x5012 | `MASTERY_UPDATE` | `[weights:1][input:1][output:1][_:1]` | Accumulate learning pressure |
| 0x5013 | `MASTERY_COMMIT` | `[weights:1][_:3]` | Apply pressure to weights |

## Control Flow

| Opcode | Mnemonic | Operands | Description |
|--------|----------|----------|-------------|
| 0x6000 | `LOOP` | `[count_hi:1][count_lo:1][_:2]` | Start loop (u16 iteration count) |
| 0x6001 | `END_LOOP` | `—` | End loop (jump back to LOOP) |
| 0x6002 | `BREAK` | `—` | Break from current loop |
| 0x6006 | `CALL` | `[target:4]` | Call subroutine (push PC) |
| 0x6007 | `RETURN` | `—` | Return from subroutine (pop PC) |
| 0x6008 | `JUMP` | `[target:4]` | Unconditional jump |

## Structural Plasticity

| Opcode | Mnemonic | Operands | Description |
|--------|----------|----------|-------------|
| 0x2003 | `WIRE_FORWARD` | `[dst:1][weights:1][input:1][_:1]` | Wire forward connection |
| 0x2004 | `WIRE_SKIP` | `[dst:1][a:1][b:1][_:1]` | Wire skip connection |
| 0x2008 | `GROW_NEURON` | `[reg:1][count:1][_:2]` | Add neurons to cold register |
| 0x2009 | `PRUNE_NEURON` | `[reg:1][index:1][_:2]` | Remove neuron from cold register |
| 0x200A | `INIT_RANDOM` | `[reg:1][seed:1][_:2]` | Initialize with random ternary weights |
