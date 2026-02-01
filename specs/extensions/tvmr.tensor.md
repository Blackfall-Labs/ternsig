# tvmr.tensor (0x0001)

**Version:** 1.0.0  
**Instructions:** 26  
**ExtID:** 0x0001

Auto-generated from the live extension registry by `ternsig-spec`.

## Instructions

| Opcode | Mnemonic | Operands | Description |
|--------|----------|----------|-------------|
| 0x0000 | `MATMUL` | `[dst:1][a:1][b:1][_:1]` | Matrix multiply: target = source @ aux |
| 0x0001 | `ADD` | `[dst:1][a:1][b:1][_:1]` | Element-wise add: target = source + aux |
| 0x0002 | `SUB` | `[dst:1][a:1][b:1][_:1]` | Element-wise subtract: target = source - aux |
| 0x0003 | `MUL` | `[dst:1][a:1][b:1][_:1]` | Element-wise multiply: target = source * aux |
| 0x0004 | `SCALE` | `[dst:1][src:1][imm16:2]` | Scale by constant: target = source * scale |
| 0x0005 | `SHIFT` | `[dst:1][a:1][b:1][_:1]` | Right shift: target = source >> amount |
| 0x0006 | `CLAMP` | `[dst:1][a:1][b:1][flags:1]` | Clamp to range: target = clamp(source, min, max) |
| 0x0007 | `CMP_GT` | `[dst:1][a:1][b:1][_:1]` | Compare greater: target = source > aux ? 1 : 0 |
| 0x0008 | `MAX_REDUCE` | `[dst:1][src:1][_:2]` | Max reduce: target[0] = max(source) |
| 0x0009 | `NEGATE` | `[dst:1][src:1][_:2]` | Negate: target = -source |
| 0x000A | `REDUCE_AVG` | `[dst:1][src:1][start:1][count:1]` | Reduce average: target[0] = mean(source[start..start+count]) |
| 0x000B | `REDUCE_MEAN_DIM` | `[dst:1][a:1][b:1][_:1]` | Reduce mean along dimension |
| 0x000C | `SLICE` | `[dst:1][src:1][start:1][len:1]` | Slice: target = source[start..start+len] |
| 0x000D | `ARGMAX` | `[dst:1][src:1][_:2]` | Argmax: target[0] = index of max in source |
| 0x000E | `CONCAT` | `[dst:1][a:1][b:1][_:1]` | Concatenate: target = concat(source, aux) |
| 0x000F | `SQUEEZE` | `[dst:1][src:1][_:2]` | Remove dimension (shape op) |
| 0x0010 | `UNSQUEEZE` | `[dst:1][src:1][_:2]` | Add dimension (shape op) |
| 0x0011 | `TRANSPOSE` | `[dst:1][src:1][_:2]` | Transpose dimensions (shape op) |
| 0x0012 | `ATTN` | `[q:1][k:1][v:1][out:1]` | Attention: out = softmax(q*k^T) * v |
| 0x0013 | `ROPE_APPLY` | `[reg:1][dim:1][pos:1][_:1]` | Apply rotary positional embedding |
| 0x0014 | `LAYERNORM` | `[src:1][dst:1][scale:1][bias:1]` | LayerNorm over last dimension |
| 0x0015 | `RMSNORM` | `[src:1][dst:1][scale:1][_:1]` | RMSNorm over last dimension |
| 0x0016 | `KV_APPEND` | `[cache:1][kv:1][pos:1][len:1]` | Append KV slice into cache |
| 0x0017 | `KV_READ` | `[cache:1][pos:1][out:1][len:1]` | Read KV slice from cache |
| 0x0018 | `KV_CLEAR` | `[cache:1][_:3]` | Clear KV cache |

## Assembly Syntax

```ternsig
.requires
  tvmr.tensor 0x0001

tvmr.tensor.MATMUL H0, H1, H2
tvmr.tensor.ADD H0, H1, H2
tvmr.tensor.SUB H0, H1, H2
; ... 15 more instructions
; Whisper extensions
tvmr.tensor.ATTN H0, H1, H2, H3
tvmr.tensor.ROPE_APPLY H0, 64, 0, 0
tvmr.tensor.LAYERNORM H0, H1, C0, C1
tvmr.tensor.RMSNORM H0, H1, C0, 0
tvmr.tensor.KV_APPEND H0, H1, 0, 0
tvmr.tensor.KV_READ H0, 0, H1, 0
tvmr.tensor.KV_CLEAR H0
```
