# tvmr.tensor (0x0001)

**Version:** 1.0.0  
**Instructions:** 18  
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

## Assembly Syntax

```ternsig
.requires
  tvmr.tensor 0x0001

tvmr.tensor.MATMUL H0, H1, H2
tvmr.tensor.ADD H0, H1, H2
tvmr.tensor.SUB H0, H1, H2
; ... 15 more instructions
```
