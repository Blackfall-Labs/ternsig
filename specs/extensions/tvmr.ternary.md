# tvmr.ternary (0x0002)

**Version:** 1.0.0  
**Instructions:** 14  
**ExtID:** 0x0002

Auto-generated from the live extension registry by `ternsig-spec`.

## Instructions

| Opcode | Mnemonic | Operands | Description |
|--------|----------|----------|-------------|
| 0x0000 | `TERNARY_MATMUL` | `[dst:1][a:1][b:1][_:1]` | Ternary matmul: target = cold_weights @ hot_input |
| 0x0001 | `TERNARY_BATCH_MATMUL` | `[dst:1][a:1][b:1][_:1]` | Batch ternary matmul: target[i] = weights @ input[i] |
| 0x0002 | `TERNARY_ADD_BIAS` | `[dst:1][a:1][b:1][_:1]` | Add signal bias: target = source + cold_bias |
| 0x0003 | `DEQUANTIZE` | `[dst:1][src:1][imm16:2]` | Dequantize: target = source >> scale |
| 0x0004 | `EMBED_LOOKUP` | `[dst:1][a:1][b:1][_:1]` | Embedding lookup: target[i] = table[indices[i]] |
| 0x0005 | `EMBED_SEQUENCE` | `[dst:1][table:1][count:1][_:1]` | Sequential embedding: target[i] = table[i] |
| 0x0006 | `GATE_UPDATE` | `[dst:1][gate:1][update:1][state:1]` | Gated update: target = gate * update + (1-gate) * state |
| 0x0007 | `QUANTIZE` | `[dst:1][src:1][_:2]` | Quantize to ternary signal |
| 0x0008 | `PACK_TERNARY` | `[dst:1][src:1][_:2]` | Pack signals to 2-bit representation |
| 0x0009 | `UNPACK_TERNARY` | `[dst:1][src:1][_:2]` | Unpack 2-bit to Signal |
| 0x000A | `APPLY_POLARITY` | `[dst:1][src:1][_:2]` | Apply polarity update to weight |
| 0x000B | `APPLY_MAGNITUDE` | `[dst:1][src:1][_:2]` | Apply magnitude update to weight |
| 0x000C | `THRESHOLD_POLARITY` | `[dst:1][src:1][_:2]` | Check polarity flip threshold (hysteresis) |
| 0x000D | `ACCUMULATE_PRESSURE` | `[dst:1][src:1][_:2]` | Accumulate polarity pressure |

## Assembly Syntax

```ternsig
.requires
  tvmr.ternary 0x0002

tvmr.ternary.TERNARY_MATMUL H0, H1, H2
tvmr.ternary.TERNARY_BATCH_MATMUL H0, H1, H2
tvmr.ternary.TERNARY_ADD_BIAS H0, H1, H2
; ... 11 more instructions
```
