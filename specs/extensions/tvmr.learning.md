# tvmr.learning (0x0004)

**Version:** 1.0.0  
**Instructions:** 21  
**ExtID:** 0x0004

Auto-generated from the live extension registry by `ternsig-spec`.

## Instructions

| Opcode | Mnemonic | Operands | Description |
|--------|----------|----------|-------------|
| 0x0000 | `MASTERY_UPDATE` | `[weights:1][activity:1][direction:1][flags:1]` | Accumulate learning pressure from error and activity |
| 0x0001 | `MASTERY_COMMIT` | `[weights:1][_:1][_:1][flags:1]` | Commit accumulated pressure to weight changes |
| 0x0002 | `ADD_BABBLE` | `[reg:1][_:3]` | Add exploration noise to activations |
| 0x0003 | `LOAD_TARGET` | `[reg:1][_:3]` | Load target buffer into hot register |
| 0x0004 | `MARK_ELIGIBILITY` | `[dst:1][src:1][_:2]` | Mark weights eligible for update based on activity |
| 0x0005 | `CHL_FREE_START` | `—` | CHL: Start free phase |
| 0x0006 | `CHL_FREE_RECORD` | `[dst:1][src:1][_:2]` | CHL: Record free phase correlations |
| 0x0007 | `CHL_CLAMP_START` | `—` | CHL: Start clamped phase |
| 0x0008 | `CHL_CLAMP_RECORD` | `[dst:1][src:1][_:2]` | CHL: Record clamped phase correlations |
| 0x0009 | `CHL_UPDATE` | `[reg:1][_:3]` | CHL: Compute weight updates |
| 0x000A | `CHL_BACKPROP_CLAMP` | `[dst:1][src:1][_:2]` | CHL: Propagate clamped signal backward |
| 0x000B | `DECAY_ELIGIBILITY` | `[reg:1][_:3]` | Decay eligibility traces |
| 0x000C | `COMPUTE_ERROR` | `[dst:1][src:1][_:2]` | Compute error between target and output |
| 0x000D | `UPDATE_WEIGHTS` | `[reg:1][_:3]` | Apply weight updates from eligibility and error |
| 0x000E | `DECAY_BABBLE` | `—` | Decay babble exploration scale |
| 0x000F | `COMPUTE_RPE` | `[dst:1][src:1][_:2]` | Compute Reward Prediction Error |
| 0x0010 | `GATE_ERROR` | `[reg:1][imm8:1][_:2]` | Gate learning based on error threshold |
| 0x0011 | `CHECKPOINT_WEIGHTS` | `[reg:1][_:3]` | Checkpoint weights for potential rollback |
| 0x0012 | `ROLLBACK_WEIGHTS` | `[reg:1][_:3]` | Rollback to checkpointed weights |
| 0x0013 | `CONSOLIDATE` | `—` | Consolidate hot → cold (Thermogram) |
| 0x0014 | `DISCARD_CHECKPOINT` | `[reg:1][_:3]` | Discard a persisted checkpoint (cleanup after successful op) |

## Assembly Syntax

```ternsig
.requires
  tvmr.learning 0x0004

tvmr.learning.MASTERY_UPDATE H0, H1, 0, 0
tvmr.learning.MASTERY_COMMIT H0, H1, 0, 0
tvmr.learning.ADD_BABBLE H0
; ... 18 more instructions
```
