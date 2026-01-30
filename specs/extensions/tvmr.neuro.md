# tvmr.neuro (0x0005)

**Version:** 1.0.0  
**Instructions:** 20  
**ExtID:** 0x0005

Auto-generated from the live extension registry by `ternsig-spec`.

## Instructions

| Opcode | Mnemonic | Operands | Description |
|--------|----------|----------|-------------|
| 0x0000 | `CHEM_READ` | `[reg:1][imm8:1][_:2]` | Read chemical level into H[reg][0]. Yields ChemRead. |
| 0x0001 | `CHEM_SET` | `[reg:1][imm8:1][_:2]` | SET chemical level from H[reg][0] (authoritative). Yields ChemSet. |
| 0x0002 | `CHEM_INJECT` | `[reg:1][imm8:1][_:2]` | Additive chemical injection (phasic). Yields ChemInject. |
| 0x0003 | `FIELD_READ` | `[reg:1][imm8:1][_:2]` | Read field region slice into H[reg]. Yields FieldRead. |
| 0x0004 | `FIELD_WRITE` | `[reg:1][imm8:1][_:2]` | Write H[reg] to field region slice. Yields FieldWrite. |
| 0x0005 | `FIELD_TICK` | `[imm8:1][_:3]` | Advance field by 1 tick (decay, age). Yields FieldTick. |
| 0x0006 | `STIM_READ` | `[reg:1][_:3]` | Read all stimulation levels into H[reg]. Yields StimRead. |
| 0x0007 | `VALENCE_READ` | `[reg:1][_:3]` | Read valence [reward, punish] into H[reg]. Yields ValenceRead. |
| 0x0008 | `CONV_READ` | `[reg:1][_:3]` | Read convergence field state into H[reg]. Yields ConvRead. |
| 0x0009 | `TEMP_READ` | `[dst:1][src:1][_:2]` | Read cold register temperatures into H[dst]. |
| 0x000A | `TEMP_WRITE` | `[dst:1][src:1][_:2]` | Write temperatures from H[src] to cold register. |
| 0x000B | `FIELD_DECAY` | `[field_id:1][retention:1][fatigue:1][_:1]` | Apply metabolic decay to field. Yields FieldDecay. |
| 0x000C | `LATERAL_INHIBIT` | `[reg:1][imm8:1][_:2]` | Winner-take-some: dominant suppresses others. Yields LateralInhibit. |
| 0x000D | `EXHAUSTION_BOOST` | `[reg:1][imm8:1][_:2]` | Apply exhaustion decay boost. Yields ExhaustionBoost. |
| 0x000E | `NOVELTY_SCORE` | `[dst:1][src:1][_:2]` | Compute novelty z-scores from region energies. Yields NoveltyScore. |
| 0x000F | `FIELD_SLICE_READ` | `[target:1][field_id:1][slice:1][_:1]` | Read specific field slice into H[target]. Yields FieldSliceRead. |
| 0x0010 | `FIELD_SLICE_WRITE` | `[source:1][field_id:1][slice:1][_:1]` | Write H[source] to specific field slice. Yields FieldSliceWrite. |
| 0x0011 | `COACT_RECORD` | `[dst:1][src:1][_:2]` | Record co-activation between two registers. Yields CoActivation. |
| 0x0012 | `COACT_READ` | `[dst:1][src:1][_:2]` | Read co-activation matrix into H[target]. Yields CoActivationRead. |
| 0x0013 | `COACT_RESET` | `—` | Reset all co-activation counters. Yields CoActivationReset. |

## Assembly Syntax

```ternsig
.requires
  tvmr.neuro 0x0005

tvmr.neuro.CHEM_READ H0, 8
tvmr.neuro.CHEM_SET H0, 8
tvmr.neuro.CHEM_INJECT H0, 8
; ... 17 more instructions
```
