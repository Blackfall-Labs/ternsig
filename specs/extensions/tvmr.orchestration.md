# tvmr.orchestration (0x0007)

**Version:** 1.0.0  
**Instructions:** 14  
**ExtID:** 0x0007

Auto-generated from the live extension registry by `ternsig-spec`.

## Instructions

| Opcode | Mnemonic | Operands | Description |
|--------|----------|----------|-------------|
| 0x0000 | `MODEL_LOAD` | `[reg:1][imm8:1][_:2]` | Load model into table slot. Yields ModelLoad. |
| 0x0001 | `MODEL_EXEC` | `[reg:1][imm8:1][_:2]` | Execute model from table slot. Yields ModelExec. |
| 0x0002 | `MODEL_INPUT` | `[reg:1][imm8:1][_:2]` | Set input register for table slot. |
| 0x0003 | `MODEL_OUTPUT` | `[reg:1][imm8:1][_:2]` | Get output from table slot into register. |
| 0x0004 | `MODEL_UNLOAD` | `[imm8:1][_:3]` | Unload model from table slot. |
| 0x0005 | `MODEL_STATUS` | `[reg:1][imm8:1][_:2]` | Get model status: 0=empty, 1=loaded, 2=running. |
| 0x0006 | `MODEL_RELOAD` | `[imm8:1][_:3]` | Hot-reload model in slot. Yields ModelReload. |
| 0x0007 | `MODEL_CHAIN` | `[slot1:1][slot2:1][_:2]` | Chain slot1 output to slot2 input. |
| 0x0008 | `ROUTE_INPUT` | `[reg:1][imm8:1][_:2]` | Route H[reg] to region ID. Yields RouteInput. |
| 0x0009 | `REGION_FIRE` | `[reg:1][imm8:1][_:2]` | Fire region, output into H[reg]. Yields RegionFire. |
| 0x000A | `COLLECT_OUTPUTS` | `[reg:1][_:3]` | Aggregate region outputs into H[reg]. Yields CollectOutputs. |
| 0x000B | `REGION_STATUS` | `[reg:1][imm8:1][_:2]` | Read region status: 0=idle, 1=active, 2=firing. |
| 0x000C | `REGION_ENABLE` | `[imm8:1][_:3]` | Enable region for routing. Yields RegionEnable. |
| 0x000D | `REGION_DISABLE` | `[imm8:1][_:3]` | Disable region. Yields RegionDisable. |

## Assembly Syntax

```ternsig
.requires
  tvmr.orchestration 0x0007

tvmr.orchestration.MODEL_LOAD H0, 8
tvmr.orchestration.MODEL_EXEC H0, 8
tvmr.orchestration.MODEL_INPUT H0, 8
; ... 11 more instructions
```
