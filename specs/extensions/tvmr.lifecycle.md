# tvmr.lifecycle (0x0008)

**Version:** 1.0.0  
**Instructions:** 13  
**ExtID:** 0x0008

Auto-generated from the live extension registry by `ternsig-spec`.

## Instructions

| Opcode | Mnemonic | Operands | Description |
|--------|----------|----------|-------------|
| 0x0000 | `PHASE_READ` | `[reg:1][_:3]` | Read current boot phase into register |
| 0x0001 | `TICK_READ` | `[reg:1][_:3]` | Read current tick count into register |
| 0x0002 | `LEVEL_READ` | `[reg:1][_:3]` | Read current neuronal level into register |
| 0x0003 | `INIT_THERMO` | `[reg:1][_:3]` | Initialize thermogram for cold register |
| 0x0004 | `SAVE_THERMO` | `[reg:1][_:3]` | Save cold register to thermogram |
| 0x0005 | `LOAD_THERMO` | `[reg:1][_:3]` | Load cold register from thermogram |
| 0x0006 | `LOG_EVENT` | `[reg:1][imm8:1][_:2]` | Log a lifecycle event |
| 0x0007 | `HALT_REGION` | `[imm8:1][_:3]` | Halt a specific brain region |
| 0x0008 | `TXN_BEGIN` | `[imm8:1][_:3]` | Begin a persistence transaction |
| 0x0009 | `TXN_COMMIT` | `[imm8:1][_:3]` | Commit all buffered writes atomically |
| 0x000A | `TXN_ROLLBACK` | `[imm8:1][_:3]` | Discard all buffered writes in transaction |
| 0x000B | `LOAD_WEIGHTS` | `[reg:1][imm8:1][_:2]` | Load weights from persistent storage by key_id (from param register) |
| 0x000C | `STORE_WEIGHTS` | `[reg:1][imm8:1][_:2]` | Store weights to persistent storage by key_id (from param register) |

## Assembly Syntax

```ternsig
.requires
  tvmr.lifecycle 0x0008

tvmr.lifecycle.PHASE_READ H0
tvmr.lifecycle.TICK_READ H0
tvmr.lifecycle.LEVEL_READ H0
; ... 10 more instructions
```
