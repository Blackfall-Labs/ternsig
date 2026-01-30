# tvmr.ipc (0x0009)

**Version:** 1.0.0  
**Instructions:** 8  
**ExtID:** 0x0009

Auto-generated from the live extension registry by `ternsig-spec`.

## Instructions

| Opcode | Mnemonic | Operands | Description |
|--------|----------|----------|-------------|
| 0x0000 | `SEND_SIGNAL` | `[reg:1][imm8:1][_:2]` | Send signal to another region |
| 0x0001 | `RECV_SIGNAL` | `[reg:1][imm8:1][_:2]` | Receive signal from another region |
| 0x0002 | `BROADCAST` | `[reg:1][imm8:1][_:2]` | Broadcast signal to all regions |
| 0x0003 | `SUBSCRIBE` | `[reg:1][imm8:1][_:2]` | Subscribe to signals from a region |
| 0x0004 | `MAILBOX_PEEK` | `[reg:1][imm8:1][_:2]` | Peek at mailbox without consuming |
| 0x0005 | `MAILBOX_POP` | `[reg:1][imm8:1][_:2]` | Pop from mailbox (consume) |
| 0x0006 | `BARRIER_WAIT` | `[imm8:1][_:3]` | Wait at synchronization barrier |
| 0x0007 | `ATOMIC_CAS` | `[dst:1][a:1][b:1][_:1]` | Atomic compare-and-swap |

## Assembly Syntax

```ternsig
.requires
  tvmr.ipc 0x0009

tvmr.ipc.SEND_SIGNAL H0, 8
tvmr.ipc.RECV_SIGNAL H0, 8
tvmr.ipc.BROADCAST H0, 8
; ... 5 more instructions
```
