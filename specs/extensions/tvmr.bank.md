# tvmr.bank (0x000B)

**Version:** 1.0.0  
**Instructions:** 12  
**ExtID:** 0x000B

Auto-generated from the live extension registry by `ternsig-spec`.

## Instructions

| Opcode | Mnemonic | Operands | Description |
|--------|----------|----------|-------------|
| 0x0000 | `BANK_QUERY` | `[target:1][source:1][bank_slot:1][top_k:1]` | Query bank by vector similarity. Yields BankQuery. |
| 0x0001 | `BANK_WRITE` | `[target:1][source:1][bank_slot:1][_:1]` | Write entry to bank. Yields BankWrite. |
| 0x0002 | `BANK_LOAD` | `[target:1][source:1][bank_slot:1][_:1]` | Load full entry vector by id. Yields BankLoad. |
| 0x0003 | `BANK_LINK` | `[source:1][edge_type:1][bank_slot:1][_:1]` | Add typed edge between entries. Yields BankLink. |
| 0x0004 | `BANK_TRAVERSE` | `[target:1][source:1][bank_slot:1][packed:1]` | Traverse edges from entry. packed=[edge_type:4][depth:4]. Yields BankTraverse. |
| 0x0005 | `BANK_TOUCH` | `[source:1][bank_slot:1][_:2]` | Touch entry to update access tick/count. Yields BankTouch. |
| 0x0006 | `BANK_DELETE` | `[source:1][bank_slot:1][_:2]` | Delete entry from bank. Yields BankDelete. |
| 0x0007 | `BANK_COUNT` | `[target:1][bank_slot:1][_:2]` | Get entry count for bank. Yields BankCount. |
| 0x0008 | `BANK_PROMOTE` | `[source:1][bank_slot:1][_:2]` | Promote entry temperature (Hot→Warm, etc). Yields BankPromote. |
| 0x0009 | `BANK_DEMOTE` | `[source:1][bank_slot:1][_:2]` | Demote entry temperature (Warm→Hot, etc). Yields BankDemote. |
| 0x000A | `BANK_EVICT` | `[bank_slot:1][count:1][_:2]` | Evict cold/low-scoring entries. Yields BankEvict. |
| 0x000B | `BANK_COMPACT` | `[bank_slot:1][_:3]` | Compact bank after pruning. Yields BankCompact. |

## Assembly Syntax

```ternsig
.requires
  tvmr.bank 0x000B

tvmr.bank.BANK_QUERY H0, H1, 0, 0
tvmr.bank.BANK_WRITE H0, H1, 0, 0
tvmr.bank.BANK_LOAD H0, H1, 0, 0
; ... 9 more instructions
```
