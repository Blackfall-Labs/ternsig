# TVMR Instruction Set Reference

Auto-generated from the live extension registry by `ternsig-spec`.

**Total: 198 instructions** (52 core ISA + 146 across 11 extensions)

## Core ISA

- [Core ISA](core/CORE_ISA.md) — 52 instructions (system, register, forward, ternary, learning, control flow, structural)

## Extensions

| ExtID | Name | Instructions | Version | Spec |
|-------|------|-------------|---------|------|
| 0x0001 | tvmr.tensor | 18 | 1.0.0 | [spec](extensions/tvmr.tensor.md) |
| 0x0002 | tvmr.ternary | 14 | 1.0.0 | [spec](extensions/tvmr.ternary.md) |
| 0x0003 | tvmr.activation | 5 | 1.0.0 | [spec](extensions/tvmr.activation.md) |
| 0x0004 | tvmr.learning | 21 | 1.0.0 | [spec](extensions/tvmr.learning.md) |
| 0x0005 | tvmr.neuro | 20 | 1.0.0 | [spec](extensions/tvmr.neuro.md) |
| 0x0006 | tvmr.arch | 13 | 1.0.0 | [spec](extensions/tvmr.arch.md) |
| 0x0007 | tvmr.orchestration | 14 | 1.0.0 | [spec](extensions/tvmr.orchestration.md) |
| 0x0008 | tvmr.lifecycle | 13 | 1.0.0 | [spec](extensions/tvmr.lifecycle.md) |
| 0x0009 | tvmr.ipc | 8 | 1.0.0 | [spec](extensions/tvmr.ipc.md) |
| 0x000A | tvmr.test | 8 | 1.0.0 | [spec](extensions/tvmr.test.md) |
| 0x000B | tvmr.bank | 12 | 1.0.0 | [spec](extensions/tvmr.bank.md) |

## Operand Patterns

All instructions use an 8-byte format: `[ExtID:2][OpCode:2][A:1][B:1][C:1][D:1]`

The 4 operand bytes `[A][B][C][D]` are interpreted per the instruction's operand pattern:

| Pattern | Layout | Typical Use |
|---------|--------|-------------|
| None | `[_:4]` | System ops (NOP, HALT) |
| Reg | `[reg:1][_:3]` | Single register ops |
| RegReg | `[dst:1][src:1][_:2]` | Unary transforms |
| RegRegReg | `[dst:1][a:1][b:1][_:1]` | Binary ops |
| RegRegRegFlags | `[dst:1][a:1][b:1][flags:1]` | Flagged binary ops |
| RegRegImm16 | `[dst:1][src:1][imm16:2]` | Register + immediate |
| RegImm16 | `[dst:1][_:1][imm16:2]` | Register + immediate |
| RegImm8 | `[reg:1][imm8:1][_:2]` | Register + small immediate |
| Imm32 | `[imm32:4]` | Jump targets |
| Imm16 | `[imm16:2][_:2]` | Loop counts |
| Imm8 | `[imm8:1][_:3]` | Small constants |
| RegCondRegReg | `[dst:1][cond:1][a:1][b:1]` | Conditional select |
| Custom | (varies) | Extension-specific |
