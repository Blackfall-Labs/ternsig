# tvmr.test (0x000A)

**Version:** 1.0.0  
**Instructions:** 8  
**ExtID:** 0x000A

Auto-generated from the live extension registry by `ternsig-spec`.

## Instructions

| Opcode | Mnemonic | Operands | Description |
|--------|----------|----------|-------------|
| 0x0000 | `ASSERT_EQ` | `[dst:1][src:1][_:2]` | Assert two registers are equal |
| 0x0001 | `ASSERT_GT` | `[dst:1][src:1][_:2]` | Assert first register greater than second |
| 0x0002 | `ASSERT_ACTIVE` | `[reg:1][_:3]` | Assert register has non-zero values |
| 0x0003 | `ASSERT_RANGE` | `[reg:1][min:1][max:1][_:1]` | Assert all values in range [min, max] |
| 0x0004 | `TEST_BEGIN` | `[imm8:1][_:3]` | Mark beginning of test case |
| 0x0005 | `TEST_END` | `—` | Mark end of test case |
| 0x0006 | `EXPECT_CHEM` | `[chemical:1][min:1][max:1][_:1]` | Assert chemical level in range |
| 0x0007 | `SNAPSHOT` | `[reg:1][_:3]` | Snapshot register state for comparison |

## Assembly Syntax

```ternsig
.requires
  tvmr.test 0x000A

tvmr.test.ASSERT_EQ H0, H1
tvmr.test.ASSERT_GT H0, H1
tvmr.test.ASSERT_ACTIVE H0
; ... 5 more instructions
```
