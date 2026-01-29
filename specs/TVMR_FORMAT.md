# TVMR Binary Format Specification v1

## Instruction Encoding

Every instruction is 8 bytes, fixed width.

```
Byte:  0     1     2     3     4     5     6     7
     [--ExtID--] [--OpCode--]  [A]   [B]   [C]   [D]
       u16 BE      u16 BE     operand bytes
```

- **ExtID = 0x0000**: Core ISA (inline dispatch, zero overhead)
- **ExtID != 0x0000**: Extension dispatch via registry

## Operand Patterns

| Pattern         | Byte Layout           | Example              |
|-----------------|-----------------------|----------------------|
| None            | `[_:4]`               | HALT, NOP            |
| Reg             | `[reg:1][_:3]`        | FREE_TENSOR          |
| RegReg          | `[dst:1][src:1][_:2]` | RELU, SIGMOID        |
| RegRegReg       | `[dst:1][src:1][aux:1][_:1]` | ADD, MATMUL   |
| RegRegRegFlags  | `[dst:1][src:1][aux:1][flags:1]` | CLAMP      |
| RegImm8         | `[reg:1][imm:1][_:2]` | CHEM_READ            |
| RegRegImm16     | `[dst:1][src:1][imm_hi:1][imm_lo:1]` | SCALE   |
| Imm8            | `[imm:1][_:3]`        | HALT_REGION          |
| Imm32           | `[imm_3:1][imm_2:1][imm_1:1][imm_0:1]` | JUMP   |
| Custom          | Per-instruction       | GROW_NEURON          |

## Register Encoding

1 byte: `[BANK:2 bits][INDEX:6 bits]`

```
0b00_xxxxxx  (0x00-0x3F)  Hot bank    Activations (volatile)
0b01_xxxxxx  (0x40-0x7F)  Cold bank   Weights (persistent)
0b10_xxxxxx  (0x80-0xBF)  Param bank  Scalars
0b11_xxxxxx  (0xC0-0xFF)  Shape bank  Dimensions
```

`0xFF` = null register (no-op target/source).

## Binary File Layout

```
Offset   Size   Content
0x00     48     TVMR Header
0x30     N*8    Extension Dependency Table
0x30+E   var    Register Definitions
0x30+E+R M*8    Instructions
```

### Header (48 bytes)

```
Offset  Size  Field               Encoding
0x00    4     Magic               "TVMR" (0x54 0x56 0x4D 0x52)
0x04    2     Version             u16 LE (currently 1)
0x06    2     Flags               u16 LE (bit 0=compressed, 1=encrypted, 2=metadata)
0x08    4     ext_table_offset    u32 LE (byte offset from file start)
0x0C    4     ext_table_count     u32 LE (number of extension entries)
0x10    4     reg_defs_offset     u32 LE
0x14    4     reg_defs_count      u32 LE (number of register definitions)
0x18    4     instr_offset        u32 LE
0x1C    4     instr_count         u32 LE
0x20    8     checksum            u64 LE (xxhash64 over ext_table + reg_defs + instructions)
0x28    8     reserved            zeroes
```

### Extension Dependency Table

Each entry is 8 bytes:

```
Offset  Size  Field
0x00    2     ext_id          u16 LE
0x02    2     version_major   u16 LE
0x04    2     version_minor   u16 LE
0x06    2     version_patch   u16 LE
```

### Register Definition

Variable-length per register:

```
Offset  Size     Field
0x00    1        register_id     u8 (bank:2 | index:6)
0x01    2        type_id         u16 LE (see Type System)
0x03    1        flags           u8 (bit 0=allocated, bit 1=frozen)
0x04    1        ndims           u8
0x05    2*N      dimensions      u16 LE each
0x05+2N 1        key_len         u8
0x06+2N key_len  thermogram_key  UTF-8 bytes
```

## Type System

TypeId is u16. Primitives are universal. Domain types are extension-registered.

```
0x0000  void          0 bytes
0x0001  bool          1 byte
0x0002  u8            1 byte
0x0003  i8            1 byte
0x0004  u16           2 bytes
0x0005  i16           2 bytes
0x0006  u32           4 bytes
0x0007  i32           4 bytes
0x0008  u64           8 bytes
0x0009  i64           8 bytes
0x000A  f16           2 bytes
0x000B  f32           4 bytes
0x000C  f64           8 bytes
0x0100  signal        2 bytes (polarity:i8, magnitude:u8)
0x0101  packed_signal 1 byte  (4 ternary values, 2 bits each)
0x0102  chemical      4 bytes (da:u8, serotonin:u8, ne:u8, gaba:u8)
```

## Assembly Source Format

```asm
; Comment
.meta
    name    "program_name"
    version 1

.requires
    extension_name  0xNNNN      ; explicit ext_id
    extension_name              ; auto-resolved

.registers
    C0: ternary[rows, cols]  key="thermo.key"
    H0: i32[size]
    P0: i32[1]

.program
    mnemonic dst, src, aux       ; core ISA (unqualified)
    ext.mnemonic dst, src        ; extension-qualified
    label:                       ; jump target
    jump label
```

## Legacy Format Detection

Deserializer checks first 4 bytes:
- `"TVMR"` (0x54564D52) -> TVMR v1 format (48-byte header)
- `"TERN"` (0x5445524E) -> Legacy format (32-byte header, 1-byte Dtype)
- Anything else -> error
