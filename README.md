# Ternsig Virtual Mainframe Runtime (TVMR)

A CPU-only virtual machine for self-modifying neural architectures.

```
Signal:  s = p * m    p in {-1, 0, +1}    m in {0..255}
Format:  [ExtID:2][OpCode:2][A:1][B:1][C:1][D:1]   8 bytes fixed
```

Two bytes per weight. Integer arithmetic. No floats. No GPU. Hot-reloadable.

## What TVMR Is

TVMR is a **virtual mainframe** — a fixed instruction set that runs programs defining neural architectures, learning rules, structural plasticity, inter-region communication, and lifecycle management. The Rust code is the kernel. Everything else is a program.

```
Rust Kernel (fixed, rarely recompiled)
  ├── TVMR instruction dispatch
  ├── Extension registry
  ├── Hot-reload engine
  └── Substrate I/O bridges

Programs (.ternsig files, hot-reloadable)
  ├── Region firmware         (forward pass, learning, gating)
  ├── Boot sequences          (region topology, wiring)
  ├── Orchestration scripts   (model loading, chaining)
  └── Test harnesses          (assertions, snapshots)
```

**Why this matters:** Changing a brain region's behavior means editing a `.ternsig` file and hot-reloading. No Rust recompilation. No cargo build. No waiting.

## Architecture

### Instruction Format

Every instruction is exactly 8 bytes:

```
[ExtID:2][OpCode:2][A:1][B:1][C:1][D:1]
 bytes 0-1  2-3     4    5    6    7
```

- **ExtID** (u16 BE): Extension identifier. `0x0000` = core ISA.
- **OpCode** (u16 BE): Extension-local opcode. 65,536 per extension.
- **A, B, C, D**: Operand bytes. Meaning defined per-instruction.

### Register Banks

```
Hot   (0x00-0x3F)  Activations, intermediates (volatile)
Cold  (0x40-0x7F)  Weights, Signal storage (persistent via Thermogram)
Param (0x80-0xBF)  Scalars (learning_rate, babble_scale)
Shape (0xC0-0xFF)  Dimension metadata
```

4 banks x 64 registers = 256 total.

### Extension System

Programs declare their dependencies. Extensions are registered at boot. Each extension owns a 2-byte ID and up to 65,536 opcodes.

```asm
.requires
    ternary    0x0002
    activation 0x0003
    learning   0x0004

.registers
    C0: ternary[32, 12]  key="region.w1"
    H0: i32[12]
    H1: i32[32]

.program
    load_input H0
    ternary_matmul H1, C0, H0
    activation.relu H1, H1
    store_output H1
    halt
```

### Standard Extensions

| ExtID  | Name          | Ops | Status     | Domain |
|--------|---------------|-----|------------|--------|
| 0x0000 | core          | 104 | Built-in   | System, registers, arithmetic, control flow, debug |
| 0x0001 | tensor        | 18  | Functional | Matrix ops, reductions, slicing |
| 0x0002 | ternary       | 14  | Functional | Signal matmul, quantization, gating |
| 0x0003 | activation    | 5   | Functional | ReLU, sigmoid, tanh, softmax, GELU |
| 0x0004 | learning      | 20  | Partial    | Mastery, CHL, babble, eligibility |
| 0x0005 | neuro         | 8   | Planned    | Chemical/field/substrate I/O |
| 0x0006 | arch          | 11  | Functional | Structural plasticity (grow/prune/wire) |
| 0x0007 | orchestration | 8   | Planned    | Model table, chaining, hot-reload |
| 0x0008 | lifecycle     | 8   | Planned    | Boot phases, tick/level reads, logging |
| 0x0009 | ipc           | 8   | Planned    | Inter-region signals, barriers |
| 0x000A | test          | 8   | Planned    | Assertions, snapshots, test harnesses |

**212 total instructions.** See `specs/` for complete per-extension documentation.

### Binary Format (TVMR v1)

48-byte header. Auto-detect deserializer reads both TVMR and legacy TERN formats.

```
HEADER (48 bytes)
  Magic "TVMR" | Version | Flags
  Extension table offset/count
  Register defs offset/count
  Instruction offset/count
  xxhash64 checksum | Reserved

EXTENSION TABLE  (8 bytes per dep)
REGISTER DEFS    (variable, TypeId-based)
INSTRUCTIONS     (8 bytes each)
```

### Cross-Language Type System

Types are u16 IDs, portable across any TVMR engine implementation:

```
0x0000-0x00FF  Primitives   (void, bool, u8..u64, i8..i64, f16..f64)
0x0100-0x01FF  Domain types (signal, packed_signal, chemical)
0x0200+        User types   (future)
```

## Signal Arithmetic

The core data type. Two bytes. Integer-only.

```rust
pub struct Signal {
    pub polarity: i8,    // -1, 0, or +1
    pub magnitude: u8,   // 0-255
}
// Effective value = polarity * magnitude
// Range: -255 to +255
```

### Ternary Matrix Multiply

Temperature-gated signal flow:

```
for each output[i]:
    accumulator = 0
    for each input[j]:
        signal = weights[i][j]
        conductance = temperature[i][j].conductance()  // 0.0-1.0
        if conductance > activation_threshold:
            accumulator += signal.polarity * signal.magnitude * input[j]
    output[i] = accumulator
```

Weights that haven't been used cool down. Cold weights conduct less. This is biological — unused synapses weaken.

## Mastery Learning

Pure integer adaptive learning. No gradients. No backpropagation.

- **Peak-relative gating**: Only neurons above `max_activation / 4` participate
- **Sustained pressure**: Changes require accumulated evidence via pressure registers
- **Weaken before flip**: Magnitude depletes before polarity changes
- **Dopamine gating**: Learning requires `chemical_state.dopamine >= threshold`
- **Temperature lifecycle**: HOT (learning) -> WARM -> COOL -> COLD (frozen)

## Usage

```toml
[dependencies]
ternsig = "0.5"
```

```rust
use ternsig::vm::{assemble, Interpreter};

let program = assemble(include_str!("region.ternsig"))?;
let mut vm = Interpreter::from_program(&program);
let output = vm.forward(&input)?;
```

### Hot Reload

```rust
use ternsig::vm::{HotReloadManager, ReloadableInterpreter};

let mut reloadable = ReloadableInterpreter::new(program);
let mut reload_mgr = HotReloadManager::watch("firmware/")?;

// In your loop:
if let Some(event) = reload_mgr.poll() {
    reloadable.reload(event)?;
}
```

## Project Structure

```
ternsig/
├── src/
│   ├── lib.rs              # Crate root
│   ├── ternary.rs          # Signal type and operations
│   ├── thermo.rs           # Thermogram integration
│   ├── learning.rs         # Mastery learning algorithm
│   ├── loader.rs           # Program loader (.card format)
│   ├── validate.rs         # Source validation
│   └── vm/
│       ├── mod.rs           # VM module root, re-exports
│       ├── instruction.rs   # 8-byte TVMR instruction format
│       ├── interpreter/     # Execution engine
│       ├── assembler.rs     # .ternsig source -> program
│       ├── binary.rs        # TVMR v1 binary serialization
│       ├── extension.rs     # Extension trait, ExecutionContext
│       ├── registry.rs      # ExtensionRegistry
│       ├── types.rs         # Cross-language TypeId system
│       ├── validator.rs     # Pre-execution validation
│       ├── hot_reload.rs    # File-watching hot reload
│       └── extensions/      # Standard extension implementations
│           ├── tensor.rs      # 0x0001: Matrix operations
│           ├── ternary.rs     # 0x0002: Signal operations
│           ├── activation.rs  # 0x0003: Activation functions
│           ├── learning.rs    # 0x0004: Learning algorithms
│           ├── neuro.rs       # 0x0005: Substrate I/O
│           ├── arch.rs        # 0x0006: Structural plasticity
│           ├── orchestration.rs # 0x0007: Model management
│           ├── lifecycle.rs   # 0x0008: Boot/phase management
│           ├── ipc.rs         # 0x0009: Inter-region comms
│           └── test_ext.rs    # 0x000A: Testing/assertions
├── specs/                    # Instruction specifications
└── Cargo.toml
```

## License

MIT OR Apache-2.0
