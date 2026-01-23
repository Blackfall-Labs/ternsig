# ternsig

Signal foundation for CPU-only neural networks.

## The Equation

```
s = p * m

p in {-1, 0, +1}    polarity
m in {0..255}       magnitude
```

Two bytes per weight. Integer arithmetic only. No floats. No GPU.

## What This Changes

Traditional neural networks use 32-bit floats (4 bytes per weight). Ternsig uses Signal (2 bytes per weight):

| Property | Float | Signal |
|----------|-------|--------|
| Size | 4 bytes | 2 bytes |
| Arithmetic | FP multiply-add | Integer multiply-add |
| Hardware | GPU preferred | CPU native |
| Training | Gradient descent | Mastery learning |

## Core Components

### Signal

```rust
pub struct Signal {
    pub polarity: i8,    // -1, 0, or +1
    pub magnitude: u8,   // 0-255
}
```

Effective value = `polarity * magnitude`. Range: -255 to +255.

### Signal ISA (TensorISA)

Hot-reloadable neural network definitions as `.ternsig` files:

```asm
.registers
    H0: i32[12]          ; input activations
    H1: i32[32]          ; hidden layer
    C0: ternary[32, 12]  key="chip.audio.w1"

.program
    load_input    H0
    ternary_matmul H1, C0, H0
    relu          H1, H1
    store_output  H1
    halt
```

### Mastery Learning

Pure integer adaptive learning. 23ms to 90% accuracy.

```rust
use ternsig::{MasteryConfig, MasteryState, mastery_update};

let mut weights = init_random_structure(16, 42);
let mut state = MasteryState::new(16);
let config = MasteryConfig::default();

// Learning loop
for sample in samples {
    let activations = forward(&weights, &sample.input);
    let direction = if sample.target > output { 1 } else { -1 };
    mastery_update(&mut weights, &mut state, &activations, direction, &config);
}
```

Key principles:
- **Peak-relative gating**: Only neurons above `max_activation / 4` participate (not percentile-based)
- **Sustained pressure**: Changes require accumulated evidence
- **Weaken before flip**: Magnitude depletes before polarity changes

### Thermogram Integration

Persistent weight storage with temperature lifecycle:

- **HOT**: Actively learning, high plasticity
- **WARM**: Recently learned, moderate plasticity
- **COOL**: Stable, low plasticity
- **COLD**: Long-term memory, frozen

## Usage

```toml
[dependencies]
ternsig = "0.5"
```

```rust
use ternsig::{Signal, assemble, TensorInterpreter};

// Load chip definition
let program = assemble(include_str!("classifier.ternsig"))?;
let mut interpreter = TensorInterpreter::from_program(&program);

// Forward pass
let input: Vec<Signal> = /* your input */;
let output = interpreter.forward(&input)?;
```

## Migration from 0.4.x

`TernarySignal` has been renamed to `Signal`. A deprecated type alias is provided for backwards compatibility:

```rust
// Old (still works but deprecated)
use ternsig::TernarySignal;

// New (preferred)
use ternsig::Signal;
```

## License

MIT OR Apache-2.0
