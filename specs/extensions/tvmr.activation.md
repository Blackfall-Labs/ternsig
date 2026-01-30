# tvmr.activation (0x0003)

**Version:** 1.0.0  
**Instructions:** 5  
**ExtID:** 0x0003

Auto-generated from the live extension registry by `ternsig-spec`.

## Instructions

| Opcode | Mnemonic | Operands | Description |
|--------|----------|----------|-------------|
| 0x0000 | `RELU` | `[dst:1][src:1][_:2]` | ReLU: target = max(0, source) |
| 0x0001 | `SIGMOID` | `[dst:1][src:1][_:2]` | Sigmoid: target = sigmoid(source) [integer approx] |
| 0x0002 | `TANH` | `[dst:1][src:1][_:2]` | Tanh: target = tanh(source) [integer approx] |
| 0x0003 | `SOFTMAX` | `[dst:1][src:1][_:2]` | Softmax: target = softmax(source) [integer approx, 0-255 range] |
| 0x0004 | `GELU` | `[dst:1][src:1][_:2]` | GELU: target = gelu(source) [integer approx] |

## Assembly Syntax

```ternsig
.requires
  tvmr.activation 0x0003

tvmr.activation.RELU H0, H1
tvmr.activation.SIGMOID H0, H1
tvmr.activation.TANH H0, H1
; ... 2 more instructions
```
