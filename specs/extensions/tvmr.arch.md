# tvmr.arch (0x0006)

**Version:** 1.0.0  
**Instructions:** 13  
**ExtID:** 0x0006

Auto-generated from the live extension registry by `ternsig-spec`.

## Instructions

| Opcode | Mnemonic | Operands | Description |
|--------|----------|----------|-------------|
| 0x0000 | `ALLOC_TENSOR` | `[reg:1][dim0:1][dim1_hi:1][dim1_lo:1]` | Allocate a register at runtime |
| 0x0001 | `FREE_TENSOR` | `[reg:1][_:3]` | Free a register |
| 0x0002 | `WIRE_FORWARD` | `[dst:1][a:1][b:1][_:1]` | Dynamic forward connection: output = weights @ input |
| 0x0003 | `WIRE_SKIP` | `[dst:1][a:1][b:1][_:1]` | Dynamic skip connection: output = input1 + input2 |
| 0x0004 | `GROW_NEURON` | `[cold_reg:1][_:1][count:1][seed:1]` | Add neurons to a cold register |
| 0x0005 | `PRUNE_NEURON` | `[cold_reg:1][_:1][neuron_idx:1][_:1]` | Remove a neuron from a cold register |
| 0x0006 | `INIT_RANDOM` | `[cold_reg:1][_:1][_:1][seed:1]` | Initialize cold register with random weights |
| 0x0007 | `DEFINE_LAYER` | `[layer:1][in_dim:1][out_hi:1][out_lo:1]` | Define layer dimensions |
| 0x0008 | `FREEZE_LAYER` | `[reg:1][_:3]` | Mark layer as non-trainable |
| 0x0009 | `UNFREEZE_LAYER` | `[reg:1][_:3]` | Unfreeze a layer for training |
| 0x000A | `SET_ACTIVATION` | `[reg:1][imm8:1][_:2]` | Set activation function for a layer |
| 0x000B | `ALLOC_DYNAMIC` | `[target:1][bank:1][size_hi:1][size_lo:1]` | Request dynamic register allocation from kernel. Yields AllocRegister. |
| 0x000C | `FREE_DYNAMIC` | `[reg:1][_:3]` | Release dynamically allocated register. Yields FreeRegister. |

## Assembly Syntax

```ternsig
.requires
  tvmr.arch 0x0006

tvmr.arch.ALLOC_TENSOR H0, H1, 0, 0
tvmr.arch.FREE_TENSOR H0
tvmr.arch.WIRE_FORWARD H0, H1, H2
; ... 10 more instructions
```
