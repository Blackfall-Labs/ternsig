# tvmr.sampling (0x00C0)

Sampling and decoding helpers for realtime inference.

| Op   | Mnemonic        | Pattern   | Description |
|------|-----------------|-----------|-------------|
| 0000 | `TEMPERATURE`   | RegRegReg | Scale logits by temperature |
| 0001 | `TOP_K`         | RegRegReg | Keep top-k logits, zero rest |
| 0002 | `TOP_P`         | RegRegReg | Nucleus filtering (top-p) |
| 0003 | `REP_PENALTY`   | `[dst:1][logits:1][history:1][penalty:1]` | Apply repetition penalty |
