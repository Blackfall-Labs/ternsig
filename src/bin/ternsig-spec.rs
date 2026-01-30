//! ternsig-spec — generates specs/*.md from the live extension registry.
//!
//! Reads every registered extension + core ISA and writes per-category
//! markdown reference files into `specs/extensions/` and `specs/core/`.
//!
//! # Usage
//!
//! ```bash
//! # From the ternsig crate root:
//! cargo run --bin ternsig-spec
//!
//! # Or with a custom output directory:
//! cargo run --bin ternsig-spec -- path/to/output
//! ```

use std::fmt::Write as FmtWrite;
use std::fs;
use std::path::{Path, PathBuf};

use ternsig::vm::extension::{Extension, OperandPattern};
use ternsig::vm::extensions;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let out_dir = if args.len() > 1 {
        PathBuf::from(&args[1])
    } else {
        PathBuf::from("specs")
    };

    let ext_dir = out_dir.join("extensions");
    let core_dir = out_dir.join("core");

    fs::create_dir_all(&ext_dir).expect("Failed to create specs/extensions/");
    fs::create_dir_all(&core_dir).expect("Failed to create specs/core/");

    let exts = extensions::standard_extensions();

    // Write per-extension files
    for ext in &exts {
        let filename = format!("{}.md", ext.name());
        let path = ext_dir.join(&filename);
        let content = render_extension(ext.as_ref());
        fs::write(&path, &content).unwrap_or_else(|e| {
            panic!("Failed to write {}: {}", path.display(), e);
        });
        println!("  wrote {}", path.display());
    }

    // Write core ISA
    let core_path = core_dir.join("CORE_ISA.md");
    let core_content = render_core_isa();
    fs::write(&core_path, &core_content).unwrap_or_else(|e| {
        panic!("Failed to write {}: {}", core_path.display(), e);
    });
    println!("  wrote {}", core_path.display());

    // Write index
    let index_path = out_dir.join("INDEX.md");
    let index_content = render_index(&exts, &out_dir);
    fs::write(&index_path, &index_content).unwrap_or_else(|e| {
        panic!("Failed to write {}: {}", index_path.display(), e);
    });
    println!("  wrote {}", index_path.display());

    let ext_total: usize = exts.iter().map(|e| e.instructions().len()).sum();
    println!();
    println!("Done. {} extension files + core ISA + index.", exts.len());
    println!("Total: {} core + {} extension = {} instructions.",
        CORE_ISA.len(), ext_total, CORE_ISA.len() + ext_total);
}

// ─── Rendering ───────────────────────────────────────────────────────────────

fn render_index(exts: &[Box<dyn Extension>], _out_dir: &Path) -> String {
    let mut s = String::new();
    let ext_total: usize = exts.iter().map(|e| e.instructions().len()).sum();
    let total = CORE_ISA.len() + ext_total;

    writeln!(s, "# TVMR Instruction Set Reference").unwrap();
    writeln!(s).unwrap();
    writeln!(s, "Auto-generated from the live extension registry by `ternsig-spec`.").unwrap();
    writeln!(s).unwrap();
    writeln!(s, "**Total: {} instructions** ({} core ISA + {} across {} extensions)", total, CORE_ISA.len(), ext_total, exts.len()).unwrap();
    writeln!(s).unwrap();
    writeln!(s, "## Core ISA").unwrap();
    writeln!(s).unwrap();
    writeln!(s, "- [Core ISA](core/CORE_ISA.md) — {} instructions (system, register, forward, ternary, learning, control flow, structural)", CORE_ISA.len()).unwrap();
    writeln!(s).unwrap();
    writeln!(s, "## Extensions").unwrap();
    writeln!(s).unwrap();
    writeln!(s, "| ExtID | Name | Instructions | Version | Spec |").unwrap();
    writeln!(s, "|-------|------|-------------|---------|------|").unwrap();

    for ext in exts {
        let (maj, min, pat) = ext.version();
        writeln!(s, "| 0x{:04X} | {} | {} | {}.{}.{} | [spec](extensions/{}.md) |",
            ext.ext_id(), ext.name(), ext.instructions().len(),
            maj, min, pat, ext.name()
        ).unwrap();
    }

    writeln!(s).unwrap();
    writeln!(s, "## Operand Patterns").unwrap();
    writeln!(s).unwrap();
    writeln!(s, "All instructions use an 8-byte format: `[ExtID:2][OpCode:2][A:1][B:1][C:1][D:1]`").unwrap();
    writeln!(s).unwrap();
    writeln!(s, "The 4 operand bytes `[A][B][C][D]` are interpreted per the instruction's operand pattern:").unwrap();
    writeln!(s).unwrap();
    writeln!(s, "| Pattern | Layout | Typical Use |").unwrap();
    writeln!(s, "|---------|--------|-------------|").unwrap();
    writeln!(s, "| None | `[_:4]` | System ops (NOP, HALT) |").unwrap();
    writeln!(s, "| Reg | `[reg:1][_:3]` | Single register ops |").unwrap();
    writeln!(s, "| RegReg | `[dst:1][src:1][_:2]` | Unary transforms |").unwrap();
    writeln!(s, "| RegRegReg | `[dst:1][a:1][b:1][_:1]` | Binary ops |").unwrap();
    writeln!(s, "| RegRegRegFlags | `[dst:1][a:1][b:1][flags:1]` | Flagged binary ops |").unwrap();
    writeln!(s, "| RegRegImm16 | `[dst:1][src:1][imm16:2]` | Register + immediate |").unwrap();
    writeln!(s, "| RegImm16 | `[dst:1][_:1][imm16:2]` | Register + immediate |").unwrap();
    writeln!(s, "| RegImm8 | `[reg:1][imm8:1][_:2]` | Register + small immediate |").unwrap();
    writeln!(s, "| Imm32 | `[imm32:4]` | Jump targets |").unwrap();
    writeln!(s, "| Imm16 | `[imm16:2][_:2]` | Loop counts |").unwrap();
    writeln!(s, "| Imm8 | `[imm8:1][_:3]` | Small constants |").unwrap();
    writeln!(s, "| RegCondRegReg | `[dst:1][cond:1][a:1][b:1]` | Conditional select |").unwrap();
    writeln!(s, "| Custom | (varies) | Extension-specific |").unwrap();

    s
}

fn render_extension(ext: &dyn Extension) -> String {
    let mut s = String::new();
    let (maj, min, pat) = ext.version();
    let instrs = ext.instructions();

    writeln!(s, "# {} (0x{:04X})", ext.name(), ext.ext_id()).unwrap();
    writeln!(s).unwrap();
    writeln!(s, "**Version:** {}.{}.{}  ", maj, min, pat).unwrap();
    writeln!(s, "**Instructions:** {}  ", instrs.len()).unwrap();
    writeln!(s, "**ExtID:** 0x{:04X}", ext.ext_id()).unwrap();
    writeln!(s).unwrap();
    writeln!(s, "Auto-generated from the live extension registry by `ternsig-spec`.").unwrap();
    writeln!(s).unwrap();
    writeln!(s, "## Instructions").unwrap();
    writeln!(s).unwrap();
    writeln!(s, "| Opcode | Mnemonic | Operands | Description |").unwrap();
    writeln!(s, "|--------|----------|----------|-------------|").unwrap();

    for meta in instrs {
        let pat_str = format_operand_pattern(&meta.operand_pattern);
        writeln!(s, "| 0x{:04X} | `{}` | `{}` | {} |",
            meta.opcode, meta.mnemonic, pat_str, meta.description
        ).unwrap();
    }

    writeln!(s).unwrap();
    writeln!(s, "## Assembly Syntax").unwrap();
    writeln!(s).unwrap();
    writeln!(s, "```ternsig").unwrap();
    writeln!(s, ".requires").unwrap();
    writeln!(s, "  {} 0x{:04X}", ext.name(), ext.ext_id()).unwrap();
    writeln!(s).unwrap();

    // Show first 3 instructions as examples
    for meta in instrs.iter().take(3) {
        let example_ops = example_operands(&meta.operand_pattern);
        writeln!(s, "{}.{} {}", ext.name(), meta.mnemonic, example_ops).unwrap();
    }
    if instrs.len() > 3 {
        writeln!(s, "; ... {} more instructions", instrs.len() - 3).unwrap();
    }
    writeln!(s, "```").unwrap();

    s
}

fn render_core_isa() -> String {
    let mut s = String::new();

    writeln!(s, "# Core ISA (0x0000)").unwrap();
    writeln!(s).unwrap();
    writeln!(s, "**Instructions:** {}  ", CORE_ISA.len()).unwrap();
    writeln!(s, "**ExtID:** 0x0000 (built-in, not an extension)", ).unwrap();
    writeln!(s).unwrap();
    writeln!(s, "Auto-generated from the live extension registry by `ternsig-spec`.").unwrap();
    writeln!(s).unwrap();

    let mut current_cat = "";
    for instr in CORE_ISA.iter() {
        if instr.category != current_cat {
            if !current_cat.is_empty() {
                writeln!(s).unwrap();
            }
            writeln!(s, "## {}", instr.category).unwrap();
            writeln!(s).unwrap();
            writeln!(s, "| Opcode | Mnemonic | Operands | Description |").unwrap();
            writeln!(s, "|--------|----------|----------|-------------|").unwrap();
            current_cat = instr.category;
        }
        writeln!(s, "| 0x{:04X} | `{}` | `{}` | {} |",
            instr.opcode, instr.mnemonic, instr.operands, instr.description
        ).unwrap();
    }

    s
}

fn format_operand_pattern(pat: &OperandPattern) -> String {
    match pat {
        OperandPattern::None => "—".to_string(),
        OperandPattern::Reg => "[reg:1][_:3]".to_string(),
        OperandPattern::RegReg => "[dst:1][src:1][_:2]".to_string(),
        OperandPattern::RegRegReg => "[dst:1][a:1][b:1][_:1]".to_string(),
        OperandPattern::RegRegRegFlags => "[dst:1][a:1][b:1][flags:1]".to_string(),
        OperandPattern::RegRegImm16 => "[dst:1][src:1][imm16:2]".to_string(),
        OperandPattern::RegImm16 => "[dst:1][_:1][imm16:2]".to_string(),
        OperandPattern::RegImm8 => "[reg:1][imm8:1][_:2]".to_string(),
        OperandPattern::Imm32 => "[imm32:4]".to_string(),
        OperandPattern::Imm16 => "[imm16:2][_:2]".to_string(),
        OperandPattern::Imm8 => "[imm8:1][_:3]".to_string(),
        OperandPattern::RegCondRegReg => "[dst:1][cond:1][a:1][b:1]".to_string(),
        OperandPattern::Custom(s) => s.to_string(),
    }
}

fn example_operands(pat: &OperandPattern) -> &'static str {
    match pat {
        OperandPattern::None => "",
        OperandPattern::Reg => "H0",
        OperandPattern::RegReg => "H0, H1",
        OperandPattern::RegRegReg => "H0, H1, H2",
        OperandPattern::RegRegRegFlags => "H0, H1, H2, 0x01",
        OperandPattern::RegRegImm16 => "H0, H1, 256",
        OperandPattern::RegImm16 => "H0, 256",
        OperandPattern::RegImm8 => "H0, 8",
        OperandPattern::Imm32 => "0x00000010",
        OperandPattern::Imm16 => "100",
        OperandPattern::Imm8 => "1",
        OperandPattern::RegCondRegReg => "H0, 1, H1, H2",
        OperandPattern::Custom(_) => "H0, H1, 0, 0",
    }
}

// ─── Core ISA table ──────────────────────────────────────────────────────────

struct CoreInstr {
    opcode: u16,
    mnemonic: &'static str,
    category: &'static str,
    operands: &'static str,
    description: &'static str,
}

const CORE_ISA: &[CoreInstr] = &[
    // System
    CoreInstr { opcode: 0x0000, mnemonic: "NOP", category: "System", operands: "—", description: "No operation" },
    CoreInstr { opcode: 0x0001, mnemonic: "HALT", category: "System", operands: "—", description: "Halt execution" },
    CoreInstr { opcode: 0x0003, mnemonic: "RESET", category: "System", operands: "—", description: "Reset PC and loop stack" },

    // Register Management
    CoreInstr { opcode: 0x1000, mnemonic: "ALLOC_TENSOR", category: "Register Management", operands: "[reg:1][dim0:1][dim1_hi:1][dim1_lo:1]", description: "Allocate tensor register with shape" },
    CoreInstr { opcode: 0x1001, mnemonic: "FREE_TENSOR", category: "Register Management", operands: "[reg:1][_:3]", description: "Free tensor register" },
    CoreInstr { opcode: 0x1004, mnemonic: "COPY_REG", category: "Register Management", operands: "[dst:1][src:1][_:2]", description: "Copy between hot registers" },
    CoreInstr { opcode: 0x1006, mnemonic: "ZERO_REG", category: "Register Management", operands: "[reg:1][_:3]", description: "Zero out register" },
    CoreInstr { opcode: 0x1007, mnemonic: "LOAD_INPUT", category: "Register Management", operands: "[dst:1][_:3]", description: "Load input buffer into hot register" },
    CoreInstr { opcode: 0x1008, mnemonic: "STORE_OUTPUT", category: "Register Management", operands: "[src:1][_:3]", description: "Store hot register to output buffer" },
    CoreInstr { opcode: 0x1009, mnemonic: "LOAD_TARGET", category: "Register Management", operands: "[dst:1][_:3]", description: "Load target buffer for learning" },

    // Forward Operations
    CoreInstr { opcode: 0x3001, mnemonic: "ADD", category: "Forward Operations", operands: "[dst:1][a:1][b:1][_:1]", description: "Element-wise addition with saturation" },
    CoreInstr { opcode: 0x300E, mnemonic: "SUB", category: "Forward Operations", operands: "[dst:1][a:1][b:1][_:1]", description: "Element-wise subtraction" },
    CoreInstr { opcode: 0x3002, mnemonic: "MUL", category: "Forward Operations", operands: "[dst:1][a:1][b:1][_:1]", description: "Element-wise multiply (>>8 fixed-point)" },
    CoreInstr { opcode: 0x3003, mnemonic: "RELU", category: "Forward Operations", operands: "[dst:1][src:1][_:2]", description: "ReLU activation" },
    CoreInstr { opcode: 0x3004, mnemonic: "SIGMOID", category: "Forward Operations", operands: "[dst:1][src:1][_:2]", description: "Integer sigmoid approximation" },
    CoreInstr { opcode: 0x3005, mnemonic: "TANH", category: "Forward Operations", operands: "[dst:1][src:1][_:2]", description: "Integer tanh approximation" },
    CoreInstr { opcode: 0x3006, mnemonic: "SOFTMAX", category: "Forward Operations", operands: "[dst:1][src:1][_:2]", description: "Integer softmax" },
    CoreInstr { opcode: 0x3007, mnemonic: "GELU", category: "Forward Operations", operands: "[dst:1][src:1][_:2]", description: "Integer GELU approximation" },
    CoreInstr { opcode: 0x3009, mnemonic: "SHIFT", category: "Forward Operations", operands: "[dst:1][src:1][amt:1][_:1]", description: "Right-shift (integer scaling)" },
    CoreInstr { opcode: 0x300A, mnemonic: "CLAMP", category: "Forward Operations", operands: "[dst:1][src:1][min:1][max:1]", description: "Clamp to range [min*256, max*256]" },
    CoreInstr { opcode: 0x300F, mnemonic: "CMP_GT", category: "Forward Operations", operands: "[dst:1][a:1][b:1][_:1]", description: "Compare greater-than (outputs 1/0)" },
    CoreInstr { opcode: 0x3010, mnemonic: "MAX_REDUCE", category: "Forward Operations", operands: "[dst:1][src:1][_:2]", description: "Reduce to maximum value" },

    // Ternary Operations
    CoreInstr { opcode: 0x4000, mnemonic: "TERNARY_MATMUL", category: "Ternary Operations", operands: "[dst:1][weights:1][input:1][_:1]", description: "Temperature-gated ternary matmul" },
    CoreInstr { opcode: 0x4002, mnemonic: "DEQUANTIZE", category: "Ternary Operations", operands: "[dst:1][src:1][shift:2]", description: "Dequantize signals (right-shift)" },
    CoreInstr { opcode: 0x4009, mnemonic: "TERNARY_ADD_BIAS", category: "Ternary Operations", operands: "[dst:1][src:1][bias:1][_:1]", description: "Add signal bias to activations" },
    CoreInstr { opcode: 0x400A, mnemonic: "EMBED_LOOKUP", category: "Ternary Operations", operands: "[dst:1][table:1][indices:1][_:1]", description: "Embedding table lookup" },
    CoreInstr { opcode: 0x400B, mnemonic: "REDUCE_AVG", category: "Ternary Operations", operands: "[dst:1][src:1][start:1][count:1]", description: "Slice average" },
    CoreInstr { opcode: 0x400C, mnemonic: "SLICE", category: "Ternary Operations", operands: "[dst:1][src:1][start:1][len:1]", description: "Slice tensor" },
    CoreInstr { opcode: 0x400D, mnemonic: "ARGMAX", category: "Ternary Operations", operands: "[dst:1][src:1][_:2]", description: "Index of maximum value" },
    CoreInstr { opcode: 0x400E, mnemonic: "CONCAT", category: "Ternary Operations", operands: "[dst:1][a:1][b:1][_:1]", description: "Concatenate two registers" },
    CoreInstr { opcode: 0x400F, mnemonic: "SQUEEZE", category: "Ternary Operations", operands: "[dst:1][src:1][_:2]", description: "Copy (shape semantics)" },
    CoreInstr { opcode: 0x4010, mnemonic: "UNSQUEEZE", category: "Ternary Operations", operands: "[dst:1][src:1][_:2]", description: "Copy (shape semantics)" },
    CoreInstr { opcode: 0x4011, mnemonic: "TRANSPOSE", category: "Ternary Operations", operands: "[dst:1][src:1][_:2]", description: "Copy (shape semantics)" },
    CoreInstr { opcode: 0x4012, mnemonic: "GATE_UPDATE", category: "Ternary Operations", operands: "[dst:1][gate:1][upd:1][state:1]", description: "Gated update: gate*upd + (1-gate)*state" },
    CoreInstr { opcode: 0x4013, mnemonic: "TERNARY_BATCH_MATMUL", category: "Ternary Operations", operands: "[dst:1][weights:1][input:1][_:1]", description: "Batch ternary matmul" },
    CoreInstr { opcode: 0x4014, mnemonic: "EMBED_SEQUENCE", category: "Ternary Operations", operands: "[dst:1][table:1][count:1][_:1]", description: "Sequential position embeddings" },
    CoreInstr { opcode: 0x4015, mnemonic: "REDUCE_MEAN_DIM", category: "Ternary Operations", operands: "[dst:1][src:1][dim:1][_:1]", description: "Reduce mean along dimension" },

    // Learning Operations
    CoreInstr { opcode: 0x5000, mnemonic: "MARK_ELIGIBILITY", category: "Learning Operations", operands: "[reg:1][_:3]", description: "Mark weights eligible by activity" },
    CoreInstr { opcode: 0x5004, mnemonic: "ADD_BABBLE", category: "Learning Operations", operands: "[dst:1][src:1][_:2]", description: "Add exploration noise" },
    CoreInstr { opcode: 0x5012, mnemonic: "MASTERY_UPDATE", category: "Learning Operations", operands: "[weights:1][input:1][output:1][_:1]", description: "Accumulate learning pressure" },
    CoreInstr { opcode: 0x5013, mnemonic: "MASTERY_COMMIT", category: "Learning Operations", operands: "[weights:1][_:3]", description: "Apply pressure to weights" },

    // Control Flow
    CoreInstr { opcode: 0x6000, mnemonic: "LOOP", category: "Control Flow", operands: "[count_hi:1][count_lo:1][_:2]", description: "Start loop (u16 iteration count)" },
    CoreInstr { opcode: 0x6001, mnemonic: "END_LOOP", category: "Control Flow", operands: "—", description: "End loop (jump back to LOOP)" },
    CoreInstr { opcode: 0x6002, mnemonic: "BREAK", category: "Control Flow", operands: "—", description: "Break from current loop" },
    CoreInstr { opcode: 0x6006, mnemonic: "CALL", category: "Control Flow", operands: "[target:4]", description: "Call subroutine (push PC)" },
    CoreInstr { opcode: 0x6007, mnemonic: "RETURN", category: "Control Flow", operands: "—", description: "Return from subroutine (pop PC)" },
    CoreInstr { opcode: 0x6008, mnemonic: "JUMP", category: "Control Flow", operands: "[target:4]", description: "Unconditional jump" },

    // Structural Plasticity
    CoreInstr { opcode: 0x2003, mnemonic: "WIRE_FORWARD", category: "Structural Plasticity", operands: "[dst:1][weights:1][input:1][_:1]", description: "Wire forward connection" },
    CoreInstr { opcode: 0x2004, mnemonic: "WIRE_SKIP", category: "Structural Plasticity", operands: "[dst:1][a:1][b:1][_:1]", description: "Wire skip connection" },
    CoreInstr { opcode: 0x2008, mnemonic: "GROW_NEURON", category: "Structural Plasticity", operands: "[reg:1][count:1][_:2]", description: "Add neurons to cold register" },
    CoreInstr { opcode: 0x2009, mnemonic: "PRUNE_NEURON", category: "Structural Plasticity", operands: "[reg:1][index:1][_:2]", description: "Remove neuron from cold register" },
    CoreInstr { opcode: 0x200A, mnemonic: "INIT_RANDOM", category: "Structural Plasticity", operands: "[reg:1][seed:1][_:2]", description: "Initialize with random ternary weights" },
];
