//! ternsig-validate - Pre-execution validation tool for .ternsig files
//!
//! # Usage
//!
//! ```bash
//! # Validate all .ternsig files in a directory (assembly only)
//! ternsig-validate path/to/ternsig/files
//!
//! # Deep validation (extensions, control flow, semantics, register bounds)
//! ternsig-validate --deep path/to/ternsig/files
//!
//! # Deep validation + shape checking (expensive but thorough)
//! ternsig-validate --deep --shapes path/to/ternsig/files
//!
//! # Verbose output with register/instruction counts
//! ternsig-validate -v path/to/ternsig/files
//! ```
//!
//! # Exit Codes
//!
//! - 0: All files validated successfully (no errors)
//! - 1: One or more files failed validation (assembly errors or error-level diagnostics)
//! - 2: Invalid arguments or IO error

use std::path::Path;
use std::process::ExitCode;
use ternsig::validate::{
    validate_file_with_config, validate_directory_with_config,
    ValidationResult, ValidationSummary,
};
use ternsig::vm::validator::{ValidationConfig, DiagnosticLevel};

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();

    let mut verbose = false;
    let mut deep = false;
    let mut shapes = false;
    let mut paths = Vec::new();

    for arg in args.iter().skip(1) {
        match arg.as_str() {
            "-v" | "--verbose" => verbose = true,
            "--deep" | "-d" => deep = true,
            "--shapes" | "-s" => shapes = true,
            "-h" | "--help" => {
                print_help();
                return ExitCode::SUCCESS;
            }
            _ if arg.starts_with('-') => {
                eprintln!("Unknown option: {}\n", arg);
                print_help();
                return ExitCode::from(2);
            }
            _ => paths.push(arg.clone()),
        }
    }

    if paths.is_empty() {
        eprintln!("Error: No path specified\n");
        print_help();
        return ExitCode::from(2);
    }

    // Build config
    let config = if deep {
        Some(if shapes {
            ValidationConfig::full()
        } else {
            ValidationConfig::default()
        })
    } else {
        None
    };

    let mut all_results = Vec::new();

    for path_str in &paths {
        let path = Path::new(path_str);

        if !path.exists() {
            eprintln!("Error: Path does not exist: {}", path.display());
            return ExitCode::from(2);
        }

        if path.is_file() {
            let result = validate_file_with_config(path, config.as_ref());
            print_result(&result, verbose, deep);
            all_results.push(result);
        } else if path.is_dir() {
            match validate_directory_with_config(path, config.as_ref()) {
                Ok(results) => {
                    for result in &results {
                        print_result(result, verbose, deep);
                    }
                    all_results.extend(results);
                }
                Err(e) => {
                    eprintln!("Error reading directory {}: {}", path.display(), e);
                    return ExitCode::from(2);
                }
            }
        }
    }

    // Print summary
    let summary = ValidationSummary::from_results(&all_results);
    eprintln!();
    summary.print_report();

    if summary.failed > 0 {
        ExitCode::FAILURE
    } else {
        ExitCode::SUCCESS
    }
}

fn print_result(result: &ValidationResult, verbose: bool, show_diagnostics: bool) {
    match result {
        ValidationResult::Ok { path, program, diagnostics } => {
            let has_errors = diagnostics.iter().any(|d| d.level == DiagnosticLevel::Error);
            let marker = if has_errors { "!" } else { "+" };

            if verbose {
                println!(
                    "{} {} ({} regs, {} instrs)",
                    marker,
                    path.display(),
                    program.registers.len(),
                    program.instructions.len()
                );
            } else {
                println!("{} {}", marker, path.display());
            }

            if show_diagnostics && !diagnostics.is_empty() {
                for d in diagnostics {
                    let level_str = match d.level {
                        DiagnosticLevel::Error => "  ERROR",
                        DiagnosticLevel::Warning => "  WARN ",
                        DiagnosticLevel::Info => "  INFO ",
                    };
                    if let Some(idx) = d.instruction_idx {
                        eprintln!("  {} [{}]: {}", level_str, idx, d.message);
                    } else {
                        eprintln!("  {}: {}", level_str, d.message);
                    }
                }
            }
        }
        ValidationResult::Err { path, error } => {
            eprintln!("x {}", path.display());
            if let Some(line) = error.line {
                eprintln!("  line {}: {}", line, error.message);
            } else {
                eprintln!("  {}", error.message);
            }
            if let Some(snippet) = &error.snippet {
                eprintln!("  | {}", snippet);
            }
        }
    }
}

fn print_help() {
    eprintln!("ternsig-validate - Validate .ternsig assembly files");
    eprintln!();
    eprintln!("USAGE:");
    eprintln!("    ternsig-validate [OPTIONS] <PATH>...");
    eprintln!();
    eprintln!("ARGS:");
    eprintln!("    <PATH>    File or directory to validate (recursive for directories)");
    eprintln!();
    eprintln!("OPTIONS:");
    eprintln!("    -v, --verbose    Show detailed output (register/instruction counts)");
    eprintln!("    -d, --deep       Run deep validation (extensions, control flow, semantics)");
    eprintln!("    -s, --shapes     Enable shape checking (requires --deep, expensive)");
    eprintln!("    -h, --help       Print this help message");
    eprintln!();
    eprintln!("EXIT CODES:");
    eprintln!("    0    All files validated successfully");
    eprintln!("    1    One or more files failed validation");
    eprintln!("    2    Invalid arguments or IO error");
    eprintln!();
    eprintln!("EXAMPLES:");
    eprintln!("    ternsig-validate firmware/          Assembly check all .ternsig files");
    eprintln!("    ternsig-validate --deep firmware/   Full validation pipeline");
    eprintln!("    ternsig-validate -dv firmware/      Deep + verbose");
}
