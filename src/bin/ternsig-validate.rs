//! ternsig-validate - Pre-compilation validation tool for .ternsig files
//!
//! # Usage
//!
//! ```bash
//! # Validate all .ternsig files in a directory
//! ternsig-validate path/to/ternsig/files
//!
//! # Validate a single file
//! ternsig-validate path/to/file.ternsig
//!
//! # Validate with verbose output
//! ternsig-validate -v path/to/ternsig/files
//! ```
//!
//! # Exit Codes
//!
//! - 0: All files validated successfully
//! - 1: One or more files failed validation
//! - 2: Invalid arguments or IO error

use std::path::Path;
use std::process::ExitCode;
use ternsig::validate::{validate_file, validate_directory, ValidationResult, ValidationSummary};

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();

    let mut verbose = false;
    let mut paths = Vec::new();

    for arg in args.iter().skip(1) {
        match arg.as_str() {
            "-v" | "--verbose" => verbose = true,
            "-h" | "--help" => {
                print_help();
                return ExitCode::SUCCESS;
            }
            _ => paths.push(arg.clone()),
        }
    }

    if paths.is_empty() {
        eprintln!("Error: No path specified\n");
        print_help();
        return ExitCode::from(2);
    }

    let mut all_results = Vec::new();

    for path_str in &paths {
        let path = Path::new(path_str);

        if !path.exists() {
            eprintln!("Error: Path does not exist: {}", path.display());
            return ExitCode::from(2);
        }

        if path.is_file() {
            // Single file
            let result = validate_file(path);
            print_result(&result, verbose);
            all_results.push(result);
        } else if path.is_dir() {
            // Directory
            match validate_directory(path) {
                Ok(results) => {
                    for result in &results {
                        print_result(result, verbose);
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

fn print_result(result: &ValidationResult, verbose: bool) {
    match result {
        ValidationResult::Ok { path, program } => {
            if verbose {
                println!(
                    "✓ {} ({} regs, {} instrs)",
                    path.display(),
                    program.registers.len(),
                    program.instructions.len()
                );
            } else {
                println!("✓ {}", path.display());
            }
        }
        ValidationResult::Err { path, error } => {
            eprintln!("✗ {}", path.display());
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
    eprintln!("    -h, --help       Print this help message");
    eprintln!();
    eprintln!("EXIT CODES:");
    eprintln!("    0    All files validated successfully");
    eprintln!("    1    One or more files failed validation");
    eprintln!("    2    Invalid arguments or IO error");
}
