//! Hot Reload support for Ternsig programs
//!
//! Enables runtime program updates without process restart:
//! - Watch .ternsig files for changes
//! - Reassemble on change
//! - Swap program while preserving cold register state (weights)
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────┐
//! │               HotReloadManager                   │
//! ├─────────────────────────────────────────────────┤
//! │  Watcher → Assembler → Swap Protocol            │
//! │                                                 │
//! │  1. File change detected                        │
//! │  2. Assemble new program                        │
//! │  3. Wait for safe swap point                    │
//! │  4. Preserve cold registers                     │
//! │  5. Load new program                            │
//! │  6. Restore cold registers                      │
//! │  7. Emit reload event                           │
//! └─────────────────────────────────────────────────┘
//! ```
//!
//! ## Usage
//!
//! ```rust,ignore
//! use ternsig::vm::{HotReloadManager, Interpreter};
//!
//! let mut interpreter = Interpreter::new();
//! let manager = HotReloadManager::new("audio_classifier.ternsig")?;
//!
//! // In main loop
//! if let Some(new_program) = manager.poll_reload()? {
//!     interpreter.hot_reload(&new_program)?;
//! }
//! ```

use super::{assemble, AssembledProgram, ColdBuffer, Interpreter};
use crate::Signal;
use anyhow::{Context, Result};
use notify::{RecommendedWatcher, RecursiveMode, Watcher};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{channel, Receiver, TryRecvError};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Reload event emitted when a program is hot-reloaded
#[derive(Debug, Clone)]
pub struct ReloadEvent {
    /// Path of the reloaded file
    pub path: PathBuf,
    /// Timestamp of reload
    pub timestamp: Instant,
    /// Number of instructions in new program
    pub instruction_count: usize,
    /// Number of registers defined
    pub register_count: usize,
    /// Whether cold registers were preserved
    pub cold_preserved: bool,
}

/// State of the interpreter for safe reload detection
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterpreterState {
    /// Not executing, safe to reload
    Idle,
    /// Currently executing instructions
    Running,
    /// Waiting for external input
    Waiting,
    /// Requested reload, waiting for safe point
    ReloadPending,
}

/// Hot reload manager for Ternsig programs
pub struct HotReloadManager {
    /// Path being watched
    watch_path: PathBuf,
    /// File watcher
    _watcher: RecommendedWatcher,
    /// Channel for file change events
    change_rx: Receiver<PathBuf>,
    /// Last successful program
    current_program: Option<AssembledProgram>,
    /// Debounce duration to avoid rapid reloads
    debounce: Duration,
    /// Last change time for debouncing
    last_change: Option<Instant>,
    /// Pending reload path
    pending_path: Option<PathBuf>,
}

impl HotReloadManager {
    /// Create a new hot reload manager watching a single file
    pub fn new(path: impl AsRef<Path>) -> Result<Self> {
        let watch_path = path.as_ref().to_path_buf();
        let (tx, rx) = channel();

        // Create watcher
        let tx_clone = tx.clone();
        let mut watcher = notify::recommended_watcher(move |res: notify::Result<notify::Event>| {
            if let Ok(event) = res {
                if matches!(
                    event.kind,
                    notify::EventKind::Modify(_) | notify::EventKind::Create(_)
                ) {
                    for path in event.paths {
                        let _ = tx_clone.send(path);
                    }
                }
            }
        })
        .context("Failed to create file watcher")?;

        // Watch the file's parent directory
        if let Some(parent) = watch_path.parent() {
            watcher
                .watch(parent, RecursiveMode::NonRecursive)
                .context("Failed to watch directory")?;
        }

        Ok(Self {
            watch_path,
            _watcher: watcher,
            change_rx: rx,
            current_program: None,
            debounce: Duration::from_millis(100),
            last_change: None,
            pending_path: None,
        })
    }

    /// Create manager watching a directory of .ternsig files
    pub fn watch_directory(dir: impl AsRef<Path>) -> Result<Self> {
        let watch_path = dir.as_ref().to_path_buf();
        let (tx, rx) = channel();

        let tx_clone = tx.clone();
        let mut watcher = notify::recommended_watcher(move |res: notify::Result<notify::Event>| {
            if let Ok(event) = res {
                if matches!(
                    event.kind,
                    notify::EventKind::Modify(_) | notify::EventKind::Create(_)
                ) {
                    for path in event.paths {
                        if path.extension().map_or(false, |e| e == "ternsig") {
                            let _ = tx_clone.send(path);
                        }
                    }
                }
            }
        })
        .context("Failed to create file watcher")?;

        watcher
            .watch(&watch_path, RecursiveMode::Recursive)
            .context("Failed to watch directory")?;

        Ok(Self {
            watch_path,
            _watcher: watcher,
            change_rx: rx,
            current_program: None,
            debounce: Duration::from_millis(100),
            last_change: None,
            pending_path: None,
        })
    }

    /// Set debounce duration
    pub fn with_debounce(mut self, duration: Duration) -> Self {
        self.debounce = duration;
        self
    }

    /// Poll for pending reloads (non-blocking)
    ///
    /// Returns Some(AssembledProgram) if a reload is ready, None otherwise.
    pub fn poll_reload(&mut self) -> Result<Option<AssembledProgram>> {
        // Check for new file changes
        loop {
            match self.change_rx.try_recv() {
                Ok(path) => {
                    // Check if this is the file we're watching (or any .ternsig in directory mode)
                    if path == self.watch_path
                        || path.extension().map_or(false, |e| e == "ternsig")
                    {
                        self.pending_path = Some(path);
                        self.last_change = Some(Instant::now());
                    }
                }
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    anyhow::bail!("File watcher disconnected");
                }
            }
        }

        // Check if debounce period has passed
        if let (Some(pending), Some(last_change)) = (&self.pending_path, self.last_change) {
            if last_change.elapsed() >= self.debounce {
                let path = pending.clone();
                self.pending_path = None;
                self.last_change = None;

                // Read and assemble the file
                let source = std::fs::read_to_string(&path)
                    .with_context(|| format!("Failed to read {}", path.display()))?;

                let program = assemble(&source)
                    .with_context(|| format!("Failed to assemble {}", path.display()))?;

                self.current_program = Some(program.clone());
                return Ok(Some(program));
            }
        }

        Ok(None)
    }

    /// Get the current program (if any)
    pub fn current_program(&self) -> Option<&AssembledProgram> {
        self.current_program.as_ref()
    }

    /// Get the watch path
    pub fn watch_path(&self) -> &Path {
        &self.watch_path
    }
}

/// Extension trait for Interpreter hot reload support
impl Interpreter {
    /// Hot reload a new program while preserving cold register state
    ///
    /// This method:
    /// 1. Saves all cold registers (weights)
    /// 2. Loads the new program
    /// 3. Restores cold registers that match the new program's layout
    ///
    /// Returns a ReloadEvent describing the reload.
    pub fn hot_reload(&mut self, new_program: &AssembledProgram) -> Result<ReloadEvent> {
        let start = Instant::now();

        // Save current cold registers
        let saved_cold: Vec<(usize, ColdBuffer)> = (0..16)
            .filter_map(|i| self.cold_reg(i).cloned().map(|buf| (i, buf)))
            .collect();

        let _old_instruction_count = self.program_len();

        // Load new program (clears registers)
        self.load_program(new_program);

        // Restore cold registers that have matching thermogram keys
        let mut cold_preserved = false;
        for (idx, saved) in &saved_cold {
            if let Some(current) = self.cold_reg_mut(*idx) {
                // Match by thermogram key if available
                let key_matches = match (&saved.thermogram_key, &current.thermogram_key) {
                    (Some(a), Some(b)) => a == b,
                    _ => false,
                };

                // Also check shape compatibility
                let shape_matches = saved.shape == current.shape;

                if key_matches && shape_matches {
                    // Restore weights
                    current.weights.copy_from_slice(&saved.weights);
                    cold_preserved = true;
                } else if shape_matches && saved.thermogram_key.is_none() {
                    // No key but shape matches - preserve anyway
                    current.weights.copy_from_slice(&saved.weights);
                    cold_preserved = true;
                }
            }
        }

        Ok(ReloadEvent {
            path: PathBuf::new(), // Caller should set this
            timestamp: start,
            instruction_count: new_program.instructions.len(),
            register_count: new_program.registers.len(),
            cold_preserved,
        })
    }

    /// Check if interpreter is at a safe reload point
    ///
    /// A safe point is when:
    /// - PC is 0 (before execution)
    /// - PC is at end (after execution)
    /// - Not in the middle of a loop
    pub fn is_safe_reload_point(&self) -> bool {
        self.pc() == 0 || self.is_ended()
    }

    /// Get interpreter state
    pub fn state(&self) -> InterpreterState {
        if self.is_ended() || self.pc() == 0 {
            InterpreterState::Idle
        } else {
            InterpreterState::Running
        }
    }

    /// Create a snapshot of cold registers for backup
    pub fn snapshot_cold(&self) -> Vec<(usize, ColdBuffer)> {
        (0..16)
            .filter_map(|i| self.cold_reg(i).cloned().map(|buf| (i, buf)))
            .collect()
    }

    /// Restore cold registers from a snapshot
    pub fn restore_cold(&mut self, snapshot: &[(usize, ColdBuffer)]) {
        for (idx, buf) in snapshot {
            if let Some(current) = self.cold_reg_mut(*idx) {
                if current.shape == buf.shape {
                    current.weights.copy_from_slice(&buf.weights);
                    current.thermogram_key = buf.thermogram_key.clone();
                }
            }
        }
    }
}

/// Managed hot-reloadable interpreter
///
/// Combines Interpreter with HotReloadManager for automatic reloading.
pub struct ReloadableInterpreter {
    /// The interpreter
    interpreter: Interpreter,
    /// Hot reload manager
    manager: HotReloadManager,
    /// Reload history
    reload_history: Vec<ReloadEvent>,
    /// Max history entries
    max_history: usize,
}

impl ReloadableInterpreter {
    /// Create from a .ternsig file path
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();

        // Read and assemble initial program
        let source = std::fs::read_to_string(path)
            .with_context(|| format!("Failed to read {}", path.display()))?;

        let program =
            assemble(&source).with_context(|| format!("Failed to assemble {}", path.display()))?;

        let interpreter = Interpreter::from_program(&program);
        let mut manager = HotReloadManager::new(path)?;
        manager.current_program = Some(program);

        Ok(Self {
            interpreter,
            manager,
            reload_history: Vec::new(),
            max_history: 100,
        })
    }

    /// Set max history entries
    pub fn with_max_history(mut self, max: usize) -> Self {
        self.max_history = max;
        self
    }

    /// Check for and apply any pending reloads
    ///
    /// Returns true if a reload occurred.
    pub fn check_reload(&mut self) -> Result<bool> {
        if let Some(new_program) = self.manager.poll_reload()? {
            let mut event = self.interpreter.hot_reload(&new_program)?;
            event.path = self.manager.watch_path().to_path_buf();

            // Add to history
            self.reload_history.push(event);
            if self.reload_history.len() > self.max_history {
                self.reload_history.remove(0);
            }

            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Get the interpreter
    pub fn interpreter(&self) -> &Interpreter {
        &self.interpreter
    }

    /// Get mutable interpreter
    pub fn interpreter_mut(&mut self) -> &mut Interpreter {
        &mut self.interpreter
    }

    /// Get reload history
    pub fn reload_history(&self) -> &[ReloadEvent] {
        &self.reload_history
    }

    /// Get last reload event
    pub fn last_reload(&self) -> Option<&ReloadEvent> {
        self.reload_history.last()
    }

    /// Forward pass with automatic reload check (Signal API)
    pub fn forward(&mut self, input: &[Signal]) -> Result<Vec<Signal>> {
        // Check for reload at safe point
        if self.interpreter.is_safe_reload_point() {
            self.check_reload()?;
        }

        self.interpreter
            .forward(input)
            .map_err(|e| anyhow::anyhow!(e))
    }

    /// Forward pass with i32 input/output
    pub fn forward_i32(&mut self, input: &[i32]) -> Result<Vec<i32>> {
        // Check for reload at safe point
        if self.interpreter.is_safe_reload_point() {
            self.check_reload()?;
        }

        self.interpreter
            .forward_i32(input)
            .map_err(|e| anyhow::anyhow!(e))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_hot_reload_preserves_weights() {
        let source1 = r#"
.registers
    C0: ternary[4, 2]  key="test.weights"
    H0: i32[2]
    H1: i32[4]

.program
    load_input H0
    ternary_matmul H1, C0, H0
    store_output H1
    halt
"#;

        let source2 = r#"
.registers
    C0: ternary[4, 2]  key="test.weights"
    H0: i32[2]
    H1: i32[4]
    H2: i32[4]

.program
    load_input H0
    ternary_matmul H1, C0, H0
    relu H2, H1
    store_output H2
    halt
"#;

        let program1 = assemble(source1).unwrap();
        let mut interp = Interpreter::from_program(&program1);

        // Set some weights
        use crate::Signal;
        if let Some(cold) = interp.cold_reg_mut(0) {
            cold.weights[0] = Signal {
                polarity: 1,
                magnitude: 128,
            };
            cold.weights[1] = Signal {
                polarity: -1,
                magnitude: 64,
            };
        }

        // Hot reload new program
        let program2 = assemble(source2).unwrap();
        let event = interp.hot_reload(&program2).unwrap();

        assert!(event.cold_preserved);
        assert_eq!(event.instruction_count, 5); // New program has 5 instructions

        // Verify weights preserved
        if let Some(cold) = interp.cold_reg(0) {
            assert_eq!(cold.weights[0].polarity, 1);
            assert_eq!(cold.weights[0].magnitude, 128);
            assert_eq!(cold.weights[1].polarity, -1);
            assert_eq!(cold.weights[1].magnitude, 64);
        } else {
            panic!("Cold register not found after reload");
        }
    }

    #[test]
    fn test_snapshot_restore() {
        let source = r#"
.registers
    C0: ternary[2, 2]  key="snap.weights"
    H0: i32[2]

.program
    halt
"#;

        let program = assemble(source).unwrap();
        let mut interp = Interpreter::from_program(&program);

        // Set weights
        use crate::Signal;
        if let Some(cold) = interp.cold_reg_mut(0) {
            cold.weights[0] = Signal {
                polarity: 1,
                magnitude: 200,
            };
        }

        // Snapshot
        let snapshot = interp.snapshot_cold();
        assert_eq!(snapshot.len(), 1);

        // Clear weights
        if let Some(cold) = interp.cold_reg_mut(0) {
            cold.weights[0] = Signal::zero();
        }

        // Restore
        interp.restore_cold(&snapshot);

        // Verify
        if let Some(cold) = interp.cold_reg(0) {
            assert_eq!(cold.weights[0].polarity, 1);
            assert_eq!(cold.weights[0].magnitude, 200);
        }
    }
}
