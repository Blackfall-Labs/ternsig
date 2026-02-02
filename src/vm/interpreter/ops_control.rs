//! Control flow operation implementations for the Interpreter

use super::{Instruction, Interpreter, LoopState, StepResult};

impl Interpreter {
    pub(super) fn execute_loop(&mut self, instr: Instruction) -> StepResult {
        let count = instr.count();
        if count > 0 {
            // start_pc = self.pc which already points to the first body instruction
            // (step() incremented PC before dispatch). Do NOT advance PC further.
            self.loop_stack.push(LoopState {
                start_pc: self.pc,
                remaining: count,
            });
        }
        // For count == 0, PC already advanced past LOOP — fall through to body
        // which will immediately hit END_LOOP and skip.
        StepResult::Continue
    }

    pub(super) fn execute_end_loop(&mut self) -> StepResult {
        if let Some(state) = self.loop_stack.last_mut() {
            state.remaining -= 1;
            if state.remaining > 0 {
                // Jump back to body start (step() already advanced past END_LOOP,
                // so we overwrite PC to the first body instruction).
                self.pc = state.start_pc;
            } else {
                // Loop complete — PC already points past END_LOOP from step().
                self.loop_stack.pop();
            }
        }
        // If no loop state, PC already advanced past END_LOOP — just continue.
        StepResult::Continue
    }

    pub(super) fn execute_break(&mut self) -> StepResult {
        self.loop_stack.pop();
        self.pc += 1;
        StepResult::Break
    }
}
