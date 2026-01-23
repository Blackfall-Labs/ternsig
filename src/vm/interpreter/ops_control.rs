//! Control flow operation implementations for the Interpreter

use super::{Instruction, Interpreter, LoopState, StepResult};

impl Interpreter {
    pub(super) fn execute_loop(&mut self, instr: Instruction) -> StepResult {
        let count = instr.count();
        if count > 0 {
            self.loop_stack.push(LoopState {
                start_pc: self.pc,
                remaining: count,
            });
            self.pc += 1;
        } else {
            self.pc += 1;
        }
        StepResult::Continue
    }

    pub(super) fn execute_end_loop(&mut self) -> StepResult {
        if let Some(state) = self.loop_stack.last_mut() {
            state.remaining -= 1;
            if state.remaining > 0 {
                let start = state.start_pc;
                self.pc = start + 1;
            } else {
                self.loop_stack.pop();
                self.pc += 1;
            }
        } else {
            self.pc += 1;
        }
        StepResult::Continue
    }

    pub(super) fn execute_break(&mut self) -> StepResult {
        self.loop_stack.pop();
        self.pc += 1;
        StepResult::Break
    }
}
