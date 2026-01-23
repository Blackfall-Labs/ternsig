//! Learning operation implementations for the Interpreter

use super::{HotBuffer, Instruction, Interpreter, StepResult};

impl Interpreter {
    pub(super) fn execute_load_target(&mut self, instr: Instruction) -> StepResult {
        let idx = instr.target.index();

        let data = self.target_buffer.clone();
        let len = data.len();
        self.hot_regs[idx] = Some(HotBuffer {
            data,
            shape: vec![len],
        });

        StepResult::Continue
    }

    pub(super) fn execute_mastery_update(&mut self, instr: Instruction) -> StepResult {
        // =========================================================================
        // DOPAMINE GATING: No learning without surprise/reward
        // =========================================================================
        // Dopamine gates whether learning happens at all.
        // Below threshold: no pressure accumulation (no learning)
        // Above threshold: learning rate scales with dopamine level
        if !self.chemical_state.learning_enabled() {
            return StepResult::Continue; // No learning without dopamine
        }

        let weights_idx = instr.target.index();
        let activity_idx = instr.source.index();
        let direction_idx = instr.aux as usize & 0x0F;
        let base_scale = if instr.modifier[0] > 0 { instr.modifier[0] as i32 } else { 15 };
        let threshold_div = if instr.modifier[1] > 0 { instr.modifier[1] as i32 } else { 4 };

        // Scale by dopamine level (1-4x multiplier based on dopamine)
        let dopamine_scale = self.chemical_state.dopamine_scale();
        let scale = base_scale * dopamine_scale;

        let activity = match &self.hot_regs[activity_idx] {
            Some(buf) => buf.data.clone(),
            None => return StepResult::Error("Activity register not allocated".to_string()),
        };

        // Get direction (error signal)
        // Direction = target - output: positive means push up, negative means push down
        // The value comparison (target > output → +1) determines learning direction
        let direction = match &self.hot_regs[direction_idx] {
            Some(buf) => {
                if buf.data.is_empty() { 0 } else { buf.data[0].signum() }
            }
            None => return StepResult::Error("Direction register not allocated".to_string()),
        };

        let pressure = self.pressure_regs[weights_idx].get_or_insert_with(|| {
            vec![0i32; activity.len()]
        });

        if pressure.len() != activity.len() {
            *pressure = vec![0i32; activity.len()];
        }

        let max_activity = activity.iter().cloned().max().unwrap_or(1).max(1);
        let threshold = max_activity / threshold_div;

        for (i, &act) in activity.iter().enumerate() {
            if act > threshold {
                let activity_strength = (act - threshold) as i64 * 256 / max_activity as i64;
                let delta = (direction as i64 * activity_strength * scale as i64 / 256) as i32;
                pressure[i] = pressure[i].saturating_add(delta);
            }
        }

        StepResult::Continue
    }

    pub(super) fn execute_mastery_commit(&mut self, instr: Instruction) -> StepResult {
        let weights_idx = instr.target.index();
        let base_threshold = if instr.modifier[0] > 0 { instr.modifier[0] as i32 } else { 50 };
        let mag_step = if instr.modifier[1] > 0 { instr.modifier[1] } else { 5 };

        let pressure = match &mut self.pressure_regs[weights_idx] {
            Some(p) => p,
            None => return StepResult::Continue,
        };

        // Get temperatures first (before mutable borrow of signals)
        let temperatures: Vec<i32> = match &self.cold_regs[weights_idx] {
            Some(buf) => (0..buf.weights.len())
                .map(|i| buf.temperature(i).threshold_multiplier())
                .collect(),
            None => return StepResult::Error("Signal register not allocated".to_string()),
        };

        let signals = match &mut self.cold_regs[weights_idx] {
            Some(buf) => &mut buf.weights,
            None => return StepResult::Error("Signal register not allocated".to_string()),
        };

        for (i, p) in pressure.iter_mut().enumerate() {
            if i >= signals.len() { break; }

            // Temperature-aware threshold:
            // HOT: base_threshold / 2 (multiplier=1, so threshold/2)
            // WARM: base_threshold (multiplier=2)
            // COOL: base_threshold * 2 (multiplier=4)
            // COLD: impossible (multiplier=MAX)
            let temp_multiplier = temperatures.get(i).copied().unwrap_or(1);
            let effective_threshold = if temp_multiplier == i32::MAX {
                i32::MAX // COLD: never update
            } else {
                (base_threshold * temp_multiplier) / 2
            };

            if p.abs() >= effective_threshold {
                let needed_polarity = if *p > 0 { 1i8 } else { -1i8 };
                let s = &mut signals[i];

                // Signal update logic: weaken-before-flip
                // Polarity is STRUCTURAL - flip only when magnitude depleted
                if s.polarity == needed_polarity {
                    // Aligned: strengthen magnitude
                    s.magnitude = s.magnitude.saturating_add(mag_step);
                } else if s.polarity == 0 {
                    // Neutral: establish polarity
                    s.polarity = needed_polarity;
                    s.magnitude = mag_step;
                } else if s.magnitude > mag_step {
                    // Misaligned: weaken first (don't flip yet)
                    s.magnitude -= mag_step;
                } else {
                    // Depleted: now flip polarity
                    s.polarity = needed_polarity;
                    s.magnitude = mag_step;
                }

                // Only clear pressure for signals that were updated
                *p = 0;
            }
            // Pressure below threshold is preserved for continued accumulation
        }

        StepResult::Continue
    }

    pub(super) fn execute_add_babble(&mut self, instr: Instruction) -> StepResult {
        let idx = instr.target.index();

        if let Some(buf) = &mut self.hot_regs[idx] {
            let babble_base = self.babble_scale * 50;

            for (i, val) in buf.data.iter_mut().enumerate() {
                let phase = self.babble_phase;
                let magnitude_factor = 77 + ((phase * 7 + i * 13) % 179);
                let sign = if (i * 7 + phase / 10) % 5 < 3 { 1i32 } else { -1i32 };
                let babble = sign * (babble_base * magnitude_factor as i32 / 255);
                let positive_bias = babble_base / 3;
                *val = val.saturating_add(babble + positive_bias);
            }

            self.babble_phase += 1;
        }

        StepResult::Continue
    }
}
