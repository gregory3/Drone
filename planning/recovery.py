"""
planning.recovery
Recovery behavior when gate is lost.

Strategy (Phase 1):
  1. Hold current position (zero velocity)
  2. Yaw sweep left/right to scan for gate
  3. Slight altitude gain to improve FOV
  4. If gate found within timeout → return to approach
  5. If timeout exceeded → advance toward last known gate position

The key principle: slow down and look before doing anything drastic.
Fast wrong moves lose the course entirely.
"""

from __future__ import annotations
import time
from enum import Enum, auto
from typing import Optional, Tuple

import numpy as np

from config.loader import cfg


class RecoveryState(Enum):
    IDLE = auto()
    YAW_SWEEP = auto()
    ADVANCE_BLIND = auto()
    COMPLETE = auto()
    FAILED = auto()


class RecoveryBehavior:

    SWEEP_DURATION_S = 3.0       # How long to hold + yaw-sweep before advancing.
                                 # Halved from 6.0: at 50Hz, 6s of holding = 300
                                 # frozen loops. 3s is enough to re-acquire or move on.
    BLIND_ADVANCE_DURATION_S = 1.5
    ALTITUDE_GAIN_M = 0.2

    def __init__(self) -> None:
        self._state = RecoveryState.IDLE
        self._start_t: Optional[float] = None
        self._last_known_gate_pos: Optional[np.ndarray] = None
        self._last_heading_rad: Optional[float] = None
        self._sweep_dir = 1.0   # +1 or -1 yaw direction
        self._frames_without_gate = 0
        # Recovery engages after this many consecutive missed frames. Decoupled
        # from the main loop's stale-detection hold (detection_history_frames),
        # which is now much shorter; falls back to it if the key is absent.
        self._trigger_frames = getattr(
            cfg.perception, "recovery_trigger_frames",
            cfg.perception.detection_history_frames)
        self._resume_confirm_frames = 0
        self._required_resume_confirm = cfg.planning.recovery_resume_confirm_frames

    # ------------------------------------------------------------------
    def update(self, gate_detected: bool,
               gate_confidence: float,
               current_pos: np.ndarray,
               last_gate_pos: Optional[np.ndarray],
               heading_rad: Optional[float] = None) -> Tuple[str, np.ndarray, float]:
        """
        Returns: (phase_name, target_pos_cmd, yaw_rate_cmd)

        Call every control cycle.
        phase_name: 'recovery' | 'resume'
        target_pos_cmd: where to fly
        yaw_rate_cmd: yaw rate in deg/s
        heading_rad: current heading; stored while the gate is visible so a
            blind advance can dead-reckon forward instead of chasing a stale
            absolute position.
        """
        if gate_detected:
            if heading_rad is not None:
                self._last_heading_rad = heading_rad
            self._frames_without_gate = 0
            self._resume_confirm_frames += 1
            if self._state == RecoveryState.IDLE:
                self._resume_confirm_frames = 0
                return ("resume", current_pos, 0.0)

            if self._resume_confirm_frames >= self._required_resume_confirm:
                self._state = RecoveryState.IDLE
                self._start_t = None
                self._resume_confirm_frames = 0
                return ("resume", current_pos, 0.0)

            return ("recovery", current_pos, 0.0)

        self._resume_confirm_frames = 0
        self._frames_without_gate += 1

        # Haven't lost it long enough to trigger recovery
        if self._frames_without_gate < self._trigger_frames:
            return ("resume", current_pos, 0.0)

        # --- recovery is active ---
        if last_gate_pos is not None:
            self._last_known_gate_pos = last_gate_pos

        if self._state == RecoveryState.IDLE:
            self._state = RecoveryState.YAW_SWEEP
            self._start_t = time.time()
            self._sweep_dir = 1.0

        now = time.time()
        elapsed = now - (self._start_t or now)

        if self._state == RecoveryState.YAW_SWEEP:
            # HOLD current position and yaw-sweep to re-find the gate. Do NOT
            # chase an absolute last-known position: in vision-only flight the
            # position estimate drifts, so targeting a stale absolute point
            # flies the drone away. Setting target == current_pos yields ~zero
            # translation error (robust to that drift); only yaw changes.
            hover_target = current_pos.copy()

            # Reverse sweep direction halfway through
            if elapsed > self.SWEEP_DURATION_S / 2:
                self._sweep_dir = -1.0

            yaw_rate = cfg.planning.recovery_search_yaw_rate * self._sweep_dir

            if elapsed >= self.SWEEP_DURATION_S:
                self._state = RecoveryState.ADVANCE_BLIND
                self._start_t = now

            return ("recovery", hover_target, yaw_rate)

        if self._state == RecoveryState.ADVANCE_BLIND:
            # Dead-reckoning (MonoRace principle): the absolute position estimate
            # has drifted by the time we get here, so DON'T chase a stale
            # last_known_gate_pos. Instead hold altitude and creep forward along
            # the heading the gate was last seen on — gentle (0.3x recovery
            # speed) and time-bounded, so a genuinely lost gate can't fly us out
            # of bounds. Falls back to a small climb if we never had a heading.
            if self._last_heading_rad is not None:
                step_fwd = cfg.drone.recovery_speed_mps * 0.3
                step = np.array([
                    np.cos(self._last_heading_rad) * step_fwd,
                    np.sin(self._last_heading_rad) * step_fwd,
                    0.0,                       # hold altitude
                ])
                target = current_pos + step
            else:
                target = current_pos.copy()
                target[2] += self.ALTITUDE_GAIN_M

            if elapsed >= self.BLIND_ADVANCE_DURATION_S:
                # Give up on this gate, advance to next
                self._state = RecoveryState.FAILED
                print("[Recovery] FAILED — gate not found after full sweep.")

            return ("recovery", target, 0.0)

        if self._state == RecoveryState.FAILED:
            # Best effort: fly toward last known gate position
            tgt = self._last_known_gate_pos if self._last_known_gate_pos is not None \
                  else current_pos
            return ("recovery", tgt, 0.0)

        return ("recovery", current_pos, 0.0)

    def reset(self) -> None:
        self._state = RecoveryState.IDLE
        self._start_t = None
        self._frames_without_gate = 0
        self._last_known_gate_pos = None
        self._last_heading_rad = None
        self._resume_confirm_frames = 0
