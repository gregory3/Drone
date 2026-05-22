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

    SWEEP_DURATION_S = 3.0       # How long to yaw-sweep before advancing
    BLIND_ADVANCE_DURATION_S = 1.5
    ALTITUDE_GAIN_M = 0.2

    def __init__(self) -> None:
        self._state = RecoveryState.IDLE
        self._start_t: Optional[float] = None
        self._last_known_gate_pos: Optional[np.ndarray] = None
        self._sweep_dir = 1.0   # +1 or -1 yaw direction
        self._frames_without_gate = 0
        self._trigger_frames = cfg.perception.detection_history_frames

    # ------------------------------------------------------------------
    def update(self, gate_detected: bool,
               current_pos: np.ndarray,
               last_gate_pos: Optional[np.ndarray]) -> Tuple[str, np.ndarray, float]:
        """
        Returns: (phase_name, target_pos_cmd, yaw_rate_cmd)

        Call every control cycle.
        phase_name: 'recovery' | 'resume'
        target_pos_cmd: where to fly
        yaw_rate_cmd: yaw rate in deg/s
        """
        if gate_detected:
            self._frames_without_gate = 0
            if self._state != RecoveryState.IDLE:
                self._state = RecoveryState.IDLE
                self._start_t = None
            return ("resume", current_pos, 0.0)

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
            # Hover near the last known gate position + climb slightly + yaw sweep
            if self._last_known_gate_pos is not None:
                hover_target = self._last_known_gate_pos.copy()
            else:
                hover_target = current_pos.copy()
            hover_target[2] += self.ALTITUDE_GAIN_M

            # Reverse sweep direction halfway through
            if elapsed > self.SWEEP_DURATION_S / 2:
                self._sweep_dir = -1.0

            yaw_rate = cfg.planning.recovery_search_yaw_rate * self._sweep_dir

            if elapsed >= self.SWEEP_DURATION_S:
                self._state = RecoveryState.ADVANCE_BLIND
                self._start_t = now

            return ("recovery", hover_target, yaw_rate)

        if self._state == RecoveryState.ADVANCE_BLIND:
            # Slowly advance toward last known gate position
            if self._last_known_gate_pos is not None:
                direction = self._last_known_gate_pos - current_pos
                norm = np.linalg.norm(direction)
                if norm > 0.1:
                    step = direction / norm * cfg.drone.recovery_speed_mps * 0.5
                    target = current_pos + step
                else:
                    target = self._last_known_gate_pos
            else:
                # No last known position — hold and yaw
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
