"""
control.controller
Velocity-command PID controller.

The outer loop converts a 3D target position (gate center) into a
commanded velocity. The inner loop is handled by the drone's onboard
firmware (velocity controller is standard on most FPV autopilots).

Output: (vx, vy, vz, yaw_rate) in world frame.
"""

from __future__ import annotations
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from config.loader import cfg


@dataclass
class ControlOutput:
    vx: float
    vy: float
    vz: float
    yaw_rate: float      # deg/s
    confidence: float    # 0..1 — used to gate max speed externally


class PIDController:

    def __init__(self) -> None:
        c = cfg.control
        kp = c.pos_kp
        ki = c.pos_ki
        kd = c.pos_kd

        self._kp = np.array(kp)
        self._ki = np.array(ki)
        self._kd = np.array(kd)

        self._integral = np.zeros(3)
        self._prev_error = np.zeros(3)
        self._last_t: float = time.time()

        self._max_speed = cfg.drone.max_speed_mps
        self._max_tilt = cfg.control.max_tilt_deg

    def compute(self,
                current_pos: np.ndarray,
                target_pos: np.ndarray,
                confidence: float = 1.0,
                max_speed: Optional[float] = None) -> ControlOutput:
        """
        Compute velocity command to drive drone from current_pos toward target_pos.
        confidence ∈ [0,1] scales max speed (low confidence → slow down).
        max_speed can override the configured top speed for phase-based control.
        """
        now = time.time()
        dt = now - self._last_t
        if dt <= 0 or dt > 0.5:
            dt = 0.02
        self._last_t = now

        error = target_pos - current_pos
        distance = np.linalg.norm(error)
        if distance < 0.05:
            return ControlOutput(0.0, 0.0, 0.0, 0.0, confidence)

        # Integrate (with windup guard)
        self._integral += error * dt
        self._integral = np.clip(self._integral, -2.0, 2.0)

        # Derivative
        derivative = (error - self._prev_error) / dt
        self._prev_error = error.copy()

        # PID output
        raw_cmd = (self._kp * error
                   + self._ki * self._integral
                   + self._kd * derivative)

        # Speed limit: scale by confidence (lower confidence → slower)
        speed_limit = self._max_speed if max_speed is None else min(self._max_speed, max_speed)
        speed_limit *= np.clip(confidence, 0.2, 1.0)
        speed = np.linalg.norm(raw_cmd)
        if speed > speed_limit and speed > 0:
            raw_cmd = raw_cmd / speed * speed_limit

        # Yaw toward gate: simple proportional to lateral error
        yaw_rate = float(np.clip(raw_cmd[1] * 15.0,
                                 -cfg.control.max_yaw_rate_dps,
                                  cfg.control.max_yaw_rate_dps))

        return ControlOutput(
            vx=float(raw_cmd[0]),
            vy=float(raw_cmd[1]),
            vz=float(raw_cmd[2]),
            yaw_rate=yaw_rate,
            confidence=confidence,
        )

    def reset(self) -> None:
        self._integral = np.zeros(3)
        self._prev_error = np.zeros(3)
        self._last_t = time.time()
