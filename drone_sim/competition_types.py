"""
sim.competition_types
Mirror of the Elodin AI Grand Prix `solver/api.py` contract.

These dataclasses match the upstream rig at github.com/elodin-sys/ai-grand-prix
(verified against v0.17.3, 2026-05-28). Values are kept identical to upstream
so a contestant solver can be moved between repos without translation.

If the official DCL simulator publishes a different contract later, this
module is the only place that needs to change.

Channel values are PWM microseconds in the standard Betaflight range:
  1000 = min (idle / disarmed)
  1500 = center (no input)
  2000 = max
  AUX1 (arm) >= 1700 = armed.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class SensorUpdate:
    """Per-tick sensor bundle handed to the autopilot.

    `world_pos` layout: [qx, qy, qz, qw, x, y, z]   (quaternion + ENU position)
    `world_vel` layout: [wx, wy, wz, vx, vy, vz]    (angular + linear velocity)

    These are the rig's "cheat" channels — the official DCL sim will likely
    expose them only for the starting position. Anything that uses them
    today should fall back to perception once the official sim drops.
    """

    t: float
    tick: int

    world_pos: np.ndarray
    world_vel: np.ndarray

    gyro: np.ndarray
    accel: np.ndarray
    gyro_fresh: bool = True
    accel_fresh: bool = True

    baro: float = 0.0
    baro_fresh: bool = False
    mag: np.ndarray = field(default_factory=lambda: np.zeros(3))
    mag_fresh: bool = False

    # The camera frame is RGBA (4-channel) when the rig is connected to a
    # render-server; None on ticks where no fresh frame is available.
    frame_rgba: Optional[np.ndarray] = None
    frame_fresh: bool = False

    last_gate_passed: int = -1
    next_gate_index: int = -1

    # ------------------------------------------------------------------
    # Convenience accessors so consumers don't sprinkle magic indices
    # ------------------------------------------------------------------
    @property
    def position(self) -> np.ndarray:
        """ENU position (x, y, z) in metres."""
        return np.asarray(self.world_pos[4:7], dtype=float)

    @property
    def linear_velocity(self) -> np.ndarray:
        """ENU linear velocity (vx, vy, vz) in m/s."""
        if self.world_vel.size < 6:
            return np.zeros(3)
        return np.asarray(self.world_vel[3:6], dtype=float)

    @property
    def angular_velocity(self) -> np.ndarray:
        """Body angular velocity (wx, wy, wz) in rad/s."""
        if self.world_vel.size < 3:
            return np.zeros(3)
        return np.asarray(self.world_vel[0:3], dtype=float)

    @property
    def quaternion(self) -> np.ndarray:
        """Body attitude quaternion (qx, qy, qz, qw)."""
        return np.asarray(self.world_pos[0:4], dtype=float)


@dataclass
class RCCommand:
    """Betaflight RC channel outputs (PWM microseconds, AETR + AUX layout)."""

    throttle: int = 1000
    roll:     int = 1500
    pitch:    int = 1500
    yaw:      int = 1500
    arm:      int = 1000      # >= 1700 to arm
    aux2:     int = 1500
    aux3:     int = 1500
    aux4:     int = 1500


class CameraSpec:
    """Forward FPV camera intrinsics per the rig README."""

    WIDTH_PX:    int   = 640
    HEIGHT_PX:   int   = 360
    HFOV_DEG:    float = 90.0
    TILT_DEG:    float = 20.0
    FRAME_HZ:    int   = 30
    FX:          float = 320.0
    FY:          float = 320.0
    CX:          float = 320.0
    CY:          float = 180.0


# ---------------------------------------------------------------------------
# RC PWM helpers
# ---------------------------------------------------------------------------

RC_MIN     = 1000
RC_MAX     = 2000
RC_CENTER  = 1500
RC_ARMED   = 1800
RC_DISARM  = 1000
RC_HOVER_THROTTLE  = 1135
RC_TAKEOFF_THROTTLE = 1300


def clamp_rc(value: float, lo: int = RC_MIN, hi: int = RC_MAX) -> int:
    """Round + clamp a float into a valid Betaflight PWM integer."""
    return int(round(max(lo, min(hi, value))))
