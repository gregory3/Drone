"""
control.velocity_to_rc
Translates a world-frame velocity command into Betaflight RC PWM values.

The Elodin AI Grand Prix rig consumes `RCCommand` PWM ints (1000-2000, center
1500). Betaflight handles the inner attitude rate loop; this translator's job
is to turn a world-frame ENU velocity setpoint into a coherent stick request
plus throttle.

Gain shape is borrowed from the rig's baseline solver (`solver/baseline.py`):
  - throttle: hover bias + altitude PD against the requested vertical speed
  - pitch / roll: world->body velocity rotated by yaw, then scaled to a
    moderate ±50 PWM offset (1450..1550) - well inside Betaflight's linear
    rate response.
  - yaw: yaw rate request normalised against MAX_YAW_DPS, clipped to ±200 PWM.

The numbers below are first-pass. Expect to tune them against the rig.
"""

from __future__ import annotations
import math
from typing import Optional

from drone_sim.competition_types import (
    RCCommand,
    RC_CENTER,
    RC_ARMED,
    RC_HOVER_THROTTLE,
    RC_TAKEOFF_THROTTLE,
    clamp_rc,
)

# ---------------------------------------------------------------------------
# Tuning constants — see solver/baseline.py for the inspiration.
# ---------------------------------------------------------------------------

# Pitch / roll: PWM offset per (m/s) of body-frame velocity command.
KP_HORIZONTAL_PWM_PER_MPS = 35.0
# Tilt stick is clamped to ±50 PWM (1450..1550) to stay in Betaflight's
# linear response region.
MAX_TILT_PWM_OFFSET = 50

# Throttle: PWM offset per (m/s) of requested vertical velocity (ENU "up").
THROTTLE_PWM_PER_VZ_MPS = 80.0
# Throttle floor / ceiling — Betaflight ignores < ~1050 and saturates motors
# above ~1700 in our race tune.
THROTTLE_MIN = 1000
THROTTLE_MAX = 1700

# Yaw stick: PWM offset per (deg/s) of yaw-rate command.
YAW_PWM_PER_DPS = 1.5
MAX_YAW_PWM_OFFSET = 200


class VelocityToRC:
    """Stateless converter — instantiate once, call `.convert()` each tick.

    A live attitude estimate is required to rotate the world-frame velocity
    into body frame (so "fly east" becomes a pitch command regardless of
    which way the drone is facing). If no attitude is supplied we assume
    yaw = 0.
    """

    def __init__(self) -> None:
        self._armed = False

    # ------------------------------------------------------------------
    def convert(self,
                vx: float, vy: float, vz: float,
                yaw_rate_dps: float = 0.0,
                yaw_rad: Optional[float] = None,
                altitude_m: Optional[float] = None,
                vertical_velocity_mps: Optional[float] = None) -> RCCommand:
        """Map an ENU velocity command into an `RCCommand`.

        altitude_m / vertical_velocity_mps are optional and used purely for
        the takeoff hand-off — if altitude is below 1 m we boost throttle to
        TAKEOFF_PWM regardless of vz request, matching the baseline solver.
        """
        # Rotate world ENU horizontal velocity into body frame using yaw.
        yaw = yaw_rad if yaw_rad is not None else 0.0
        cy = math.cos(yaw)
        sy = math.sin(yaw)
        body_forward = cy * vx + sy * vy
        body_right   = sy * vx - cy * vy

        # Sign convention for this Betaflight tune (confirmed against the
        # rig's baseline solver, which does roll = 1500 - KP * dy and
        # pitch = 1500 + KP * dx):
        #   pitch stick > 1500 -> drone moves +X (forward in ENU world)
        #   roll  stick > 1500 -> drone moves -Y (right in body when yaw=0)
        # If the official sim ships with a different tune, this is the only
        # place that needs flipping.
        pitch_offset = body_forward * KP_HORIZONTAL_PWM_PER_MPS
        roll_offset  = body_right   * KP_HORIZONTAL_PWM_PER_MPS

        pitch_offset = max(-MAX_TILT_PWM_OFFSET, min(MAX_TILT_PWM_OFFSET, pitch_offset))
        roll_offset  = max(-MAX_TILT_PWM_OFFSET, min(MAX_TILT_PWM_OFFSET, roll_offset))

        # Throttle: takeoff handoff -> hover bias + vz term.
        if altitude_m is not None and altitude_m < 1.0:
            throttle = RC_TAKEOFF_THROTTLE
        else:
            throttle = RC_HOVER_THROTTLE + vz * THROTTLE_PWM_PER_VZ_MPS
            # Bleed off a touch of throttle if we're already climbing faster
            # than requested - prevents runaway altitude.
            if vertical_velocity_mps is not None:
                throttle -= (vertical_velocity_mps - vz) * 8.0

        # Yaw stick: clamp the requested rate offset to a sane range.
        yaw_offset = yaw_rate_dps * YAW_PWM_PER_DPS
        yaw_offset = max(-MAX_YAW_PWM_OFFSET, min(MAX_YAW_PWM_OFFSET, yaw_offset))

        return RCCommand(
            arm=RC_ARMED,
            throttle=clamp_rc(throttle, lo=THROTTLE_MIN, hi=THROTTLE_MAX),
            roll=clamp_rc(RC_CENTER + roll_offset),
            pitch=clamp_rc(RC_CENTER + pitch_offset),
            yaw=clamp_rc(RC_CENTER + yaw_offset),
        )
