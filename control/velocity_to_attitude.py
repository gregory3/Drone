"""
control.velocity_to_attitude
Translates a navigation-frame velocity command into MAVLink body-rate + thrust
commands for an ACRO-mode flight controller.

Why this exists: the official AI-GP sim runs the vehicle in ACRO (rate) mode.
It ignores SET_POSITION_TARGET_LOCAL_NED velocity/position setpoints entirely
(verified live: a vx=2 command made the drone climb to 193 m and fly off). The
only control input it honours is SET_ATTITUDE_TARGET with body rates + collective
thrust (the example's `update_attitude_flight_control`).

This translator turns the outer PID's (vx, vy, vz, yaw_rate) request into:
  - desired body tilt angles proportional to the demanded horizontal velocity,
  - body **rates** via a proportional attitude loop against the live attitude
    (ACRO takes rates, not angles, so we close the angle loop ourselves),
  - collective thrust from a hover bias + a climb-rate PD term.

Frame: NED (x north, y east, z down) — matches the sim. Vertical: NED vz>0 is
descending, so climb_rate = -vz.

Sign conventions (from the example's `update_attitude_flight_control`, which uses
PITCH_RATE = -0.3 with the comment "negative = pitch forward"):
  - Forward motion  -> NOSE DOWN -> **negative** pitch angle / rate.
  - Right motion    -> roll right -> **positive** roll angle / rate.
  - yaw_rate is passed straight through (deg/s -> rad/s).
If the sim's tune differs, flip the SIGN_* constants below — they are the only
place that needs changing.

All gains are first-pass and config-driven; expect to tune on the rig.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from config.loader import cfg

# Sign conventions — see module docstring. Flip these if the sim tune differs.
SIGN_PITCH_FORWARD = -1.0   # forward velocity -> negative pitch (nose down)
                            # VERIFIED live 2026-06-03: forward cmd -> dN=+0.84m (correct).
SIGN_ROLL_RIGHT = +1.0      # rightward velocity -> positive roll (standard NED).
                            # The 2026-06-03 "verified -1.0" displacement test is VOID:
                            # it was run with an unstable roll loop (see SIGN_ROLL_RATE)
                            # — the drone was tumbling, so net drift measured chaos, not
                            # the convention. Re-verify live now that the loop is stable.
SIGN_YAW = +1.0             # positive yaw_rate request -> positive body yaw rate

# Inner-loop feedback sign for the roll axis. tools/attitude_sysid (2026-06-12,
# open-loop) settled this: a +0.5 rad/s roll-rate pulse produced POSITIVE
# ATTITUDE roll — the plant is STANDARD (+1). The fly17/18 tumble was the
# inner loop closing on the ODOMETRY quaternion, whose euler decode is
# (-roll, -pitch, +yaw) vs ATTITUDE on this build (fixed at the source in
# mavlink_adapter._current_attitude_and_vz). Keep this hook at +1.0 unless a
# future sim build measurably inverts the plant — re-run attitude_sysid first.
SIGN_ROLL_RATE = +1.0


@dataclass
class AttitudeRateCommand:
    """Body-rate + collective-thrust setpoint for SET_ATTITUDE_TARGET."""
    roll_rate: float    # rad/s, body frame
    pitch_rate: float   # rad/s, body frame
    yaw_rate: float     # rad/s, body frame
    thrust: float       # 0.0 .. 1.0 collective


class VelocityToAttitude:
    """Stateless converter — instantiate once, call `.convert()` each tick.

    Reads gains from cfg.control. Needs the live attitude (roll/pitch/yaw) and
    vertical velocity to close the inner angle loop and the climb-rate loop.
    """

    def __init__(self) -> None:
        c = cfg.control
        self._tilt_per_mps = math.radians(getattr(c, "att_tilt_per_mps_deg", 5.0))
        self._max_tilt = math.radians(getattr(c, "max_tilt_deg", 35.0))
        self._rate_kp = float(getattr(c, "att_rate_kp", 6.0))
        self._max_rate = math.radians(getattr(c, "att_max_rate_dps", 400.0))
        self._max_yaw_rate = math.radians(getattr(c, "max_yaw_rate_dps", 120.0))

        self._hover_thrust = float(getattr(c, "hover_thrust", 0.5))
        self._thrust_kp_climb = float(getattr(c, "thrust_kp_climb", 0.05))
        self._thrust_kd_climb = float(getattr(c, "thrust_kd_climb", 0.03))
        self._thrust_min = float(getattr(c, "thrust_min", 0.1))
        self._thrust_max = float(getattr(c, "thrust_max", 0.9))

    # ------------------------------------------------------------------
    def convert(self,
                vx: float, vy: float, vz: float,
                yaw_rate_dps: float = 0.0,
                yaw_rad: float = 0.0,
                roll_rad: float = 0.0,
                pitch_rad: float = 0.0,
                vertical_velocity_mps: Optional[float] = None) -> AttitudeRateCommand:
        """Map an NED velocity command to a body-rate + thrust command.

        vx, vy, vz          : desired NED velocity (m/s). vz>0 = descend.
        yaw_rate_dps        : desired yaw rate (deg/s).
        yaw_rad             : current heading (rad), to rotate world->body.
        roll_rad, pitch_rad : current attitude (rad), for the angle loop.
        vertical_velocity_mps: current NED vz (m/s), for climb damping.
        """
        # --- rotate desired NED horizontal velocity into body frame ---
        # NED: x north, y east. Heading yaw measured from north toward east.
        cy = math.cos(yaw_rad)
        sy = math.sin(yaw_rad)
        body_forward = cy * vx + sy * vy
        body_right = -sy * vx + cy * vy

        # --- desired tilt angles proportional to demanded body velocity ---
        desired_pitch = SIGN_PITCH_FORWARD * body_forward * self._tilt_per_mps
        desired_roll = SIGN_ROLL_RIGHT * body_right * self._tilt_per_mps
        desired_pitch = _clamp(desired_pitch, -self._max_tilt, self._max_tilt)
        desired_roll = _clamp(desired_roll, -self._max_tilt, self._max_tilt)

        # --- inner P loop: angle error -> body rate (ACRO consumes rates) ---
        pitch_rate = self._rate_kp * (desired_pitch - pitch_rad)
        roll_rate = SIGN_ROLL_RATE * self._rate_kp * (desired_roll - roll_rad)
        pitch_rate = _clamp(pitch_rate, -self._max_rate, self._max_rate)
        roll_rate = _clamp(roll_rate, -self._max_rate, self._max_rate)

        # --- yaw rate: pass through (deg/s -> rad/s), clamped ---
        yaw_rate = SIGN_YAW * math.radians(yaw_rate_dps)
        yaw_rate = _clamp(yaw_rate, -self._max_yaw_rate, self._max_yaw_rate)

        # --- collective thrust: hover bias + climb-rate PD ---
        climb_des = -vz                                   # NED vz>0 descends
        thrust = self._hover_thrust + self._thrust_kp_climb * climb_des
        if vertical_velocity_mps is not None:
            climb_act = -vertical_velocity_mps
            thrust -= self._thrust_kd_climb * climb_act   # damp actual climb
        thrust = _clamp(thrust, self._thrust_min, self._thrust_max)

        return AttitudeRateCommand(
            roll_rate=float(roll_rate),
            pitch_rate=float(pitch_rate),
            yaw_rate=float(yaw_rate),
            thrust=float(thrust),
        )


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v
