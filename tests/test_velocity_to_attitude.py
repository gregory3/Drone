"""
tests/test_velocity_to_attitude.py
Unit tests for the velocity -> body-rate + thrust translator (ACRO control).
"""
import sys
import math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from control.velocity_to_attitude import VelocityToAttitude, AttitudeRateCommand
from config.loader import cfg


def test_forward_velocity_pitches_nose_down():
    # Forward (NED north) command with level attitude -> negative pitch rate
    # (the sim convention: negative pitch = nose down = accelerate forward).
    v = VelocityToAttitude()
    cmd = v.convert(vx=2.0, vy=0.0, vz=0.0, yaw_rad=0.0,
                    roll_rad=0.0, pitch_rad=0.0)
    assert isinstance(cmd, AttitudeRateCommand)
    assert cmd.pitch_rate < 0, "forward -> nose down -> negative pitch rate"
    assert abs(cmd.roll_rate) < 1e-6


def _measured_plant_step(angle, rate_state, commanded_rate, dt):
    """One step of the plant tools/attitude_sysid measured on the live sim
    (2026-06-12): rate commands are followed with STANDARD sign but ~2x gain
    and momentum — the achieved rate lags the command (rotation continued
    ~10 deg after a pulse stopped). First-order rate lag, tau ~0.25s."""
    tau = 0.25
    gain = 2.0
    rate_state += (gain * commanded_rate - rate_state) * (dt / tau)
    return angle + rate_state * dt, rate_state


def test_roll_loop_stable_on_measured_plant():
    """fly17/18 regression, closed-loop against the MEASURED plant. The tumble
    root cause was inverted attitude feedback from the ODOMETRY quaternion
    (fixed in mavlink_adapter); the plant itself is standard-sign. The loop
    must converge to the commanded bank on the 2x-gain laggy plant without
    winding up — this is what bounds att_rate_kp."""
    from control.velocity_to_attitude import SIGN_ROLL_RIGHT
    v = VelocityToAttitude()
    tilt_per_mps = math.radians(cfg.control.att_tilt_per_mps_deg)
    desired = SIGN_ROLL_RIGHT * 2.0 * tilt_per_mps

    roll, rate = 0.0, 0.0
    dt = 0.02  # 50 Hz, matching the live command rate
    peak = 0.0
    for _ in range(300):  # 6 simulated seconds
        cmd = v.convert(vx=0.0, vy=2.0, vz=0.0, yaw_rad=0.0,
                        roll_rad=roll, pitch_rad=0.0)
        roll, rate = _measured_plant_step(roll, rate, cmd.roll_rate, dt)
        peak = max(peak, abs(roll))
    assert abs(roll - desired) < math.radians(1.5), \
        f"roll must converge (got {math.degrees(roll):.1f}°, " \
        f"want {math.degrees(desired):.1f}°)"
    assert peak < math.radians(45.0), \
        "no tumble: transient must stay far from the rate-clamp wind-up regime"


def test_pitch_loop_stable_on_measured_plant():
    v = VelocityToAttitude()
    tilt_per_mps = math.radians(cfg.control.att_tilt_per_mps_deg)
    desired = -1.0 * 2.0 * tilt_per_mps  # SIGN_PITCH_FORWARD * v * gain

    pitch, rate = 0.0, 0.0
    dt = 0.02
    for _ in range(300):
        cmd = v.convert(vx=2.0, vy=0.0, vz=0.0, yaw_rad=0.0,
                        roll_rad=0.0, pitch_rad=pitch)
        pitch, rate = _measured_plant_step(pitch, rate, cmd.pitch_rate, dt)
    assert abs(pitch - desired) < math.radians(1.5)


def test_yaw_world_to_body_rotation():
    # Facing east (yaw=90deg): a world-north velocity becomes body-LEFT, so it
    # should produce a roll command, not pitch.
    v = VelocityToAttitude()
    cmd = v.convert(vx=2.0, vy=0.0, vz=0.0, yaw_rad=math.radians(90),
                    roll_rad=0.0, pitch_rad=0.0)
    assert abs(cmd.pitch_rate) < 1e-3, "north while facing east -> not forward"
    assert abs(cmd.roll_rate) > 1e-3


def test_climb_increases_thrust():
    # NED vz < 0 means climbing -> more than hover thrust.
    v = VelocityToAttitude()
    hover = cfg.control.hover_thrust
    cmd = v.convert(vx=0.0, vy=0.0, vz=-2.0)
    assert cmd.thrust > hover


def test_descend_decreases_thrust():
    v = VelocityToAttitude()
    hover = cfg.control.hover_thrust
    cmd = v.convert(vx=0.0, vy=0.0, vz=2.0)
    assert cmd.thrust < hover


def test_thrust_clamped_to_limits():
    v = VelocityToAttitude()
    cmd = v.convert(vx=0.0, vy=0.0, vz=-1000.0)
    assert cfg.control.thrust_min <= cmd.thrust <= cfg.control.thrust_max


def test_rates_clamped_to_max():
    v = VelocityToAttitude()
    max_rate = math.radians(cfg.control.att_max_rate_dps)
    cmd = v.convert(vx=1000.0, vy=1000.0, vz=0.0,
                    roll_rad=0.0, pitch_rad=0.0)
    assert abs(cmd.pitch_rate) <= max_rate + 1e-9
    assert abs(cmd.roll_rate) <= max_rate + 1e-9


def test_angle_loop_zero_rate_when_at_target():
    # If the drone is already tilted to the angle the demanded velocity wants,
    # the rate command should be ~0 (inner P loop satisfied).
    v = VelocityToAttitude()
    tilt_per_mps = math.radians(cfg.control.att_tilt_per_mps_deg)
    desired_pitch = -1.0 * 2.0 * tilt_per_mps   # SIGN_PITCH_FORWARD * v * gain
    cmd = v.convert(vx=2.0, vy=0.0, vz=0.0, yaw_rad=0.0,
                    roll_rad=0.0, pitch_rad=desired_pitch)
    assert abs(cmd.pitch_rate) < 1e-6


def test_yaw_rate_passthrough_sign_and_units():
    v = VelocityToAttitude()
    cmd = v.convert(vx=0.0, vy=0.0, vz=0.0, yaw_rate_dps=30.0)
    assert cmd.yaw_rate > 0
    assert cmd.yaw_rate == pytest.approx(math.radians(30.0), abs=1e-6)


def test_climb_damping_reduces_thrust_when_already_climbing():
    v = VelocityToAttitude()
    # Same desired climb, but one already climbing fast -> less thrust.
    base = v.convert(vx=0, vy=0, vz=-2.0, vertical_velocity_mps=0.0)
    damped = v.convert(vx=0, vy=0, vz=-2.0, vertical_velocity_mps=-5.0)
    assert damped.thrust < base.thrust
