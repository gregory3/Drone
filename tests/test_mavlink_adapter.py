"""
tests/test_mavlink_adapter.py
Unit + integration tests for the official AI-GP MAVLink adapter.

These run with no pymavlink, no sockets and no running sim: wire formats are
exercised via crafted bytes, MAVLink messages via lightweight fakes, and the
control path via a recording fake connection.

Run: pytest tests/test_mavlink_adapter.py -v
"""

import sys
import struct
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import math
import numpy as np
import pytest

from drone_sim.mavlink_adapter import (
    MavlinkSimInterface,
    VisionFrameAssembler,
    parse_race_status,
    parse_track_data,
    quaternion_to_euler_deg,
    RACE_STATUS_FMT,
    TRACK_GATE_FMT,
    VISION_HEADER_FMT,
    VELOCITY_YAWRATE_MASK,
    MAV_FRAME_LOCAL_NED,
    ENCAPSULATED_RACE_STATUS_MSG_ID,
    ENCAPSULATED_TRACK_INFO_MSG_ID,
    COLLISION_ID_ENVIRONMENT,
    COLLISION_ID_GATE,
)


# ---------------------------------------------------------------------------
# Test helpers: fake MAVLink message + recording connection
# ---------------------------------------------------------------------------

class FakeMsg:
    """Minimal stand-in for a pymavlink message: a type tag + attributes."""

    def __init__(self, msg_type, **fields):
        self._type = msg_type
        for k, v in fields.items():
            setattr(self, k, v)

    def get_type(self):
        return self._type


class RecordingMav:
    def __init__(self):
        self.setpoints = []
        self.attitude_targets = []
        self.commands = []

    def set_position_target_local_ned_send(self, *args):
        self.setpoints.append(args)

    def set_attitude_target_send(self, *args):
        self.attitude_targets.append(args)

    def command_long_send(self, *args):
        self.commands.append(args)

    def timesync_send(self, *args):
        pass


class FakeConn:
    def __init__(self):
        self.mav = RecordingMav()
        self.target_system = 1
        self.target_component = 1


def _connected_adapter(control_mode="velocity"):
    """An adapter with a fake connection, bypassing the network connect().

    Defaults to the post-arming "armed" state so control-path tests exercise
    real command output. Auto-arm tests reset _arm_state/_race_armed to idle.
    """
    a = MavlinkSimInterface(control_mode=control_mode)
    a._conn = FakeConn()
    a._connected = True
    a._arm_state = "armed"
    a._race_armed = True
    if control_mode == "attitude":
        from control.velocity_to_attitude import VelocityToAttitude
        a._vel_to_att = VelocityToAttitude()
    return a


# ---------------------------------------------------------------------------
# Pure parsers
# ---------------------------------------------------------------------------

def test_parse_race_status_roundtrip():
    payload = struct.pack(
        RACE_STATUS_FMT,
        ENCAPSULATED_RACE_STATUS_MSG_ID,
        123456,        # sim_boot_time_ms
        1000,          # race_start_boot_time_ms
        -1,            # race_finish_time_ns (ongoing)
        3,             # active_gate_index
        42,            # last_gate_race_time
    )
    status = parse_race_status(payload)
    assert status.sim_boot_time_ms == 123456
    assert status.active_gate_index == 3
    assert status.race_started is True
    assert status.race_finished is False


def test_parse_race_status_not_started():
    payload = struct.pack(RACE_STATUS_FMT,
                          ENCAPSULATED_RACE_STATUS_MSG_ID, 10, -1, -1, 0, -1)
    status = parse_race_status(payload)
    assert status.race_started is False
    assert status.race_finished is False


def test_parse_track_data_positions_and_order():
    num_gates = 3
    payload = struct.pack("<H", num_gates)
    specs = [
        (0, 1.0, 2.0, -3.0, 1.0, 0.0, 0.0, 0.0, 1.5, 1.5),
        (1, 4.0, 5.0, -3.0, 1.0, 0.0, 0.0, 0.0, 1.2, 1.2),
        (2, 7.0, 8.0, -3.0, 1.0, 0.0, 0.0, 0.0, 2.0, 2.0),
    ]
    for s in specs:
        payload += struct.pack(TRACK_GATE_FMT, *s)

    gates = parse_track_data(payload)
    assert len(gates) == 3
    assert gates[0].gate_id == 0
    np.testing.assert_allclose(gates[0].position_ned, [1.0, 2.0, -3.0])
    np.testing.assert_allclose(gates[2].position_ned, [7.0, 8.0, -3.0])
    assert gates[1].width_m == pytest.approx(1.2)
    # NED z is negative for altitude — confirms frame convention carried through
    assert gates[0].position_ned[2] < 0


def test_quaternion_to_euler_identity():
    roll, pitch, yaw = quaternion_to_euler_deg(1.0, 0.0, 0.0, 0.0)
    assert roll == pytest.approx(0.0)
    assert pitch == pytest.approx(0.0)
    assert yaw == pytest.approx(0.0)


def test_quaternion_to_euler_yaw_90():
    # 90 deg yaw about Z: q = (cos45, 0, 0, sin45)
    c = math.cos(math.radians(45))
    roll, pitch, yaw = quaternion_to_euler_deg(c, 0.0, 0.0, c)
    assert yaw == pytest.approx(90.0, abs=1e-6)
    assert roll == pytest.approx(0.0, abs=1e-6)
    assert pitch == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Vision reassembly
# ---------------------------------------------------------------------------

def _make_vision_packets(image, frame_id=1, n_chunks=2, sim_time_ns=1_000):
    import cv2
    ok, buf = cv2.imencode(".jpg", image)
    assert ok
    jpeg = buf.tobytes()
    jpeg_size = len(jpeg)
    # Split into n_chunks roughly-equal pieces.
    step = math.ceil(jpeg_size / n_chunks)
    packets = []
    for i in range(n_chunks):
        piece = jpeg[i * step:(i + 1) * step]
        header = struct.pack(VISION_HEADER_FMT, frame_id, i, n_chunks,
                             jpeg_size, len(piece), sim_time_ns)
        packets.append(header + piece)
    return packets


def test_vision_assembler_reassembles_and_decodes():
    img = np.random.randint(0, 255, (360, 640, 3), dtype=np.uint8)
    asm = VisionFrameAssembler()
    packets = _make_vision_packets(img, n_chunks=3)

    result = None
    for p in packets:
        result = asm.add_packet(p)
    assert result is not None, "Final chunk should complete the frame"
    frame_id, decoded, sim_time_ns = result
    assert frame_id == 1
    assert decoded.shape == (360, 640, 3)
    assert sim_time_ns == 1_000


def test_vision_assembler_partial_returns_none():
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    asm = VisionFrameAssembler()
    packets = _make_vision_packets(img, n_chunks=2)
    assert asm.add_packet(packets[0]) is None  # only first chunk


def test_vision_assembler_ignores_runt_packet():
    asm = VisionFrameAssembler()
    assert asm.add_packet(b"\x00\x01") is None


# ---------------------------------------------------------------------------
# Message dispatch -> snapshot -> observation/ground-truth
# ---------------------------------------------------------------------------

def test_highres_imu_flows_to_observation():
    a = _connected_adapter()
    a._apply_message(FakeMsg("HIGHRES_IMU",
                             xacc=0.1, yacc=0.2, zacc=-9.81,
                             xgyro=0.01, ygyro=0.02, zgyro=0.03,
                             time_usec=2_000_000))
    obs = a.get_observation()
    assert obs.imu.accel == (0.1, 0.2, -9.81)
    assert obs.imu.gyro == (0.01, 0.02, 0.03)
    assert obs.imu.timestamp == pytest.approx(2.0)
    assert len(obs.rpm) == 4


def test_observation_synthesizes_black_frame_when_no_camera():
    from config.loader import cfg
    a = _connected_adapter()
    obs = a.get_observation()
    assert obs.image.shape == (cfg.perception.image_height,
                               cfg.perception.image_width, 3)
    assert np.all(obs.image == 0)


def test_get_observation_requires_connection():
    a = MavlinkSimInterface()
    with pytest.raises(RuntimeError):
        a.get_observation()


def test_odometry_flows_to_ground_truth_ned():
    a = _connected_adapter()
    # 90 deg yaw quaternion (w, x, y, z)
    c = math.cos(math.radians(45))
    a._apply_message(FakeMsg("ODOMETRY",
                             x=1.0, y=2.0, z=-3.0,
                             vx=0.5, vy=0.0, vz=0.0,
                             q=[c, 0.0, 0.0, c]))
    gt = a.get_ground_truth()
    np.testing.assert_allclose(gt.pos, [1.0, 2.0, -3.0])
    np.testing.assert_allclose(gt.vel, [0.5, 0.0, 0.0])
    assert gt.att_deg[2] == pytest.approx(90.0, abs=1e-4)


def test_attitude_message_sets_euler():
    a = _connected_adapter()
    a._apply_message(FakeMsg("ATTITUDE",
                             roll=0.0, pitch=math.radians(10.0), yaw=0.0,
                             rollspeed=0, pitchspeed=0, yawspeed=0,
                             time_boot_ms=0))
    gt = a.get_ground_truth()
    assert gt.att_deg[1] == pytest.approx(10.0, abs=1e-4)


def test_local_position_does_not_override_odometry():
    a = _connected_adapter()
    a._apply_message(FakeMsg("ODOMETRY", x=1.0, y=1.0, z=-1.0,
                             vx=0, vy=0, vz=0, q=[1.0, 0, 0, 0]))
    a._apply_message(FakeMsg("LOCAL_POSITION_NED", x=9.0, y=9.0, z=-9.0,
                             vx=0, vy=0, vz=0, time_boot_ms=0))
    gt = a.get_ground_truth()
    np.testing.assert_allclose(gt.pos, [1.0, 1.0, -1.0])  # odometry wins


def test_ground_truth_none_before_any_telemetry():
    a = _connected_adapter()
    assert a.get_ground_truth() is None


def test_attitude_and_velocity_flow_to_observation():
    """Spec-legal telemetry (VADR-TS-002 §4.5) must reach the autonomy loop.

    fly14 root cause: the loop reprojected gates with pitch=0 / EKF yaw while
    the real ATTITUDE sat unused in the adapter snapshot -> phantom +20°
    target above every gate -> runaway climb off the course."""
    a = _connected_adapter()
    a._apply_message(FakeMsg("ATTITUDE",
                             roll=0.0, pitch=math.radians(10.0),
                             yaw=math.radians(45.0),
                             rollspeed=0, pitchspeed=0, yawspeed=0,
                             time_boot_ms=0))
    a._apply_message(FakeMsg("ODOMETRY",
                             x=1.0, y=2.0, z=-3.0,
                             vx=0.5, vy=-0.25, vz=0.1,
                             q=[1.0, 0.0, 0.0, 0.0]))
    obs = a.get_observation()
    assert obs.att_deg is not None, "ATTITUDE must flow into Observation"
    assert obs.att_deg[1] == pytest.approx(10.0, abs=1e-4)
    assert obs.att_deg[2] == pytest.approx(45.0, abs=1e-4)
    assert obs.vel_ned is not None, "NED velocity must flow into Observation"
    np.testing.assert_allclose(obs.vel_ned, [0.5, -0.25, 0.1])


def test_observation_telemetry_none_before_messages():
    a = _connected_adapter()
    obs = a.get_observation()
    assert obs.att_deg is None
    assert obs.vel_ned is None


def test_actuator_output_status_sets_rpm():
    a = _connected_adapter()
    a._apply_message(FakeMsg("ACTUATOR_OUTPUT_STATUS",
                             actuator=[100.0, 200.0, 300.0, 400.0, 0, 0, 0, 0],
                             time_usec=0))
    obs = a.get_observation()
    assert obs.rpm == [100.0, 200.0, 300.0, 400.0]


# ---------------------------------------------------------------------------
# Collisions
# ---------------------------------------------------------------------------

def test_environment_collision_sets_is_crashed():
    a = _connected_adapter()
    a._apply_message(FakeMsg("COLLISION", id=COLLISION_ID_ENVIRONMENT,
                             threat_level=2, horizontal_minimum_delta=5.0))
    obs = a.get_observation()
    assert obs.is_crashed is True
    last = a.get_last_collision()
    assert last.is_environment and last.threat_level == 2


def test_gate_collision_does_not_set_crashed():
    a = _connected_adapter()
    a._apply_message(FakeMsg("COLLISION", id=COLLISION_ID_GATE,
                             threat_level=1, horizontal_minimum_delta=1.0))
    obs = a.get_observation()
    assert obs.is_crashed is False


def test_old_collision_outside_window_clears_crashed():
    a = _connected_adapter()
    a._apply_message(FakeMsg("COLLISION", id=COLLISION_ID_ENVIRONMENT,
                             threat_level=2, horizontal_minimum_delta=5.0))
    # Backdate the collision beyond the crash window.
    with a._lock:
        a._snap.last_env_collision_t = time.time() - 5.0
    obs = a.get_observation()
    assert obs.is_crashed is False


# ---------------------------------------------------------------------------
# Encapsulated race status + track data (chunked)
# ---------------------------------------------------------------------------

def test_encapsulated_race_status_updates_snapshot():
    a = _connected_adapter()
    payload = struct.pack(RACE_STATUS_FMT,
                          ENCAPSULATED_RACE_STATUS_MSG_ID, 50, 10, -1, 2, 7)
    a._apply_message(FakeMsg("ENCAPSULATED_DATA", data=payload))
    rs = a.get_race_status()
    assert rs is not None and rs.active_gate_index == 2


def test_track_data_chunked_to_course():
    a = _connected_adapter()
    transfer_id = 5

    # Build a 2-gate track payload.
    track = struct.pack("<H", 2)
    track += struct.pack(TRACK_GATE_FMT, 1, 4.0, 0.0, -2.0, 1, 0, 0, 0, 1.5, 1.5)
    track += struct.pack(TRACK_GATE_FMT, 0, 1.0, 0.0, -2.0, 1, 0, 0, 0, 1.5, 1.5)

    # Announce the transfer (DATA_TRANSMISSION_HANDSHAKE repurposed).
    a._apply_message(FakeMsg("DATA_TRANSMISSION_HANDSHAKE",
                             width=transfer_id, packets=1))
    # Single chunk: 3-byte header (data_type, transfer_id) + track payload.
    data = struct.pack("<BH", ENCAPSULATED_TRACK_INFO_MSG_ID, transfer_id) + track
    a._apply_message(FakeMsg("ENCAPSULATED_DATA", data=data, seqnr=0))

    course = a.get_course()
    assert course is not None and len(course) == 2
    # Sorted by gate_id: gate 0 first.
    np.testing.assert_allclose(course[0], [1.0, 0.0, -2.0])
    np.testing.assert_allclose(course[1], [4.0, 0.0, -2.0])


def test_track_chunk_without_handshake_ignored():
    a = _connected_adapter()
    data = struct.pack("<BH", ENCAPSULATED_TRACK_INFO_MSG_ID, 99) + struct.pack("<H", 0)
    a._apply_message(FakeMsg("ENCAPSULATED_DATA", data=data, seqnr=0))
    assert a.get_course() is None


# ---------------------------------------------------------------------------
# Control: velocity setpoint construction
# ---------------------------------------------------------------------------

def test_send_velocity_command_builds_ned_setpoint():
    a = _connected_adapter()
    a.send_velocity_command(vx=2.0, vy=-1.0, vz=0.5, yaw_rate=90.0)

    assert len(a._conn.mav.setpoints) == 1
    args = a._conn.mav.setpoints[0]
    # (time_boot_ms, sys, comp, frame, mask, x,y,z, vx,vy,vz, afx,afy,afz, yaw, yaw_rate)
    assert args[3] == MAV_FRAME_LOCAL_NED
    assert args[4] == VELOCITY_YAWRATE_MASK
    assert args[8] == pytest.approx(2.0)    # vx
    assert args[9] == pytest.approx(-1.0)   # vy
    assert args[10] == pytest.approx(0.5)   # vz
    assert args[14] == pytest.approx(0.0)   # yaw ignored
    assert args[15] == pytest.approx(math.pi / 2)  # 90 deg/s -> rad/s


def test_velocity_mask_ignores_position_accel_yaw_only():
    # Velocity bits and yaw-rate bit must NOT be set; position/accel/yaw must be.
    assert not (VELOCITY_YAWRATE_MASK & 8)     # VX not ignored
    assert not (VELOCITY_YAWRATE_MASK & 16)    # VY not ignored
    assert not (VELOCITY_YAWRATE_MASK & 32)    # VZ not ignored
    assert not (VELOCITY_YAWRATE_MASK & 2048)  # yaw rate not ignored
    assert VELOCITY_YAWRATE_MASK & 1           # X ignored
    assert VELOCITY_YAWRATE_MASK & 1024        # yaw ignored


def test_send_velocity_requires_connection():
    a = MavlinkSimInterface()
    with pytest.raises(RuntimeError):
        a.send_velocity_command(1.0, 0.0, 0.0)


# ---------------------------------------------------------------------------
# Control: attitude (body-rate + thrust) path — the ACRO-mode default
# ---------------------------------------------------------------------------

def test_attitude_mode_sends_attitude_target_not_setpoint():
    from drone_sim.mavlink_adapter import ATT_TYPEMASK_ATTITUDE_IGNORE
    a = _connected_adapter(control_mode="attitude")
    a.send_velocity_command(vx=2.0, vy=0.0, vz=0.0, yaw_rate=0.0)

    assert len(a._conn.mav.attitude_targets) == 1
    assert len(a._conn.mav.setpoints) == 0, "attitude mode must not use velocity setpoint"
    args = a._conn.mav.attitude_targets[0]
    # (time, sys, comp, type_mask, q[4], roll_rate, pitch_rate, yaw_rate, thrust)
    assert args[3] == ATT_TYPEMASK_ATTITUDE_IGNORE
    pitch_rate = args[6]
    thrust = args[8]
    assert pitch_rate < 0, "forward command -> nose-down (negative pitch rate)"
    assert 0.0 <= thrust <= 1.0


def test_attitude_mode_uses_live_attitude_for_angle_loop():
    # Feed an ODOMETRY attitude already tilted to the target; the angle loop
    # should then command ~0 pitch rate.
    import math
    from control.velocity_to_attitude import VelocityToAttitude
    from config.loader import cfg
    a = _connected_adapter(control_mode="attitude")

    tilt_per_mps = math.radians(cfg.control.att_tilt_per_mps_deg)
    desired_pitch = -1.0 * 2.0 * tilt_per_mps
    # Build a quaternion for pitch = desired_pitch (roll=yaw=0): q=(cos(p/2), 0, sin(p/2), 0)
    half = desired_pitch / 2.0
    a._apply_message(FakeMsg("ODOMETRY", x=0, y=0, z=0, vx=0, vy=0, vz=0,
                             q=[math.cos(half), 0.0, math.sin(half), 0.0]))
    a.send_velocity_command(vx=2.0, vy=0.0, vz=0.0)
    args = a._conn.mav.attitude_targets[0]
    assert abs(args[6]) < 1e-3, "already at target pitch -> ~0 pitch rate"


def test_velocity_mode_still_uses_setpoint():
    a = _connected_adapter(control_mode="velocity")
    a.send_velocity_command(vx=1.0, vy=0.0, vz=0.0)
    assert len(a._conn.mav.setpoints) == 1
    assert len(a._conn.mav.attitude_targets) == 0


def test_default_control_mode_is_attitude():
    from config.loader import cfg
    a = MavlinkSimInterface()
    assert a._control_mode == "attitude" == cfg.control.mavlink_control_mode


# ---------------------------------------------------------------------------
# Auto-arm on race-active
# ---------------------------------------------------------------------------

def test_auto_arm_throttle_down_then_arms_once():
    a = _connected_adapter()
    a._arm_state = "idle"; a._race_armed = False   # start pre-arm
    # No IMU yet -> not active -> no arm.
    a._maybe_auto_arm()
    assert len(a._conn.mav.commands) == 0

    # IMU arrives -> first tick enters throttle_down, NOT armed yet.
    a._apply_message(FakeMsg("HIGHRES_IMU", xacc=0, yacc=0, zacc=-9.8,
                             xgyro=0, ygyro=0, zgyro=0, time_usec=1_000))
    a._maybe_auto_arm()
    assert a._arm_state == "throttle_down"
    assert a._race_armed is False
    assert len([c for c in a._conn.mav.commands if c[2] == 400]) == 0

    # After the throttle-down window, it arms exactly once.
    a._throttle_down_start -= 1.0   # simulate window elapsed
    a._maybe_auto_arm()
    a._maybe_auto_arm()  # repeated ticks must NOT re-arm
    arm_cmds = [c for c in a._conn.mav.commands if c[2] == 400]
    assert len(arm_cmds) == 1, "auto-arm must fire exactly once per race"
    assert a._race_armed is True and a._arm_state == "armed"


def test_auto_arm_waits_for_race_status_not_imu():
    """fly15-fly18 + sysid-probe regression: newer sim builds stream
    HIGHRES_IMU at the PRE-FLIGHT screen, so IMU presence must NOT trigger
    arming once an authoritative race-status message has been seen. The pilot
    armed at the menu on every one of those flights and started each race
    mid-sequence / into stale state."""
    a = _connected_adapter()
    a._arm_state = "idle"; a._race_armed = False

    # IMU streams (pre-flight screen) but race-status says NOT started (-1).
    a._apply_message(FakeMsg("HIGHRES_IMU", xacc=0, yacc=0, zacc=-9.8,
                             xgyro=0, ygyro=0, zgyro=0, time_usec=1_000))
    payload = struct.pack(RACE_STATUS_FMT,
                          ENCAPSULATED_RACE_STATUS_MSG_ID, 50, -1, -1, 0, 0)
    a._apply_message(FakeMsg("ENCAPSULATED_DATA", data=payload))
    a._maybe_auto_arm()
    assert a._arm_state == "idle", \
        "IMU at the pre-flight screen must not start the arming sequence"

    # Race actually starts (race_start_boot_time_ms >= 0) -> arming proceeds.
    payload = struct.pack(RACE_STATUS_FMT,
                          ENCAPSULATED_RACE_STATUS_MSG_ID, 50, 10, -1, 0, 0)
    a._apply_message(FakeMsg("ENCAPSULATED_DATA", data=payload))
    a._maybe_auto_arm()
    assert a._arm_state == "throttle_down"
    a._throttle_down_start -= 1.0
    a._maybe_auto_arm()
    assert a._race_armed is True

    # Race ends (back to -1) -> state machine resets for the next race.
    payload = struct.pack(RACE_STATUS_FMT,
                          ENCAPSULATED_RACE_STATUS_MSG_ID, 99, -1, -1, 0, 0)
    a._apply_message(FakeMsg("ENCAPSULATED_DATA", data=payload))
    a._maybe_auto_arm()
    assert a._arm_state == "idle" and a._race_armed is False


def test_send_forces_throttle_down_until_armed():
    a = _connected_adapter(control_mode="attitude")
    a._arm_state = "idle"; a._race_armed = False   # not armed yet
    # Not armed yet -> control output is forced to zero thrust.
    a.send_velocity_command(vx=2.0, vy=0.0, vz=-1.0)
    assert len(a._conn.mav.attitude_targets) == 1
    thrust = a._conn.mav.attitude_targets[0][8]
    assert thrust == 0.0, "throttle must be down before arming"


def test_auto_arm_rearms_after_race_ends():
    import time as _t
    a = _connected_adapter()
    a._arm_state = "idle"; a._race_armed = False   # start pre-arm
    a._apply_message(FakeMsg("HIGHRES_IMU", xacc=0, yacc=0, zacc=-9.8,
                             xgyro=0, ygyro=0, zgyro=0, time_usec=1_000))
    a._maybe_auto_arm()
    a._throttle_down_start -= 1.0
    a._maybe_auto_arm()
    assert a._race_armed is True

    # Telemetry goes stale (race ended) -> state resets so next race re-arms.
    a._last_imu_wall = _t.time() - 5.0
    a._maybe_auto_arm()
    assert a._race_armed is False and a._arm_state == "idle"


def test_auto_arm_disabled():
    a = MavlinkSimInterface(auto_arm=False)
    a._conn = FakeConn()
    a._connected = True
    a._apply_message(FakeMsg("HIGHRES_IMU", xacc=0, yacc=0, zacc=-9.8,
                             xgyro=0, ygyro=0, zgyro=0, time_usec=1_000))
    a._maybe_auto_arm()
    assert len(a._conn.mav.commands) == 0


def test_arm_and_reset_emit_commands():
    a = _connected_adapter()
    a.arm()
    a.reset()
    cmds = a._conn.mav.commands
    assert len(cmds) == 2
    from drone_sim.mavlink_adapter import (MAV_CMD_COMPONENT_ARM_DISARM,
                                           MAVLINK_CMD_SIM_RESET)
    arm_cmd = cmds[0]
    assert arm_cmd[2] == MAV_CMD_COMPONENT_ARM_DISARM
    assert arm_cmd[4] == 1            # arm param
    reset_cmd = cmds[1]
    assert reset_cmd[2] == MAVLINK_CMD_SIM_RESET


# ---------------------------------------------------------------------------
# Factory wiring
# ---------------------------------------------------------------------------

def test_make_interface_mavlink_returns_adapter():
    from drone_sim.interface import make_interface
    sim = make_interface(mode="mavlink")
    assert isinstance(sim, MavlinkSimInterface)


def test_make_interface_mavlink_ignores_force_ned(capsys):
    from drone_sim.interface import make_interface
    sim = make_interface(mode="mavlink", force_ned=True)
    assert isinstance(sim, MavlinkSimInterface)  # not wrapped in NEDAdapter
    out = capsys.readouterr().out
    assert "force-ned ignored" in out


def test_config_has_mavlink_ports():
    from config.loader import cfg
    assert cfg.sim.mavlink_port == 14550
    assert cfg.sim.vision_port == 5600
