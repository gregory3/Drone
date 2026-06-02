"""
drone_sim.mavlink_adapter
Adapter that bridges the official AI Grand Prix (AI-GP) flight simulator
into our SimInterface contract so the existing autonomy stack
(perception / state / control / planning / telemetry) runs unchanged.

The AI-GP sim (``FlightSim.exe``, shipped in ``PLTR Drone Package/AIGP_*.zip``)
is a MAVLink server. The interface is fully specified by the official
``PyAIPilotExample`` template:

  * MAVLink (telemetry in + control out) over UDP, default 127.0.0.1:14550.
  * Camera frames as chunked JPEG over a separate UDP socket, default :5600.

This adapter owns three background threads — exactly mirroring the official
example — and folds their output into one thread-safe snapshot:

  * MAVLink receive loop : ATTITUDE / LOCAL_POSITION_NED / ODOMETRY /
                           HIGHRES_IMU / ACTUATOR_OUTPUT_STATUS / COLLISION /
                           ENCAPSULATED_DATA (race status + track gates).
  * Vision receive loop  : reassembles JPEG chunks -> BGR np.ndarray.
  * Timesync loop        : 10 Hz TIMESYNC requests (matches example).

Control mode: VELOCITY SETPOINTS.
``send_velocity_command()`` maps 1:1 onto SET_POSITION_TARGET_LOCAL_NED with a
velocity + yaw-rate type mask, in the sim's native NED frame. Our PIDController
already outputs (vx, vy, vz, yaw_rate), so no RC translation layer is needed
(unlike the Elodin Betaflight path).

Frame note: the AI-GP sim is NED end-to-end (gates, odometry, velocity
setpoints). This adapter stays in NED internally; do NOT wrap it in NEDAdapter.

Rules note (see PLTR Drone Package/README.md): the real competition provides
"no GPS or absolute coordinate data". ``get_ground_truth()`` (odometry) and
``get_course()`` (absolute gate coordinates) are parsed and exposed for
bring-up/validation, but Round-1 only. The autonomy loop gates both behind
mock mode, so by default an mavlink-mode run flies vision-only — competition
legal. Opt into the cheat channels explicitly for debugging.

Testability: every wire-format parse is a pure function, and message handling
goes through ``_apply_message()`` which accepts any object exposing the right
attributes. The live ``pymavlink``/socket dependencies are confined to
``connect()`` and the loop bodies, so the parsing/command logic is unit-tested
without the SDK or a running sim.
"""

from __future__ import annotations

import math
import socket
import struct
import threading
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from config.loader import cfg
from .interface import DroneState, IMUReading, Observation, SimInterface


# ---------------------------------------------------------------------------
# Wire-format constants (from the official PyAIPilotExample). These are fixed
# by the MAVLink spec / the AI-GP encapsulated-message contract and are defined
# locally so parsing/command building needs no pymavlink import.
# ---------------------------------------------------------------------------

# Encapsulated payload discriminators (first byte of ENCAPSULATED_DATA.data).
ENCAPSULATED_RACE_STATUS_MSG_ID = 1
ENCAPSULATED_TRACK_INFO_MSG_ID = 2

# struct layouts (little-endian) — must match the sim byte-for-byte.
RACE_STATUS_FMT = "<BQqqIq"          # 37 bytes
TRACK_PACKET_HEADER_FMT = "<BH"       # data_type, transfer_id (then 3-byte skip)
TRACK_DATA_NUM_GATES_FMT = "<H"
TRACK_GATE_FMT = "<Hfffffffff"        # 38 bytes per gate
TRACK_GATE_SZ = struct.calcsize(TRACK_GATE_FMT)
VISION_HEADER_FMT = "<IHHIIQ"          # 24 bytes
VISION_HEADER_SZ = struct.calcsize(VISION_HEADER_FMT)

# Custom command id (repurposed) used by the sim to reset a run.
MAVLINK_CMD_SIM_RESET = 31000

# MAVLink standard constants we depend on (avoids importing mavutil for these).
MAV_FRAME_LOCAL_NED = 1
MAV_CMD_COMPONENT_ARM_DISARM = 400

# SET_ATTITUDE_TARGET type mask: ignore the attitude quaternion and use the
# body rates + thrust instead (ACRO/rate control). MAVLink standard value.
ATT_TYPEMASK_ATTITUDE_IGNORE = 128

# SET_POSITION_TARGET_LOCAL_NED type-mask bits (MAVLink standard).
_POS_X_IGNORE = 1
_POS_Y_IGNORE = 2
_POS_Z_IGNORE = 4
_POS_VX_IGNORE = 8
_POS_VY_IGNORE = 16
_POS_VZ_IGNORE = 32
_POS_AX_IGNORE = 64
_POS_AY_IGNORE = 128
_POS_AZ_IGNORE = 256
_POS_FORCE = 512
_POS_YAW_IGNORE = 1024
_POS_YAW_RATE_IGNORE = 2048

# Use velocity (vx, vy, vz) + yaw_rate; ignore position, acceleration and
# absolute yaw. This lets the sim's inner loop stabilise to our setpoints.
VELOCITY_YAWRATE_MASK = (
    _POS_X_IGNORE | _POS_Y_IGNORE | _POS_Z_IGNORE
    | _POS_AX_IGNORE | _POS_AY_IGNORE | _POS_AZ_IGNORE
    | _POS_YAW_IGNORE
)

# Collision ids (from the example).
COLLISION_ID_GATE = 1001
COLLISION_ID_ENVIRONMENT = 1002


# ---------------------------------------------------------------------------
# Parsed data types
# ---------------------------------------------------------------------------

@dataclass
class GateInfo:
    """A single race gate, as broadcast in the track-data packet (NED)."""
    gate_id: int
    position_ned: np.ndarray              # (3,) x, y, z metres
    orientation_ned: np.ndarray           # (4,) quaternion w, x, y, z
    width_m: float
    height_m: float


@dataclass
class RaceStatus:
    """Decoded ENCAPSULATED race-status message."""
    sim_boot_time_ms: int
    race_start_boot_time_ms: int          # < 0 / None-ish until race starts
    race_finish_time_ns: int              # < 0 while race ongoing
    active_gate_index: int
    last_gate_race_time_s: int

    @property
    def race_started(self) -> bool:
        return self.race_start_boot_time_ms is not None and self.race_start_boot_time_ms >= 0

    @property
    def race_finished(self) -> bool:
        return self.race_finish_time_ns is not None and self.race_finish_time_ns >= 0


@dataclass
class CollisionEvent:
    collision_id: int
    threat_level: int                     # 1-2, 2 = higher impact
    impulse_kg_m_s: float                 # impulse magnitude (see example)
    received_t: float                     # local wall-clock time of receipt

    @property
    def is_gate(self) -> bool:
        return self.collision_id == COLLISION_ID_GATE

    @property
    def is_environment(self) -> bool:
        return self.collision_id == COLLISION_ID_ENVIRONMENT


# ---------------------------------------------------------------------------
# Pure wire-format parsers (no I/O, no SDK) — directly unit-testable
# ---------------------------------------------------------------------------

def parse_race_status(raw_payload: bytes) -> RaceStatus:
    """Decode an ENCAPSULATED race-status payload (leading byte = msg id)."""
    (_data_type, sim_boot_time_ms, race_start_boot_time_ms, race_finish_time_ns,
     active_gate_index, last_gate_race_time) = struct.unpack_from(
        RACE_STATUS_FMT, raw_payload)
    return RaceStatus(
        sim_boot_time_ms=sim_boot_time_ms,
        race_start_boot_time_ms=race_start_boot_time_ms,
        race_finish_time_ns=race_finish_time_ns,
        active_gate_index=active_gate_index,
        last_gate_race_time_s=last_gate_race_time,
    )


def parse_track_data(payload: bytes) -> List[GateInfo]:
    """Decode a fully-reassembled track-data payload into gate list.

    Layout: ``<H`` num_gates, then ``num_gates`` x 38-byte gate records.
    """
    (num_gates,) = struct.unpack_from(TRACK_DATA_NUM_GATES_FMT, payload)
    payload = payload[2:]
    gates: List[GateInfo] = []
    for _ in range(num_gates):
        (gate_id, px, py, pz, ow, ox, oy, oz, width, height) = struct.unpack_from(
            TRACK_GATE_FMT, payload)
        payload = payload[TRACK_GATE_SZ:]
        gates.append(GateInfo(
            gate_id=gate_id,
            position_ned=np.array([px, py, pz], dtype=float),
            orientation_ned=np.array([ow, ox, oy, oz], dtype=float),
            width_m=float(width),
            height_m=float(height),
        ))
    return gates


def quaternion_to_euler_deg(qw: float, qx: float, qy: float, qz: float
                            ) -> Tuple[float, float, float]:
    """Convert a (w, x, y, z) quaternion to (roll, pitch, yaw) in degrees."""
    roll = math.atan2(2.0 * (qw * qx + qy * qz),
                      1.0 - 2.0 * (qx * qx + qy * qy))
    sinp = 2.0 * (qw * qy - qz * qx)
    sinp = max(-1.0, min(1.0, sinp))
    pitch = math.asin(sinp)
    yaw = math.atan2(2.0 * (qw * qz + qx * qy),
                     1.0 - 2.0 * (qy * qy + qz * qz))
    return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)


# ---------------------------------------------------------------------------
# Vision JPEG reassembly (no SDK; cv2 imported lazily on decode)
# ---------------------------------------------------------------------------

class VisionFrameAssembler:
    """Reassembles chunked JPEG vision packets into decoded BGR frames.

    Mirrors the official example's chunking protocol. Keeps a small ring of
    in-flight frames and drops any that never complete to bound memory.
    """

    def __init__(self, max_inflight: int = 8) -> None:
        self._frames: dict[int, dict] = {}
        self._max_inflight = max_inflight

    def add_packet(self, packet: bytes) -> Optional[Tuple[int, np.ndarray, int]]:
        """Feed one raw UDP packet. Returns (frame_id, bgr_image, sim_time_ns)
        when a frame completes and decodes, else None.
        """
        if len(packet) < VISION_HEADER_SZ:
            return None
        header = packet[:VISION_HEADER_SZ]
        payload = packet[VISION_HEADER_SZ:]
        (frame_id, chunk_id, total_chunks, jpeg_size,
         payload_size, sim_time_ns) = struct.unpack(VISION_HEADER_FMT, header)

        if total_chunks == 0:
            return None

        entry = self._frames.get(frame_id)
        if entry is None:
            # Bound memory: if we are tracking too many partial frames, drop
            # the oldest (lowest frame_id) before starting a new one.
            if len(self._frames) >= self._max_inflight:
                oldest = min(self._frames)
                del self._frames[oldest]
            entry = {"chunks": {}, "total": total_chunks, "time": sim_time_ns}
            self._frames[frame_id] = entry

        entry["chunks"][chunk_id] = payload

        if len(entry["chunks"]) < entry["total"]:
            return None

        # All chunks present — assemble in order.
        jpeg_bytes = bytearray()
        for i in range(entry["total"]):
            chunk = entry["chunks"].get(i)
            if chunk is None:
                # Incomplete despite count match (duplicate chunk ids); wait.
                return None
            jpeg_bytes.extend(chunk)

        del self._frames[frame_id]

        import cv2
        img_array = np.frombuffer(bytes(jpeg_bytes), dtype=np.uint8)
        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if image is None:
            return None
        return frame_id, image, sim_time_ns


# ---------------------------------------------------------------------------
# Thread-safe latest-state snapshot
# ---------------------------------------------------------------------------

@dataclass
class _Snapshot:
    # IMU (body frame, m/s^2 and rad/s) from HIGHRES_IMU
    accel: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    gyro: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    imu_t: float = 0.0

    # Pose / velocity (NED) — from ODOMETRY (preferred) or LOCAL_POSITION_NED
    pos_ned: Optional[np.ndarray] = None
    vel_ned: Optional[np.ndarray] = None
    quat_wxyz: Optional[np.ndarray] = None        # from ODOMETRY
    att_deg: Optional[np.ndarray] = None          # from ATTITUDE (roll,pitch,yaw)

    rpm: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])

    # Vision
    image: Optional[np.ndarray] = None
    image_t: float = 0.0

    # Race / track
    gates: Optional[List[GateInfo]] = None
    race_status: Optional[RaceStatus] = None

    # Collisions
    last_collision: Optional[CollisionEvent] = None
    last_env_collision_t: float = -1.0
    collision_count: int = 0

    armed: bool = False


# ---------------------------------------------------------------------------
# The adapter
# ---------------------------------------------------------------------------

class MavlinkSimInterface(SimInterface):
    """SimInterface implementation for the official AI-GP MAVLink simulator."""

    # How long get_observation will wait for the first camera frame / IMU
    # sample after connect before falling back to a synthetic black frame.
    _FIRST_FRAME_TIMEOUT_S = 2.0

    def __init__(self,
                 host: Optional[str] = None,
                 mavlink_port: Optional[int] = None,
                 vision_port: Optional[int] = None,
                 vision_bind_ip: str = "0.0.0.0",
                 heartbeat_timeout_s: Optional[float] = None,
                 timesync_hz: float = 10.0,
                 env_collision_crash_window_s: float = 1.0,
                 control_mode: Optional[str] = None,
                 auto_arm: bool = True) -> None:
        s = getattr(cfg, "sim", None)
        # Control backend: the official AI-GP sim is ACRO (rate) mode, so the
        # default routes velocity commands through a body-rate + thrust
        # translator. "velocity" keeps the raw SET_POSITION_TARGET path for any
        # future GUIDED-mode sim.
        self._control_mode = (control_mode if control_mode is not None
                              else getattr(cfg.control, "mavlink_control_mode",
                                           "attitude")).lower()
        self._vel_to_att = None  # lazily built in connect() (attitude mode)
        self._host = host if host is not None else getattr(s, "host", "127.0.0.1")
        self._mavlink_port = int(mavlink_port if mavlink_port is not None
                                 else getattr(s, "mavlink_port", 14550))
        self._vision_port = int(vision_port if vision_port is not None
                                else getattr(s, "vision_port", 5600))
        self._vision_bind_ip = vision_bind_ip
        self._heartbeat_timeout_s = float(
            heartbeat_timeout_s if heartbeat_timeout_s is not None
            else getattr(s, "connect_timeout_s", 10.0))
        self._timesync_hz = float(timesync_hz)
        self._env_collision_crash_window_s = float(env_collision_crash_window_s)

        self._conn = None                     # pymavlink connection (set in connect)
        self._vision_assembler = VisionFrameAssembler()

        self._snap = _Snapshot()
        self._lock = threading.Lock()
        # pymavlink's mav.*_send is NOT thread-safe. Control (main thread),
        # arming (rx thread) and timesync (timesync thread) all transmit, so
        # every send must hold this lock or the outgoing byte stream corrupts
        # and commands (e.g. arm) are silently dropped by the sim.
        self._send_lock = threading.Lock()

        self._running = False
        self._connected = False
        self._threads: List[threading.Thread] = []
        self._vision_sock: Optional[socket.socket] = None

        # Auto-arm: the sim only accepts arming AFTER the race goes active, and
        # re-sending arm every tick breaks the FC. So we arm exactly once when
        # the race starts (detected via HIGHRES_IMU streaming, which only flows
        # during an active race) and re-arm if a new race begins.
        self._auto_arm = auto_arm
        self._race_armed = False
        self._last_imu_wall = -1.0
        # Arming state machine. The sim refuses to arm while throttle is up
        # ("THROTTLE DOWN please"), so on race start we hold zero thrust for a
        # moment, then arm, then allow real control.
        self._arm_state = "idle"          # idle -> throttle_down -> armed
        self._throttle_down_start = 0.0
        self._arm_throttle_down_s = 0.6

        # Track-data reassembly (mirrors the example's DATA_TRANSMISSION_HANDSHAKE
        # + ENCAPSULATED_DATA chunking).
        self._track_chunks: dict[int, dict[int, bytes]] = {}
        self._expected_track_chunks: dict[int, int] = {}

        # Camera frame size for the synthetic fallback frame.
        p = cfg.perception
        self._frame_h = int(p.image_height)
        self._frame_w = int(p.image_width)

    # ------------------------------------------------------------------ runtime
    def connect(self) -> None:
        try:
            from pymavlink import mavutil
        except Exception as exc:  # pragma: no cover - depends on environment
            raise NotImplementedError(
                "MavlinkSimInterface requires the `pymavlink` package. "
                "Install it with `pip install pymavlink`. "
                f"Import error: {exc}"
            )

        self._conn = mavutil.mavlink_connection(
            "udpin:%s:%d" % (self._host, self._mavlink_port))
        print(f"[MavlinkSim] Waiting for heartbeat on "
              f"{self._host}:{self._mavlink_port} ...", flush=True)
        hb = self._conn.wait_heartbeat(timeout=self._heartbeat_timeout_s)
        if hb is None:
            raise TimeoutError(
                f"No MAVLink heartbeat within {self._heartbeat_timeout_s:.0f}s. "
                "Is FlightSim.exe running and the virtual qualifier loaded?")
        print(f"[MavlinkSim] Connected to system {self._conn.target_system}",
              flush=True)

        if self._control_mode == "attitude":
            from control.velocity_to_attitude import VelocityToAttitude
            self._vel_to_att = VelocityToAttitude()
            print("[MavlinkSim] Control mode: attitude (body rates + thrust, "
                  "ACRO-compatible)", flush=True)
        else:
            print("[MavlinkSim] Control mode: velocity (SET_POSITION_TARGET)",
                  flush=True)

        self._running = True
        self._connected = True
        self._start_thread(self._mavlink_loop, "mavlink-rx")
        self._start_thread(self._vision_loop, "vision-rx")
        if self._timesync_hz > 0:
            self._start_thread(self._timesync_loop, "timesync")

    def _start_thread(self, target, name: str) -> None:
        t = threading.Thread(target=target, name=name, daemon=True)
        t.start()
        self._threads.append(t)

    def disconnect(self) -> None:
        self._running = False
        self._connected = False
        sock = self._vision_sock
        if sock is not None:
            try:
                sock.close()
            except Exception:
                pass
        for t in self._threads:
            t.join(timeout=1.0)
        self._threads.clear()
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
        print("[MavlinkSim] Disconnected.", flush=True)

    def reset(self) -> None:
        """Reset the simulated run via the sim's custom reset command."""
        if self._conn is None:
            return
        with self._send_lock:
            self._conn.mav.command_long_send(
                self._conn.target_system,
                self._conn.target_component,
                MAVLINK_CMD_SIM_RESET,
                0,                       # confirmation
                0, 0, 0, 0, 0, 0, 0,
            )

    def arm(self) -> None:
        if self._conn is None:
            raise RuntimeError("MavlinkSimInterface not connected")
        with self._send_lock:
            self._conn.mav.command_long_send(
                self._conn.target_system,
                self._conn.target_component,
                MAV_CMD_COMPONENT_ARM_DISARM,
                0,
                1,                       # 1 = arm
                0, 0, 0, 0, 0, 0,
            )

    # ------------------------------------------------------------------ API
    def get_observation(self) -> Observation:
        if not self._connected:
            raise RuntimeError("MavlinkSimInterface not connected")

        with self._lock:
            snap = self._snap
            image = None if snap.image is None else snap.image
            accel = snap.accel
            gyro = snap.gyro
            imu_t = snap.imu_t
            rpm = list(snap.rpm)
            image_t = snap.image_t
            last_env_t = snap.last_env_collision_t

        if image is None:
            # No camera frame yet — synthesize a black BGR frame so the
            # perception pipeline runs without crashing (it returns no
            # detections). Mirrors the Elodin adapter's fallback.
            image = np.zeros((self._frame_h, self._frame_w, 3), dtype=np.uint8)

        timestamp = image_t or imu_t or time.time()
        imu = IMUReading(accel=accel, gyro=gyro, timestamp=imu_t or timestamp)

        is_crashed = (last_env_t >= 0.0 and
                      (time.time() - last_env_t) < self._env_collision_crash_window_s)

        return Observation(image=image, imu=imu, rpm=rpm, timestamp=timestamp,
                           is_crashed=is_crashed)

    def send_velocity_command(self, vx: float, vy: float, vz: float,
                              yaw_rate: float = 0.0) -> None:
        """Command an NED velocity (+ yaw rate) to the sim.

        ``vx, vy, vz`` are NED m/s. ``yaw_rate`` is **deg/s** (the convention of
        our ControlOutput).

        In the default "attitude" mode the request is translated to body rates +
        collective thrust and sent via SET_ATTITUDE_TARGET (the only path the
        ACRO-mode sim honours). In "velocity" mode it is sent raw via
        SET_POSITION_TARGET_LOCAL_NED.
        """
        if self._conn is None:
            raise RuntimeError("MavlinkSimInterface not connected")

        # Until the arming sequence completes, hold throttle down so the sim
        # will actually arm (it rejects arming with throttle up). This overrides
        # whatever the controller asks for during the brief throttle-down window.
        if self._auto_arm and self._arm_state != "armed":
            self._send_throttle_down()
            return

        if self._control_mode == "attitude" and self._vel_to_att is not None:
            self._send_attitude_command(vx, vy, vz, yaw_rate)
        else:
            self._send_velocity_setpoint(vx, vy, vz, yaw_rate)

    def _send_throttle_down(self) -> None:
        """Send a zero-thrust, zero-rate attitude command (throttle down)."""
        time_boot_ms = int((time.time() * 1000)) & 0xFFFFFFFF
        with self._send_lock:
            self._conn.mav.set_attitude_target_send(
                time_boot_ms,
                self._conn.target_system,
                self._conn.target_component,
                ATT_TYPEMASK_ATTITUDE_IGNORE,
                [1.0, 0.0, 0.0, 0.0],
                0.0, 0.0, 0.0, 0.0,                  # zero rates, zero thrust
            )

    def _send_velocity_setpoint(self, vx: float, vy: float, vz: float,
                                yaw_rate: float) -> None:
        yaw_rate_rad = math.radians(float(yaw_rate))
        time_boot_ms = int((time.time() * 1000)) & 0xFFFFFFFF
        with self._send_lock:
            self._conn.mav.set_position_target_local_ned_send(
                time_boot_ms,
                self._conn.target_system,
                self._conn.target_component,
                MAV_FRAME_LOCAL_NED,
                VELOCITY_YAWRATE_MASK,
                0.0, 0.0, 0.0,                       # position (ignored)
                float(vx), float(vy), float(vz),     # velocity NED
                0.0, 0.0, 0.0,                       # acceleration (ignored)
                0.0,                                  # yaw (ignored)
                yaw_rate_rad,                         # yaw rate
            )

    def _send_attitude_command(self, vx: float, vy: float, vz: float,
                               yaw_rate: float) -> None:
        # Pull the live attitude + vertical velocity for the inner angle loop.
        roll, pitch, yaw, vz_actual = self._current_attitude_and_vz()
        cmd = self._vel_to_att.convert(
            vx=vx, vy=vy, vz=vz, yaw_rate_dps=yaw_rate,
            yaw_rad=yaw, roll_rad=roll, pitch_rad=pitch,
            vertical_velocity_mps=vz_actual,
        )
        time_boot_ms = int((time.time() * 1000)) & 0xFFFFFFFF
        with self._send_lock:
            self._conn.mav.set_attitude_target_send(
                time_boot_ms,
                self._conn.target_system,
                self._conn.target_component,
                ATT_TYPEMASK_ATTITUDE_IGNORE,
                [1.0, 0.0, 0.0, 0.0],                # quaternion (ignored)
                float(cmd.roll_rate),
                float(cmd.pitch_rate),
                float(cmd.yaw_rate),
                float(cmd.thrust),
            )

    def _current_attitude_and_vz(self):
        """Return (roll_rad, pitch_rad, yaw_rad, vz_ned) from the snapshot.

        Prefers ODOMETRY quaternion for attitude; falls back to ATTITUDE
        (stored in degrees). vz is NED (positive = descending) or None.
        """
        with self._lock:
            att = None if self._snap.att_deg is None else self._snap.att_deg.copy()
            quat = None if self._snap.quat_wxyz is None else self._snap.quat_wxyz.copy()
            vel = None if self._snap.vel_ned is None else self._snap.vel_ned.copy()

        if quat is not None:
            r, p, y = quaternion_to_euler_deg(quat[0], quat[1], quat[2], quat[3])
        elif att is not None:
            r, p, y = att[0], att[1], att[2]
        else:
            r = p = y = 0.0
        vz = float(vel[2]) if vel is not None else None
        return math.radians(r), math.radians(p), math.radians(y), vz

    def get_ground_truth(self) -> Optional[DroneState]:
        """NED pose/velocity from ODOMETRY/LOCAL_POSITION_NED.

        Round-1 dev aid only: the real competition provides no absolute
        coordinates. The autonomy loop does not use this in mavlink mode.
        """
        with self._lock:
            snap = self._snap
            pos = None if snap.pos_ned is None else snap.pos_ned.copy()
            vel = None if snap.vel_ned is None else snap.vel_ned.copy()
            quat = None if snap.quat_wxyz is None else snap.quat_wxyz.copy()
            att = None if snap.att_deg is None else snap.att_deg.copy()

        if pos is None and vel is None and quat is None and att is None:
            return None

        if att is None and quat is not None:
            roll, pitch, yaw = quaternion_to_euler_deg(
                quat[0], quat[1], quat[2], quat[3])
            att = np.array([roll, pitch, yaw], dtype=float)

        return DroneState(
            pos=pos if pos is not None else np.zeros(3),
            vel=vel if vel is not None else np.zeros(3),
            att_deg=att if att is not None else np.zeros(3),
        )

    def get_course(self) -> Optional[List[np.ndarray]]:
        """Gate centre positions (NED), in gate order.

        Round-1 dev aid only (absolute coordinates). Returns None until the
        sim has broadcast the track-data packet.
        """
        with self._lock:
            gates = self._snap.gates
        if not gates:
            return None
        ordered = sorted(gates, key=lambda g: g.gate_id)
        return [g.position_ned.copy() for g in ordered]

    # -------------------------------------------------- introspection helpers
    def is_armed(self) -> bool:
        """True once the race is active and the drone has been armed.

        Used by the autonomy loop to hold its flight logic (takeoff, control,
        camera watchdog) until the race actually starts. When auto-arm is off,
        always report ready so manual flows aren't blocked.
        """
        if not self._auto_arm:
            return True
        return self._race_armed

    def get_gates(self) -> Optional[List[GateInfo]]:
        with self._lock:
            gates = self._snap.gates
        return None if gates is None else list(gates)

    def get_race_status(self) -> Optional[RaceStatus]:
        with self._lock:
            return self._snap.race_status

    def get_last_collision(self) -> Optional[CollisionEvent]:
        with self._lock:
            return self._snap.last_collision

    # ------------------------------------------------------------------ loops
    def _mavlink_loop(self) -> None:
        conn = self._conn
        while self._running:
            try:
                msg = conn.recv_match(blocking=False)
            except ConnectionResetError:
                print("[MavlinkSim] MAVLink connection reset.", flush=True)
                return
            except Exception:
                if not self._running:
                    return
                time.sleep(0.001)
                continue

            if msg is None:
                self._maybe_auto_arm()
                time.sleep(0.001)
                continue
            if msg.get_type() == "BAD_DATA":
                continue
            try:
                self._apply_message(msg)
            except Exception:
                # A single malformed message must never kill the rx thread.
                continue
            self._maybe_auto_arm()

    def _maybe_auto_arm(self) -> None:
        """Arming state machine, driven each rx loop iteration.

        Race-active is inferred from a live HIGHRES_IMU stream (only present
        during an active race). The sim refuses to arm with throttle up, so on
        race start we enter ``throttle_down`` (send_velocity_command forces zero
        thrust during this window), then arm exactly once. Arming is sent once
        per race — repeated arm commands leave the FC non-responsive.
        """
        if not self._auto_arm or self._conn is None:
            return
        now = time.time()
        active = self._last_imu_wall > 0 and (now - self._last_imu_wall) < 0.5

        if not active:
            # Race ended / not started → reset so the next race re-arms.
            if self._arm_state != "idle":
                self._arm_state = "idle"
                self._race_armed = False
            return

        if self._arm_state == "idle":
            self._arm_state = "throttle_down"
            self._throttle_down_start = now
            print("[MavlinkSim] Race active — throttle down before arm...",
                  flush=True)
        elif self._arm_state == "throttle_down":
            # Drive throttle-down from here (the autonomy loop is holding until
            # armed, so it isn't sending), then arm once the window elapses.
            self._send_throttle_down()
            if now - self._throttle_down_start >= self._arm_throttle_down_s:
                self.arm()
                self._arm_state = "armed"
                self._race_armed = True
                print("[MavlinkSim] Armed (throttle down satisfied).", flush=True)

    def _vision_loop(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((self._vision_bind_ip, self._vision_port))
        sock.settimeout(0.5)
        self._vision_sock = sock
        print(f"[MavlinkSim] Listening for camera frames on "
              f"{self._vision_bind_ip}:{self._vision_port} ...", flush=True)

        while self._running:
            try:
                packet, _addr = sock.recvfrom(65536)
            except socket.timeout:
                continue
            except OSError:
                # Socket closed during disconnect.
                return
            try:
                result = self._vision_assembler.add_packet(packet)
            except Exception:
                continue
            if result is None:
                continue
            _frame_id, image, sim_time_ns = result
            with self._lock:
                self._snap.image = image
                self._snap.image_t = sim_time_ns / 1e9 if sim_time_ns else time.time()

    def _timesync_loop(self) -> None:
        conn = self._conn
        period = 1.0 / self._timesync_hz
        while self._running:
            try:
                with self._send_lock:
                    conn.mav.timesync_send(int(time.time_ns()), 0)
            except Exception:
                pass
            time.sleep(period)

    # ---------------------------------------------------- message dispatch
    def _apply_message(self, msg) -> None:
        """Fold one MAVLink message into the snapshot.

        ``msg`` only needs ``get_type()`` plus the per-type attributes, so this
        is exercised in tests with lightweight fakes (no pymavlink required).
        """
        mtype = msg.get_type()

        if mtype == "HIGHRES_IMU":
            self._last_imu_wall = time.time()   # race-active heartbeat for auto-arm
            with self._lock:
                self._snap.accel = (float(msg.xacc), float(msg.yacc), float(msg.zacc))
                self._snap.gyro = (float(msg.xgyro), float(msg.ygyro), float(msg.zgyro))
                self._snap.imu_t = float(msg.time_usec) / 1e6

        elif mtype == "ATTITUDE":
            roll = math.degrees(float(msg.roll))
            pitch = math.degrees(float(msg.pitch))
            yaw = math.degrees(float(msg.yaw))
            with self._lock:
                self._snap.att_deg = np.array([roll, pitch, yaw], dtype=float)

        elif mtype == "LOCAL_POSITION_NED":
            pos = np.array([float(msg.x), float(msg.y), float(msg.z)], dtype=float)
            vel = np.array([float(msg.vx), float(msg.vy), float(msg.vz)], dtype=float)
            with self._lock:
                # ODOMETRY is preferred; only fill if odometry hasn't set these.
                if self._snap.pos_ned is None or self._snap.quat_wxyz is None:
                    self._snap.pos_ned = pos
                    self._snap.vel_ned = vel

        elif mtype == "ODOMETRY":
            pos = np.array([float(msg.x), float(msg.y), float(msg.z)], dtype=float)
            vel = np.array([float(msg.vx), float(msg.vy), float(msg.vz)], dtype=float)
            # MAVLink ODOMETRY.q is [w, x, y, z].
            q = msg.q
            quat = np.array([float(q[0]), float(q[1]), float(q[2]), float(q[3])],
                            dtype=float)
            with self._lock:
                self._snap.pos_ned = pos
                self._snap.vel_ned = vel
                self._snap.quat_wxyz = quat

        elif mtype == "ACTUATOR_OUTPUT_STATUS":
            act = msg.actuator
            with self._lock:
                self._snap.rpm = [float(act[0]), float(act[1]),
                                  float(act[2]), float(act[3])]

        elif mtype == "COLLISION":
            event = CollisionEvent(
                collision_id=int(msg.id),
                threat_level=int(msg.threat_level),
                impulse_kg_m_s=float(msg.horizontal_minimum_delta),
                received_t=time.time(),
            )
            with self._lock:
                self._snap.last_collision = event
                self._snap.collision_count += 1
                if event.is_environment:
                    self._snap.last_env_collision_t = event.received_t

        elif mtype == "HEARTBEAT":
            armed = bool(int(msg.base_mode) & 0x80)  # MAV_MODE_FLAG_SAFETY_ARMED
            with self._lock:
                self._snap.armed = armed

        elif mtype == "ENCAPSULATED_DATA":
            self._on_encapsulated_data(msg)

        elif mtype == "DATA_TRANSMISSION_HANDSHAKE":
            # Repurposed as a track-data transfer announcement.
            transfer_id = int(msg.width)
            self._track_chunks[transfer_id] = {}
            self._expected_track_chunks[transfer_id] = int(msg.packets)

    def _on_encapsulated_data(self, msg) -> None:
        raw_payload = bytes(msg.data)
        if not raw_payload:
            return
        data_type = raw_payload[0]
        if data_type == ENCAPSULATED_RACE_STATUS_MSG_ID:
            status = parse_race_status(raw_payload)
            with self._lock:
                self._snap.race_status = status
        elif data_type == ENCAPSULATED_TRACK_INFO_MSG_ID:
            self._on_track_chunk(msg, raw_payload)

    def _on_track_chunk(self, msg, raw_payload: bytes) -> None:
        # header: data_type (B), transfer_id (H); then 3-byte skip per example.
        _data_type, transfer_id = struct.unpack_from(TRACK_PACKET_HEADER_FMT,
                                                      raw_payload)
        if transfer_id not in self._expected_track_chunks:
            return
        chunk = raw_payload[3:]
        self._track_chunks[transfer_id][int(msg.seqnr)] = chunk
        if len(self._track_chunks[transfer_id]) != self._expected_track_chunks[transfer_id]:
            return
        full = bytearray()
        for i in range(len(self._track_chunks[transfer_id])):
            piece = self._track_chunks[transfer_id].get(i)
            if piece is None:
                return
            full.extend(piece)
        del self._track_chunks[transfer_id]
        del self._expected_track_chunks[transfer_id]
        gates = parse_track_data(bytes(full))
        with self._lock:
            self._snap.gates = gates
