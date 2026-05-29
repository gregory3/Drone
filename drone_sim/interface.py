"""
drone_sim.interface
Abstraction layer for the DCL simulator API.

Real SDK:
    When Anthropic/DCL release their Python package, implement
    RealSimInterface(SimInterface) and swap it in via config.

Mock (now):
    MockSimInterface runs a local physics sim so we can develop
    perception and control pipelines before the official SDK arrives.

Usage:
    from drone_sim.interface import make_interface
    sim = make_interface(mode="mock")   # or "real"
    sim.connect()

    obs = sim.get_observation()
    # obs.image     → np.ndarray (H, W, 3) BGR
    # obs.imu       → IMUReading
    # obs.rpm       → List[float] (4 motors)
    # obs.timestamp → float

    sim.send_velocity_command(vx=1.0, vy=0.0, vz=0.0, yaw_rate=0.0)
    sim.disconnect()
"""

from __future__ import annotations
import time
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Data types returned by the sim
# ---------------------------------------------------------------------------

@dataclass
class IMUReading:
    accel: Tuple[float, float, float]   # m/s² (x, y, z)
    gyro: Tuple[float, float, float]    # rad/s (x, y, z)
    timestamp: float


@dataclass
class Observation:
    image: np.ndarray                   # BGR, shape (H, W, 3)
    imu: IMUReading
    rpm: List[float]                    # motor RPMs (4 motors)
    timestamp: float
    # Optional extras the real module might provide
    battery_pct: float = 100.0
    is_crashed: bool = False


@dataclass
class DroneState:
    """Ground-truth state — only available in mock; never on real drone."""
    pos: np.ndarray = field(default_factory=lambda: np.zeros(3))     # m
    vel: np.ndarray = field(default_factory=lambda: np.zeros(3))     # m/s
    att_deg: np.ndarray = field(default_factory=lambda: np.zeros(3)) # roll, pitch, yaw


# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------

class SimInterface(ABC):

    @abstractmethod
    def connect(self) -> None: ...

    @abstractmethod
    def get_observation(self) -> Observation: ...

    @abstractmethod
    def send_velocity_command(self,
                               vx: float, vy: float, vz: float,
                               yaw_rate: float = 0.0) -> None: ...

    @abstractmethod
    def disconnect(self) -> None: ...

    # Optional: only meaningful in mock
    def get_ground_truth(self) -> Optional[DroneState]:
        return None

    def get_course(self) -> Optional[List[np.ndarray]]:
        return None

    def reset(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Mock sim — simple integrator with synthetic gate imagery
# ---------------------------------------------------------------------------

class MockSimInterface(SimInterface):
    """
    Local mock that simulates a drone flying through a gate course.
    Produces synthetic camera frames with coloured gate markers so
    the perception pipeline can be developed immediately.

    Physics: simple Euler integration, no aerodynamics.
    Gates   : defined as 3D waypoints; rendered as colored rectangles
              in the projected image frame.
    """

    # Gate positions in world frame (x, y, z) in meters
    # Laid out in a simple line for Phase 1 baseline testing
    DEFAULT_GATES = [
        np.array([5.0,  0.0, 1.5]),
        np.array([10.0, 0.0, 1.5]),
        np.array([15.0, 2.0, 1.5]),
        np.array([20.0, 0.0, 1.5]),
        np.array([25.0, 0.0, 1.5]),
    ]

    GATE_SIZE_M = 1.2   # square gate, side length

    def __init__(self,
                 image_size: Optional[Tuple[int, int]] = None,
                 gates: Optional[List[np.ndarray]] = None,
                 fov_deg: Optional[float] = None,
                 tilt_deg: Optional[float] = None) -> None:
        from config.loader import cfg as _cfg
        p = _cfg.perception
        if image_size is None:
            image_size = (p.image_width, p.image_height)
        if fov_deg is None:
            fov_deg = p.camera_fov_deg
        if tilt_deg is None:
            tilt_deg = getattr(p, "camera_tilt_deg", 0.0)

        self._W, self._H = image_size
        self._gates = gates or self.DEFAULT_GATES
        self._state = DroneState(
            pos=np.array([0.0, 0.0, 1.5]),
            vel=np.zeros(3),
            att_deg=np.zeros(3),
        )
        self._cmd_vel = np.zeros(3)
        self._cmd_yaw_rate = 0.0
        self._last_t = time.time()
        self._connected = False
        self._lock = threading.Lock()

        fov_rad = np.radians(fov_deg)
        self._fx = (self._W / 2) / np.tan(fov_rad / 2)
        self._fy = self._fx
        self._cx = self._W / 2
        self._cy = self._H / 2
        self._tilt_rad = np.radians(tilt_deg)

    # ------------------------------------------------------------------
    def connect(self) -> None:
        self._connected = True
        self._last_t = time.time()
        print(f"[MockSim] Connected. {len(self._gates)} gates loaded.")

    def disconnect(self) -> None:
        self._connected = False
        print("[MockSim] Disconnected.")

    # ------------------------------------------------------------------
    def send_velocity_command(self,
                               vx: float, vy: float, vz: float,
                               yaw_rate: float = 0.0) -> None:
        with self._lock:
            self._cmd_vel = np.array([vx, vy, vz])
            self._cmd_yaw_rate = yaw_rate

    # ------------------------------------------------------------------
    def get_observation(self) -> Observation:
        now = time.time()
        with self._lock:
            dt = now - self._last_t
            self._last_t = now

            # Euler integrate position
            self._state.vel = self._cmd_vel.copy()
            self._state.pos += self._state.vel * dt
            self._state.att_deg[2] += self._cmd_yaw_rate * dt

        # Synthesize image
        image = self._render_frame()

        # Synthesize IMU (add small noise)
        imu = IMUReading(
            accel=(0.0 + np.random.randn() * 0.02,
                   0.0 + np.random.randn() * 0.02,
                   -9.81 + np.random.randn() * 0.05),
            gyro=(np.radians(self._cmd_yaw_rate) + np.random.randn() * 0.001,
                  np.random.randn() * 0.001,
                  np.random.randn() * 0.001),
            timestamp=now,
        )

        rpm_base = np.linalg.norm(self._state.vel) * 1000 + 5000
        rpm = [rpm_base + np.random.randn() * 50 for _ in range(4)]

        return Observation(image=image, imu=imu, rpm=rpm, timestamp=now)

    # ------------------------------------------------------------------
    def get_ground_truth(self) -> DroneState:
        with self._lock:
            return DroneState(
                pos=self._state.pos.copy(),
                vel=self._state.vel.copy(),
                att_deg=self._state.att_deg.copy(),
            )

    def get_course(self) -> list[np.ndarray]:
        return self._gates

    def reset(self) -> None:
        with self._lock:
            self._state = DroneState(
                pos=np.array([0.0, 0.0, 1.5]),
                vel=np.zeros(3),
                att_deg=np.zeros(3),
            )
            self._cmd_vel = np.zeros(3)
            self._cmd_yaw_rate = 0.0
        print("[MockSim] Reset to start position.")

    # ------------------------------------------------------------------
    def _render_frame(self) -> np.ndarray:
        """
        Project gates into camera space and draw colored rectangles.
        Round 1 style: orange gates on a simple gray background.
        """
        frame = np.full((self._H, self._W, 3), (80, 80, 80), dtype=np.uint8)

        with self._lock:
            drone_pos = self._state.pos.copy()
            yaw = np.radians(self._state.att_deg[2])

        # Rotation matrix: world -> camera frame.
        # Camera convention: X=right, Y=down, Z=forward (into scene)
        # Step 1: yaw (around world Z) aligns drone heading with camera forward.
        # Step 2: apply +tilt about the camera X axis so the lens looks "up"
        #         relative to the body forward vector (matches Elodin rig spec).
        R_yaw = np.array([
            [-np.sin(yaw),  np.cos(yaw), 0],
            [           0,            0,-1],
            [ np.cos(yaw),  np.sin(yaw), 0],
        ])
        t = self._tilt_rad
        R_tilt = np.array([
            [1.0,         0.0,          0.0],
            [0.0,  np.cos(t),    np.sin(t)],
            [0.0, -np.sin(t),    np.cos(t)],
        ])
        R = R_tilt @ R_yaw

        for i, gate_world in enumerate(self._gates):
            rel = gate_world - drone_pos
            cam = R @ rel

            if cam[2] <= 0.3:   # behind or too close
                continue

            # Project gate corners
            corners_cam = []
            half = self.GATE_SIZE_M / 2
            offsets = [(-half, -half), (half, -half), (half, half), (-half, half)]
            for du, dv in offsets:
                # Gate lies in the XZ plane (vertical square)
                c = cam.copy()
                c[0] += du   # horizontal offset
                c[1] += dv   # vertical offset
                if c[2] <= 0:
                    continue
                u = int(self._fx * c[0] / c[2] + self._cx)
                v = int(self._fy * c[1] / c[2] + self._cy)
                corners_cam.append((u, v))

            if len(corners_cam) < 4:
                continue

            import cv2
            pts = np.array(corners_cam, dtype=np.int32)

            # Gate fill (semi-transparent orange for Round 1 highlighted style)
            overlay = frame.copy()
            cv2.fillPoly(overlay, [pts], (0, 120, 220))   # BGR orange
            cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

            # Gate border
            cv2.polylines(frame, [pts], isClosed=True,
                          color=(0, 165, 255), thickness=3)

            # Gate number label
            center_u = int(np.mean([p[0] for p in corners_cam]))
            center_v = int(np.mean([p[1] for p in corners_cam]))
            cv2.putText(frame, str(i + 1), (center_u - 8, center_v + 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return frame


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_interface(mode: str = "mock", **kwargs) -> SimInterface:
    force_ned = kwargs.pop("force_ned", False)
    if mode == "mock":
        base = MockSimInterface(**kwargs)
    elif mode == "real":
        base = RealSimInterface(**kwargs)
    elif mode == "elodin":
        base = ElodinSimInterface(**kwargs)
    else:
        raise ValueError(
            f"Unknown sim mode: {mode!r}. Use 'mock', 'real', or 'elodin'."
        )
    if force_ned:
        return NEDAdapter(base, underlying_frame="enu")
    return base


class RealSimInterface(SimInterface):
    """Placeholder stub for the real DCL simulator adapter."""

    def __init__(self, host: str = "127.0.0.1", port: int = 5760,
                 connect_timeout_s: float = 10.0) -> None:
        self.host = host
        self.port = port
        self.connect_timeout_s = connect_timeout_s
        self._connected = False

    def connect(self) -> None:
        raise NotImplementedError(
            "RealSimInterface is not implemented. "
            "Install the DCL simulator SDK and implement a real simulator adapter."
        )

    def get_observation(self) -> Observation:
        raise NotImplementedError(
            "RealSimInterface.get_observation() is not implemented."
        )

    def send_velocity_command(self,
                               vx: float, vy: float, vz: float,
                               yaw_rate: float = 0.0) -> None:
        raise NotImplementedError(
            "RealSimInterface.send_velocity_command() is not implemented."
        )

    def disconnect(self) -> None:
        self._connected = False

    def reset(self) -> None:
        pass


class ElodinSimInterface(SimInterface):
    """Adapter for the open-source Elodin AI Grand Prix practice rig.

    The Elodin rig runs Betaflight SITL + 6-DOF physics and asks the solver
    to implement a single callback:

        def autopilot(update: SensorUpdate) -> RCCommand

    This adapter inverts that callback so the rest of the stack can keep
    using the polling `get_observation()` / `send_velocity_command()` shape.
    Each call to `get_observation()` blocks until the next `SensorUpdate`
    arrives from the rig; `send_velocity_command()` is bridged through a
    `velocity_to_rc` translator (see `control.velocity_to_rc`) and the
    resulting `RCCommand` is returned the next time the rig invokes our
    autopilot.

    SDK availability:
      - The `elodin` PyPI package is macOS / glibc >= 2.35 Linux only.
        On Windows the rig must be run inside WSL. The adapter itself is
        portable; only `connect()` requires the runtime SDK.
    """

    def __init__(self, sim_main_module: Optional[str] = None,
                 control_hz: float = 50.0,
                 frame_timeout_s: float = 1.0,
                 synchronous: bool = False) -> None:
        """sim_main_module: when set, `connect()` launches a worker thread
        running the rig's `sim/main.py` (production path).
        synchronous: when True, skip the inter-thread queues entirely.
            push_sensor_update/get_observation work via `_last_update` and
            send_velocity_command/pop_command via `_last_rc`. Use this for
            solver-bridge mode where one callback invocation = one tick.
        """
        self._sim_main_module = sim_main_module
        self._control_hz = control_hz
        self._frame_timeout_s = frame_timeout_s
        self._synchronous = synchronous
        self._connected = False
        self._last_update: Optional["SensorUpdate"] = None  # type: ignore[name-defined]
        self._last_rc = None
        self._cmd_vel = np.zeros(3)
        self._cmd_yaw_rate = 0.0
        self._velocity_to_rc = None  # lazy-imported to avoid circulars
        self._update_queue = None
        self._command_queue = None
        self._sim_thread = None

    # ---------------------------------------------------------------- runtime
    def connect(self) -> None:
        try:
            import elodin  # type: ignore  # noqa: F401
        except Exception as exc:
            raise NotImplementedError(
                "ElodinSimInterface requires the `elodin` SDK package "
                "(macOS or glibc >= 2.35 Linux; on Windows install inside WSL). "
                f"Install it with `pip install -U elodin`. Import error: {exc}"
            )

        from control.velocity_to_rc import VelocityToRC
        self._velocity_to_rc = VelocityToRC()

        if not self._synchronous:
            import queue
            self._update_queue = queue.Queue(maxsize=8)
            self._command_queue = queue.Queue(maxsize=8)

        if self._sim_main_module is not None:
            self._launch_sim_thread(self._sim_main_module)

        self._connected = True

    def _launch_sim_thread(self, module_name: str) -> None:
        """Import the rig's `sim/main.py` in a worker thread and feed it the
        adapter's autopilot callback. The Elodin runtime calls the callback
        every physics tick; the callback drains a single (update, rc) pair
        through the queues exposed by `get_observation` /
        `send_velocity_command`.
        """
        import importlib
        import threading

        def _autopilot_callback(update):
            try:
                self._update_queue.put(update, timeout=self._frame_timeout_s)
            except Exception:
                pass
            try:
                rc = self._command_queue.get(timeout=self._frame_timeout_s)
            except Exception:
                from .competition_types import RCCommand
                rc = RCCommand()  # safe defaults (disarmed, throttle 1000)
            return rc

        def _runner():
            mod = importlib.import_module(module_name)
            if hasattr(mod, "run"):
                mod.run(autopilot=_autopilot_callback)
            elif hasattr(mod, "main"):
                mod.main(_autopilot_callback)
            else:
                raise RuntimeError(
                    f"Elodin sim module {module_name!r} does not expose "
                    "`run(autopilot=...)` or `main(autopilot)`."
                )

        self._sim_thread = threading.Thread(
            target=_runner, name="elodin-sim", daemon=True
        )
        self._sim_thread.start()

    # ---------------------------------------------------------------- inbound
    def push_sensor_update(self, update) -> None:
        """Test/integration hook: inject a SensorUpdate manually (used when
        the Elodin rig is driven from the same process rather than a thread).
        """
        self._last_update = update
        if self._update_queue is not None:
            try:
                self._update_queue.put_nowait(update)
            except Exception:
                pass

    def pop_command(self):
        """Test/integration hook: pull the next RCCommand the adapter wants
        to issue. Returns a zero command if none queued.
        """
        from .competition_types import RCCommand
        if self._command_queue is None:
            return self._last_rc or RCCommand(0.0, 0.0, 0.0, 0.0)
        try:
            return self._command_queue.get_nowait()
        except Exception:
            return self._last_rc or RCCommand(0.0, 0.0, 0.0, 0.0)

    # ---------------------------------------------------------------- API
    def get_observation(self) -> Observation:
        if not self._connected:
            raise RuntimeError("ElodinSimInterface not connected")

        update = self._last_update
        if self._update_queue is not None:
            try:
                update = self._update_queue.get(timeout=self._frame_timeout_s)
                self._last_update = update
            except Exception:
                if update is None:
                    raise RuntimeError(
                        "Timed out waiting for SensorUpdate from Elodin rig"
                    )

        return self._to_observation(update)

    def send_velocity_command(self,
                               vx: float, vy: float, vz: float,
                               yaw_rate: float = 0.0) -> None:
        if not self._connected:
            raise RuntimeError("ElodinSimInterface not connected")

        self._cmd_vel = np.array([vx, vy, vz])
        self._cmd_yaw_rate = float(yaw_rate)

        # Extract yaw + altitude + vz from the latest SensorUpdate so the
        # translator can do takeoff hand-off and altitude clamping cleanly.
        # Use raw indexing so this works against both our SensorUpdate and
        # the upstream rig's solver.api.SensorUpdate (which lacks our
        # convenience properties).
        yaw_rad = None
        altitude_m = None
        vz_actual = None
        u = self._last_update
        if u is not None:
            wp = np.asarray(u.world_pos, dtype=float)
            wv = np.asarray(u.world_vel, dtype=float)
            qx, qy, qz, qw = wp[0], wp[1], wp[2], wp[3]
            yaw_rad = float(np.arctan2(
                2.0 * (qw * qz + qx * qy),
                1.0 - 2.0 * (qy * qy + qz * qz),
            ))
            altitude_m = float(wp[6])
            if wv.size >= 6:
                vz_actual = float(wv[5])

        rc = self._velocity_to_rc.convert(
            vx=vx, vy=vy, vz=vz, yaw_rate_dps=yaw_rate,
            yaw_rad=yaw_rad,
            altitude_m=altitude_m,
            vertical_velocity_mps=vz_actual,
        )
        self._last_rc = rc
        if self._command_queue is not None:
            try:
                if self._command_queue.full():
                    try:
                        self._command_queue.get_nowait()
                    except Exception:
                        pass
                self._command_queue.put_nowait(rc)
            except Exception:
                pass

    def disconnect(self) -> None:
        self._connected = False
        # Worker thread is daemon — exits with the process.

    def reset(self) -> None:
        self._cmd_vel = np.zeros(3)
        self._cmd_yaw_rate = 0.0
        self._last_update = None
        self._last_rc = None

    def get_ground_truth(self) -> Optional[DroneState]:
        u = self._last_update
        if u is None:
            return None
        # Use raw layout (`world_pos = [qx, qy, qz, qw, x, y, z]`,
        # `world_vel = [wx, wy, wz, vx, vy, vz]`) so this works against
        # both our `drone_sim.competition_types.SensorUpdate` and the
        # upstream Elodin `solver.api.SensorUpdate` (which doesn't carry
        # our convenience properties).
        wp = np.asarray(u.world_pos, dtype=float)
        wv = np.asarray(u.world_vel, dtype=float)
        pos = wp[4:7].copy()
        vel = wv[3:6].copy() if wv.size >= 6 else np.zeros(3)
        qx, qy, qz, qw = wp[0], wp[1], wp[2], wp[3]
        roll  = np.degrees(np.arctan2(2 * (qw * qx + qy * qz),
                                       1 - 2 * (qx * qx + qy * qy)))
        sinp  = 2 * (qw * qy - qz * qx)
        pitch = np.degrees(np.arcsin(np.clip(sinp, -1.0, 1.0)))
        yaw   = np.degrees(np.arctan2(2 * (qw * qz + qx * qy),
                                       1 - 2 * (qy * qy + qz * qz)))
        return DroneState(
            pos=pos,
            vel=vel,
            att_deg=np.array([roll, pitch, yaw], dtype=float),
        )

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _to_observation(update) -> Observation:
        if update is None:
            raise RuntimeError("No SensorUpdate available yet")
        imu = IMUReading(
            accel=tuple(np.asarray(update.accel, dtype=float).tolist()),
            gyro=tuple(np.asarray(update.gyro, dtype=float).tolist()),
            timestamp=update.t,
        )

        frame = update.frame_rgba
        if frame is None:
            # No fresh camera this tick - synthesize a black BGR frame so
            # the perception pipeline can run without crashing. The detector
            # will simply return no detections.
            from .competition_types import CameraSpec
            image = np.zeros((CameraSpec.HEIGHT_PX, CameraSpec.WIDTH_PX, 3),
                             dtype=np.uint8)
        else:
            # SensorUpdate.frame_rgba is RGBA uint8; drop alpha + flip to BGR
            # so the rest of the stack (OpenCV-based) can consume it.
            arr = np.asarray(frame)
            if arr.ndim == 3 and arr.shape[2] == 4:
                image = arr[..., [2, 1, 0]].copy()  # RGBA -> BGR
            elif arr.ndim == 3 and arr.shape[2] == 3:
                image = arr[..., ::-1].copy()       # RGB  -> BGR
            else:
                image = arr.copy()

        return Observation(
            image=image,
            imu=imu,
            rpm=[0.0, 0.0, 0.0, 0.0],
            timestamp=update.t,
        )


class NEDAdapter(SimInterface):
    """Wrap a SimInterface and convert outputs from another frame (e.g., ENU)
    into NED before returning to callers.

    The adapter implements a minimal, conservative mapping for position,
    velocity and accelerometer Z. More complete attitude/gyro mappings
    should be implemented when integrating a real SDK.
    """

    def __init__(self, underlying: SimInterface, underlying_frame: str = "enu") -> None:
        self._underlying = underlying
        self._frame = underlying_frame.lower()

    def connect(self) -> None:
        self._underlying.connect()

    def disconnect(self) -> None:
        self._underlying.disconnect()

    def reset(self) -> None:
        self._underlying.reset()

    def get_observation(self) -> Observation:
        obs = self._underlying.get_observation()
        if self._frame == "enu":
            # Convert accel: ENU (ax east, ay north, az up) -> NED (ax north, ay east, az down)
            ax_e, ay_e, az_e = obs.imu.accel
            ax_n = ay_e
            ay_n = ax_e
            az_n = -az_e
            imu = IMUReading(accel=(ax_n, ay_n, az_n), gyro=obs.imu.gyro, timestamp=obs.imu.timestamp)

            # No change to image; timestamps identical
            return Observation(image=obs.image, imu=imu, rpm=obs.rpm, timestamp=obs.timestamp,
                               battery_pct=obs.battery_pct, is_crashed=obs.is_crashed)
        return obs

    def send_velocity_command(self, vx: float, vy: float, vz: float, yaw_rate: float = 0.0) -> None:
        # Commands expected by higher layers are in NED (x north, y east, z down).
        # Map NED -> underlying ENU if needed.
        if self._frame == "enu":
            # NED (xn, ye, zd) -> ENU (xe, yn, zu)
            xn, ye, zd = vx, vy, vz
            xe = ye
            ye = xn
            zu = -zd
            self._underlying.send_velocity_command(xe, ye, zu, yaw_rate)
        else:
            self._underlying.send_velocity_command(vx, vy, vz, yaw_rate)

    def get_ground_truth(self) -> Optional[DroneState]:
        gt = self._underlying.get_ground_truth()
        if gt is None:
            return None
        if self._frame == "enu":
            # ENU -> NED mapping: x_north = y_enu, y_east = x_enu, z_down = -z_enu
            x_e, y_e, z_e = gt.pos
            vx_e, vy_e, vz_e = gt.vel
            x_n = y_e
            y_n = x_e
            z_n = -z_e
            vx_n = vy_e
            vy_n = vx_e
            vz_n = -vz_e
            # Attitude conversion left as-is; implement when needed
            return DroneState(pos=np.array([x_n, y_n, z_n]),
                              vel=np.array([vx_n, vy_n, vz_n]),
                              att_deg=gt.att_deg.copy())
        return gt

    def get_course(self) -> Optional[List[np.ndarray]]:
        course = self._underlying.get_course()
        if course is None:
            return None
        if self._frame == "enu":
            converted = []
            for p in course:
                x_e, y_e, z_e = p
                converted.append(np.array([y_e, x_e, -z_e]))
            return converted
        return course
