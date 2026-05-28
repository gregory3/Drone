"""
sim.interface
Abstraction layer for the DCL simulator API.

Real SDK:
    When Anthropic/DCL release their Python package, implement
    RealSimInterface(SimInterface) and swap it in via config.

Mock (now):
    MockSimInterface runs a local physics sim so we can develop
    perception and control pipelines before the official SDK arrives.

Usage:
    from sim.interface import make_interface
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
                 image_size: Tuple[int, int] = (640, 480),
                 gates: Optional[List[np.ndarray]] = None) -> None:
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

        # Intrinsics for a 120° HFOV camera
        fov_rad = np.radians(120.0)
        self._fx = (self._W / 2) / np.tan(fov_rad / 2)
        self._fy = self._fx
        self._cx = self._W / 2
        self._cy = self._H / 2

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

        # Rotation matrix: world → camera frame (yaw only)
        # Camera convention: X=right, Y=down, Z=forward (into scene)
        # World forward (X) → camera Z; world left (Y) → camera -X; world up (Z) → camera -Y
        R = np.array([
            [-np.sin(yaw),  np.cos(yaw), 0],   # cam X = world lateral
            [           0,            0,-1],    # cam Y = world down
            [ np.cos(yaw),  np.sin(yaw), 0],   # cam Z = world forward
        ])

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
        if force_ned:
            return NEDAdapter(base, underlying_frame="enu")
        return base
    elif mode == "real":
        # Swap in RealSimInterface once the DCL SDK is available
        base = RealSimInterface(**kwargs)
        if force_ned:
            return NEDAdapter(base, underlying_frame="enu")
        return base
    elif mode == "elodin":
        # Adapter for the open-source Elodin practice rig. If the
        # Elodin SDK/package isn't installed, the interface is still
        # importable but `connect()` will raise so callers can detect
        # that the runtime dependency is missing.
        base = ElodinSimInterface(**kwargs)
        if force_ned:
            return NEDAdapter(base, underlying_frame="enu")
        return base
    else:
        raise ValueError(f"Unknown sim mode: {mode!r}. Use 'mock' or 'real'.")


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
    """Adapter stub for the Elodin practice rig.

    This class is importable when the codebase is loaded. At runtime the
    `connect()` method will attempt to import the Elodin SDK (or related
    package) and raise a clear error if it's not available. The real
    implementation should wrap the Elodin client to produce `Observation`
    records compatible with the rest of the stack and ensure the NED
    coordinate frame is preserved.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 15000,
                 connect_timeout_s: float = 10.0) -> None:
        self.host = host
        self.port = port
        self.connect_timeout_s = connect_timeout_s
        self._client = None
        self._connected = False

    def connect(self) -> None:
        try:
            # Try to import the Elodin SDK (name may vary depending on the
            # upstream package). The adapter uses a thin translation layer
            # so the rest of the stack sees `Observation` / `IMUReading`.
            import elodin  # type: ignore
        except Exception as exc:
            raise NotImplementedError(
                "ElodinSimInterface requires the Elodin SDK package. "
                "Install it or run in 'mock' mode. Original import error: %s" % exc
            )

        # Create client and open streams. The exact API will depend on the
        # installed Elodin package; try common methods and raise helpful
        # errors if they are not present so integrators know what to adapt.
        try:
            self._client = elodin.Client(host=self.host, port=self.port)
        except Exception as exc:
            raise RuntimeError(f"Failed to construct Elodin client: {exc}")

        # Try to enable streaming modes commonly exposed by sim clients.
        try:
            if hasattr(self._client, "open_streams"):
                self._client.open_streams()
            elif hasattr(self._client, "start"):
                self._client.start()
        except Exception:
            # Not fatal; many toy SDKs don't require explicit start.
            pass

        self._connected = True

    def get_observation(self) -> Observation:
        if not self._connected or self._client is None:
            raise RuntimeError("Elodin client not connected")

        # Many sim SDKs provide a single call that returns the latest
        # timestep as a dict-like object. Allow multiple candidate method
        # names so this adapter is tolerant to the upstream API.
        frame = None
        for method in ("get_frame", "recv_frame", "pop_frame", "read_frame"):
            if hasattr(self._client, method):
                frame = getattr(self._client, method)()
                break

        if frame is None:
            # As a last resort, try a `poll()` that may return (ok, data)
            if hasattr(self._client, "poll"):
                try:
                    ok, frame = self._client.poll(timeout=0.1)
                    if not ok:
                        raise RuntimeError("No frame available from Elodin client")
                except Exception as exc:
                    raise RuntimeError(f"Failed to read frame from Elodin client: {exc}")
            else:
                raise NotImplementedError(
                    "Elodin client does not expose known frame retrieval methods."
                )

        # Expect frame to be a mapping with keys: 'image', 'imu', 'rpm', 'timestamp'
        try:
            img = frame.get("image") if isinstance(frame, dict) else None
        except Exception:
            img = None

        # Convert image to numpy BGR if needed
        if img is None:
            raise RuntimeError("Elodin frame did not contain 'image' field")

        import numpy as _np
        # Some SDKs return RGB; check dtype and shape and convert when needed.
        image = _np.array(img)
        if image.ndim == 3 and image.shape[2] == 3:
            # Heuristic: if image appears to be RGB, try converting by checking
            # whether values look like normalized floats or 0..255 ints. We
            # assume upstream supplies uint8; if it's not BGR convert RGB->BGR.
            try:
                # If the SDK documents a field, use it; otherwise assume RGB and
                # convert to BGR for OpenCV compatibility.
                image = image[..., ::-1].copy()  # RGB -> BGR
            except Exception:
                pass

        # IMU mapping: accept either dict or object with attributes
        imu_src = frame.get("imu") if isinstance(frame, dict) else None
        if imu_src is None:
            raise RuntimeError("Elodin frame missing imu data")

        # Normalize IMU values
        try:
            ax, ay, az = imu_src.get("accel")
            gx, gy, gz = imu_src.get("gyro")
            ts = frame.get("timestamp", frame.get("time", None))
        except Exception:
            # Try attribute access
            ax, ay, az = imu_src.accel
            gx, gy, gz = imu_src.gyro
            ts = getattr(frame, "timestamp", None)

        # Upstream Elodin rig follows NED by spec; expose the IMU as-is.
        imu = IMUReading(accel=(float(ax), float(ay), float(az)),
                         gyro=(float(gx), float(gy), float(gz)),
                         timestamp=float(ts) if ts is not None else time.time())

        # RPM telemetry optional
        rpm = frame.get("rpm") or frame.get("motors") or [0.0, 0.0, 0.0, 0.0]

        timestamp = frame.get("timestamp") or frame.get("time") or time.time()

        return Observation(image=image, imu=imu, rpm=rpm, timestamp=float(timestamp))

    def send_velocity_command(self,
                               vx: float, vy: float, vz: float,
                               yaw_rate: float = 0.0) -> None:
        if not self._connected or self._client is None:
            raise RuntimeError("Elodin client not connected")

        # Most sim flight APIs accept a velocity or setpoint command. Try
        # a few common method names. Commands are expected in NED (north,
        # east, down). If the upstream client expects different axes, the
        # integrator should adapt here.
        cmd = {
            "vx": float(vx),
            "vy": float(vy),
            "vz": float(vz),
            "yaw_rate": float(yaw_rate),
        }

        for method in ("send_velocity", "set_velocity", "send_command"):
            if hasattr(self._client, method):
                try:
                    getattr(self._client, method)(cmd)
                    return
                except TypeError:
                    # maybe method expects separate args
                    try:
                        getattr(self._client, method)(vx, vy, vz, yaw_rate)
                        return
                    except Exception:
                        continue

        # Fallback: if client exposes a generic command interface
        if hasattr(self._client, "command"):
            try:
                self._client.command("velocity", **cmd)
                return
            except Exception:
                pass

        raise NotImplementedError(
            "ElodinSimInterface could not find a supported command method on the client."
        )

    def disconnect(self) -> None:
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass
        self._connected = False

    def reset(self) -> None:
        # Optional: instruct the Elodin rig to reset
        try:
            if hasattr(self._client, "reset"):
                self._client.reset()
            elif hasattr(self._client, "home"):
                self._client.home()
        except Exception:
            pass


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
