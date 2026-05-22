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
    if mode == "mock":
        return MockSimInterface(**kwargs)
    elif mode == "real":
        # Swap in RealSimInterface once the DCL SDK is available
        raise NotImplementedError(
            "Real interface not yet implemented. "
            "Download the DCL simulator package first, then implement "
            "RealSimInterface(SimInterface) wrapping their Python API."
        )
    else:
        raise ValueError(f"Unknown sim mode: {mode!r}. Use 'mock' or 'real'.")
