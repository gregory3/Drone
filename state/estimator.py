"""
state.estimator
Extended Kalman Filter fusing IMU + vision detections.

State vector: [x, y, z, vx, vy, vz]  (6-DOF position + velocity)

Prediction step: driven by IMU accelerations (dt from IMU rate)
Update step    : driven by gate detection center + distance estimate

In Phase 1 the position estimate is approximate — we're flying
gate-to-gate visually. The EKF's job is to smooth jitter and
provide velocity estimates for the control loop.
"""

from __future__ import annotations
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from filterpy.kalman import KalmanFilter

from config.loader import cfg


@dataclass
class StateEstimate:
    pos: np.ndarray           # (3,) m
    vel: np.ndarray           # (3,) m/s
    att_deg: np.ndarray       # (3,) roll, pitch, yaw deg
    covariance: np.ndarray    # (6, 6)
    timestamp: float
    is_healthy: bool = True


class DroneStateEstimator:
    """
    Lightweight EKF for drone state.
    Linear model (position + velocity), making this technically
    a standard Kalman filter — 'Extended' ready for when we add
    attitude to the state vector.
    """

    DIM_X = 6   # [x, y, z, vx, vy, vz]
    DIM_Z_IMU = 3  # accelerometer observation
    DIM_Z_VIS = 3  # (cx_norm, cy_norm, distance) from gate detection

    def __init__(self) -> None:
        sc = cfg.state_estimation
        self._kf = KalmanFilter(dim_x=self.DIM_X, dim_z=3)

        # State transition: x' = F * x (constant velocity model)
        self._kf.F = np.eye(self.DIM_X)
        # F is updated each step with current dt

        # Observation matrix: we observe [x, y, z] directly when gate visible
        self._kf.H = np.zeros((3, self.DIM_X))
        self._kf.H[0, 0] = 1.0
        self._kf.H[1, 1] = 1.0
        self._kf.H[2, 2] = 1.0

        # Process noise
        q_pos = sc.ekf_process_noise_pos
        q_vel = sc.ekf_process_noise_vel
        self._kf.Q = np.diag([q_pos, q_pos, q_pos, q_vel, q_vel, q_vel])

        # Measurement noise (vision-based position update)
        r = sc.ekf_measurement_noise_vision
        self._kf.R = np.eye(3) * r

        # Initial state and covariance
        self._kf.x = np.zeros((self.DIM_X, 1))
        self._kf.P = np.eye(self.DIM_X) * 1.0

        self._last_t: Optional[float] = None
        self._att_deg = np.zeros(3)         # attitude tracked separately (gyro integration)
        self._gyro_bias = np.zeros(3)       # crude bias estimate

        # Velocity damping (when no control input, slow naturally)
        self._cmd_vel = np.zeros(3)

    # ------------------------------------------------------------------
    def predict(self, accel: Tuple, gyro: Tuple, timestamp: float) -> None:
        """IMU prediction step — call at IMU rate (~250 Hz in real, each obs in mock)."""
        if self._last_t is None:
            self._last_t = timestamp
            return

        dt = timestamp - self._last_t
        if dt <= 0 or dt > 0.5:
            self._last_t = timestamp
            return
        self._last_t = timestamp

        # Update F for this dt
        self._kf.F = np.eye(self.DIM_X)
        self._kf.F[0, 3] = dt
        self._kf.F[1, 4] = dt
        self._kf.F[2, 5] = dt

        # Control input: commanded velocity drives the velocity state
        # (in real system we'd use IMU; in Phase 1 mock, commanded vel is reliable)
        u = self._cmd_vel.copy()
        self._kf.x[3, 0] = u[0]
        self._kf.x[4, 0] = u[1]
        self._kf.x[5, 0] = u[2]

        self._kf.predict()

        # Integrate attitude from gyro
        g = np.array(gyro) - self._gyro_bias
        self._att_deg += np.degrees(g) * dt

    def update_from_vision(self,
                            gate_center_px: Tuple[float, float],
                            distance_m: float,
                            frame_w: int,
                            frame_h: int) -> None:
        """
        Vision update step. Gate center gives lateral offset; distance
        gives depth. Together they approximate (x, y, z) of the gate
        relative to the drone, from which we can update position.
        """
        # Normalize image coordinates to [-0.5, 0.5]
        nx = (gate_center_px[0] - frame_w / 2) / frame_w
        ny = (gate_center_px[1] - frame_h / 2) / frame_h

        # Gate position relative to drone
        gate_rel_x = distance_m * 1.0           # forward
        gate_rel_y = -distance_m * nx * 2.0      # lateral (approx)
        gate_rel_z = -distance_m * ny * 1.5      # vertical

        # Absolute gate position estimate
        cur_pos = self._kf.x[:3, 0]
        gate_abs = cur_pos + np.array([gate_rel_x, gate_rel_y, gate_rel_z])

        # We update toward the gate position (not the drone position directly)
        # — this smooths jitter from detection noise
        z = gate_abs.reshape(3, 1)
        self._kf.update(z)

    def set_command_velocity(self, vel: np.ndarray) -> None:
        """Feed commanded velocity for the prediction step."""
        self._cmd_vel = vel.copy()

    # ------------------------------------------------------------------
    def get_estimate(self) -> StateEstimate:
        x = self._kf.x.flatten()
        return StateEstimate(
            pos=x[:3].copy(),
            vel=x[3:].copy(),
            att_deg=self._att_deg.copy(),
            covariance=self._kf.P.copy(),
            timestamp=self._last_t or time.time(),
            is_healthy=self._is_healthy(),
        )

    def _is_healthy(self) -> bool:
        # Flag if covariance has blown up
        return float(np.trace(self._kf.P)) < 1000.0
