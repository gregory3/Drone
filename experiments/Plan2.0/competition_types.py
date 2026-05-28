"""
Copied experimental competition types from Plan2.0 for archival.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Tuple
import numpy as np


# (File contents preserved from Plan2.0/competition_types.py)

@dataclass
class IMUData:
    accel_mps2: Tuple[float, float, float]
    gyro_rps:   Tuple[float, float, float]
    timestamp:  float

@dataclass
class AttitudeEstimate:
    roll_rad:  float
    pitch_rad: float
    yaw_rad:   float
    timestamp: float

@dataclass
class SensorUpdate:
    timestamp:   float
    imu:         IMUData
    attitude:    AttitudeEstimate
    baro_alt_m:  float
    mag_gauss:   Tuple[float, float, float]
    camera_frame: Optional[np.ndarray]
    world_pos_ned: Optional[Tuple[float, float, float]] = None
    world_vel_ned: Optional[Tuple[float, float, float]] = None

@dataclass
class RCCommand:
    roll:     float
    pitch:    float
    yaw:      float
    throttle: float

    MAX_ROLL_DPS:  float = field(default=720.0, compare=False, repr=False)
    MAX_PITCH_DPS: float = field(default=720.0, compare=False, repr=False)
    MAX_YAW_DPS:   float = field(default=360.0, compare=False, repr=False)

    def __post_init__(self) -> None:
        import numpy as _np
        self.roll     = float(_np.clip(self.roll,     -1.0,  1.0))
        self.pitch    = float(_np.clip(self.pitch,    -1.0,  1.0))
        self.yaw      = float(_np.clip(self.yaw,      -1.0,  1.0))
        self.throttle = float(_np.clip(self.throttle,  0.0,  1.0))

class CameraSpec:
    WIDTH_PX:    int   = 640
    HEIGHT_PX:   int   = 360
    HFOV_DEG:    float = 90.0
    TILT_DEG:    float = 20.0
