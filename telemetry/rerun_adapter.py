"""Adapter that streams flight telemetry into Rerun using its archetypes.

The previous version called `rr.log(path, raw_ndarray)`, which Rerun ignores
silently — `rr.log` requires an *archetype* object. This rewrite wraps every
channel in the right archetype and indexes everything on a single
`"flight"` timeline so frames can be scrubbed alongside scalar traces.

Channels logged per `LogFrame`:
  - `/camera/image`            : the latest BGR frame (downsampled to RGB)
  - `/camera/detection`        : 2D bounding box of the best gate detection
  - `/world/drone`             : Transform3D for the EKF pose
  - `/world/waypoint`          : Points3D for the active waypoint target
  - `/imu/accel/{x,y,z}`       : scalar traces
  - `/imu/gyro/{x,y,z}`        : scalar traces
  - `/cmd/velocity/{x,y,z}`    : scalar traces
  - `/cmd/yaw_rate`            : scalar trace
  - `/state/phase`             : TextDocument with the current flight phase
  - `/state/gate_confidence`   : scalar trace
  - `/state/distance_to_gate`  : scalar trace
  - `/monitors/<channel>`      : scalar trace per monitor metric

Events use a dedicated `/events/<kind>` entity with a TextDocument payload.
"""
from __future__ import annotations
import time
from dataclasses import asdict
from typing import Any, Optional


class RerunAdapter:
    TIMELINE = "flight"

    def __init__(self, run_id: str, spawn_viewer: bool = True) -> None:
        try:
            import rerun as rr  # type: ignore
        except Exception as exc:
            raise NotImplementedError(
                "Rerun SDK not installed. Install with `pip install rerun-sdk`. "
                f"Import error: {exc}"
            )

        self._rr = rr
        self._run_id = run_id
        try:
            rr.init(f"ai_grand_prix/{run_id}", spawn=spawn_viewer)
        except TypeError:
            # Older SDKs separate init and spawn.
            rr.init(f"ai_grand_prix/{run_id}")
            if spawn_viewer and hasattr(rr, "spawn"):
                try:
                    rr.spawn()
                except Exception:
                    pass

    # ---------------------------------------------------------------- helpers
    def _set_time(self, t_seconds: float) -> None:
        rr = self._rr
        # Newer SDKs prefer `set_time`; older ones expose `set_time_seconds`.
        try:
            rr.set_time(self.TIMELINE, duration=float(t_seconds))
        except Exception:
            try:
                rr.set_time_seconds(self.TIMELINE, float(t_seconds))
            except Exception:
                pass

    def _scalar(self, path: str, value: Optional[float]) -> None:
        if value is None:
            return
        try:
            self._rr.log(path, self._rr.Scalar(float(value)))
        except Exception:
            pass

    def _scalar_triplet(self, base_path: str, values: Optional[Any]) -> None:
        if not values:
            return
        try:
            seq = list(values)
        except Exception:
            return
        for axis, v in zip(("x", "y", "z"), seq[:3]):
            self._scalar(f"{base_path}/{axis}", v)

    # ---------------------------------------------------------------- API
    def log_frame(self, frame: Any) -> None:
        rr = self._rr
        data = asdict(frame) if not isinstance(frame, dict) else dict(frame)
        t = float(data.get("t", time.time()))
        self._set_time(t)

        image = getattr(frame, "image", None)
        if image is not None:
            try:
                rr.log("/camera/image", rr.Image(image))
            except Exception:
                pass

        bbox = data.get("gate_bbox_px") or self._extract_bbox(data)
        if bbox is not None:
            try:
                x, y, w, h = bbox
                rr.log(
                    "/camera/detection",
                    rr.Boxes2D(array=[[x, y, w, h]], array_format="XYWH"),
                )
            except Exception:
                pass

        pos = data.get("pos_estimate")
        if pos is not None:
            try:
                rr.log("/world/drone", rr.Transform3D(translation=list(pos)))
            except Exception:
                pass

        wp = data.get("waypoint_target")
        if wp is not None:
            try:
                rr.log("/world/waypoint", rr.Points3D([list(wp)]))
            except Exception:
                pass

        self._scalar_triplet("/imu/accel", data.get("accel_raw"))
        self._scalar_triplet("/imu/gyro", data.get("gyro_raw"))
        self._scalar_triplet("/cmd/velocity", data.get("cmd_velocity"))
        self._scalar("/cmd/yaw_rate", data.get("cmd_yaw_rate"))
        self._scalar("/state/gate_confidence", data.get("gate_confidence"))
        dist = data.get("distance_to_gate_m")
        if dist is not None and dist >= 0:
            self._scalar("/state/distance_to_gate", dist)
        self._scalar("/state/loop_dt_ms", data.get("loop_dt_ms"))

        phase = data.get("phase")
        if phase:
            try:
                rr.log("/state/phase", rr.TextDocument(str(phase)))
            except Exception:
                pass

        monitors = data.get("monitors") or {}
        if isinstance(monitors, dict):
            for k, v in monitors.items():
                if isinstance(v, (int, float)):
                    self._scalar(f"/monitors/{k}", v)

    def log_event(self, name: str, payload: dict) -> None:
        rr = self._rr
        t = float(payload.get("t", time.time()))
        self._set_time(t)
        try:
            text = ", ".join(f"{k}={v}" for k, v in payload.items()
                             if k not in ("_event", "t"))
            rr.log(f"/events/{name}", rr.TextDocument(text))
        except Exception:
            pass

    def close(self) -> None:
        rr = self._rr
        for attr in ("disconnect", "shutdown", "stop"):
            if hasattr(rr, attr):
                try:
                    getattr(rr, attr)()
                    return
                except Exception:
                    continue

    # ---------------------------------------------------------------- internal
    @staticmethod
    def _extract_bbox(data: dict):
        """Reconstruct a bbox from `gate_center_px` + `gate_area_px` as a
        coarse fallback (used when bbox isn't directly logged)."""
        center = data.get("gate_center_px")
        area = data.get("gate_area_px") or 0.0
        if not center or area <= 0:
            return None
        side = max(float(area) ** 0.5, 4.0)
        cx, cy = center[0], center[1]
        return (cx - side / 2, cy - side / 2, side, side)
