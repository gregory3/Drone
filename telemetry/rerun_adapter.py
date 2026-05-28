"""Lightweight adapter for the Rerun SDK.

This adapter is intentionally tolerant: it attempts to import the
Rerun package (`rerun` or `rerun_sdk`) and exposes a simple API used by
`FlightLogger` so we can stream images, IMU and basic scalar channels.

If the SDK is not installed, constructing `RerunAdapter` raises
`NotImplementedError` with guidance for the integrator.
"""
from __future__ import annotations
import time
from typing import Any

class RerunAdapter:
    def __init__(self, run_id: str) -> None:
        # Try multiple import names to be robust to packaging differences
        try:
            import rerun as rr  # type: ignore
        except Exception:
            try:
                import rerun_sdk as rr  # type: ignore
            except Exception as exc:
                raise NotImplementedError(
                    "Rerun SDK not installed. Install with `pip install rerun-sdk`. "
                    f"Import error: {exc}"
                )

        self._rr = rr
        self._run_id = run_id
        # Start a Rerun recording session if the API provides a simple init
        try:
            if hasattr(rr, "init"):
                rr.init(self._run_id)
            elif hasattr(rr, "start"):
                rr.start(self._run_id)
        except Exception:
            # Not fatal; some runtimes don't require a start call
            pass

    def log_frame(self, frame: Any) -> None:
        """Log a flight frame. `frame` is expected to be a mapping or
        object that contains at least `image` (numpy array) and optional
        `imu` dict with `accel` and `gyro`.
        """
        rr = self._rr
        try:
            img = frame.image if hasattr(frame, "image") else frame.get("image")
            if img is not None and hasattr(rr, "log"):
                # Generic log: most rerun APIs accept named events
                try:
                    rr.log("/camera/image", img)
                except Exception:
                    # best-effort: skip if API differs
                    pass

            imu = None
            if hasattr(frame, "imu"):
                imu = frame.imu
            elif isinstance(frame, dict):
                imu = frame.get("imu")

            if imu is not None:
                try:
                    rr.log("/imu", imu)
                except Exception:
                    pass
        except Exception:
            # Never let Rerun errors crash flight
            return

    def log_event(self, name: str, payload: dict) -> None:
        try:
            if hasattr(self._rr, "log"):
                self._rr.log(f"/events/{name}", payload)
        except Exception:
            pass

    def close(self) -> None:
        try:
            if hasattr(self._rr, "shutdown"):
                self._rr.shutdown()
            elif hasattr(self._rr, "stop"):
                self._rr.stop()
        except Exception:
            pass
