"""
telemetry.logger
Structured, timestamped flight log writer.

Every control cycle emits a LogFrame. Logs are written as newline-delimited
JSON (NDJSON) so they can be streamed, tailed, and loaded incrementally.

Usage:
    from telemetry.logger import FlightLogger, LogFrame

    logger = FlightLogger(run_id="run_001")
    logger.log(LogFrame(
        t=0.123,
        gate_detected=True,
        gate_confidence=0.82,
        gate_center_px=(320, 240),
        pos_estimate=(1.0, 0.0, 1.5),
        vel_estimate=(2.1, 0.0, 0.0),
        cmd_velocity=(2.5, 0.0, 0.0),
        phase="approach",
        notes="nominal",
    ))
    logger.close()
"""

from __future__ import annotations
import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional, Tuple

from config.loader import cfg


@dataclass
class LogFrame:
    """One timestamped snapshot of the full autonomy state."""

    # --- time ---
    t: float                                     # seconds since run start

    # --- perception ---
    gate_detected: bool = False
    gate_confidence: float = 0.0
    gate_center_px: Optional[Tuple[float, float]] = None  # (u, v) in image
    gate_area_px: float = 0.0
    gate_id: int = -1                            # which gate in the sequence

    # --- state estimate ---
    pos_estimate: Optional[Tuple[float, float, float]] = None   # (x, y, z) m
    vel_estimate: Optional[Tuple[float, float, float]] = None   # (vx, vy, vz) m/s
    att_estimate: Optional[Tuple[float, float, float]] = None   # (roll, pitch, yaw) deg

    # --- raw IMU ---
    accel_raw: Optional[Tuple[float, float, float]] = None
    gyro_raw: Optional[Tuple[float, float, float]] = None

    # --- planning ---
    waypoint_target: Optional[Tuple[float, float, float]] = None
    distance_to_gate_m: float = -1.0

    # --- control output ---
    cmd_velocity: Optional[Tuple[float, float, float]] = None   # commanded (vx, vy, vz)
    cmd_yaw_rate: float = 0.0                    # commanded yaw rate deg/s
    confidence_score: float = 1.0               # 0..1, gates max speed

    # --- flight phase ---
    phase: str = "idle"   # idle | search | approach | through | recovery | complete

    # --- misc ---
    loop_dt_ms: float = 0.0                      # actual loop execution time
    notes: str = ""
    monitors: Optional[dict] = None  # freeform monitor channels (e.g., {'battery':0.9})


class FlightLogger:
    """
    Writes LogFrames to an NDJSON file in real time.

    File layout:
        logs/
          <run_id>/
            flight.ndjson     ← one JSON object per line, one per control cycle
            metadata.json     ← run metadata written at close()
    """

    def __init__(self, run_id: Optional[str] = None, use_rerun: bool = False) -> None:
        self._run_id = run_id or f"run_{int(time.time())}"
        log_base = Path(cfg.telemetry.log_dir) / self._run_id
        log_base.mkdir(parents=True, exist_ok=True)

        self._flight_path = log_base / "flight.ndjson"
        self._meta_path = log_base / "metadata.json"
        self._fh = open(self._flight_path, "w", buffering=1)  # line-buffered

        self._start_wall = time.time()
        self._frame_count = 0
        self._last_flush = time.time()

        print(f"[Logger] Run '{self._run_id}' → {self._flight_path}")
        # Optional Rerun integration (best-effort)
        self._rerun = None
        if use_rerun:
            try:
                from telemetry.rerun_adapter import RerunAdapter
                self._rerun = RerunAdapter(self._run_id)
            except Exception as exc:
                print(f"[Logger] Rerun adapter unavailable: {exc}")

    # ------------------------------------------------------------------
    def log(self, frame: LogFrame) -> None:
        self._fh.write(json.dumps(asdict(frame), default=_json_fallback) + "\n")
        self._frame_count += 1

        now = time.time()
        if now - self._last_flush >= cfg.telemetry.flush_interval_s:
            self._fh.flush()
            self._last_flush = now
        # Best-effort stream to Rerun
        if self._rerun is not None:
            try:
                self._rerun.log_frame(frame)
            except Exception:
                pass

    def event(self, t: float, kind: str, **kwargs) -> None:
        """Log a named event (gate passed, recovery triggered, etc.)."""
        payload = {"_event": kind, "t": t, **kwargs}
        self._fh.write(json.dumps(payload) + "\n")
        self._fh.flush()
        print(f"[Event @{t:.3f}s] {kind} {kwargs}")
        if self._rerun is not None:
            try:
                self._rerun.log_event(kind, payload)
            except Exception:
                pass

    # ------------------------------------------------------------------
    def close(self) -> None:
        elapsed = time.time() - self._start_wall
        meta = {
            "run_id": self._run_id,
            "wall_start": self._start_wall,
            "wall_duration_s": round(elapsed, 3),
            "frame_count": self._frame_count,
            "avg_hz": round(self._frame_count / elapsed, 1) if elapsed > 0 else 0,
            "log_file": str(self._flight_path),
        }
        with open(self._meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        self._fh.close()
        print(f"[Logger] Closed. {self._frame_count} frames in {elapsed:.1f}s "
              f"({meta['avg_hz']} Hz avg) → {self._flight_path}")
        if self._rerun is not None:
            try:
                self._rerun.close()
            except Exception:
                pass

    def export_dataset(self, out_dir: Optional[str] = None) -> str:
        """Export the run dataset to `out_dir` (defaults to run folder).

        Produces:
          - flight.csv : per-frame flattened CSV of LogFrame entries
          - events.ndjson : all event lines
        Returns the output directory path as string.
        """
        import csv
        out_base = Path(out_dir) if out_dir else Path(cfg.telemetry.log_dir) / self._run_id
        out_base.mkdir(parents=True, exist_ok=True)

        flight_in = Path(cfg.telemetry.log_dir) / self._run_id / "flight.ndjson"
        events_out = out_base / "events.ndjson"
        csv_out = out_base / "flight.csv"

        # Read lines and separate frames vs events
        frames = []
        events = []
        with open(flight_in, "r") as fh:
            for line in fh:
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if isinstance(obj, dict) and obj.get("_event"):
                    events.append(obj)
                else:
                    frames.append(obj)

        # Write events
        with open(events_out, "w") as eh:
            for ev in events:
                eh.write(json.dumps(ev) + "\n")

        # Flatten frames and write CSV
        if frames:
            # determine header from keys of first frame
            header = list(frames[0].keys())
            # ensure monitors present
            if "monitors" in header:
                # keep monitors as JSON string
                pass
            with open(csv_out, "w", newline="") as cf:
                writer = csv.DictWriter(cf, fieldnames=header)
                writer.writeheader()
                for f in frames:
                    row = {k: (json.dumps(f[k]) if isinstance(f[k], (dict, list)) else f[k])
                           for k in header}
                    writer.writerow(row)

        return str(out_base)

    @property
    def run_id(self) -> str:
        return self._run_id

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def export_run_dataset(run_id: str, out_dir: Optional[str] = None) -> str:
    """Export an existing run's flight log to CSV and event NDJSON."""
    import csv
    run_dir = Path(cfg.telemetry.log_dir) / run_id
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")

    out_base = Path(out_dir) if out_dir else run_dir
    out_base.mkdir(parents=True, exist_ok=True)

    flight_in = run_dir / "flight.ndjson"
    if not flight_in.exists():
        raise FileNotFoundError(f"Flight log not found: {flight_in}")

    events_out = out_base / "events.ndjson"
    csv_out = out_base / "flight.csv"

    frames = []
    events = []
    with open(flight_in, "r") as fh:
        for line in fh:
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict) and obj.get("_event"):
                events.append(obj)
            else:
                frames.append(obj)

    with open(events_out, "w") as eh:
        for ev in events:
            eh.write(json.dumps(ev) + "\n")

    if frames:
        header = list(frames[0].keys())
        with open(csv_out, "w", newline="") as cf:
            writer = csv.DictWriter(cf, fieldnames=header)
            writer.writeheader()
            for f in frames:
                row = {k: (json.dumps(f[k]) if isinstance(f[k], (dict, list)) else f[k])
                       for k in header}
                writer.writerow(row)

    return str(out_base)


def _json_fallback(obj):
    """Handle types json.dumps can't serialize (numpy arrays, etc.)."""
    if hasattr(obj, "tolist"):
        return obj.tolist()
    return str(obj)
