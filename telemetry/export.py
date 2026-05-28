"""telemetry.export
Small helpers to load exported flight datasets for analysis.

Usage:
    from telemetry.export import load_flight_csv
    df = load_flight_csv("logs/run_123/flight.csv")
"""
from __future__ import annotations
import csv
import json
from math import sqrt
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

VECTOR_FIELDS = {
    "gate_center_px": ("gate_center_u", "gate_center_v"),
    "pos_estimate": ("pos_x", "pos_y", "pos_z"),
    "vel_estimate": ("vel_x", "vel_y", "vel_z"),
    "att_estimate": ("att_roll", "att_pitch", "att_yaw"),
    "accel_raw": ("accel_x", "accel_y", "accel_z"),
    "gyro_raw": ("gyro_x", "gyro_y", "gyro_z"),
    "waypoint_target": ("waypoint_x", "waypoint_y", "waypoint_z"),
    "cmd_velocity": ("cmd_vx", "cmd_vy", "cmd_vz"),
}

def load_flight_csv(path: str) -> List[Dict[str, Any]]:
    """Load a CSV exported by `FlightLogger.export_dataset` into a list of dicts."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Flight CSV not found: {path}")
    rows: List[Dict[str, Any]] = []
    with open(p, newline="") as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            rows.append(dict(r))
    return rows


def load_events_ndjson(path: str) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    events: List[Dict[str, Any]] = []
    with open(p, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return events


def load_run(run_dir: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    run_path = Path(run_dir)
    flight_csv = run_path / "flight.csv"
    events_ndjson = run_path / "events.ndjson"
    frames = load_flight_csv(str(flight_csv))
    events = load_events_ndjson(str(events_ndjson))
    return frames, events


def list_runs(logs_dir: str) -> List[str]:
    p = Path(logs_dir)
    if not p.exists():
        return []
    return [str(x) for x in sorted(p.iterdir()) if x.is_dir()]


def augment_flight_csv(input_csv: str, output_csv: str) -> str:
    rows = load_flight_csv(input_csv)
    augmented_rows: List[Dict[str, Any]] = []
    fieldnames = set()

    for row in rows:
        flat = _flatten_frame(row)
        augmented_rows.append(flat)
        fieldnames.update(flat.keys())

    fieldnames = sorted(fieldnames)
    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in augmented_rows:
            writer.writerow({k: _serialize_value(row.get(k)) for k in fieldnames})

    return str(out_path)


def augment_run(run_dir: str, out_dir: Optional[str] = None) -> str:
    run_path = Path(run_dir)
    if not run_path.exists():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")

    out_base = Path(out_dir) if out_dir else run_path
    out_base.mkdir(parents=True, exist_ok=True)

    input_csv = run_path / "flight.csv"
    if not input_csv.exists():
        raise FileNotFoundError(f"Flight CSV not found: {input_csv}")

    output_csv = out_base / "flight_augmented.csv"
    return augment_flight_csv(str(input_csv), str(output_csv))


def _parse_json_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def _flatten_frame(frame: Dict[str, Any]) -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    parsed_frame = {k: _parse_json_value(v) for k, v in frame.items()}

    for key, value in parsed_frame.items():
        if key in VECTOR_FIELDS and value is not None:
            components = _flatten_vector(key, value, VECTOR_FIELDS[key])
            flat.update(components)
            flat[key] = value
            continue

        if key == "monitors":
            flat[key] = value
            if isinstance(value, dict):
                for mk, mv in value.items():
                    flat[f"monitor_{mk}"] = mv
            continue

        flat[key] = value

    if flat.get("gate_confidence") is not None:
        try:
            flat["is_gate_detected"] = float(flat.get("gate_confidence", 0.0)) > 0.0
        except (TypeError, ValueError):
            flat["is_gate_detected"] = False

    if all(k in flat for k in ("vel_x", "vel_y", "vel_z")):
        try:
            vx = float(flat["vel_x"])
            vy = float(flat["vel_y"])
            vz = float(flat["vel_z"])
            flat["speed_mps"] = sqrt(vx * vx + vy * vy + vz * vz)
        except (TypeError, ValueError):
            flat["speed_mps"] = None

    return flat


def _flatten_vector(field_name: str, values: Any, suffixes: Tuple[str, ...]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    if isinstance(values, (list, tuple)):
        for idx, suffix in enumerate(suffixes):
            result[suffix] = values[idx] if idx < len(values) else None
    elif isinstance(values, str):
        parsed = _parse_json_value(values)
        if isinstance(parsed, (list, tuple)):
            return _flatten_vector(field_name, parsed, suffixes)
        result[suffixes[0]] = parsed
    else:
        result[suffixes[0]] = values
    return result


def _serialize_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value)
    return value
