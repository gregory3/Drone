"""telemetry.analyze
Run analysis and augmentation on exported flight datasets.

Usage:
    python -m telemetry.analyze logs/<run_id>
    python -m telemetry.analyze logs/<run_id> --augment --augment-out out/
"""
from __future__ import annotations
import argparse
import csv
import json
from collections import Counter
from math import sqrt
from pathlib import Path
from typing import Dict, List, Optional

from telemetry.export import augment_run, load_events_ndjson, load_flight_csv


def _safe_float(value: Optional[str], default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _parse_monitors(monitors: Optional[str]) -> Dict[str, float]:
    if not monitors:
        return {}
    if isinstance(monitors, dict):
        return {k: _safe_float(v, 0.0) for k, v in monitors.items()}
    try:
        parsed = json.loads(monitors)
        if isinstance(parsed, dict):
            return {k: _safe_float(v, 0.0) for k, v in parsed.items()}
    except json.JSONDecodeError:
        pass
    return {}


def _parse_vector(value: Optional[str]) -> Optional[List[float]]:
    if not value:
        return None
    try:
        parsed = json.loads(value)
        if isinstance(parsed, list):
            return [float(x) for x in parsed[:3]]
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    return None


def summarize_run(run_dir: str) -> Dict[str, Optional[float]]:
    run_path = Path(run_dir)
    csv_path = run_path / "flight.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Exported flight.csv not found in {run_dir}")

    rows = load_flight_csv(str(csv_path))
    events = load_events_ndjson(str(run_path / "events.ndjson"))
    event_counts = Counter(ev.get("_event") for ev in events if isinstance(ev, dict))

    if not rows:
        return {
            "frame_count": 0,
            "avg_confidence": None,
            "detection_rate": None,
            "avg_detection_latency_ms": None,
            "avg_speed_mps": None,
            "recovery_fraction": None,
            "search_fraction": None,
            "approach_fraction": None,
            "through_fraction": None,
            "event_count": len(events),
        }

    count = len(rows)
    confidences = []
    detected = 0
    latencies = []
    recovery_count = 0
    phase_counts: Dict[str, int] = Counter()
    speeds = []

    for row in rows:
        conf = _safe_float(row.get("gate_confidence"), 0.0)
        confidences.append(conf)
        if conf > 0.0:
            detected += 1

        monitors = _parse_monitors(row.get("monitors"))
        latency = monitors.get("detection_latency_ms")
        if latency is not None:
            latencies.append(latency)

        if row.get("phase") == "recovery":
            recovery_count += 1
        phase_counts[row.get("phase", "unknown")] += 1

        velocity = _parse_vector(row.get("vel_estimate"))
        if velocity is not None and len(velocity) == 3:
            speeds.append(sqrt(velocity[0] ** 2 + velocity[1] ** 2 + velocity[2] ** 2))

    return {
        "frame_count": count,
        "avg_confidence": sum(confidences) / len(confidences) if confidences else None,
        "detection_rate": detected / count if count else None,
        "avg_detection_latency_ms": sum(latencies) / len(latencies) if latencies else None,
        "avg_speed_mps": sum(speeds) / len(speeds) if speeds else None,
        "recovery_fraction": recovery_count / count if count else None,
        "search_fraction": phase_counts.get("search", 0) / count if count else None,
        "approach_fraction": phase_counts.get("approach", 0) / count if count else None,
        "through_fraction": phase_counts.get("through", 0) / count if count else None,
        "event_count": len(events),
        "event_counts": dict(event_counts),
    }


def print_summary(summary: Dict[str, Optional[float]]) -> None:
    print("Flight export summary:")
    for key, value in summary.items():
        if key == "event_counts":
            print("  event_counts:")
            for event_name, count in value.items():
                print(f"    {event_name}: {count}")
        else:
            print(f"  {key}: {value}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze exported flight dataset")
    parser.add_argument("run_dir", help="Path to exported run folder (logs/<run_id>)")
    parser.add_argument("--augment", action="store_true",
                        help="Write augmented dataset for training and diagnostics")
    parser.add_argument("--augment-out", default=None,
                        help="Output directory for the augmented dataset")
    args = parser.parse_args()

    if args.augment:
        augmented_path = augment_run(args.run_dir, args.augment_out)
        print(f"Augmented dataset written to: {augmented_path}")

    summary = summarize_run(args.run_dir)
    print_summary(summary)


if __name__ == "__main__":
    main()
