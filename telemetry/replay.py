"""
telemetry.replay
Load and inspect flight logs. Run as a script for a CLI summary.

Usage (script):
    python3 -m telemetry.replay logs/run_1234567890

Usage (library):
    from telemetry.replay import FlightReplay
    replay = FlightReplay("logs/run_1234567890")
    for frame in replay.frames:
        print(frame["t"], frame["phase"], frame["gate_confidence"])
    replay.summary()
    replay.plot_confidence()   # requires matplotlib
"""

from __future__ import annotations
import json
import sys
from pathlib import Path
from typing import Iterator, List, Optional


class FlightReplay:

    def __init__(self, run_dir: str | Path) -> None:
        self._dir = Path(run_dir)
        self._flight_path = self._dir / "flight.ndjson"
        self._meta_path = self._dir / "metadata.json"

        if not self._flight_path.exists():
            raise FileNotFoundError(f"No flight.ndjson in {self._dir}")

        self._frames: Optional[List[dict]] = None
        self._events: Optional[List[dict]] = None
        self._meta: Optional[dict] = None

    # ------------------------------------------------------------------
    @property
    def meta(self) -> dict:
        if self._meta is None:
            if self._meta_path.exists():
                with open(self._meta_path) as f:
                    self._meta = json.load(f)
            else:
                self._meta = {}
        return self._meta

    def _load(self) -> None:
        frames, events = [], []
        with open(self._flight_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if "_event" in obj:
                    events.append(obj)
                else:
                    frames.append(obj)
        self._frames = frames
        self._events = events

    @property
    def frames(self) -> List[dict]:
        if self._frames is None:
            self._load()
        return self._frames

    @property
    def events(self) -> List[dict]:
        if self._events is None:
            self._load()
        return self._events

    def frame_at(self, t: float) -> Optional[dict]:
        """Return the frame closest to time t."""
        if not self.frames:
            return None
        return min(self.frames, key=lambda f: abs(f["t"] - t))

    def frames_between(self, t_start: float, t_end: float) -> List[dict]:
        return [f for f in self.frames if t_start <= f["t"] <= t_end]

    def phase_segments(self) -> dict:
        """Return dict of phase → list of (t_start, t_end) intervals."""
        segments: dict = {}
        if not self.frames:
            return segments
        current_phase = self.frames[0]["phase"]
        seg_start = self.frames[0]["t"]
        for frame in self.frames[1:]:
            if frame["phase"] != current_phase:
                segments.setdefault(current_phase, []).append(
                    (seg_start, frame["t"])
                )
                current_phase = frame["phase"]
                seg_start = frame["t"]
        segments.setdefault(current_phase, []).append((seg_start, self.frames[-1]["t"]))
        return segments

    # ------------------------------------------------------------------
    def summary(self) -> None:
        print(f"\n{'='*60}")
        print(f"  Flight Replay: {self._dir.name}")
        print(f"{'='*60}")
        if self.meta:
            print(f"  Duration   : {self.meta.get('wall_duration_s', '?'):.1f}s")
            print(f"  Frames     : {self.meta.get('frame_count', len(self.frames))}")
            print(f"  Avg rate   : {self.meta.get('avg_hz', '?')} Hz")
        print(f"  Log frames : {len(self.frames)}")
        print(f"  Events     : {len(self.events)}")

        if self.frames:
            confidences = [f["gate_confidence"] for f in self.frames]
            detected = [f for f in self.frames if f["gate_detected"]]
            recovery = [f for f in self.frames if f["phase"] == "recovery"]
            print(f"\n  Gate detections : {len(detected)} / {len(self.frames)} frames"
                  f"  ({100*len(detected)/len(self.frames):.1f}%)")
            print(f"  Avg confidence  : {sum(confidences)/len(confidences):.3f}")
            print(f"  Min confidence  : {min(confidences):.3f}")
            print(f"  Recovery frames : {len(recovery)}")

        if self.events:
            print(f"\n  Events:")
            for ev in self.events:
                print(f"    [{ev['t']:.3f}s] {ev['_event']} "
                      f"{' '.join(f'{k}={v}' for k, v in ev.items() if k not in ('_event','t'))}")

        segs = self.phase_segments()
        if segs:
            print(f"\n  Phase breakdown:")
            for phase, intervals in segs.items():
                total = sum(e - s for s, e in intervals)
                print(f"    {phase:<12} {total:.2f}s  ({len(intervals)} segment(s))")
        print()

    def plot_confidence(self) -> None:
        """Plot gate confidence and phase over time (requires matplotlib)."""
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
        except ImportError:
            print("matplotlib not available — pip install matplotlib")
            return

        if not self.frames:
            print("No frames to plot.")
            return

        ts = [f["t"] for f in self.frames]
        conf = [f["gate_confidence"] for f in self.frames]
        detected = [f["gate_detected"] for f in self.frames]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
        fig.suptitle(f"Flight Replay — {self._dir.name}", fontsize=12)

        # Confidence trace
        ax1.plot(ts, conf, color="#1a7abf", linewidth=1.2, label="Gate confidence")
        ax1.axhline(0.55, color="orange", linestyle="--", linewidth=0.8, label="Threshold")
        ax1.fill_between(ts, conf,
                         where=[d for d in detected],
                         alpha=0.2, color="green", label="Detected")
        ax1.set_ylabel("Confidence")
        ax1.set_ylim(0, 1.05)
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)

        # Phase timeline
        phase_colors = {
            "idle": "#cccccc", "search": "#f5a623", "approach": "#7ed321",
            "through": "#4a90e2", "recovery": "#d0021b", "complete": "#417505",
        }
        phases = [f["phase"] for f in self.frames]
        unique_phases = list(dict.fromkeys(phases))
        phase_idx = [list(phase_colors.keys()).index(p)
                     if p in phase_colors else 0 for p in phases]
        ax2.step(ts, phase_idx, where="post", linewidth=1.5, color="#333")
        ax2.set_yticks(range(len(phase_colors)))
        ax2.set_yticklabels(list(phase_colors.keys()), fontsize=8)
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Phase")
        ax2.grid(True, alpha=0.3, axis="x")

        # Event markers
        for ev in self.events:
            ax1.axvline(ev["t"], color="red", linestyle=":", alpha=0.6, linewidth=0.8)
            ax1.text(ev["t"], 0.95, ev["_event"][:8], fontsize=6,
                     rotation=90, va="top", color="red", alpha=0.7)

        plt.tight_layout()
        out_path = self._dir / "replay_plot.png"
        plt.savefig(out_path, dpi=150)
        print(f"Plot saved → {out_path}")
        plt.show()


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        # List available runs
        log_dir = Path("logs")
        if not log_dir.exists():
            print("No logs/ directory found. Run main.py first.")
            sys.exit(0)
        runs = sorted(log_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
        print(f"Available runs in {log_dir}/:")
        for r in runs:
            if (r / "flight.ndjson").exists():
                print(f"  {r.name}")
        print(f"\nUsage: python3 -m telemetry.replay logs/<run_id>")
        sys.exit(0)

    replay = FlightReplay(sys.argv[1])
    replay.summary()
    if "--plot" in sys.argv:
        replay.plot_confidence()
