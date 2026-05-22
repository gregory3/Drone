"""
main.py
AI Grand Prix — Main autonomy loop.

Run with:
    python3 main.py              # mock sim, default settings
    python3 main.py --mode real  # real DCL sim (once SDK installed)
    python3 main.py --replay     # just replay last run (no flight)
"""

from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# Add project root to path so imports work from any directory
sys.path.insert(0, str(Path(__file__).parent))

from config.loader import cfg
from sim.interface import make_interface
from sim.viewer import SimViewer
from perception.gate_detector import make_detector
from state.estimator import DroneStateEstimator
from control.controller import PIDController
from planning.recovery import RecoveryBehavior
from telemetry.logger import FlightLogger, LogFrame


# ---------------------------------------------------------------------------
# Flight phases
# ---------------------------------------------------------------------------
PHASE_IDLE     = "idle"
PHASE_SEARCH   = "search"
PHASE_APPROACH = "approach"
PHASE_THROUGH  = "through"
PHASE_RECOVERY = "recovery"
PHASE_COMPLETE = "complete"


class AutonomyLoop:

    def __init__(self, mode: str = "mock", run_id: Optional[str] = None,
                 show_view: bool = False) -> None:
        print("\n" + "="*60)
        print("  AI GRAND PRIX — Autonomy Stack v0.1.0")
        print("="*60)

        self._sim = make_interface(mode=mode)
        self._detector = make_detector()
        self._estimator = DroneStateEstimator()
        self._controller = PIDController()
        self._recovery = RecoveryBehavior()
        self._logger = FlightLogger(run_id=run_id)

        self._show_view = show_view
        self._viewer: Optional[SimViewer] = None
        self._view_window = "Autonomy Sim View"
        self._stop_requested = False
        self._course = self._sim.get_course() if hasattr(self._sim, "get_course") else None

        self._phase = PHASE_IDLE
        self._current_gate_idx = 0
        self._last_gate_pos: Optional[np.ndarray] = None
        self._gate_passed_threshold_m = cfg.planning.waypoint_tolerance_m
        self._loop_hz = cfg.control.control_hz
        self._loop_dt = 1.0 / self._loop_hz

        self._camera_fx = (cfg.perception.image_width / 2) / np.tan(
            np.radians(cfg.perception.camera_fov_deg / 2)
        )
        self._camera_fy = self._camera_fx
        self._camera_cx = cfg.perception.image_width / 2
        self._camera_cy = cfg.perception.image_height / 2

        # Stats
        self._gates_passed = 0
        self._loop_count = 0
        self._start_t: Optional[float] = None

    # ------------------------------------------------------------------
    def run(self, max_gates: Optional[int] = None) -> None:
        self._sim.connect()
        self._sim.reset()
        self._start_t = time.time()

        print(f"\n[Main] Starting run '{self._logger.run_id}'")
        print(f"[Main] Loop rate: {self._loop_hz} Hz | Detector: "
              f"{cfg.perception.detector_backend}\n")

        if self._show_view:
            view_dir = Path(cfg.telemetry.log_dir) / self._logger.run_id
            self._viewer = SimViewer(self._view_window, view_dir)

        try:
            self._phase = PHASE_SEARCH
            while True:
                t0 = time.time()

                done = self._tick()
                if done:
                    break

                if self._stop_requested:
                    print("[Main] Stop requested from view. Ending run.")
                    break

                if max_gates and self._gates_passed >= max_gates:
                    print(f"[Main] Reached {max_gates} gates — stopping.")
                    break

                # Pace to target Hz
                elapsed = time.time() - t0
                sleep = self._loop_dt - elapsed
                if sleep > 0:
                    time.sleep(sleep)

        except KeyboardInterrupt:
            print("\n[Main] Interrupted by user.")
        finally:
            self._sim.send_velocity_command(0, 0, 0, 0)  # stop drone
            self._sim.disconnect()
            self._logger.close()
            if self._viewer is not None:
                self._viewer.close()
            self._print_summary()

    # ------------------------------------------------------------------
    def _tick(self) -> bool:
        """One control cycle. Returns True if run is complete."""
        t0 = time.time()
        t_rel = t0 - (self._start_t or t0)

        # --- observe ---
        obs = self._sim.get_observation()

        # --- predict (EKF) ---
        self._estimator.predict(
            accel=obs.imu.accel,
            gyro=obs.imu.gyro,
            timestamp=obs.timestamp,
        )
        state = self._estimator.get_estimate()

        # Mock-only fallback: detect gate passage from course ground truth
        gt = None
        if self._course is not None and self._current_gate_idx < len(self._course):
            gt = self._sim.get_ground_truth()
            if gt is not None:
                gate_world = self._course[self._current_gate_idx]
                if np.linalg.norm(gt.pos - gate_world) < self._gate_passed_threshold_m:
                    self._on_gate_passed(t_rel)
                    state = self._estimator.get_estimate()

        # Use ground truth in mock mode for control and target computation.
        if gt is not None:
            state.pos = gt.pos.copy()
            state.vel = gt.vel.copy()
            state.att_deg = gt.att_deg.copy()

        # --- perceive ---
        detections = self._detector.detect(obs.image)
        gate_detected = bool(detections and
                             detections[0].confidence >=
                             cfg.perception.gate_confidence_threshold)
        best_det = detections[0] if gate_detected else None

        # EKF vision update
        if gate_detected and best_det.distance_est_m is not None:
            self._estimator.update_from_vision(
                gate_center_px=best_det.center_px,
                distance_m=best_det.distance_est_m,
                frame_w=obs.image.shape[1],
                frame_h=obs.image.shape[0],
            )

        # --- compute target position ---
        target_pos, yaw_rate_override = self._compute_target(
            gate_detected=gate_detected,
            best_det=best_det,
            state=state,
            obs=obs,
        )

        # --- control ---
        confidence = best_det.confidence if gate_detected else 0.3
        ctrl = self._controller.compute(
            current_pos=state.pos,
            target_pos=target_pos,
            confidence=confidence,
        )
        if yaw_rate_override is not None:
            ctrl.yaw_rate = yaw_rate_override

        # --- command drone ---
        self._sim.send_velocity_command(ctrl.vx, ctrl.vy, ctrl.vz, ctrl.yaw_rate)
        self._estimator.set_command_velocity(np.array([ctrl.vx, ctrl.vy, ctrl.vz]))

        # --- check gate passage ---
        if gate_detected and best_det.distance_est_m is not None:
            if best_det.distance_est_m < self._gate_passed_threshold_m:
                self._on_gate_passed(t_rel)

        if self._last_gate_pos is not None and self._phase in {PHASE_APPROACH, PHASE_THROUGH}:
            if np.linalg.norm(state.pos - self._last_gate_pos) < self._gate_passed_threshold_m:
                self._on_gate_passed(t_rel)

        if self._course is not None and self._current_gate_idx < len(self._course):
            gt = self._sim.get_ground_truth()
            if gt is not None:
                gate_world = self._course[self._current_gate_idx]
                if np.linalg.norm(gt.pos - gate_world) < self._gate_passed_threshold_m:
                    self._on_gate_passed(t_rel)

        # --- log ---
        loop_dt_ms = (time.time() - t0) * 1000
        self._logger.log(LogFrame(
            t=t_rel,
            gate_detected=gate_detected,
            gate_confidence=best_det.confidence if best_det else 0.0,
            gate_center_px=best_det.center_px if best_det else None,
            gate_area_px=best_det.area_px if best_det else 0.0,
            gate_id=self._current_gate_idx,
            pos_estimate=tuple(state.pos),
            vel_estimate=tuple(state.vel),
            att_estimate=tuple(state.att_deg),
            accel_raw=obs.imu.accel,
            gyro_raw=obs.imu.gyro,
            waypoint_target=tuple(target_pos),
            distance_to_gate_m=best_det.distance_est_m if best_det else -1.0,
            cmd_velocity=(ctrl.vx, ctrl.vy, ctrl.vz),
            cmd_yaw_rate=ctrl.yaw_rate,
            confidence_score=confidence,
            phase=self._phase,
            loop_dt_ms=loop_dt_ms,
        ))

        self._loop_count += 1

        if self._show_view and self._show_sim_view(obs.image, detections,
                                                   gate_detected, confidence):
            self._stop_requested = True

        # Print progress every 50 loops
        if self._loop_count % 50 == 0:
            gt = self._sim.get_ground_truth()
            pos_str = (f"pos=({gt.pos[0]:.1f},{gt.pos[1]:.1f},{gt.pos[2]:.1f})"
                       if gt else "pos=unknown")
            conf_str = f"conf={confidence:.2f}" if gate_detected else "conf=--"
            print(f"  [{t_rel:6.1f}s] phase={self._phase:<10} gate={self._current_gate_idx} "
                  f"{conf_str}  {pos_str}  dt={loop_dt_ms:.1f}ms")

        return self._phase == PHASE_COMPLETE

    # ------------------------------------------------------------------
    def _compute_target(self, gate_detected, best_det, state, obs):
        """Determine where to fly this cycle. Returns (target_pos, yaw_rate_override)."""

        # --- recovery check ---
        phase_out, recovery_target, recovery_yaw = self._recovery.update(
            gate_detected=gate_detected,
            current_pos=state.pos,
            last_gate_pos=self._last_gate_pos,
        )
        if phase_out == "recovery":
            self._phase = PHASE_RECOVERY
            return recovery_target, recovery_yaw

        # --- normal flight ---
        if gate_detected and best_det.distance_est_m is not None:
            # Compute gate world position from image and distance
            d = best_det.distance_est_m
            u, v = best_det.center_px
            cam_x = (u - self._camera_cx) * d / self._camera_fx
            cam_y = (v - self._camera_cy) * d / self._camera_fy
            yaw_rad = np.radians(state.att_deg[2])
            rel_x = np.cos(yaw_rad) * d - np.sin(yaw_rad) * cam_x
            rel_y = np.sin(yaw_rad) * d + np.cos(yaw_rad) * cam_x
            rel_z = -cam_y
            gate_center = state.pos + np.array([rel_x, rel_y, rel_z])
            self._last_gate_pos = gate_center.copy()

            # Aim to a point slightly before the gate until we're close enough
            if d > cfg.planning.gate_approach_distance_m:
                alpha = max((d - cfg.planning.gate_approach_distance_m) / d, 0.1)
                target_rel = (gate_center - state.pos) * alpha
                self._phase = PHASE_APPROACH
            else:
                target_rel = gate_center - state.pos
                self._phase = PHASE_THROUGH

            gate_target = state.pos + target_rel
            return gate_target, None

        # No gate — search: hold position and yaw slowly.
        # In mock mode, use the next course gate as a fallback target.
        self._phase = PHASE_SEARCH
        if self._course is not None and self._current_gate_idx < len(self._course):
            return self._course[self._current_gate_idx], cfg.planning.recovery_search_yaw_rate * 0.5
        hold = state.pos.copy()
        return hold, cfg.planning.recovery_search_yaw_rate * 0.5

    def _show_sim_view(self, image: np.ndarray,
                       detections: list,
                       gate_detected: bool,
                       confidence: float) -> bool:
        frame = image.copy()
        if detections:
            frame = self._detector.annotate(frame, detections)

        status = f"phase={self._phase} gate={self._current_gate_idx} " \
                 f"conf={confidence:.2f}"
        cv2.putText(frame, status, (10, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"detector={cfg.perception.detector_backend}",
                    (10, 46), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1, cv2.LINE_AA)

        if self._viewer is None:
            return False

        if self._viewer.display(frame):
            self._show_view = False
            self._stop_requested = True
            return True
        return False

    def _on_gate_passed(self, t: float) -> None:
        self._gates_passed += 1
        self._logger.event(t, "gate_passed",
                           gate_id=self._current_gate_idx,
                           total_passed=self._gates_passed)
        print(f"  [Gate {self._current_gate_idx + 1}] PASSED ✓  "
              f"(t={t:.2f}s, total={self._gates_passed})")
        self._current_gate_idx += 1
        self._last_gate_pos = None
        self._recovery.reset()
        self._controller.reset()

    # ------------------------------------------------------------------
    def _print_summary(self) -> None:
        elapsed = time.time() - (self._start_t or time.time())
        print(f"\n{'='*60}")
        print(f"  Run complete: {self._logger.run_id}")
        print(f"  Gates passed : {self._gates_passed}")
        print(f"  Total time   : {elapsed:.2f}s")
        print(f"  Avg loop     : {elapsed/max(self._loop_count,1)*1000:.1f}ms")
        print(f"  Log          : logs/{self._logger.run_id}/flight.ndjson")
        print(f"{'='*60}\n")
        print(f"Replay with: python3 -m telemetry.replay logs/{self._logger.run_id} --plot")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="AI Grand Prix autonomy stack")
    parser.add_argument("--mode", default="mock", choices=["mock", "real"],
                        help="Simulation mode")
    parser.add_argument("--run-id", default=None, help="Custom run identifier")
    parser.add_argument("--gates", type=int, default=None,
                        help="Stop after N gates (default: run until complete)")
    parser.add_argument("--view", action="store_true",
                        help="Show mock simulator camera view")
    parser.add_argument("--replay", action="store_true",
                        help="Replay last run instead of flying")
    args = parser.parse_args()

    if args.replay:
        from telemetry.replay import FlightReplay
        log_dir = Path("logs")
        runs = sorted(log_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
        if not runs:
            print("No runs to replay.")
            return
        replay = FlightReplay(runs[0])
        replay.summary()
        replay.plot_confidence()
        return

    loop = AutonomyLoop(mode=args.mode, run_id=args.run_id,
                         show_view=args.view)
    loop.run(max_gates=args.gates)


if __name__ == "__main__":
    main()
