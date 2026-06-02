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

# Force UTF-8 stdout/stderr so the Unicode arrows and check marks used in
# log lines don't crash a default Windows cp1252 console.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import cv2
import numpy as np

# Add project root to path so imports work from any directory
sys.path.insert(0, str(Path(__file__).parent))

from config.loader import cfg
from drone_sim.interface import make_interface
from drone_sim.viewer import SimViewer
from perception.gate_detector import make_detector
from state.estimator import DroneStateEstimator
from control.controller import PIDController
from planning.recovery import RecoveryBehavior
from telemetry.logger import FlightLogger, LogFrame, export_run_dataset


# ---------------------------------------------------------------------------
# Flight phases
# ---------------------------------------------------------------------------
PHASE_IDLE     = "idle"
PHASE_TAKEOFF  = "takeoff"
PHASE_SEARCH   = "search"
PHASE_APPROACH = "approach"
PHASE_THROUGH  = "through"
PHASE_RECOVERY = "recovery"
PHASE_COMPLETE = "complete"


class AutonomyLoop:

    def __init__(self, mode: str = "mock", run_id: Optional[str] = None,
                 show_view: bool = False, force_ned: bool = False,
                 use_rerun: bool = False,
                 elodin_sim_module: Optional[str] = None) -> None:
        print("\n" + "="*60)
        print("  AI GRAND PRIX — Autonomy Stack v0.1.0")
        print("="*60)

        sim_kwargs = {}
        if mode == "elodin" and elodin_sim_module is not None:
            sim_kwargs["sim_main_module"] = elodin_sim_module
        self._sim = make_interface(mode=mode, force_ned=force_ned, **sim_kwargs)
        self._detector = make_detector()
        self._estimator = DroneStateEstimator()
        self._controller = PIDController()
        self._recovery = RecoveryBehavior()
        self._logger = FlightLogger(run_id=run_id, use_rerun=use_rerun)

        self._show_view = show_view
        self._viewer: Optional[SimViewer] = None
        self._view_window = "Autonomy Sim View"
        self._stop_requested = False
        self._course = self._sim.get_course() if hasattr(self._sim, "get_course") else None
        self._use_ground_truth = (mode == "mock" and
                                  getattr(cfg.sim, "mock_use_ground_truth", True))
        self._use_course = (mode == "mock" and
                            getattr(cfg.sim, "mock_use_course", True))
        if self._course is not None and not self._use_course:
            self._course = None

        self._phase = PHASE_IDLE
        self._current_gate_idx = 0
        self._last_gate_pos: Optional[np.ndarray] = None
        self._gate_passed_threshold_m = cfg.planning.waypoint_tolerance_m
        self._loop_hz = cfg.control.control_hz
        self._loop_dt = 1.0 / self._loop_hz

        self._gate_passed_this_cycle = False
        self._last_detection = None
        self._detection_age = 0

        self._camera_fx = (cfg.perception.image_width / 2) / np.tan(
            np.radians(cfg.perception.camera_fov_deg / 2)
        )
        self._camera_fy = self._camera_fx
        self._camera_cx = cfg.perception.image_width / 2
        self._camera_cy = cfg.perception.image_height / 2
        # Camera is pitched up by this angle; must be compensated when
        # reprojecting a gate pixel to a 3D direction, otherwise a level gate
        # reads as "above" and the drone climbs away from it.
        self._camera_tilt_rad = np.radians(
            getattr(cfg.perception, "camera_tilt_deg", 0.0))

        # Stats
        self._gates_passed = 0
        self._loop_count = 0
        self._start_t: Optional[float] = None

    # ------------------------------------------------------------------
    def run(self, max_gates: Optional[int] = None) -> None:
        self._sim.connect()
        self._sim.reset()
        # Real flight controllers (e.g. the AI-GP MAVLink sim) must be armed
        # before they accept setpoints. arm() is interface-specific. Skip the
        # manual pre-race arm when the adapter auto-arms — the AI-GP sim only
        # accepts arming once the race is active, and arming twice (here + at
        # race start) leaves its FC non-responsive.
        if hasattr(self._sim, "arm") and not getattr(self._sim, "_auto_arm", False):
            print("[Main] Arming drone...")
            self._sim.arm()
        self._start_t = time.time()

        print(f"\n[Main] Starting run '{self._logger.run_id}'")
        print(f"[Main] Loop rate: {self._loop_hz} Hz | Detector: "
              f"{cfg.perception.detector_backend}\n")

        if self._show_view:
            view_dir = Path(cfg.telemetry.log_dir) / self._logger.run_id
            self._viewer = SimViewer(self._view_window, view_dir)

        try:
            self._phase = PHASE_TAKEOFF
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

        self._gate_passed_this_cycle = False

        # --- takeoff: climb hard off the pad before gate-following begins ---
        # The sim won't translate until airborne, and the confidence-scaled
        # speed limit otherwise throttles the climb too much to lift off.
        if self._phase == PHASE_TAKEOFF:
            if t_rel < cfg.planning.takeoff_duration_s:
                self._sim.send_velocity_command(
                    0.0, 0.0, -cfg.planning.takeoff_climb_mps, 0.0)
                self._loop_count += 1
                if self._loop_count % 50 == 0:
                    print(f"  [{t_rel:6.1f}s] phase=takeoff  climbing off pad")
                return False
            self._phase = PHASE_SEARCH
            self._controller.reset()

        # --- observe ---
        obs = self._sim.get_observation()

        # --- predict (EKF) ---
        self._estimator.predict(
            accel=obs.imu.accel,
            gyro=obs.imu.gyro,
            timestamp=obs.timestamp,
        )
        state = self._estimator.get_estimate()

        # Use ground truth in mock mode for control and target computation.
        gt = self._sim.get_ground_truth()
        if self._use_ground_truth and gt is not None:
            state.pos = gt.pos.copy()
            state.vel = gt.vel.copy()
            state.att_deg = gt.att_deg.copy()

        # --- perceive ---
        det_t0 = time.time()
        detections = self._detector.detect(obs.image)
        det_t1 = time.time()
        detection_latency_ms = (det_t1 - det_t0) * 1000.0

        gate_detected_actual = bool(detections and
                                    detections[0].confidence >=
                                    cfg.perception.gate_confidence_threshold)
        best_det = detections[0] if gate_detected_actual else None

        if gate_detected_actual:
            self._last_detection = best_det
            self._detection_age = 0
            gate_detected = True
        elif self._last_detection is not None and \
                self._detection_age < cfg.perception.detection_history_frames:
            self._detection_age += 1
            best_det = self._last_detection
            gate_detected = True
        else:
            self._last_detection = None
            best_det = None
            gate_detected = False

        # EKF vision update only on fresh detections
        if gate_detected_actual and best_det is not None \
                and best_det.distance_est_m is not None:
            self._estimator.update_from_vision(
                gate_center_px=best_det.center_px,
                distance_m=best_det.distance_est_m,
                frame_w=obs.image.shape[1],
                frame_h=obs.image.shape[0],
            )

        # --- monitors (for dataset/diagnostics) ---
        try:
            import psutil
            cpu = psutil.cpu_percent(interval=None)
            mem = psutil.virtual_memory().percent
        except Exception:
            cpu = None
            mem = None

        rpm_mean = float(np.mean(obs.rpm)) if obs.rpm is not None else None
        battery_pct = getattr(obs, "battery_pct", None)
        frame_gray = cv2.cvtColor(obs.image, cv2.COLOR_BGR2GRAY)
        frame_brightness = float(np.mean(frame_gray))

        monitors = {
            "rpm_mean": rpm_mean,
            "battery_pct": battery_pct,
            "detection_latency_ms": float(detection_latency_ms),
            "detection_count": len(detections) if detections is not None else 0,
            "detection_confidence": float(best_det.confidence) if best_det is not None else 0.0,
            "frame_brightness": frame_brightness,
            "cpu_percent": cpu,
            "mem_percent": mem,
        }

        # --- compute target position ---
        target_pos, yaw_rate_override = self._compute_target(
            gate_detected=gate_detected,
            best_det=best_det,
            state=state,
            obs=obs,
            gt=gt,
        )

        # --- control ---
        confidence = best_det.confidence if gate_detected else 0.3
        max_speed = cfg.drone.max_speed_mps
        if self._phase == PHASE_APPROACH:
            max_speed = cfg.drone.approach_speed_mps
        elif self._phase == PHASE_THROUGH:
            max_speed = cfg.drone.through_speed_mps
        elif self._phase == PHASE_RECOVERY:
            max_speed = cfg.drone.recovery_speed_mps

        ctrl = self._controller.compute(
            current_pos=state.pos,
            target_pos=target_pos,
            confidence=confidence,
            max_speed=max_speed,
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
                self._gate_passed_this_cycle = True

        if not self._gate_passed_this_cycle and self._last_gate_pos is not None \
                and self._phase in {PHASE_APPROACH, PHASE_THROUGH}:
            if np.linalg.norm(state.pos - self._last_gate_pos) < self._gate_passed_threshold_m:
                self._on_gate_passed(t_rel)
                self._gate_passed_this_cycle = True

        if not self._gate_passed_this_cycle and self._course is not None \
                and self._current_gate_idx < len(self._course):
            if self._gate_passed_by_plane(state.pos, self._course[self._current_gate_idx]):
                self._on_gate_passed(t_rel)
                self._gate_passed_this_cycle = True

        if not self._gate_passed_this_cycle and self._course is not None \
                and self._current_gate_idx < len(self._course) and gt is not None:
            gate_world = self._course[self._current_gate_idx]
            if np.linalg.norm(gt.pos - gate_world) < self._gate_passed_threshold_m:
                self._on_gate_passed(t_rel)
                self._gate_passed_this_cycle = True

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
            monitors=monitors,
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
    def _compute_target(self, gate_detected, best_det, state, obs, gt=None):
        """Determine where to fly this cycle. Returns (target_pos, yaw_rate_override)."""

        # --- recovery check ---
        if self._phase == PHASE_THROUGH and self._last_gate_pos is not None:
            # If the gate was recently visible, continue to the through target
            # using the last known gate position rather than switching to recovery.
            if self._course is not None and self._current_gate_idx < len(self._course):
                gate_world = self._course[self._current_gate_idx]
                direction = self._course_direction(state.pos, gate_world)
                through_target = gate_world + direction * cfg.planning.gate_through_offset_m
                return through_target, None
            return self._last_gate_pos.copy(), None

        confidence = best_det.confidence if best_det is not None else 0.0
        phase_out, recovery_target, recovery_yaw = self._recovery.update(
            gate_detected=gate_detected,
            gate_confidence=confidence,
            current_pos=state.pos,
            last_gate_pos=self._last_gate_pos,
        )
        if phase_out == "recovery":
            self._phase = PHASE_RECOVERY
            return recovery_target, recovery_yaw

        # --- normal flight ---
        if gate_detected:
            gate_target = None
            if self._course is not None and self._current_gate_idx < len(self._course):
                gate_world = self._course[self._current_gate_idx]
                self._last_gate_pos = gate_world.copy()
                direction = self._course_direction(state.pos, gate_world)
                approach_target = gate_world - direction * cfg.planning.gate_approach_distance_m
                through_target = gate_world + direction * cfg.planning.gate_through_offset_m
                distance_to_gate = np.linalg.norm(gate_world - state.pos)

                if distance_to_gate > cfg.planning.gate_approach_distance_m + 0.1:
                    gate_target = approach_target
                    self._phase = PHASE_APPROACH
                else:
                    gate_target = through_target
                    self._phase = PHASE_THROUGH
            elif best_det.distance_est_m is not None:
                # Vision-based reprojection when no course is available.
                # Account for the camera's upward tilt + the drone's live pitch,
                # otherwise a level gate reads as "above" and we climb away.
                d = best_det.distance_est_m
                u, v = best_det.center_px

                # Pixel offsets -> angles relative to the optical axis.
                alpha = np.arctan2(v - self._camera_cy, self._camera_fy)  # down +
                beta = np.arctan2(u - self._camera_cx, self._camera_fx)   # right +

                # Use the drone's measured attitude (legal sensor data) when
                # available; pitch>0 = nose up in our convention.
                if gt is not None:
                    pitch_rad = np.radians(gt.att_deg[1])
                    yaw_rad = np.radians(gt.att_deg[2])
                else:
                    pitch_rad = 0.0
                    yaw_rad = np.radians(state.att_deg[2])

                # Ray elevation above horizontal = camera tilt + body pitch - pixel offset.
                elevation = self._camera_tilt_rad + pitch_rad - alpha
                horiz = d * np.cos(elevation)
                rel_up = d * np.sin(elevation)
                body_forward = horiz * np.cos(beta)
                body_right = horiz * np.sin(beta)

                rel_x = np.cos(yaw_rad) * body_forward - np.sin(yaw_rad) * body_right
                rel_y = np.sin(yaw_rad) * body_forward + np.cos(yaw_rad) * body_right
                rel_z = -rel_up
                gate_center = state.pos + np.array([rel_x, rel_y, rel_z])
                self._last_gate_pos = gate_center.copy()

                if d > cfg.planning.gate_approach_distance_m:
                    alpha = max((d - cfg.planning.gate_approach_distance_m) / d, 0.1)
                    gate_target = state.pos + (gate_center - state.pos) * alpha
                    self._phase = PHASE_APPROACH
                else:
                    gate_target = gate_center
                    self._phase = PHASE_THROUGH

            if gate_target is not None:
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

        if self._course is not None and self._current_gate_idx >= len(self._course):
            self._phase = PHASE_COMPLETE
            self._logger.event(t, "flight_complete",
                               total_passed=self._gates_passed)
            print(f"  [Main] Course complete. {self._gates_passed} gates passed.")
            return

        self._last_gate_pos = None
        self._recovery.reset()
        self._controller.reset()

    def _course_direction(self, state_pos: np.ndarray, gate_world: np.ndarray) -> np.ndarray:
        if self._course is None or len(self._course) == 0:
            return np.array([1.0, 0.0, 0.0])

        if self._current_gate_idx > 0:
            # Use the incoming path from the prior gate to define gate plane and
            # through direction. This is the actual approach vector for the
            # current gate.
            direction = gate_world - self._course[self._current_gate_idx - 1]
        elif self._current_gate_idx + 1 < len(self._course):
            direction = self._course[self._current_gate_idx + 1] - gate_world
        else:
            direction = gate_world - state_pos

        norm = np.linalg.norm(direction)
        if norm < 1e-3:
            return np.array([1.0, 0.0, 0.0])
        return direction / norm

    def _gate_passed_by_plane(self, state_pos: np.ndarray, gate_world: np.ndarray) -> bool:
        if self._course is None or len(self._course) == 0:
            return False

        direction = self._course_direction(state_pos, gate_world)
        rel = state_pos - gate_world
        signed_dist = float(np.dot(rel, direction))
        return signed_dist > cfg.planning.gate_pass_plane_dist_m

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
    parser.add_argument("--mode", default="mock",
                        choices=["mock", "real", "elodin", "mavlink"],
                        help="Simulation mode ('mavlink' = official AI-GP "
                             "FlightSim.exe)")
    parser.add_argument("--elodin-sim-module", default=None,
                        help="Importable Python module for the Elodin rig's "
                             "sim/main.py (e.g. 'sim.main')")
    parser.add_argument("--realistic", action="store_true",
                        help="In mock mode, disable ground truth state injection")
    parser.add_argument("--blind", action="store_true",
                        help="In mock mode, ignore mock course positions and use vision only")
    parser.add_argument("--run-id", default=None, help="Custom run identifier")
    parser.add_argument("--gates", type=int, default=None,
                        help="Stop after N gates (default: run until complete)")
    parser.add_argument("--view", action="store_true",
                        help="Show mock simulator camera view")
    parser.add_argument("--export", metavar="RUN_ID", default=None,
                        help="Export a completed run's dataset and exit")
    parser.add_argument("--export-out", default=None,
                        help="Output directory for exported dataset")
    parser.add_argument("--force-ned", action="store_true",
                        help="Wrap sim interface to present NED frame to stack")
    parser.add_argument("--rerun", action="store_true",
                        help="Stream telemetry to Rerun (requires rerun-sdk installed)")
    parser.add_argument("--replay", action="store_true",
                        help="Replay last run instead of flying")
    args = parser.parse_args()

    if args.export is not None:
        out_dir = args.export_out
        exported = export_run_dataset(args.export, out_dir)
        print(f"Exported run {args.export} to {exported}")
        return

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

    if args.realistic and args.mode == "mock":
        cfg.sim.mock_use_ground_truth = False
    if args.blind and args.mode == "mock":
        cfg.sim.mock_use_course = False

    loop = AutonomyLoop(mode=args.mode, run_id=args.run_id,
                         show_view=args.view, force_ned=args.force_ned,
                         use_rerun=args.rerun,
                         elodin_sim_module=args.elodin_sim_module)
    loop.run(max_gates=args.gates)


if __name__ == "__main__":
    main()
