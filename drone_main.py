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
                 elodin_sim_module: Optional[str] = None,
                 dump_frames: bool = False, dump_hz: float = 5.0) -> None:
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

        # Vision-only gate-pass tracking (no global course/map, i.e. live AI-GP
        # MAVLink sim). We can't measure when state.pos crosses the gate plane
        # because EKF position is unreliable, so we infer the pass from the
        # image: the gate grows large + centered (we're about to fly through it)
        # and then drops out of view (we're through). Same sequence-stepping idea
        # as Swift/MonoRace, done from monocular appearance alone.
        self._gate_was_close = False
        self._img_area_px = float(cfg.perception.image_width *
                                  cfg.perception.image_height)
        self._pass_area_px = (getattr(cfg.planning, "gate_pass_area_frac", 0.10)
                              * self._img_area_px)
        self._pass_center_frac = getattr(cfg.planning, "gate_pass_center_frac", 0.40)

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

        # Annotated frame dump (so we can SEE what the drone sees offline).
        self._dump_frames = dump_frames
        self._dump_dt = 1.0 / dump_hz if dump_hz > 0 else 0.2
        self._last_dump_t = 0.0
        self._frames_dir: Optional[Path] = None
        self._frames_raw_dir: Optional[Path] = None
        self._dump_idx = 0

        # Camera-dead watchdog: abort if frames stay black (sim camera died).
        self._black_since: Optional[float] = None
        self._camera_dead_timeout = getattr(cfg.sim, "camera_dead_timeout_s", 2.5)
        self._camera_dead = False

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

        if self._dump_frames:
            self._frames_dir = Path(cfg.telemetry.log_dir) / self._logger.run_id / "frames"
            self._frames_dir.mkdir(parents=True, exist_ok=True)
            # Raw (unannotated) copies form the offline perception-regression
            # corpus: the HUD overlay pollutes HSV re-runs (green text/boxes
            # trip the green-twin start-light rejection).
            self._frames_raw_dir = Path(cfg.telemetry.log_dir) / self._logger.run_id / "frames_raw"
            self._frames_raw_dir.mkdir(parents=True, exist_ok=True)
            print(f"[Main] Dumping annotated frames → {self._frames_dir}")

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

        # --- wait for the race to go active before doing anything ---
        # On the AI-GP sim there is no telemetry/camera and the drone isn't
        # armed until the user starts the race. Hold here (resetting the clock
        # and the camera watchdog) so takeoff begins exactly when the race does
        # and the watchdog doesn't fire on the pre-race black screen.
        if hasattr(self._sim, "is_armed") and not self._sim.is_armed():
            self._start_t = t0
            self._black_since = None
            if self._loop_count % 200 == 0:
                print("  [waiting] race not active yet — click RACE in the sim")
            self._loop_count += 1
            return False

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
        else:
            # Spec-legal telemetry override (VADR-TS-002 §4.5: attitude +
            # linear velocities are provided). The IMU-only EKF attitude/vel
            # diverge within seconds on the real sim (fly14: 28 m position
            # drift at t=1.2s), and a wrong yaw here rotates every velocity
            # command. Position stays EKF: targets are built as pos+relative,
            # so the absolute term cancels in the controller error.
            if getattr(obs, "att_deg", None) is not None:
                state.att_deg = obs.att_deg.copy()
            if getattr(obs, "vel_ned", None) is not None:
                state.vel = obs.vel_ned.copy()

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

        # Camera-dead watchdog: the AI-GP sim's video stream sometimes stops
        # after repeated crashes. Black frames forever = blind run. Detect it
        # fast and abort with a clear message instead of wasting the run.
        if frame_brightness < 1.0:
            if self._black_since is None:
                self._black_since = t0
            elif (t0 - self._black_since) > self._camera_dead_timeout:
                print(f"\n[Main] ⚠ CAMERA DEAD — no video for "
                      f"{self._camera_dead_timeout:.1f}s (all-black frames). "
                      f"The sim's camera stream stopped; restart FlightSim.exe.\n")
                self._camera_dead = True
                self._sim.send_velocity_command(0, 0, 0, 0)
                return True
        else:
            self._black_since = None

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
        max_speed = self._speed_cap_for_phase(confidence)

        # _speed_cap_for_phase already folds detection confidence into max_speed,
        # so pass confidence=1.0 here to avoid throttling by confidence twice
        # (the double penalty made moderate-confidence flight slower than the old
        # flat caps). The cap is the single confidence authority now.
        ctrl = self._controller.compute(
            current_pos=state.pos,
            target_pos=target_pos,
            confidence=1.0,
            max_speed=max_speed,
        )
        if yaw_rate_override is not None:
            ctrl.yaw_rate = yaw_rate_override

        # Vision-only recovery hold: with no course/map the EKF has no absolute
        # reference, so feeding a position target through the PID just chases a
        # drifting garbage estimate. Live runs showed this as a constant
        # backward+climb drift out of bounds into the black void (never
        # re-acquiring). Instead, hover in place (zero translation, hold
        # altitude) and let the yaw-search sweep bring a gate back into frame.
        if self._phase == PHASE_RECOVERY and self._course is None:
            ctrl.vx = 0.0
            ctrl.vy = 0.0
            ctrl.vz = 0.0

        # --- command drone ---
        self._sim.send_velocity_command(ctrl.vx, ctrl.vy, ctrl.vz, ctrl.yaw_rate)
        self._estimator.set_command_velocity(np.array([ctrl.vx, ctrl.vy, ctrl.vz]))

        # --- check gate passage (single prioritized decision) ---
        # Replaces five overlapping if-blocks whose firing order depended on
        # which sensors happened to be healthy. _should_advance_gate has one
        # clear priority chain.
        if self._should_advance_gate(gate_detected_actual, best_det, state, gt):
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

        if self._dump_frames and (t0 - self._last_dump_t) >= self._dump_dt:
            self._last_dump_t = t0
            self._dump_frame(obs.image, detections, t_rel, confidence,
                             ctrl, frame_brightness)

        # Print progress every 50 loops
        if self._loop_count % 50 == 0:
            gt = self._sim.get_ground_truth()
            pos_str = (f"pos=({gt.pos[0]:.1f},{gt.pos[1]:.1f},{gt.pos[2]:.1f})"
                       if gt else "pos=unknown")
            conf_str = f"conf={confidence:.2f}" if gate_detected else "conf=--"
            print(f"  [{t_rel:6.1f}s] phase={self._phase:<10} gate={self._current_gate_idx} "
                  f"{conf_str}  {pos_str}  dt={loop_dt_ms:.1f}ms")

        return self._phase == PHASE_COMPLETE

    def _dump_frame(self, image, detections, t_rel, confidence, ctrl,
                    brightness) -> None:
        """Save an annotated FPV frame to logs/<run>/frames for offline review."""
        if self._frames_dir is None:
            return
        frame = self._detector.annotate(image.copy(), detections) \
            if detections else image.copy()
        # Crosshair at image centre (where the camera points).
        h, w = frame.shape[:2]
        cv2.drawMarker(frame, (w // 2, h // 2), (200, 200, 200),
                       cv2.MARKER_CROSS, 16, 1)
        lines = [
            f"t={t_rel:5.1f}s  phase={self._phase}  gate={self._current_gate_idx}",
            f"conf={confidence:.2f}  bright={brightness:.0f}",
            f"cmd v=({ctrl.vx:+.1f},{ctrl.vy:+.1f},{ctrl.vz:+.1f}) yaw={ctrl.yaw_rate:+.0f}",
        ]
        for i, ln in enumerate(lines):
            cv2.putText(frame, ln, (8, 18 + i * 18), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 1, cv2.LINE_AA)
        out = self._frames_dir / f"f{self._dump_idx:05d}.jpg"
        cv2.imwrite(str(out), frame)
        if self._frames_raw_dir is not None:
            cv2.imwrite(str(self._frames_raw_dir / f"f{self._dump_idx:05d}.jpg"),
                        image)
        self._dump_idx += 1

    # ------------------------------------------------------------------
    def _compute_target(self, gate_detected, best_det, state, obs, gt=None):
        """Determine where to fly this cycle. Returns (target_pos, yaw_rate_override)."""

        # THROUGH is a *speed mode*, not a blanket targeting override. The old
        # code early-returned a frozen through-target for EVERY tick in THROUGH,
        # skipping detection entirely — so a fresh detection of the next gate was
        # ignored and the pass checks never re-evaluated.
        #
        # Corrected: only carry momentum when there is NO fresh detection this
        # tick (we're flying through the gate and briefly lost sight of it). We
        # drive to a FIXED point just past the gate plane so we coast across it
        # instead of stalling in a recovery hold. A fresh/held detection falls
        # through to full re-targeting below (re-acquiring the next gate). The
        # target is bounded (gate + offset), so a genuinely lost gate just means
        # we stop past the plane — no runaway, no out-of-bounds DSQ.
        if self._phase == PHASE_THROUGH and not gate_detected \
                and self._last_gate_pos is not None:
            if self._course is not None and self._current_gate_idx < len(self._course):
                gate_world = self._course[self._current_gate_idx]
                direction = self._course_direction(state.pos, gate_world)
                return self._racing_through_target(state.pos, gate_world, direction), None
            return self._last_gate_pos.copy(), None

        confidence = best_det.confidence if best_det is not None else 0.0
        phase_out, recovery_target, recovery_yaw = self._recovery.update(
            gate_detected=gate_detected,
            gate_confidence=confidence,
            current_pos=state.pos,
            last_gate_pos=self._last_gate_pos,
            heading_rad=float(np.radians(state.att_deg[2])),
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
                through_target = self._racing_through_target(state.pos, gate_world, direction)
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

                # state.att_deg is ground truth in mock mode and the sim's
                # ATTITUDE telemetry in mavlink mode (set in _tick), so pitch
                # and yaw are always real here; pitch>0 = nose up. pitch=0 with
                # the 20° up-tilt camera put every centred gate ~20° above the
                # drone — the fly14 runaway-climb root cause.
                pitch_rad = np.radians(state.att_deg[1])
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
                # Perception-aware yaw (Swift's r_perc, as a control law): turn
                # to keep the detected gate centered in the frame. Without this,
                # a gate drifting to the frame edge (the hard right turn after
                # gate 0) slides out of view and is lost. Yaw rate is
                # proportional to the gate's horizontal pixel offset.
                yaw_override = self._gate_centering_yaw(best_det)
                return gate_target, yaw_override

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
        # Drop the held detection so we don't keep steering toward the gate we
        # just passed; the detector re-acquires the next (now nearest) gate.
        self._gate_was_close = False
        self._last_detection = None
        self._detection_age = 0
        if hasattr(self._detector, "reset"):
            self._detector.reset()
        self._recovery.reset()
        self._controller.reset()

    def _gate_centering_yaw(self, best_det):
        """Yaw rate (deg/s) that turns the drone to face the detected gate.

        Proportional to the gate's horizontal offset from image centre, so the
        camera tracks the gate as it drifts toward the frame edge (keeping it in
        FOV through turns). Returns None if no usable detection.
        """
        if best_det is None or best_det.center_px is None:
            return None
        u = best_det.center_px[0]
        offset_frac = (u - self._camera_cx) / self._camera_cx  # -1 (left) .. +1 (right)
        kp = getattr(cfg.planning, "gate_yaw_kp_dps", 70.0)
        max_yaw = cfg.control.max_yaw_rate_dps
        return float(np.clip(offset_frac * kp, -max_yaw, max_yaw))

    def _speed_cap_for_phase(self, confidence: float) -> float:
        """Confidence-scaled speed ceiling for the current phase.

        The phase sets the *ceiling* (approach/through/recovery); detection
        confidence sets how much of it we're allowed to use. Below
        speed_confidence_floor the cap collapses to recovery_speed (creep when
        blind); at/above speed_confidence_ceil we get the full phase ceiling.
        This is "confidence is speed authority" — fast when locked on, cautious
        when the gate is uncertain.
        """
        d = cfg.drone
        if self._phase == PHASE_APPROACH:
            phase_max = d.approach_speed_mps
        elif self._phase == PHASE_THROUGH:
            phase_max = d.through_speed_mps
        elif self._phase == PHASE_RECOVERY:
            phase_max = d.recovery_speed_mps
        else:
            phase_max = d.max_speed_mps

        floor = getattr(d, "speed_confidence_floor", 0.4)
        ceil = getattr(d, "speed_confidence_ceil", 0.85)
        recovery = d.recovery_speed_mps
        if ceil <= floor:
            return phase_max
        conf_scale = float(np.clip((confidence - floor) / (ceil - floor), 0.0, 1.0))
        return recovery + conf_scale * (phase_max - recovery)

    def _should_advance_gate(self, gate_detected_actual: bool, best_det,
                             state, gt) -> bool:
        """Single, prioritized decision: did we just pass the current gate?

        Priority chain, most reliable source first:
          - No course/map (live AI-GP sim): vision appearance pass only.
          - Course available: ground-truth proximity (mock/debug) ->
            EKF plane crossing (only when the filter is healthy) ->
            fresh-detection distance fallback.
        """
        # Vision-only path (no global course/map — live AI-GP sim). Infer the
        # pass from appearance: gate grows large + centered, then drops out.
        if self._course is None:
            return self._check_vision_gate_pass(gate_detected_actual, best_det)

        if self._current_gate_idx >= len(self._course):
            return False
        gate_world = self._course[self._current_gate_idx]

        # 1) Ground-truth proximity — perfect position, mock/debug only.
        if gt is not None and \
                np.linalg.norm(gt.pos - gate_world) < self._gate_passed_threshold_m:
            return True

        # 2) Plane crossing via EKF position — trust only when EKF is healthy.
        if state.is_healthy and self._gate_passed_by_plane(state.pos, gate_world):
            return True

        # 3) Distance fallback from a fresh detection.
        if best_det is not None and best_det.distance_est_m is not None \
                and best_det.distance_est_m < self._gate_passed_threshold_m:
            return True

        return False

    def _check_vision_gate_pass(self, gate_detected_actual: bool,
                                best_det) -> bool:
        """Detect a gate pass from monocular appearance alone.

        Returns True on the frame where we judge the current gate was just
        flown through. Logic: arm a 'close' flag once a *fresh* detection is
        both large (we're nearly on top of it) and roughly centered (we're
        lined up to pass), then fire when that gate drops out of fresh view
        (we're through it). Used only when no global course/map is available.
        """
        if gate_detected_actual and best_det is not None:
            u, v = best_det.center_px
            off_x = abs(u - self._camera_cx) / cfg.perception.image_width
            off_y = abs(v - self._camera_cy) / cfg.perception.image_height
            centered = (off_x < self._pass_center_frac and
                        off_y < self._pass_center_frac)
            if best_det.area_px >= self._pass_area_px and centered:
                self._gate_was_close = True
            return False

        # No fresh detection this frame. If the gate was close just before it
        # vanished, we flew through it.
        if self._gate_was_close:
            self._gate_was_close = False
            return True
        return False

    def _racing_through_target(self, state_pos: np.ndarray,
                               gate_world: np.ndarray,
                               direction_in: np.ndarray) -> np.ndarray:
        """Through-target that bends toward the NEXT gate to carry momentum.

        Reactive flight aims straight through the gate then hard-turns for the
        next one, bleeding speed. World-class lines (Swift/MonoRace) curve the
        exit toward the next gate. As we close on the current gate we blend the
        entry direction toward the exit (next-gate) direction, so the through
        point sits on the racing line. Falls back to a straight through-target
        when there is no next gate.
        """
        offset = cfg.planning.gate_through_offset_m
        next_idx = self._current_gate_idx + 1
        if self._course is not None and next_idx < len(self._course):
            out = self._course[next_idx] - gate_world
            n = float(np.linalg.norm(out))
            if n > 1e-3:
                direction_out = out / n
                dist = float(np.linalg.norm(gate_world - state_pos))
                blend = float(np.clip(
                    1.0 - dist / cfg.planning.gate_approach_distance_m, 0.0, 1.0))
                racing_dir = (1.0 - blend) * direction_in + blend * direction_out
                rn = float(np.linalg.norm(racing_dir))
                if rn > 1e-3:
                    return gate_world + (racing_dir / rn) * offset
        return gate_world + direction_in * offset

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
    parser.add_argument("--dump-frames", action="store_true",
                        help="Save annotated FPV frames to logs/<run>/frames for review")
    parser.add_argument("--dump-hz", type=float, default=5.0,
                        help="Frame dump rate when --dump-frames is set (default 5)")
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
    parser.add_argument("--calibrate", action="store_true",
                        help="Run hover-thrust calibration against the live sim and exit "
                             "(writes hover_thrust to settings.yaml)")
    parser.add_argument("--verify-signs", action="store_true",
                        help="Verify attitude sign conventions against the live sim and exit "
                             "(patches control/velocity_to_attitude.py if inverted)")
    args = parser.parse_args()

    if args.calibrate:
        from tools.calibrate_hover import calibrate, write_back
        value = calibrate(port=cfg.sim.mavlink_port, timeout=cfg.sim.connect_timeout_s,
                          lo=0.15, hi=0.55, hold_s=3.0, tol_m=0.1, rate_hz=50.0,
                          max_iters=10)
        write_back(value)
        return

    if args.verify_signs:
        from tools.verify_attitude_signs import main as verify_main
        verify_main()
        return

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
                         elodin_sim_module=args.elodin_sim_module,
                         dump_frames=args.dump_frames, dump_hz=args.dump_hz)
    loop.run(max_gates=args.gates)


if __name__ == "__main__":
    main()
