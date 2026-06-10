"""
tests/test_stack.py
Core unit + integration tests for the autonomy stack.

Run: pytest tests/ -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import csv
import numpy as np
import pytest
import time


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def test_config_loads():
    from config.loader import cfg
    assert cfg.drone.max_speed_mps > 0
    assert cfg.perception.gate_confidence_threshold > 0
    assert cfg.control.control_hz == 50


def test_config_reload():
    from config.loader import reload, cfg
    reload()
    assert cfg.drone.max_speed_mps > 0


# ---------------------------------------------------------------------------
# Telemetry
# ---------------------------------------------------------------------------

def test_logger_writes_and_closes(tmp_path, monkeypatch):
    monkeypatch.setattr("config.loader.cfg.telemetry.log_dir", str(tmp_path))
    from importlib import reload as rl
    import telemetry.logger as tl_module
    rl(tl_module)
    from telemetry.logger import FlightLogger, LogFrame

    with FlightLogger(run_id="test_001") as logger:
        logger.log(LogFrame(t=0.1, gate_detected=True, gate_confidence=0.8))
        logger.log(LogFrame(t=0.2, gate_detected=False, gate_confidence=0.2,
                            phase="recovery"))
        logger.event(0.3, "gate_passed", gate_id=0)

    log_file = tmp_path / "test_001" / "flight.ndjson"
    assert log_file.exists()
    lines = log_file.read_text().strip().split("\n")
    assert len(lines) == 3   # 2 frames + 1 event


def test_replay_summary(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("config.loader.cfg.telemetry.log_dir", str(tmp_path))
    from importlib import reload as rl
    import telemetry.logger as tl_module
    rl(tl_module)
    from telemetry.logger import FlightLogger, LogFrame
    from telemetry.replay import FlightReplay

    with FlightLogger(run_id="replay_test") as logger:
        for i in range(20):
            logger.log(LogFrame(t=i * 0.02, gate_detected=(i % 3 != 0),
                                gate_confidence=0.7 if i % 3 != 0 else 0.1,
                                phase="approach"))

    replay = FlightReplay(tmp_path / "replay_test")
    assert len(replay.frames) == 20
    replay.summary()
    captured = capsys.readouterr()
    assert "Flight Replay" in captured.out


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------

def test_mock_sim_connects_and_observes():
    from drone_sim.interface import MockSimInterface
    from config.loader import cfg
    sim = MockSimInterface()
    sim.connect()
    obs = sim.get_observation()
    assert obs.image.shape == (cfg.perception.image_height,
                              cfg.perception.image_width, 3)
    assert len(obs.rpm) == 4
    assert obs.imu.timestamp > 0
    sim.disconnect()


def test_mock_sim_moves_on_command():
    from drone_sim.interface import MockSimInterface
    sim = MockSimInterface()
    sim.connect()
    sim.reset()
    gt_start = sim.get_ground_truth()

    sim.send_velocity_command(vx=2.0, vy=0.0, vz=0.0)
    time.sleep(0.2)
    obs = sim.get_observation()   # triggers integration

    gt_end = sim.get_ground_truth()
    assert gt_end.pos[0] > gt_start.pos[0]   # moved forward
    sim.disconnect()


def test_mock_sim_renders_gates():
    from drone_sim.interface import MockSimInterface
    import cv2
    sim = MockSimInterface()
    sim.connect()
    # Position drone close to first gate
    sim._state.pos = np.array([3.5, 0.0, 1.5])
    obs = sim.get_observation()
    # Frame should not be all gray (gate should be visible)
    frame = obs.image
    assert not np.all(frame == 80), "Expected gate pixels, got blank frame"
    sim.disconnect()


def test_make_interface_real_mode_stub():
    from drone_sim.interface import make_interface, RealSimInterface
    sim = make_interface(mode="real")
    assert isinstance(sim, RealSimInterface)
    with pytest.raises(NotImplementedError):
        sim.connect()


def test_make_interface_elodin_raises_when_sdk_missing():
    from drone_sim.interface import make_interface, ElodinSimInterface
    sim = make_interface(mode="elodin")
    assert isinstance(sim, ElodinSimInterface)
    with pytest.raises(NotImplementedError):
        sim.connect()


def test_elodin_adapter_round_trips_sensor_update(monkeypatch):
    """Drive the adapter without the SDK: push a SensorUpdate, pop the
    resulting RCCommand. Exercises the queue-based callback bridge."""
    import sys, types
    fake_elodin = types.ModuleType("elodin")
    monkeypatch.setitem(sys.modules, "elodin", fake_elodin)

    from drone_sim.interface import ElodinSimInterface
    from drone_sim.competition_types import SensorUpdate

    adapter = ElodinSimInterface()
    adapter.connect()

    # world_pos = [qx, qy, qz, qw, x, y, z]  (level attitude => qw=1)
    # world_vel = [wx, wy, wz, vx, vy, vz]
    update = SensorUpdate(
        t=0.0, tick=0,
        world_pos=np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.5]),
        world_vel=np.zeros(6),
        gyro=np.zeros(3),
        accel=np.array([0.0, 0.0, -9.81]),
        frame_rgba=np.zeros((360, 640, 4), dtype=np.uint8),
        frame_fresh=True,
    )
    adapter.push_sensor_update(update)
    obs = adapter.get_observation()
    assert obs.image.shape == (360, 640, 3), "frame_rgba must be reduced to BGR"
    assert obs.imu.accel == (0.0, 0.0, -9.81)

    adapter.send_velocity_command(vx=2.0, vy=0.0, vz=0.0, yaw_rate=10.0)
    rc = adapter.pop_command()
    # PWM ints in [1000, 2000]. The rig's Betaflight tune uses
    # pitch > 1500 to accelerate forward (matches baseline solver), so a
    # +vx command should push pitch above center.
    assert 1000 <= rc.pitch <= 2000
    assert 1000 <= rc.roll  <= 2000
    assert 1000 <= rc.throttle <= 2000
    assert rc.pitch > 1500, "Forward velocity -> pitch stick above center"
    assert rc.arm >= 1700, "Adapter should arm motors when commanding velocity"


def test_velocity_to_rc_clamps_and_directions():
    from control.velocity_to_rc import VelocityToRC

    v = VelocityToRC()
    # Pure forward velocity (vx>0): pitch stick above center (1500).
    rc = v.convert(vx=1.0, vy=0.0, vz=0.0, altitude_m=2.0)
    assert rc.pitch > 1500
    assert rc.roll == 1500

    # Pure world-east velocity while facing north (yaw=90 deg) ->
    # becomes forward in body frame -> pitch above center.
    rc = v.convert(vx=0.0, vy=1.0, vz=0.0, yaw_rad=math_pi_half(),
                   altitude_m=2.0)
    assert rc.pitch > 1500

    # Below 1 m altitude -> takeoff throttle handoff.
    rc = v.convert(vx=0.0, vy=0.0, vz=0.5, altitude_m=0.5)
    from drone_sim.competition_types import RC_TAKEOFF_THROTTLE
    assert rc.throttle == RC_TAKEOFF_THROTTLE

    # Huge command -> still clamped into Betaflight PWM range.
    rc = v.convert(vx=100.0, vy=100.0, vz=100.0, yaw_rate_dps=10000.0,
                   altitude_m=5.0)
    assert 1000 <= rc.roll  <= 2000
    assert 1000 <= rc.pitch <= 2000
    assert 1000 <= rc.yaw   <= 2000
    assert 1000 <= rc.throttle <= 2000


def math_pi_half():
    import math
    return math.pi / 2


def test_export_run_dataset_creates_csv_and_events(tmp_path, monkeypatch):
    import json
    from telemetry.logger import export_run_dataset

    run_dir = tmp_path / "run_001"
    run_dir.mkdir(parents=True)
    flight_file = run_dir / "flight.ndjson"
    data = [
        {"t": 0.0, "gate_detected": False, "monitors": {"rpm_mean": 5000}},
        {"_event": "gate_passed", "t": 1.0, "gate_id": 0}
    ]
    flight_file.write_text("\n".join(json.dumps(x) for x in data) + "\n")

    monkeypatch.setattr("config.loader.cfg.telemetry.log_dir", str(tmp_path))
    out_dir = export_run_dataset("run_001")

    assert Path(out_dir).exists()
    assert (Path(out_dir) / "flight.csv").exists()
    assert (Path(out_dir) / "events.ndjson").exists()


def test_export_load_and_augment_flight_dataset(tmp_path):
    import json
    from telemetry.export import augment_flight_csv, augment_run, load_flight_csv

    flight_csv = tmp_path / "flight.csv"
    rows = [
        {
            "t": "0.0",
            "gate_confidence": "0.7",
            "gate_center_px": "[320, 240]",
            "vel_estimate": "[1.0, 0.0, 0.0]",
            "monitors": "{\"detection_latency_ms\": 15.0}"
        },
    ]
    fieldnames = ["t", "gate_confidence", "gate_center_px", "vel_estimate", "monitors"]
    with open(flight_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    output_csv = tmp_path / "flight_augmented.csv"
    augmented_path = augment_flight_csv(str(flight_csv), str(output_csv))
    assert augmented_path == str(output_csv)
    augmented_rows = load_flight_csv(str(output_csv))
    assert len(augmented_rows) == 1
    assert augmented_rows[0]["gate_center_u"] == "320"
    assert augmented_rows[0]["monitor_detection_latency_ms"] == "15.0"
    assert augmented_rows[0]["speed_mps"] == "1.0"

    run_dir = tmp_path / "run_003"
    run_dir.mkdir()
    with open(run_dir / "flight.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    (run_dir / "events.ndjson").write_text(json.dumps({"_event": "gate_passed", "t": 1.0}) + "\n")

    augmented_csv_path = augment_run(str(run_dir))
    assert Path(augmented_csv_path).exists()
    assert Path(augmented_csv_path).name == "flight_augmented.csv"


def test_analyze_exported_run_summary(tmp_path, monkeypatch):
    import json
    from telemetry.analyze import summarize_run

    run_dir = tmp_path / "run_002"
    run_dir.mkdir(parents=True)
    flight_csv = run_dir / "flight.csv"
    rows = [
        {"t": "0.0", "gate_confidence": "0.0", "phase": "search", "monitors": "{\"detection_latency_ms\": 10.0}"},
        {"t": "0.1", "gate_confidence": "0.8", "phase": "approach", "monitors": "{\"detection_latency_ms\": 20.0}"},
        {"t": "0.2", "gate_confidence": "0.5", "phase": "recovery", "monitors": "{\"detection_latency_ms\": 30.0}"},
    ]
    fieldnames = ["t", "gate_confidence", "phase", "monitors"]
    with open(flight_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    summary = summarize_run(str(run_dir))
    assert summary["frame_count"] == 3
    assert abs(summary["avg_confidence"] - 0.4333) < 1e-3
    assert abs(summary["avg_detection_latency_ms"] - 20.0) < 1e-3
    assert abs(summary["recovery_fraction"] - (1/3)) < 1e-6


# ---------------------------------------------------------------------------
# Gate detector
# ---------------------------------------------------------------------------

def test_classical_detector_no_crash_on_blank():
    from perception.gate_detector import ClassicalGateDetector
    det = ClassicalGateDetector()
    blank = np.full((480, 640, 3), 80, dtype=np.uint8)
    results = det.detect(blank)
    assert isinstance(results, list)
    assert all(r.confidence >= 0 for r in results)


def test_classical_detector_finds_orange_rect():
    from perception.gate_detector import ClassicalGateDetector
    import cv2

    det = ClassicalGateDetector()
    frame = np.full((480, 640, 3), 80, dtype=np.uint8)

    # Round-1 gates render as SOLID red/orange squares, so draw a filled gate.
    orange_bgr = (0, 165, 255)   # OpenCV BGR orange
    cv2.rectangle(frame, (250, 150), (400, 330), orange_bgr, -1)

    results = det.detect(frame)
    assert len(results) > 0, "Expected at least one detection on orange gate"
    assert results[0].confidence > 0.1


def test_classical_detector_finds_red_gate():
    """Round-1 gates render as solid red squares; multi-range HSV must catch them."""
    from perception.gate_detector import ClassicalGateDetector
    import cv2

    det = ClassicalGateDetector()
    frame = np.full((480, 640, 3), 80, dtype=np.uint8)
    # Solid red square (BGR). Red wraps the hue boundary -> needs the
    # configured red sub-ranges, not just the orange primary.
    cv2.rectangle(frame, (250, 150), (400, 330), (0, 0, 255), -1)

    results = det.detect(frame)
    assert len(results) > 0, "Expected a detection on the red gate"
    assert results[0].confidence > 0.1


def test_classical_detector_rejects_red_with_green_twin():
    """A red blob beside a comparable green blob is the start light, not a gate.

    This is the fly16 failure: the detector chased a red/green light pair at
    conf 0.76 instead of a gate, and the drone flew off."""
    from perception.gate_detector import ClassicalGateDetector
    import cv2

    det = ClassicalGateDetector()
    # A lone red gate is detected.
    gate_frame = np.full((480, 640, 3), 80, dtype=np.uint8)
    cv2.rectangle(gate_frame, (290, 210), (350, 270), (0, 0, 255), -1)
    gate = det.detect(gate_frame)
    assert len(gate) > 0, "lone red gate should be detected"

    # Same red blob, now with a green twin right beside it -> rejected.
    light_frame = np.full((480, 640, 3), 80, dtype=np.uint8)
    cv2.rectangle(light_frame, (290, 210), (350, 270), (0, 0, 255), -1)
    cv2.rectangle(light_frame, (360, 210), (420, 270), (0, 255, 0), -1)
    light = det.detect(light_frame)
    # The red blob with a green twin must not survive as a confident gate.
    assert not any(d.center_px[0] < 360 and d.confidence > 0.1 for d in light), \
        "red blob with adjacent green twin must be rejected as a start light"


def test_classical_detector_temporal_smoothing_persists_through_short_dropout():
    from perception.gate_detector import make_detector
    import cv2

    det = make_detector("classical")
    frame = np.full((480, 640, 3), 80, dtype=np.uint8)
    orange_bgr = (0, 165, 255)
    cv2.rectangle(frame, (250, 150), (400, 330), orange_bgr, 14)  # hollow gate

    initial = det.detect(frame)
    assert len(initial) == 1
    assert initial[0].confidence > 0.1
    center_initial = initial[0].center_px

    blank = np.full((480, 640, 3), 80, dtype=np.uint8)
    second = det.detect(blank)
    assert len(second) == 1
    assert second[0].confidence > 0.0
    assert np.allclose(second[0].center_px, center_initial, atol=30.0)

    third = det.detect(blank)
    assert len(third) == 1
    fourth = det.detect(blank)
    assert len(fourth) == 0, "Temporal smoothing should expire after repeated misses"


class _FixedDetector:
    """Returns a preset list of detections each call (highest-conf first)."""
    def __init__(self, batches):
        self._batches = list(batches)
        self._i = 0

    def detect(self, frame_bgr):
        out = self._batches[min(self._i, len(self._batches) - 1)]
        self._i += 1
        return list(out)


def _det(cx, cy, area, conf):
    from perception.gate_detector import GateDetection
    w = h = int(area ** 0.5)
    return GateDetection(center_px=(cx, cy),
                         bbox_px=(int(cx - w / 2), int(cy - h / 2), w, h),
                         area_px=float(area), confidence=conf)


def test_temporal_detector_picks_nearest_not_highest_confidence():
    # Two gates in view: a near one (big area, lower conf) and a far one
    # (small area, higher conf). "nearest" policy must pick the near one.
    from perception.gate_detector import TemporalGateDetector
    near = _det(300, 180, area=40000, conf=0.6)
    far = _det(500, 120, area=4000, conf=0.95)
    backend = _FixedDetector([[far, near]])
    det = TemporalGateDetector(backend, history_frames=1, max_missed_frames=0,
                               max_center_jump_px=120, select="nearest")
    out = det.detect(None)
    assert out[0].area_px == 40000, "should lock the nearest (largest) gate"


def test_temporal_detector_keeps_lock_via_continuity():
    # Locked on near gate; next frame a far gate spikes in confidence. Continuity
    # must keep us on the near gate (it only moved a little), not jump to far.
    from perception.gate_detector import TemporalGateDetector
    near0 = _det(300, 180, area=40000, conf=0.6)
    near1 = _det(310, 185, area=46000, conf=0.6)   # same gate, grown slightly
    far = _det(560, 90, area=3000, conf=0.99)      # far gate, far away in image
    backend = _FixedDetector([[near0], [far, near1]])
    det = TemporalGateDetector(backend, history_frames=1, max_missed_frames=0,
                               max_center_jump_px=120, select="nearest",
                               continuity=True)
    det.detect(None)                # lock near0
    out = det.detect(None)          # far spikes, but near1 is close to lock
    assert abs(out[0].center_px[0] - 310) < 30, "continuity should hold the lock"


def test_vision_gate_pass_fires_on_large_then_lost(monkeypatch):
    from drone_main import AutonomyLoop
    import config.loader as loader
    monkeypatch.setattr(loader.cfg.sim, "mock_use_course", False)
    loop = AutonomyLoop(mode="mock")
    loop._course = None  # force the vision-only path

    big_centered = _det(loop._camera_cx, loop._camera_cy,
                        area=loop._pass_area_px + 1000, conf=0.9)
    # Gate is large + centered -> arm "close", no pass yet.
    assert loop._check_vision_gate_pass(True, big_centered) is False
    assert loop._gate_was_close is True
    # Gate drops out of fresh view -> declare a pass.
    assert loop._check_vision_gate_pass(False, None) is True
    assert loop._gate_was_close is False


def test_make_detector_onnx_raises_when_model_missing(monkeypatch, tmp_path):
    from perception.gate_detector import make_detector
    import config.loader as loader

    monkeypatch.setattr(loader.cfg.perception, "onnx_model_path",
                        str(tmp_path / "missing_model.onnx"))
    with pytest.raises(FileNotFoundError) as excinfo:
        make_detector("onnx")
    assert "ONNX model not found" in str(excinfo.value)


def test_detector_annotate_returns_same_shape():
    from perception.gate_detector import ClassicalGateDetector, GateDetection
    det = ClassicalGateDetector()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    detections = [GateDetection(center_px=(320, 240), bbox_px=(200, 160, 240, 160),
                                area_px=38400, confidence=0.75)]
    annotated = det.annotate(frame, detections)
    assert annotated.shape == frame.shape


# ---------------------------------------------------------------------------
# State estimator
# ---------------------------------------------------------------------------

def test_estimator_returns_healthy_initial():
    from state.estimator import DroneStateEstimator
    est = DroneStateEstimator()
    t = time.time()
    est.predict((0, 0, -9.81), (0, 0, 0), t)
    state = est.get_estimate()
    assert state.is_healthy
    assert state.pos.shape == (3,)
    assert state.vel.shape == (3,)


def test_estimator_integrates_imu_acceleration():
    """With body accel of 5 m/s^2 forward (plus gravity hold) the EKF
    should report a positive forward velocity after a few ticks."""
    from state.estimator import DroneStateEstimator
    est = DroneStateEstimator()

    t = time.time()
    # First call only seeds _last_t; second call onwards integrates.
    est.predict((0.0, 0.0, -9.81), (0, 0, 0), t)
    for i in range(1, 20):
        est.predict((5.0, 0.0, -9.81), (0, 0, 0), t + i * 0.02)

    state = est.get_estimate()
    assert state.vel[0] > 0.1, f"Expected forward velocity, got {state.vel[0]}"
    assert state.pos[0] > 0.0, f"Expected forward position, got {state.pos[0]}"


def test_estimator_holds_still_under_gravity_only():
    """Stationary drone (only gravity on the accelerometer) must not drift."""
    from state.estimator import DroneStateEstimator
    est = DroneStateEstimator()

    t = time.time()
    est.predict((0.0, 0.0, -9.81), (0, 0, 0), t)
    for i in range(1, 100):
        est.predict((0.0, 0.0, -9.81), (0, 0, 0), t + i * 0.02)

    state = est.get_estimate()
    assert abs(state.vel[0]) < 1e-6
    assert abs(state.vel[1]) < 1e-6
    assert abs(state.vel[2]) < 1e-6


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------

def test_pid_moves_toward_target():
    from control.controller import PIDController
    ctrl = PIDController()
    current = np.array([0.0, 0.0, 1.5])
    target = np.array([5.0, 0.0, 1.5])
    out = ctrl.compute(current, target, confidence=1.0)
    assert out.vx > 0, "Should command positive vx toward target"
    assert abs(out.vy) < 0.5, "Should not drift laterally"


def test_pid_respects_speed_limit():
    from control.controller import PIDController
    from config.loader import cfg
    ctrl = PIDController()
    # Far away target
    out = ctrl.compute(np.zeros(3), np.array([1000.0, 0.0, 0.0]), confidence=1.0)
    speed = np.sqrt(out.vx**2 + out.vy**2 + out.vz**2)
    assert speed <= cfg.drone.max_speed_mps + 0.01


def test_pid_respects_max_speed_override():
    from control.controller import PIDController

    ctrl = PIDController()
    out = ctrl.compute(np.zeros(3), np.array([1000.0, 0.0, 0.0]),
                       confidence=1.0, max_speed=1.0)
    speed = np.sqrt(out.vx**2 + out.vy**2 + out.vz**2)
    assert speed <= 1.01


def test_pid_scales_speed_with_confidence():
    from control.controller import PIDController
    ctrl_hi = PIDController()
    ctrl_lo = PIDController()
    target = np.array([100.0, 0.0, 0.0])
    cur = np.zeros(3)
    hi = ctrl_hi.compute(cur, target, confidence=1.0)
    lo = ctrl_lo.compute(cur, target, confidence=0.3)
    assert hi.vx > lo.vx, "High confidence should allow higher speed"


# ---------------------------------------------------------------------------
# Recovery
# ---------------------------------------------------------------------------

def test_recovery_activates_after_lost_frames():
    from planning.recovery import RecoveryBehavior
    from config.loader import cfg
    rec = RecoveryBehavior()
    pos = np.array([5.0, 0.0, 1.5])

    for _ in range(rec._trigger_frames + 1):
        phase, _, _ = rec.update(gate_detected=False, gate_confidence=0.0,
                                 current_pos=pos, last_gate_pos=None)

    assert phase == "recovery"


def test_recovery_clears_on_gate_found():
    from planning.recovery import RecoveryBehavior
    from config.loader import cfg
    rec = RecoveryBehavior()
    pos = np.array([5.0, 0.0, 1.5])

    # Trigger recovery
    for _ in range(rec._trigger_frames + 2):
        rec.update(gate_detected=False, gate_confidence=0.0,
                   current_pos=pos, last_gate_pos=None)

    # Gate found
    phase, _, _ = rec.update(gate_detected=True, gate_confidence=0.8,
                              current_pos=pos, last_gate_pos=None)
    assert phase == "recovery"

    phase, _, _ = rec.update(gate_detected=True, gate_confidence=0.9,
                              current_pos=pos, last_gate_pos=None)
    assert phase == "resume"


def test_recovery_requires_stable_detection_to_resume():
    from planning.recovery import RecoveryBehavior
    from config.loader import cfg
    rec = RecoveryBehavior()
    pos = np.array([5.0, 0.0, 1.5])

    # Trigger recovery
    for _ in range(rec._trigger_frames + 2):
        rec.update(gate_detected=False, gate_confidence=0.0,
                   current_pos=pos, last_gate_pos=None)

    # One good detection should not immediately exit recovery
    phase, _, _ = rec.update(gate_detected=True, gate_confidence=0.8,
                              current_pos=pos, last_gate_pos=None)
    assert phase == "recovery"

    phase, _, _ = rec.update(gate_detected=True, gate_confidence=0.9,
                              current_pos=pos, last_gate_pos=None)
    assert phase == "resume"


def test_autonomy_loop_completes_when_last_gate_passed():
    from drone_main import AutonomyLoop, PHASE_COMPLETE

    loop = AutonomyLoop(mode="mock", run_id="test_complete", show_view=False)
    assert loop._course is not None
    loop._current_gate_idx = len(loop._course) - 1
    loop._gates_passed = len(loop._course) - 1

    loop._on_gate_passed(0.1)

    assert loop._phase == PHASE_COMPLETE
    assert loop._gates_passed == len(loop._course)


def test_autonomy_loop_uses_course_gate_when_visible():
    from drone_main import AutonomyLoop

    loop = AutonomyLoop(mode="mock", run_id="test_course_target", show_view=False)
    assert loop._course is not None
    loop._current_gate_idx = 0

    state = loop._estimator.get_estimate()
    target, yaw_rate = loop._compute_target(
        gate_detected=True,
        best_det=None,
        state=state,
        obs=None,
    )

    expected_gate = loop._course[0]
    assert np.allclose(loop._last_gate_pos, expected_gate)
    assert yaw_rate is None
    assert loop._phase in {"approach", "through"}
    assert np.linalg.norm(target - expected_gate) <= np.linalg.norm(expected_gate - state.pos)


def test_autonomy_loop_aims_past_gate_when_close():
    from drone_main import AutonomyLoop
    from config.loader import cfg

    loop = AutonomyLoop(mode="mock", run_id="test_close_target", show_view=False)
    assert loop._course is not None
    loop._current_gate_idx = 0

    state = loop._estimator.get_estimate()
    state.pos = np.array([4.2, 0.0, 1.5])

    target, _ = loop._compute_target(
        gate_detected=True,
        best_det=None,
        state=state,
        obs=None,
    )

    expected_gate = loop._course[0]
    assert np.linalg.norm(target - expected_gate) < cfg.planning.gate_through_offset_m + 0.1
    assert np.linalg.norm(target - expected_gate) > 0.0


def test_autonomy_loop_transitions_to_through_at_approach_boundary():
    from drone_main import AutonomyLoop

    loop = AutonomyLoop(mode="mock", run_id="test_approach_boundary", show_view=False)
    assert loop._course is not None
    loop._current_gate_idx = 0

    state = loop._estimator.get_estimate()
    gate_world = loop._course[0]
    direction = loop._course_direction(state.pos, gate_world)
    state.pos = gate_world - direction * loop._gate_passed_threshold_m

    target, _ = loop._compute_target(
        gate_detected=True,
        best_det=None,
        state=state,
        obs=None,
    )

    assert loop._phase == "through"
    assert np.linalg.norm(target - gate_world) > 0.0


def test_autonomy_loop_can_disable_mock_ground_truth(monkeypatch):
    import config.loader as loader
    from drone_main import AutonomyLoop

    monkeypatch.setattr(loader.cfg.sim, "mock_use_ground_truth", False)
    loop = AutonomyLoop(mode="mock", run_id="test_realistic", show_view=False)
    assert loop._use_ground_truth is False


def test_autonomy_loop_can_disable_mock_course(monkeypatch):
    import config.loader as loader
    from drone_main import AutonomyLoop

    monkeypatch.setattr(loader.cfg.sim, "mock_use_course", False)
    loop = AutonomyLoop(mode="mock", run_id="test_blind", show_view=False)
    assert loop._use_course is False
    assert loop._course is None


def test_autonomy_loop_persists_last_detection_short_loss(monkeypatch):
    from drone_main import AutonomyLoop
    from perception.gate_detector import GateDetection
    import numpy as np

    loop = AutonomyLoop(mode="mock", run_id="test_detection_history", show_view=False)
    loop._use_ground_truth = False
    loop._sim._state.pos = np.array([3.5, 0.0, 1.5])

    detection = GateDetection(
        center_px=(320, 240),
        bbox_px=(250, 150, 400, 330),
        area_px=20000,
        confidence=0.95,
        distance_est_m=4.0,
    )
    calls = {"count": 0}

    def fake_detect(image):
        calls["count"] += 1
        return [detection] if calls["count"] == 1 else []

    monkeypatch.setattr(loop._detector, "detect", fake_detect)

    loop._tick()
    assert loop._last_detection is not None
    assert loop._detection_age == 0
    assert loop._phase != "recovery"

    loop._tick()
    assert calls["count"] == 2
    assert loop._last_detection is not None
    assert loop._detection_age == 1
    assert loop._phase != "recovery"


def test_course_direction_prefers_previous_gate_path():
    from drone_main import AutonomyLoop

    loop = AutonomyLoop(mode="mock", run_id="test_direction_path", show_view=False)
    assert loop._course is not None
    loop._current_gate_idx = 2

    state = loop._estimator.get_estimate()
    gate_world = loop._course[2]
    direction = loop._course_direction(state.pos, gate_world)

    expected = gate_world - loop._course[1]
    assert np.allclose(direction, expected / np.linalg.norm(expected))


# ---------------------------------------------------------------------------
# Integration: one full control cycle
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Phase 1 bug-fix regression tests
# ---------------------------------------------------------------------------

def test_yaw_decoupled_from_lateral_pid():
    """Bug 2: pure lateral position error must not manufacture a big yaw rate.

    Yaw is owned by the perception-aware override upstream; the controller
    emits no yaw from the lateral PID term."""
    from control.controller import PIDController
    ctrl = PIDController()
    out = ctrl.compute(np.array([0.0, 0.0, 1.5]),
                       np.array([0.0, 2.0, 1.5]), confidence=1.0)
    assert abs(out.yaw_rate) < 15.0
    assert out.vy > 0.0, "still commands lateral velocity to close the offset"


def test_through_phase_uses_fresh_detection_vision_only(monkeypatch):
    """Bug 3: THROUGH must not freeze on a stale last_gate_pos — it should run
    the normal detection-driven targeting every tick."""
    import config.loader as loader
    monkeypatch.setattr(loader.cfg.sim, "mock_use_course", False)
    from drone_main import AutonomyLoop, PHASE_THROUGH
    from perception.gate_detector import GateDetection

    loop = AutonomyLoop(mode="mock", run_id="t_through_vis", show_view=False)
    loop._course = None
    loop._phase = PHASE_THROUGH
    stale = np.array([99.0, 99.0, 99.0])
    loop._last_gate_pos = stale.copy()

    state = loop._estimator.get_estimate()
    state.pos = np.array([0.0, 0.0, 1.5])
    det = GateDetection(center_px=(loop._camera_cx, loop._camera_cy),
                        bbox_px=(0, 0, 100, 100), area_px=20000.0,
                        confidence=0.9, distance_est_m=2.0)

    target, _ = loop._compute_target(gate_detected=True, best_det=det,
                                     state=state, obs=None, gt=None)
    assert not np.allclose(target, stale), \
        "THROUGH must re-target from the fresh detection, not freeze on stale pos"


def test_classical_detector_distance_uses_spec_gate_size():
    """VADR-TS-002 §3.7: the outer gate frame is 2.7 m and the red contour's
    bbox spans it. dist = gate_real_size_m * fx / w_px. The old hard-coded
    1.2 m guess read every distance ~2.25x too short, corrupting phase
    transitions and EKF vision updates."""
    import cv2
    from perception.gate_detector import ClassicalGateDetector
    from config.loader import cfg

    assert cfg.perception.gate_real_size_m == pytest.approx(2.7)

    det = ClassicalGateDetector()
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    cv2.rectangle(frame, (280, 140), (359, 219), (0, 0, 255), thickness=-1)
    dets = det.detect(frame)
    assert dets, "a solid red square must be detected"
    w_px = dets[0].bbox_px[2]
    expected = cfg.perception.gate_real_size_m * 320.0 / w_px  # fx=320 @90° HFOV
    assert dets[0].distance_est_m == pytest.approx(expected, rel=0.05)
    assert dets[0].distance_est_m == pytest.approx(10.8, rel=0.15)


def test_vision_reprojection_uses_real_pitch(monkeypatch):
    """fly14 regression: with the 20° up-tilt camera, a centred gate is level
    with the drone only when the nose is pitched DOWN 20°. The loop must take
    pitch from state.att_deg (sim ATTITUDE telemetry), never assume 0 — the
    old pitch=0 fallback placed a phantom target d*sin(20°) above every gate
    and the drone climbed off the course chasing it."""
    import config.loader as loader
    monkeypatch.setattr(loader.cfg.sim, "mock_use_course", False)
    from drone_main import AutonomyLoop
    from perception.gate_detector import GateDetection

    loop = AutonomyLoop(mode="mock", run_id="t_reproj_pitch", show_view=False)
    loop._course = None
    d = 10.0
    det = GateDetection(center_px=(loop._camera_cx, loop._camera_cy),
                        bbox_px=(0, 0, 100, 100), area_px=10000.0,
                        confidence=0.9, distance_est_m=d)

    state = loop._estimator.get_estimate()
    state.pos = np.array([0.0, 0.0, 0.0])
    state.att_deg = np.array([0.0, -20.0, 0.0])  # nose-down cancels camera tilt
    t_level, _ = loop._compute_target(gate_detected=True, best_det=det,
                                      state=state, obs=None, gt=None)
    assert abs(t_level[2]) < 0.2, \
        "tilt-compensated pitch must yield a level target"

    state.att_deg = np.array([0.0, 0.0, 0.0])
    t_zero, _ = loop._compute_target(gate_detected=True, best_det=det,
                                     state=state, obs=None, gt=None)
    assert abs(t_zero[2]) > 2.0, \
        "pitch=0 must show the 20° camera-tilt bias (sanity check)"


def test_should_advance_gate_vision_only_no_false_pass(monkeypatch):
    """Bug 6: with no course and no detection, no gate pass is declared."""
    import config.loader as loader
    monkeypatch.setattr(loader.cfg.sim, "mock_use_course", False)
    from drone_main import AutonomyLoop
    loop = AutonomyLoop(mode="mock", run_id="t_adv_vis", show_view=False)
    loop._course = None
    state = loop._estimator.get_estimate()
    assert loop._should_advance_gate(False, None, state, None) is False


def test_should_advance_gate_ground_truth_proximity():
    """Bug 6: ground-truth proximity to the current gate advances."""
    from drone_main import AutonomyLoop
    loop = AutonomyLoop(mode="mock", run_id="t_adv_gt", show_view=False)
    loop._current_gate_idx = 0
    state = loop._estimator.get_estimate()
    gate = loop._course[0]

    gt_on = SimpleNamespaceGT(gate.copy())
    assert loop._should_advance_gate(False, None, state, gt_on) is True


class SimpleNamespaceGT:
    def __init__(self, pos):
        self.pos = pos


def test_detection_history_resets_on_gate_pass():
    """Bug 5: passing a gate clears the held detection so we don't keep
    steering toward the gate we just passed."""
    from drone_main import AutonomyLoop
    loop = AutonomyLoop(mode="mock", run_id="t_hist", show_view=False)
    loop._current_gate_idx = 0
    loop._last_detection = object()
    loop._detection_age = 3
    loop._on_gate_passed(0.5)
    assert loop._last_detection is None
    assert loop._detection_age == 0


def test_detection_history_frames_is_short():
    """Bug 5: the main-loop stale-hold is short (was 20)."""
    from config.loader import cfg
    assert cfg.perception.detection_history_frames <= 5


def test_vision_update_projects_gate_to_correct_side():
    """Bug 4: a gate pixel to the RIGHT of center (heading north, NED) must pull
    the estimate toward +east, not -east. The old code inverted this."""
    from state.estimator import DroneStateEstimator
    est = DroneStateEstimator()
    est._kf.x[:3, 0] = np.array([0.0, 0.0, 0.0])
    est._att_deg = np.zeros(3)
    w, h = 640, 360
    est.update_from_vision(gate_center_px=(w * 0.75, h / 2.0),
                           distance_m=4.0, frame_w=w, frame_h=h)
    pos = est.get_estimate().pos
    assert pos[0] > 0.0, "forward gate should pull estimate north (+x)"
    assert pos[1] > 0.0, "right-of-center gate should pull estimate east (+y)"


# ---------------------------------------------------------------------------
# Phase 2 upgrade tests
# ---------------------------------------------------------------------------

def test_speed_cap_scales_with_confidence():
    """Upgrade 3: low confidence collapses the cap to recovery speed; high
    confidence yields the full phase ceiling."""
    from drone_main import AutonomyLoop, PHASE_APPROACH
    from config.loader import cfg
    loop = AutonomyLoop(mode="mock", run_id="t_speedcap", show_view=False)
    loop._phase = PHASE_APPROACH
    lo = loop._speed_cap_for_phase(0.0)
    hi = loop._speed_cap_for_phase(1.0)
    assert abs(lo - cfg.drone.recovery_speed_mps) < 1e-6
    assert abs(hi - cfg.drone.approach_speed_mps) < 1e-6
    assert lo < hi


def test_racing_target_bends_toward_next_gate():
    """Upgrade 2: the through-point curves toward the next gate, not straight."""
    from drone_main import AutonomyLoop
    loop = AutonomyLoop(mode="mock", run_id="t_racing", show_view=False)
    loop._current_gate_idx = 0
    gate = np.array([5.0, 0.0, 1.5])
    nxt = np.array([8.0, 4.0, 1.5])      # next gate is off to +y
    loop._course = [gate, nxt]
    state_pos = np.array([4.9, 0.0, 1.5])  # close -> high blend toward next
    target = loop._racing_through_target(state_pos, gate, np.array([1.0, 0.0, 0.0]))
    assert target[1] > 0.0, "racing line should bend toward the next gate (+y)"


def test_racing_target_straight_when_no_next_gate():
    """Upgrade 2: with no next gate, fall back to a straight through-target."""
    from drone_main import AutonomyLoop
    from config.loader import cfg
    loop = AutonomyLoop(mode="mock", run_id="t_racing_last", show_view=False)
    loop._current_gate_idx = 0
    gate = np.array([5.0, 0.0, 1.5])
    loop._course = [gate]
    direction = np.array([1.0, 0.0, 0.0])
    target = loop._racing_through_target(np.array([4.0, 0.0, 1.5]), gate, direction)
    expected = gate + direction * cfg.planning.gate_through_offset_m
    assert np.allclose(target, expected)


def test_recovery_dead_reckons_along_heading():
    """Upgrade 8: a blind advance creeps forward along the last-seen heading
    and holds altitude, rather than chasing a stale absolute position."""
    from planning.recovery import RecoveryBehavior, RecoveryState
    rec = RecoveryBehavior()
    pos = np.array([5.0, 0.0, 1.5])
    # See the gate once facing +x (heading 0) so the heading is stored.
    rec.update(gate_detected=True, gate_confidence=0.9, current_pos=pos,
               last_gate_pos=np.array([10.0, 0.0, 1.5]), heading_rad=0.0)
    # Force into the blind-advance state with enough missed frames to act.
    rec._frames_without_gate = rec._trigger_frames
    rec._state = RecoveryState.ADVANCE_BLIND
    rec._start_t = time.time()

    phase, target, _ = rec.update(gate_detected=False, gate_confidence=0.0,
                                  current_pos=pos, last_gate_pos=None)
    assert phase == "recovery"
    assert target[0] > pos[0], "should creep forward along heading (+x)"
    assert abs(target[2] - pos[2]) < 1e-6, "altitude held during dead-reckon"


def test_recovery_sweep_duration_reduced():
    """Upgrade 8: the yaw-sweep hold was halved."""
    from planning.recovery import RecoveryBehavior
    assert RecoveryBehavior.SWEEP_DURATION_S <= 3.0


def test_vision_only_recovery_holds_position(monkeypatch):
    """Live fix: in vision-only (no course) mode, recovery must hover in place,
    not feed a garbage-EKF position target through the PID (which drifted the
    real drone backward+up out of bounds). Translation must be zeroed."""
    import config.loader as loader
    from planning.recovery import RecoveryState
    monkeypatch.setattr(loader.cfg.sim, "mock_use_course", False)
    monkeypatch.setattr(loader.cfg.sim, "mock_use_ground_truth", False)
    from drone_main import AutonomyLoop

    loop = AutonomyLoop(mode="mock", run_id="t_rec_hold", show_view=False)
    loop._course = None
    monkeypatch.setattr(loop._detector, "detect", lambda img: [])
    # Force the recovery behaviour into an active sweep.
    loop._recovery._frames_without_gate = loop._recovery._trigger_frames + 5
    loop._recovery._state = RecoveryState.YAW_SWEEP
    loop._recovery._start_t = time.time()

    sent = {}
    def cap(vx, vy, vz, yaw_rate=0.0):
        sent.update(vx=vx, vy=vy, vz=vz, yaw=yaw_rate)
    monkeypatch.setattr(loop._sim, "send_velocity_command", cap)
    loop._start_t = time.time() - 10.0   # past the takeoff window

    loop._tick()
    assert loop._phase == "recovery"
    assert sent["vx"] == 0.0 and sent["vy"] == 0.0 and sent["vz"] == 0.0


def test_full_control_cycle():
    """Smoke test: run one tick through all modules without crashing."""
    from drone_sim.interface import MockSimInterface
    from perception.gate_detector import ClassicalGateDetector
    from state.estimator import DroneStateEstimator
    from control.controller import PIDController
    from planning.recovery import RecoveryBehavior

    sim = MockSimInterface()
    sim.connect()
    sim._state.pos = np.array([3.5, 0.0, 1.5])  # near gate 1

    detector = ClassicalGateDetector()
    estimator = DroneStateEstimator()
    controller = PIDController()
    recovery = RecoveryBehavior()

    obs = sim.get_observation()
    estimator.predict(obs.imu.accel, obs.imu.gyro, obs.timestamp)
    state = estimator.get_estimate()

    detections = detector.detect(obs.image)
    gate_detected = bool(detections)

    pos = state.pos
    target = pos + np.array([1.0, 0.0, 0.0])
    ctrl = controller.compute(pos, target, confidence=0.8)

    sim.send_velocity_command(ctrl.vx, ctrl.vy, ctrl.vz, ctrl.yaw_rate)
    sim.disconnect()

    assert ctrl.vx != 0 or ctrl.vy != 0 or ctrl.vz != 0 or True  # just no crash
