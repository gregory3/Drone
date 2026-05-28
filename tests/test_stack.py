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
    from sim.interface import MockSimInterface
    sim = MockSimInterface()
    sim.connect()
    obs = sim.get_observation()
    assert obs.image.shape == (480, 640, 3)
    assert len(obs.rpm) == 4
    assert obs.imu.timestamp > 0
    sim.disconnect()


def test_mock_sim_moves_on_command():
    from sim.interface import MockSimInterface
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
    from sim.interface import MockSimInterface
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
    from sim.interface import make_interface, RealSimInterface
    sim = make_interface(mode="real")
    assert isinstance(sim, RealSimInterface)
    with pytest.raises(NotImplementedError):
        sim.connect()


def test_make_interface_elodin_raises_when_sdk_missing():
    from sim.interface import make_interface, ElodinSimInterface
    sim = make_interface(mode="elodin")
    assert isinstance(sim, ElodinSimInterface)
    with pytest.raises(NotImplementedError):
        sim.connect()


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

    # Draw a synthetic orange rectangle matching our HSV range
    # config gate_hsv_lower=[20,80,80], gate_hsv_upper=[40,255,255]
    # Hue 30 (orange) in OpenCV HSV
    orange_bgr = (0, 165, 255)   # OpenCV BGR orange
    cv2.rectangle(frame, (250, 150), (400, 330), orange_bgr, -1)

    results = det.detect(frame)
    assert len(results) > 0, "Expected at least one detection on orange rect"
    assert results[0].confidence > 0.1


def test_classical_detector_temporal_smoothing_persists_through_short_dropout():
    from perception.gate_detector import make_detector
    import cv2

    det = make_detector("classical")
    frame = np.full((480, 640, 3), 80, dtype=np.uint8)
    orange_bgr = (0, 165, 255)
    cv2.rectangle(frame, (250, 150), (400, 330), orange_bgr, -1)

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


def test_estimator_tracks_commanded_velocity():
    from state.estimator import DroneStateEstimator
    est = DroneStateEstimator()
    est.set_command_velocity(np.array([2.0, 0.0, 0.0]))

    t = time.time()
    for i in range(10):
        est.predict((0, 0, -9.81), (0, 0, 0), t + i * 0.02)

    state = est.get_estimate()
    # Position should have moved in x
    assert state.pos[0] > 0.1


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

    for _ in range(cfg.perception.detection_history_frames + 1):
        phase, _, _ = rec.update(gate_detected=False, gate_confidence=0.0,
                                 current_pos=pos, last_gate_pos=None)

    assert phase == "recovery"


def test_recovery_clears_on_gate_found():
    from planning.recovery import RecoveryBehavior
    from config.loader import cfg
    rec = RecoveryBehavior()
    pos = np.array([5.0, 0.0, 1.5])

    # Trigger recovery
    for _ in range(cfg.perception.detection_history_frames + 2):
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
    for _ in range(cfg.perception.detection_history_frames + 2):
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
    from main import AutonomyLoop, PHASE_COMPLETE

    loop = AutonomyLoop(mode="mock", run_id="test_complete", show_view=False)
    assert loop._course is not None
    loop._current_gate_idx = len(loop._course) - 1
    loop._gates_passed = len(loop._course) - 1

    loop._on_gate_passed(0.1)

    assert loop._phase == PHASE_COMPLETE
    assert loop._gates_passed == len(loop._course)


def test_autonomy_loop_uses_course_gate_when_visible():
    from main import AutonomyLoop

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
    from main import AutonomyLoop
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
    from main import AutonomyLoop

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
    from main import AutonomyLoop

    monkeypatch.setattr(loader.cfg.sim, "mock_use_ground_truth", False)
    loop = AutonomyLoop(mode="mock", run_id="test_realistic", show_view=False)
    assert loop._use_ground_truth is False


def test_autonomy_loop_can_disable_mock_course(monkeypatch):
    import config.loader as loader
    from main import AutonomyLoop

    monkeypatch.setattr(loader.cfg.sim, "mock_use_course", False)
    loop = AutonomyLoop(mode="mock", run_id="test_blind", show_view=False)
    assert loop._use_course is False
    assert loop._course is None


def test_autonomy_loop_persists_last_detection_short_loss(monkeypatch):
    from main import AutonomyLoop
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
    from main import AutonomyLoop

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

def test_full_control_cycle():
    """Smoke test: run one tick through all modules without crashing."""
    from sim.interface import MockSimInterface
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
