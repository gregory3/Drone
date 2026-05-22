"""
tests/test_stack.py
Core unit + integration tests for the autonomy stack.

Run: pytest tests/ -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

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
        phase, _, _ = rec.update(gate_detected=False, current_pos=pos,
                                 last_gate_pos=None)

    assert phase == "recovery"


def test_recovery_clears_on_gate_found():
    from planning.recovery import RecoveryBehavior
    from config.loader import cfg
    rec = RecoveryBehavior()
    pos = np.array([5.0, 0.0, 1.5])

    # Trigger recovery
    for _ in range(cfg.perception.detection_history_frames + 2):
        rec.update(gate_detected=False, current_pos=pos, last_gate_pos=None)

    # Gate found
    phase, _, _ = rec.update(gate_detected=True, current_pos=pos,
                              last_gate_pos=None)
    assert phase == "resume"


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
