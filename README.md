# AI Grand Prix — Autonomy Stack

Software-only autonomous drone racing for the [AI Grand Prix](https://www.theaigrandprix.com/).

## Quick Start

```bash
# Install dependencies
python -m pip install --user -r requirements.txt

# Run mock simulation (no hardware needed)
python main.py

# Run through N gates then stop
python main.py --gates 3

# Run with live simulator view
python main.py --gates 3 --view  # press ESC to exit

# If GUI is unavailable, the simulator will record view frames to logs/<run_id>/view.mp4 or logs/<run_id>/view_frames

# Replay last run
python main.py --replay

# Or replay directly with plot
python -m telemetry.replay logs/<run_id> --plot

# Run tests
python -m pytest tests/ -v
├── state/
│   ├── __init__.py
│   └── estimator.py       # EKF fusing IMU + vision
├── planning/
│   ├── __init__.py
│   └── recovery.py        # Lost-gate recovery behavior
├── control/
│   ├── __init__.py
│   └── controller.py      # PID velocity controller
├── telemetry/
│   ├── __init__.py
│   ├── logger.py          # FlightLogger → logs/<run_id>/flight.ndjson
│   └── replay.py          # FlightReplay CLI + plot tool
├── main.py                # AutonomyLoop — orchestrates everything
├── requirements.txt       # Python dependencies for local setup
└── tests/
    └── test_stack.py      # 18 unit + integration tests
```

## Competition Timeline

| Phase | Window | Target |
|---|---|---|
| Virtual Round 1 | Now – July | Gate detection working, clean course completion |
| Virtual Round 2 | ~July cutoff | ONNX detector, Round 2 realistic environment |
| Physical Qualifier | Sept 2026, SoCal | sim-to-real transfer, hardware tuning |
| Finals | Nov 2026, Ohio | Competitive race speed, RL optimization |

## Development Workflow

1. **Make a change** to any module
2. **Run** `python -m pytest test_stack.py -v` — must stay green
3. **Fly** `python main.py --gates 5`
4. **Replay** `python -m telemetry.replay logs/<last_run> --plot`
5. Look at the confidence + phase plot and understand *why* something failed
6. Fix the right thing

> Build so that every run produces a structured log you can scrub through frame by frame. That's the competitive asset.

## Swapping to the Real DCL Simulator

Once the official SDK arrives:
1. Implement `RealSimInterface(SimInterface)` in `sim/interface.py`
2. Set `sim.interface.mode = "real"` in `config/settings.yaml` (or `--mode real`)
3. All other modules stay identical

## Switching to ONNX Detector (Round 2)

1. Export/download a YOLO gate detector to `perception/models/gate_detector.onnx`
2. Change `config.perception.detector_backend` to `"onnx"` in `settings.yaml`
3. Run `python3 main.py` — detector swaps automatically
