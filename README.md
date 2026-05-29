# AI Grand Prix — Autonomy Stack

Software-only autonomous drone racing for the [AI Grand Prix](https://www.theaigrandprix.com/).

## Quick Start

```bash
# Install dependencies
python -m pip install --user -r requirements.txt

# Run mock simulation (no hardware needed)
python drone_main.py

# Run through N gates then stop
python drone_main.py --gates 3

# Run with live simulator view
python drone_main.py --gates 3 --view  # press ESC to exit

# To test the perception + estimator pipeline without perfect ground truth:
python drone_main.py --gates 3 --view --realistic

# To test the mock sim in a blind mode that ignores gate world coordinates:
python drone_main.py --gates 3 --view --realistic --blind

# If GUI is unavailable, the simulator will record view frames to logs/<run_id>/view.mp4 or logs/<run_id>/view_frames

# Run in real simulator mode (placeholder stub until SDK integration)
python drone_main.py --mode real

# Use the Elodin practice rig (Betaflight SITL + rolling-shutter camera)

`ElodinSimInterface` in `sim/interface.py` adapts the rig's
`autopilot(SensorUpdate) -> RCCommand` callback into the polling
`SimInterface` shape the rest of the stack uses.

### Install the rig

The `elodin` SDK supports **macOS** and **glibc >= 2.35 Linux** only. Pick
the path that matches your host:

**Windows (current host):**
```powershell
# One-time: enable WSL with Ubuntu 22.04+
wsl --install -d Ubuntu-22.04
# Then inside WSL:
git clone https://github.com/elodin-sys/ai-grand-prix
cd ai-grand-prix
bash scripts/install_elodin.sh
uv venv --python 3.13 && source .venv/bin/activate && uv sync
git submodule update --init --recursive --depth 1 betaflight
bash scripts/build_betaflight.sh
```

**macOS (future host):**
```bash
git clone https://github.com/elodin-sys/ai-grand-prix
cd ai-grand-prix
bash scripts/install_elodin.sh
uv venv --python 3.13 && source .venv/bin/activate && uv sync
git submodule update --init --recursive --depth 1 betaflight
bash scripts/build_betaflight.sh
```

### Run our stack as the rig's solver

The rig invokes a Python module named by the `RACE_SOLVER` env var as its
autopilot. Our bridge module at [drone_sim/elodin_solver.py](drone_sim/elodin_solver.py)
exposes `autopilot(SensorUpdate) -> RCCommand` and re-uses the same
`AutonomyLoop` we test against the mock.

One-time inside the rig's venv (WSL `~/ai-grand-prix`):

```bash
source .venv/bin/activate
uv pip install filterpy matplotlib pyyaml rerun-sdk
```

Then run, pointing at our Drone project on `/mnt/c/...`:

```bash
cd ~/ai-grand-prix && source .venv/bin/activate
PYTHONPATH="/mnt/c/Users/<You>/G/Drone" \
DRONE_PROJECT_ROOT="/mnt/c/Users/<You>/G/Drone" \
RACE_SOLVER=drone_sim.elodin_solver \
elodin run sim/main.py
```

Tunables exposed to the bridge:
- `DRONE_CONTROL_HZ` (default `50`) — heavy-tick rate. The rig calls
    `autopilot()` at 1000 Hz; we run our perception/EKF/controller on
    1/N of those ticks and cache the resulting `RCCommand`.
- `DRONE_LOG_DIR` (default `~/drone_solver_logs`) — where the
    `FlightLogger` writes NDJSON. Keep it in WSL-native filesystem;
    writing to `/mnt/c` via 9P costs ~10 ms per write.
- `RACE_RUN_ID` (default `elodin_smoke`) — name of the run.

Notes:
- The Python code is OS-agnostic — everything except the `elodin`
    runtime install runs natively on Windows.
- `--force-ned` wraps the adapter so the stack sees NED instead of the
    rig's native ENU. Leave it off until the official DCL sim publishes a
    confirmed frame convention.
- The bridge mirrors the rig's arming sequence verbatim: throttle 1000 /
    arm 1000 for t < 0.5 s, then arm transitions to 1800 with throttle
    still at 1000 until t = 0.75 s, then the autonomy loop takes over.
    Betaflight refuses to arm if throttle is non-minimum during the
    transition.

## Rerun telemetry (optional)

Rerun provides powerful multi-modal visualization and time-indexed querying
for frames, IMU traces, poses and annotations. Install the optional SDK:

```bash
pip install rerun-sdk
```

Then enable streaming when running the stack:

```bash
python drone_main.py --gates 3 --rerun
```

The system will continue writing NDJSON logs locally; Rerun is an additional
best-effort streaming sink for quicker debugging and collaboration.

# Replay last run
python drone_main.py --replay

# Export a completed run for dataset analysis
python drone_main.py --export <run_id>

# Analyze an exported run
python -m telemetry.analyze logs/<run_id>

# Augment an exported run for training/feature extraction
python -m telemetry.analyze logs/<run_id> --augment --augment-out logs/<run_id>/augmented

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
3. **Fly** `python drone_main.py --gates 5`
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
