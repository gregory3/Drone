"""drone_sim.elodin_solver
Bridge that lets the Elodin AI Grand Prix rig call our autonomy stack via
its `autopilot(SensorUpdate) -> RCCommand` contract.

The rig runs in a Python 3.13 venv under WSL/macOS. The contestant solver
is imported via the `RACE_SOLVER` env var, e.g.

    RACE_SOLVER=drone_sim.elodin_solver elodin run sim/main.py

The rig's own `sim/` package and our `drone_sim/` package have distinct
names so they coexist on `sys.path` without shadowing each other.

For this to work, the rig's Python process must be able to import this
module. That requires our Drone project root on `sys.path`. We resolve that
via the `DRONE_PROJECT_ROOT` env var, defaulting to the canonical Windows
checkout path under `/mnt/c/Users/Fine Finish/G/Drone`.

The bridge keeps long-lived state — perception, EKF, controller, recovery —
so each `autopilot()` call is one tick of the same `AutonomyLoop` we test
with the mock sim. The only thing it can't do is `time.sleep` to pace the
loop; the rig drives that.
"""

from __future__ import annotations
import os
import sys
import time
from pathlib import Path
from typing import Optional


_DEFAULT_DRONE_ROOT = "/mnt/c/Users/Fine Finish/G/Drone"


def _ensure_drone_on_path() -> None:
    drone_root = os.environ.get("DRONE_PROJECT_ROOT", _DEFAULT_DRONE_ROOT)
    if not Path(drone_root).is_dir():
        raise RuntimeError(
            f"DRONE_PROJECT_ROOT={drone_root!r} does not exist. Set the "
            "DRONE_PROJECT_ROOT env var to the absolute path of your Drone "
            "project so the Elodin solver can import the autonomy modules."
        )
    if drone_root not in sys.path:
        sys.path.insert(0, drone_root)


_ensure_drone_on_path()

# Imports below come from the Drone project, made available by the path
# bootstrap above. Keep these imports after `_ensure_drone_on_path()`.
from drone_sim.interface import ElodinSimInterface  # noqa: E402
from drone_sim.competition_types import RCCommand as DroneRCCommand  # noqa: E402

# The rig matches on its OWN RCCommand class via `isinstance` before
# accepting the returned RC. Try to import that class so we can hand back
# the right type; fall back to ours if the rig isn't around (e.g. unit
# tests). See ~/ai-grand-prix/solver/api.py for the upstream definition.
try:
    from solver.api import RCCommand  # type: ignore
except Exception:
    RCCommand = DroneRCCommand  # noqa: F811


def _to_rig_rc(rc: DroneRCCommand) -> "RCCommand":
    """Translate our internal RCCommand into the rig's expected class."""
    if RCCommand is DroneRCCommand:
        return rc
    return RCCommand(
        throttle=int(rc.throttle),
        roll=int(rc.roll),
        pitch=int(rc.pitch),
        yaw=int(rc.yaw),
        arm=int(rc.arm),
        aux2=int(rc.aux2),
        aux3=int(rc.aux3),
        aux4=int(rc.aux4),
    )


class _Bridge:
    """Persistent state for the duration of a sim run.

    The rig calls `autopilot()` at the Betaflight PID rate (1000 Hz). Our
    autonomy logic targets ~50 Hz; running it every rig tick wastes 95 %
    of the budget on duplicate detector / EKF work. We rate-limit:

      - Heavy logic (perception / EKF / controller) runs every N rig ticks
        where N = 1000 / DRONE_CONTROL_HZ (default 50).
      - Every other tick simply re-issues the last RCCommand so Betaflight
        keeps a fresh RC stream.
    """

    def __init__(self) -> None:
        self._adapter: Optional[ElodinSimInterface] = None
        self._loop = None  # type: ignore[assignment]
        self._initialised = False
        self._tick_count = 0
        self._start_wall = time.time()

        rate_hz = float(os.environ.get("DRONE_CONTROL_HZ", "50"))
        self._stride = max(1, int(round(1000.0 / rate_hz)))
        self._cached_rc: Optional[RCCommand] = None

    def _init_once(self) -> None:
        if self._initialised:
            return

        # Lazy-import AutonomyLoop after path setup so circular imports don't
        # bite during module load.
        from drone_main import AutonomyLoop  # noqa: E402  (drone project)

        # Push logs to a WSL-native path. Writing NDJSON to the 9P-mounted
        # /mnt/c path costs ~10ms per write and dominates the tick budget.
        # Default to a fixed location under the user's home so the rig's
        # private-tmp namespace doesn't hide them.
        log_dir = os.environ.get(
            "DRONE_LOG_DIR",
            os.path.expanduser("~/drone_solver_logs"),
        )
        from config.loader import cfg as _cfg
        _cfg.telemetry.log_dir = log_dir

        adapter = ElodinSimInterface(synchronous=True)
        adapter.connect()

        loop = AutonomyLoop(
            mode="mock",
            run_id=os.environ.get("RACE_RUN_ID", "elodin_smoke"),
            show_view=False,
            force_ned=False,
            use_rerun=False,
        )
        loop._sim = adapter

        # Match the rig's race gates as a "course" for the controller.
        # These are EASY_COURSE in the rig's sim/course.py.
        import numpy as np  # noqa: E402
        loop._course = [
            np.array([10.0, 0.0, 1.8]),
            np.array([20.0, 0.0, 1.8]),
            np.array([30.0, 0.0, 1.8]),
        ]
        loop._use_course = True
        # We DON'T have ground truth from the perception layer, but the
        # rig's SensorUpdate carries world_pos. The Elodin adapter exposes
        # it via get_ground_truth(); enabling this lets the EKF/controller
        # use the cheat-pos for the first smoke flight while perception is
        # still wired up against a camera-less rig.
        loop._use_ground_truth = True

        adapter.reset()
        loop._start_t = time.time()
        loop._phase = "search"

        self._adapter = adapter
        self._loop = loop
        self._initialised = True

    def reset(self) -> None:
        """Called by the rig between sim runs."""
        self._adapter = None
        self._loop = None
        self._initialised = False
        self._tick_count = 0
        self._cached_rc = None
        self._start_wall = time.time()

    # Betaflight requires a clean throttle-low transition between disarmed
    # and armed states or it will reject the arm switch and refuse motors.
    # Mirror the baseline solver's arming sequence verbatim.
    _T_DISARMED_END = 0.50
    _T_ARM_IDLE_END = 0.75
    _T_LAND_END     = 14.00

    def step(self, update) -> RCCommand:
        try:
            self._init_once()
        except Exception as exc:
            import traceback
            print(f"[ELODIN_SOLVER] init failed: {exc}", flush=True)
            traceback.print_exc()
            return _to_rig_rc(DroneRCCommand())

        assert self._adapter is not None and self._loop is not None

        # Arming sequence overrides the autopilot for the first 0.75s.
        t = float(update.t)
        if t < self._T_DISARMED_END:
            self._tick_count += 1
            rc = RCCommand(arm=1000, throttle=1000)
            self._cached_rc = rc
            return rc
        if t < self._T_ARM_IDLE_END:
            self._tick_count += 1
            rc = RCCommand(arm=1800, throttle=1000)
            self._cached_rc = rc
            return rc
        if t >= self._T_LAND_END:
            self._tick_count += 1
            rc = RCCommand(arm=1000, throttle=1000)
            self._cached_rc = rc
            return rc

        # Fast path: most ticks just re-issue the last RC command.
        if self._cached_rc is not None and (self._tick_count % self._stride) != 0:
            self._tick_count += 1
            return self._cached_rc

        # Heavy path: one full autonomy tick.
        self._adapter.push_sensor_update(update)
        if update.next_gate_index >= 0:
            self._loop._current_gate_idx = int(update.next_gate_index)

        try:
            self._loop._tick()
        except Exception as exc:
            import traceback
            print(f"[ELODIN_SOLVER] tick error at t={update.t:.3f}: {exc}",
                  flush=True)
            traceback.print_exc()
            return self._cached_rc or _to_rig_rc(DroneRCCommand())

        rc_internal = self._adapter.pop_command()
        rc = _to_rig_rc(rc_internal)
        self._cached_rc = rc
        self._tick_count += 1

        # Heartbeat: surface what we are commanding every second of sim time
        # so the smoke test is observable from the rig log.
        if self._tick_count % (self._stride * 50) == 0:
            try:
                wp = update.world_pos
                print(
                    f"[ELODIN_SOLVER] t={update.t:5.2f}s tick={self._tick_count} "
                    f"pos=({wp[4]:+.2f},{wp[5]:+.2f},{wp[6]:+.2f}) "
                    f"rc=(thr={rc.throttle},rol={rc.roll},pit={rc.pitch},"
                    f"yaw={rc.yaw},arm={rc.arm})",
                    flush=True,
                )
            except Exception:
                pass

        return rc


_BRIDGE = _Bridge()


# ---------------------------------------------------------------------------
# Public API the rig expects
# ---------------------------------------------------------------------------

def reset_state() -> None:
    _BRIDGE.reset()


def autopilot(update) -> RCCommand:
    return _BRIDGE.step(update)
