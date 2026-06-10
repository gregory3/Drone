# AI Grand Prix — Competition Rules & Technical Spec (distilled)

Sources (fetched 2026-06-10):
- https://www.theaigrandprix.com/previousupdates/ (rules updates, Feb–May 2026)
- VADR-TS-001 (2026-03-09) and VADR-TS-002 (2026-05-08) technical specifications:
  - https://www.theaigrandprix.com/wp-content/uploads/2026/03/260318_Technical_Spec_0001.pdf
  - https://www.theaigrandprix.com/wp-content/uploads/2026/05/260508_Technical_Spec_0002.pdf

## Format & timeline

| Stage | When | Scoring |
|---|---|---|
| Virtual Qualifier 1 | May 2026 → ~end of July | **Completion only** — pass all gates in order. Speed NOT scored. |
| Virtual Qualifier 2 | June 2026 → mid/late July | **Fastest valid time** wins. |
| Physical Qualifier | Sept 2026, SoCal | — |
| Final | Nov 2026, Ohio | $500K pool |

- **Max run duration: 8 minutes per attempt.** Unlimited attempts within each window.
- VQ1 course: <10 gates, standardized appearance, highlighted gates, visual guidance aids on, minimal distractions.
- VQ2 course: <20 gates, guidance aids OFF, lighting changes/obstacles/distractions, lower signal-to-noise.
- Course: start gate → sequential intermediate gates → finish gate. Gates must be passed in correct sequence.

## Disqualification / fair play

- **No human interaction during a submitted timed run** (immediate DQ). Clicking RACE to start is the operator action; everything after must be autonomous.
- No rewriting/altering game files, no color changes, screen tricks, or disabling collision detection.
- Code must remain accessible for review; anti-cheat requires an active internet connection.

## Technical interface (VADR-TS-002, authoritative numbers)

**Simulation:** rigid-body physics @ 120 Hz; local Cartesian frame, no GPS/absolute position. Deterministic course/physics for all teams. Windows 11 only.

**Coordinate frames: NED.** `MAV_FRAME_LOCAL_NED` origin = arming point. `MAV_FRAME_BODY_NED`: X fwd, Y right, Z down. Body→IMU = identity.

**Drone chassis:** 280 × 280 × 160 mm.

**Gate dimensions:** outer boundary **2.7 m × 2.7 m**, inner (flyable) opening **1.5 m × 1.5 m**, depth 0.26 m.
→ The full red structure seen by the detector spans 2.7 m; the safe corridor is the central 1.5 m.

**Camera:** tilted **upwards 20°**, same origin as body frame. Pinhole, no distortion:
640×360 px, [cx,cy]=[320,180], [fx,fy]=[320,320] (⇒ 90° horizontal FOV, ~58.7° vertical — the doc's "VFoV=90°" is inconsistent with fy=320; intrinsics are authoritative). Stream: 30 Hz UDP :5600, chunked JPEG, 24-byte little-endian header (frame_id u32, chunk_id u16, total_chunks u16, jpeg_size u32, payload_size u32, sim_time_ns u64).

**MAVLink:** v2 over UDP. Sim→client: HEARTBEAT, ATTITUDE, HIGHRES_IMU, ODOMETRY, TIMESYNC. Client→sim: SET_POSITION_TARGET_LOCAL_NED, SET_ATTITUDE_TARGET.
- Command rate **< 100 Hz** (we run 50 Hz ✓)
- **Minimum heartbeat rate 2 Hz — a CLIENT responsibility** ("maintain heartbeat messages")

**Runtime:** Python assumed (3.14.2 known-good); external libs and AI coding tools allowed; any framework allowed.

## Compliance status of our stack (audited 2026-06-10)

| Spec item | Our stack | Status |
|---|---|---|
| NED frames | mavlink_adapter handles NED↔ENU | ✓ (verified live 06/02) |
| ACRO control via SET_ATTITUDE_TARGET | control_mode="attitude" default | ✓ (verified live) |
| Camera 20° up-tilt | `camera_tilt_deg: 20`, elevation = +tilt+pitch−α | ✓ |
| Intrinsics fx=fy=320, c=(320,180) | derived from `camera_fov_deg: 90` | ✓ |
| Command rate <100 Hz | `step_hz: 50` | ✓ |
| 30 Hz chunked-JPEG vision | adapter reassembles on :5600 | ✓ (verified live) |
| **Gate size for monocular distance** | `_gate_real_size_m = 1.2` hard-coded | ✗ **spec = 2.7 m outer → distances ~2.25× short. Fix queued (Phase 2).** |
| **Client heartbeat ≥2 Hz** | we never send HEARTBEAT | ✗ **add GCS heartbeat sender (Phase 2).** |
| 8-minute run cap | no mission timer; recovery can loop forever | ✗ **bound recovery/mission to the 8-min budget (Phase 2).** |
| No human interaction mid-run | fully autonomous after RACE click | ✓ |

## Strategy implications

- **VQ1 is pass/fail on completion — submit a conservative, reliable run first.** Speed work targets VQ2.
- Unlimited attempts ⇒ iterate freely, but every submitted run must be hands-off.
- 8-min cap ⇒ a stuck recovery wastes the attempt but costs nothing; recovery should keep trying (smartly, escalating) for the full budget rather than abort.
- Inner opening is 1.5 m vs our 280 mm chassis ⇒ ~0.6 m clearance either side through the center; aim-point accuracy matters more than speed in VQ1.
