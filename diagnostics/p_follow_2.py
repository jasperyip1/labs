"""
MIT BWSI Autonomous Drone Racing Course - UAV Neo

p_follow — P-only baseline line follower, tuned for 1-2 m flight height,
with rich in-flight logging so you can read REAL values and correct the
assumed ones.

Control (unchanged from the working baseline):
  - P control only (no I, no D)
  - roll from the line's horizontal offset (centroid)
  - yaw from the line's angle
  - constant slow forward pitch, constant throttle
  - WHOLE frame, bright_mask (Value channel)

Constants set for HIGHER flight (1-2 m). The pixel thresholds are EXTRAPOLATED
from bench data taken at 0-0.5 m, so they are best-guesses, not measurements.
The whole point of the logging below is to replace those guesses with the real
numbers from the first flight:
  - watch the `px` column: MIN_PIXELS should be ~40% of the typical count,
    MAX_PIXELS ~5-8x it. If px reads ~1800 in flight, set MIN_PIXELS ~700.
  - watch `offset` and `angle`: these drive roll and yaw. If they are jumpy,
    that's perception noise from fewer pixels at height, not a gain problem.
  - watch `yaw` vs `rate`: is the command reaching the cap, and what rad/s
    does the drone actually achieve.

A per-frame CSV is written, and a live summary prints every SUMMARY_EVERY sec
so you can read real-vs-assumed values without parsing the CSV mid-session.

Run:
    drone sim p_follow.py
    drone p_follow.py            (real)
"""

import time

import drone_core
import drone_utils as uav_utils
import cv2
import numpy as np

# -- Course setup: makes the shared `neo_lab` helper importable. --
import os as _os, sys as _sys
_d = _os.path.dirname(_os.path.realpath(__file__))
while _os.path.basename(_d) != "labs" and _os.path.dirname(_d) != _d:
    _d = _os.path.dirname(_d)
if _d not in _sys.path:
    _sys.path.insert(0, _d)
import neo_lab

# -- Perception thresholds (EXTRAPOLATED for 1-2 m — verify from the log) ----
V_MIN        = 200        # bright_mask threshold. SOLID: V_p99 was ~230 at
                          # every bench height, so this transfers with height.
MIN_PIXELS   = 2000        # floor. GUESS for 1-2 m (line ~600-1000 px there).
                          # Reset to ~40% of the real typical count once you
                          # see the `px` column in flight.
MAX_PIXELS   = 25000      # ceiling: above this the frame is flooded (phantom).
                          # Lowered from 40000 because counts are far smaller
                          # at height. Raise if real px approaches it.
IMAGE_CENTER = 320        # 640-wide image -> center column

# -- Control ----------------------------------------------------------------
PITCH    = 0.15           # constant, slow. lower = slower = tighter turns
THROTTLE = 0.0            # hold height; launcher already set it

KP_ROLL = 0.30            # strafe per unit of normalized centroid offset
KP_YAW  = 1.20            # yaw per radian of line angle

MAX_ROLL = 0.70
MAX_YAW  = 1.00

# Near +/-90deg the line is nearly horizontal and the fitted angle's sign is
# noise-dominated (wraps +89 <-> -89). Past this limit, clamp the error to the
# boundary instead of chasing the wraparound.
YAW_ANGLE_LIMIT = np.radians(80.0)

FOLLOW_TIME   = 1000000.0
SUMMARY_EVERY = 5.0       # seconds between running-summary prints
LOG_PATH      = "p_follow_log.csv"


# -- Module state -----------------------------------------------------------
_timer = 0.0
_done  = False
_log   = None
_next_summary = 0.0
_stats = None             # accumulates real values for the running summary


def _new_stats():
    return {
        "frames": 0, "seen": 0, "lost_low": 0, "lost_high": 0,
        "px": [], "offset": [], "angle": [],
        "roll": [], "yaw": [], "yaw_raw": [], "rate": [], "height": [],
    }


def reset():
    global _timer, _done, _log, _next_summary, _stats
    _timer = 0.0
    _done  = False
    _next_summary = SUMMARY_EVERY
    _stats = _new_stats()
    _log = open(LOG_PATH, "w")
    _log.write("t,dt,height,pixels,seen,offset,angle_deg,centroid,"
               "roll_cmd,yaw_cmd_raw,yaw_cmd,yaw_rate_meas\n")


# -- Perception -------------------------------------------------------------
def find_line(drone):
    """
    Whole-frame line fit on the bright mask.
    Returns (angle_rad, centroid_col, pixel_count, reject) where reject is
    None if a line was found, or 'low'/'high' if the pixel count was out of
    bounds (so the caller can log WHY the line was dropped).
    """
    image = drone.camera.get_downward_image()
    mask = neo_lab.bright_mask(image, V_MIN)

    pts = np.argwhere(mask).astype(np.float64)   # rows of (row, col)
    count = len(pts)
    if count < MIN_PIXELS:
        return None, None, count, "low"
    if count > MAX_PIXELS:
        return None, None, count, "high"

    ys, xs = pts[:, 0], pts[:, 1]
    xy = np.column_stack([xs, ys]).astype(np.float32)
    vx, vy, _x0, _y0 = cv2.fitLine(xy, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
    if vy > 0:                       # keep direction pointing up the image
        vx, vy = -vx, -vy

    angle = float(np.arctan2(vx, -vy))
    centroid = float(xs.mean())
    return angle, centroid, count, None


# -- Control (P only) -------------------------------------------------------
def roll_cmd(centroid):
    offset = (centroid - IMAGE_CENTER) / IMAGE_CENTER    # -1 .. +1
    return uav_utils.clamp(KP_ROLL * offset, -MAX_ROLL, MAX_ROLL), offset


def yaw_cmd(angle):
    if abs(angle) > YAW_ANGLE_LIMIT:            # guard near-horizontal wraparound
        angle = np.sign(angle) * YAW_ANGLE_LIMIT
    raw = KP_YAW * angle
    clamped = uav_utils.clamp(raw, -MAX_YAW, MAX_YAW)
    return raw, clamped


# -- Main loop --------------------------------------------------------------
def update(drone):
    global _timer, _done, _next_summary

    if _done:
        return True

    dt = drone.get_delta_time()
    angle, centroid, count, reject = find_line(drone)

    height = float(drone.physics.get_altitude())
    # get_angular_velocity() is (pitch, yaw, roll) -> yaw is index [1].
    yaw_rate_meas = float(drone.physics.get_angular_velocity()[1])

    _stats["frames"] += 1
    _stats["px"].append(count)
    _stats["height"].append(height)
    _stats["rate"].append(yaw_rate_meas)

    if reject is not None:
        drone.flight.stop()
        _stats["lost_low" if reject == "low" else "lost_high"] += 1
        _write(_timer, dt, height, count, 0, float("nan"), float("nan"),
               float("nan"), 0.0, 0.0, 0.0, yaw_rate_meas)
    else:
        r, offset = roll_cmd(centroid)
        yraw, y = yaw_cmd(angle)
        drone.flight.send_pcmd(PITCH, r, y, THROTTLE)

        _stats["seen"] += 1
        _stats["offset"].append(offset)
        _stats["angle"].append(np.degrees(angle))
        _stats["roll"].append(r)
        _stats["yaw"].append(y)
        _stats["yaw_raw"].append(yraw)

        _write(_timer, dt, height, count, 1, offset, np.degrees(angle),
               centroid, r, yraw, y, yaw_rate_meas)

    _timer += dt
    if _timer >= _next_summary:
        _next_summary += SUMMARY_EVERY
        _print_summary()

    if _timer >= FOLLOW_TIME:
        _done = True
        if _log:
            _log.close()
    return _done


def _write(t, dt, height, pixels, seen, offset, angle_deg, centroid,
           r, yraw, y, yaw_rate_meas):
    if _log is None:
        return
    _log.write(f"{t:.3f},{dt:.4f},{height:.3f},{pixels},{seen},"
               f"{offset:.4f},{angle_deg:.3f},{centroid:.2f},"
               f"{r:.4f},{yraw:.4f},{y:.4f},{yaw_rate_meas:.4f}\n")
    # No per-frame print — the CSV has every frame, and the running summary
    # (every SUMMARY_EVERY sec) is what you read live.


def _print_summary():
    """Running snapshot of REAL values, to compare against the assumed ones."""
    s = _stats
    if s["frames"] == 0:
        return

    def stat(key):
        vals = [v for v in s[key] if v == v]   # drop NaN
        if not vals:
            return "--"
        return f"{min(vals):+.0f}/{np.median(vals):+.0f}/{max(vals):+.0f}"

    px = s["px"]
    seen_pct = 100.0 * s["seen"] / s["frames"]
    print("  " + "-" * 68)
    print(f"  SUMMARY @ t={_timer:.0f}s   line seen {seen_pct:.0f}% of frames "
          f"(lost: {s['lost_low']} low-px, {s['lost_high']} high-px)")
    print(f"    px (min/med/max)   {min(px)}/{int(np.median(px))}/{max(px)}"
          f"   [MIN_PIXELS={MIN_PIXELS}, MAX_PIXELS={MAX_PIXELS}]")
    if px:
        med = np.median([p for p in px if p >= MIN_PIXELS] or [0])
        if med:
            print(f"      -> suggested MIN_PIXELS ~ {int(med * 0.4)} "
                  f"(40% of typical seen count)")
    print(f"    height   min/med/max   {stat('height')} m")
    print(f"    offset (min/med/max)   {stat('offset')}   [-1..+1, 0=centered]")
    print(f"    angle  (min/med/max)   {stat('angle')} deg")
    print(f"    roll   (min/med/max)   {stat('roll')}   [cap +/-{MAX_ROLL}]")
    print(f"    yaw    (min/med/max)   {stat('yaw')}   [cap +/-{MAX_YAW}]")
    print(f"    yaw_rate achieved      {stat('rate')} rad/s")
    print("  " + "-" * 68, flush=True)


# -- Entry point ------------------------------------------------------------
if __name__ == "__main__":
    _drone = drone_core.create_drone()
    _launcher = neo_lab.Launcher(0.7)          # fly higher: 1.5 m

    def start():
        _launcher.reset()
        reset()
        print("p_follow — P-only baseline, 1-2 m flight, verbose logging")

    def _update():
        if not _launcher.done:
            _launcher.update(_drone)
            return
        if update(_drone):
            _drone.flight.land()

    _drone.set_start_update(start, _update)
    _drone.go(not neo_lab._is_sim(_drone))