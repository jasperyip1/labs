"""
MIT BWSI Autonomous Drone Racing Course - UAV Neo

p_follow — the stripped-to-the-studs baseline line follower.

Deliberately minimal. This is the day-one version that WORKED, plus logging:
  - P control only (no I, no D)
  - roll from the line's horizontal offset (centroid)
  - yaw from the line's angle
  - constant slow forward pitch, constant throttle
  - WHOLE frame (no near-band crop)
  - bright_mask (Value channel)
  - NO variable speed, NO search states, NO gate logic

Do not add anything to this file. Get it flying and logged first. Add features
in SEPARATE files, one at a time, each compared against this baseline's log.

Every frame is written to a CSV. The columns that matter most:
  yaw_cmd_raw   — what the P controller asked for, BEFORE the clamp
  yaw_cmd       — what was actually sent (after clamp to +/-MAX_YAW)
  yaw_rate_meas — what the drone ACTUALLY did, from the IMU (rad/s)
That triplet answers the two-day question: is the commanded yaw reaching the
cap, and is the achieved rate what MAX_YAW_RATE should now allow?

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

# -- Constants --------------------------------------------------------------
V_MIN        = 200        # bright_mask threshold (tune on the bench)
MIN_PIXELS   = 200        # minimum bright pixels to trust a line
IMAGE_CENTER = 320        # 640-wide image -> center column

PITCH   = 0.15           # constant, slow. lower = slower = tighter turns
THROTTLE = 0.0           # hold height; launcher already set it

KP_ROLL = 0.30           # strafe per unit of normalized centroid offset
KP_YAW  = 1.20           # yaw per radian of line angle

MAX_ROLL = 0.30
MAX_YAW  = 1.00

FOLLOW_TIME = 1000000.0
LOG_PATH    = "p_follow_log.csv"


# -- Module state -----------------------------------------------------------
_timer = 0.0
_done  = False
_log   = None


def reset():
    global _timer, _done, _log
    _timer = 0.0
    _done  = False
    _log = open(LOG_PATH, "w")
    _log.write("t,dt,fresh,pixels,angle_deg,centroid,"
               "roll_cmd,yaw_cmd_raw,yaw_cmd,yaw_rate_meas\n")


# -- Perception -------------------------------------------------------------
def find_line(drone):
    """
    Whole-frame line fit on the bright mask.
    Returns (angle_rad, centroid_col, pixel_count) or None.
      angle    0 = line runs straight up the image, + leaning right going up
      centroid mean column of the bright pixels
    """
    image = drone.camera.get_downward_image()
    mask = neo_lab.bright_mask(image, V_MIN)

    pts = np.argwhere(mask).astype(np.float64)   # rows of (row, col)
    count = len(pts)
    if count < MIN_PIXELS:
        return None

    ys, xs = pts[:, 0], pts[:, 1]
    xy = np.column_stack([xs, ys]).astype(np.float32)
    vx, vy, _x0, _y0 = cv2.fitLine(xy, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
    if vy > 0:                       # keep direction pointing up the image
        vx, vy = -vx, -vy

    angle = float(np.arctan2(vx, -vy))
    centroid = float(xs.mean())
    return angle, centroid, count


# -- Control (P only) -------------------------------------------------------
def roll_cmd(centroid):
    error = (centroid - IMAGE_CENTER) / IMAGE_CENTER    # -1 .. +1
    return uav_utils.clamp(KP_ROLL * error, -MAX_ROLL, MAX_ROLL)


def yaw_cmd(angle):
    raw = KP_YAW * angle
    clamped = uav_utils.clamp(raw, -MAX_YAW, MAX_YAW)
    return raw, clamped


# -- Main loop --------------------------------------------------------------
def update(drone):
    global _timer, _done

    if _done:
        return True

    dt = drone.get_delta_time()
    fit = find_line(drone)

    # Achieved yaw rate from the IMU. NOTE: get_angular_velocity() is
    # (pitch, yaw, roll) -> yaw is index [1], not [2].
    yaw_rate_meas = float(drone.physics.get_angular_velocity()[1])

    if fit is None:
        drone.flight.stop()
        _write(_timer, dt, 1, 0, float("nan"), float("nan"),
               0.0, 0.0, 0.0, yaw_rate_meas)
    else:
        angle, centroid, count = fit
        r = roll_cmd(centroid)
        yraw, y = yaw_cmd(angle)
        drone.flight.send_pcmd(PITCH, r, y, THROTTLE)
        _write(_timer, dt, 1, count, np.degrees(angle), centroid,
               r, yraw, y, yaw_rate_meas)

    _timer += dt
    if _timer >= FOLLOW_TIME:
        _done = True
        if _log:
            _log.close()
    return _done


def _write(t, dt, fresh, pixels, angle_deg, centroid,
           r, yraw, y, yaw_rate_meas):
    if _log is None:
        return
    _log.write(f"{t:.3f},{dt:.4f},{fresh},{pixels},"
               f"{angle_deg:.3f},{centroid:.2f},"
               f"{r:.4f},{yraw:.4f},{y:.4f},{yaw_rate_meas:.4f}\n")
    # Also print a compact live line so you can watch it fly.
    print(f"t={t:6.1f} px={pixels:5d} angle={angle_deg:+6.1f} "
          f"cen={centroid:5.0f} roll={r:+.3f} "
          f"yaw={y:+.3f}(raw {yraw:+.3f}) rate={yaw_rate_meas:+.3f}",
          flush=True)


# -- Entry point ------------------------------------------------------------
if __name__ == "__main__":
    _drone = drone_core.create_drone()
    _launcher = neo_lab.Launcher(1.0)

    def start():
        _launcher.reset()
        reset()
        print("p_follow — P-only baseline (roll + yaw, constant slow pitch)")

    def _update():
        if not _launcher.done:
            _launcher.update(_drone)
            return
        if update(_drone):
            _drone.flight.land()

    _drone.set_start_update(start, _update)
    _drone.go(not neo_lab._is_sim(_drone))