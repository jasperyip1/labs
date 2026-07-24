"""
MIT BWSI Autonomous Drone Racing Course - UAV Neo
GNU General Public License v3.0

Week 2/3 Lab — Metric Look-Ahead ("Pure Pursuit") Line Follower
An aggressive, low-tuning rewrite of velocity_line_follow.py that tracks sharp turns and
is tuned in real, physical units (m/s, rad/s, meters).

Why this beats a single global line fit
----------------------------------------
velocity_line_follow.py fits ONE straight line (x = m*y + b) to every bright pixel and
steers off that slope. On a curve or a corner the single slope is a blend of the whole
line, so the yaw command is wrong exactly when it matters, and "curviness" only slows the
drone instead of turning it.

Instead this splits the image into horizontal BANDS and takes the bright-pixel centroid of
each band, tracing the line even as it bends. Then it works in METERS on the ground plane,
projecting pixels with the pinhole model (the downward camera looks straight down, so a
pixel offset maps to a ground offset of pixel_offset / FOCAL_PX * altitude):

  * TOP band centroid  -> a look-ahead "carrot" on the ground AHEAD of the drone (top of the
                          downward frame is forward). Steer YAW RATE by the true heading angle
                          to that carrot -> the drone turns toward where the line is GOING,
                          before the body gets there. This is what tracks sharp turns.
  * CENTER band centroid -> where the line sits right now under the drone. A light STRAFE
                            keeps the body over the line.

So the top edge is held at top-middle (via yaw) and the body stays centered (via strafe) --
your description, realized in meters. Forward speed eases off automatically as the heading
angle grows, so there is no separate curvature gain to hand-tune.

Because the gains are physical (rad/s per rad, m/s, m), there is very little to tune and it
transfers between sim and real: start with K_PSI (steer strength) and V_CRUISE (speed).

Command path: this uses drone.flight.send_body_velocity(v_forward, v_right, v_up, yaw_rate),
the SI body-velocity setpoint -- true m/s and rad/s, published straight to the flight
controller (no [-1,1] normalization, unlike send_pcmd/send_velocity).
"""

import math

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

# -- Perception -------------------------------------------------------------
V_MIN         = 200      # brightness threshold for the glowing line (bright_mask)
NUM_BANDS     = 6        # horizontal slices; the top slice is the look-ahead point
BAND_MIN_PX   = 25       # bright pixels a band needs to count as "line here"
MIN_TOTAL_PX  = 120      # below this total, treat the line as lost

# -- Camera / ground projection (pinhole, downward camera) -------------------
FOCAL_PX      = 320.0    # camera focal length in pixels (approx; matches module5 distance est.)
CX            = 320.0    # principal point x (640-wide image)
CY            = 240.0    # principal point y (480-tall image)
MIN_AGL_M     = 0.30     # floor on altitude used for the pixel->meter scale (avoids blow-up low down)

# -- Steering (look-ahead yaw + light strafe), physical units ----------------
K_PSI         = 2.0      # yaw rate (rad/s) per rad of heading error to the carrot  (TUNE FIRST)
YAW_RATE_MAX  = 2.0      # rad/s cap on yaw
PSI_SLOW      = 0.70     # heading error (rad, ~40 deg) at which forward speed hits its minimum
K_LAT         = 1.2      # strafe (m/s) per meter of lateral offset under the drone
V_STRAFE_MAX  = 0.40     # m/s cap on strafe

# -- Forward speed (auto-eases in turns; no separate curvature gain) ---------
V_CRUISE      = 0.6      # m/s on straights
V_FWD_MIN     = 0.20     # m/s held through the sharpest turn

# -- Altitude & run ----------------------------------------------------------
TARGET_HEIGHT  = 0.75        # meters above launch ground
FOLLOW_TIME    = 1000000.0   # seconds to follow before landing
LOST_YAW_DECAY = 0.6         # keep turning toward the last-seen side while the line is lost

# -- Module-level state -----------------------------------------------------
_timer    = 0.0
_done     = False
_last_yaw = 0.0          # remembered yaw rate, so a briefly-lost line keeps turning the right way


def reset():
    global _timer, _done, _last_yaw
    _timer    = 0.0
    _done     = False
    _last_yaw = 0.0


# -- Trace the line as band centroids ---------------------------------------
def line_bands(mask):
    """Return (points, total, h, w): points is a list of (row, col) bright-pixel centroids,
    one per band with enough line in it, ordered top (ahead) -> bottom; total is the bright
    pixel count across all bands."""
    h, w = mask.shape
    band_h = max(1, h // NUM_BANDS)
    points = []
    total = 0
    for r0 in range(0, h, band_h):
        band = mask[r0:r0 + band_h]
        rc = np.argwhere(band)               # (row_in_band, col) of bright pixels
        n = len(rc)
        total += n
        if n >= BAND_MIN_PX:
            points.append((r0 + rc[:, 0].mean(), rc[:, 1].mean()))
    return points, total, h, w


def _ground_offset(row, col, meters_per_px):
    """Pinhole projection of an image point to a ground offset from the point directly below
    the drone: returns (forward_m, right_m). Top-of-frame is forward; right-of-frame is right."""
    forward_m = (CY - row) * meters_per_px
    right_m = (col - CX) * meters_per_px
    return forward_m, right_m


# -- Main loop --------------------------------------------------------------
def update(drone):
    global _timer, _done, _last_yaw
    if _done:
        return True

    dt   = drone.get_delta_time()
    v_up = neo_lab.altitude_hold_velocity(drone, TARGET_HEIGHT)   # m/s, holds height on sim & real

    frame = drone.camera.get_downward_image()
    mask  = neo_lab.bright_mask(frame, V_MIN)
    points, total, _h, _w = line_bands(mask)

    if not points or total < MIN_TOTAL_PX:
        # Line lost: keep yawing toward where it went (decayed), stop translating, hold height,
        # so the drone rotates to reacquire instead of flying blindly off the line.
        _last_yaw *= LOST_YAW_DECAY
        drone.flight.send_body_velocity(0.0, 0.0, v_up, _last_yaw)
    else:
        agl = max(neo_lab.height(drone), MIN_AGL_M)
        meters_per_px = agl / FOCAL_PX

        # Carrot = ground point under the top band (ahead of the drone).
        fwd_L, right_L = _ground_offset(points[0][0], points[0][1], meters_per_px)
        # Body point = band nearest the frame center (directly under the drone).
        near = min(points, key=lambda p: abs(p[0] - CY))
        _, near_right = _ground_offset(near[0], near[1], meters_per_px)

        # Yaw: true heading angle to the carrot -> steer toward the line's future direction.
        psi_err = math.atan2(right_L, fwd_L)                       # rad; + = carrot to the right
        yaw_rate = uav_utils.clamp(K_PSI * psi_err, -YAW_RATE_MAX, YAW_RATE_MAX)

        # Forward: full cruise when aimed straight, easing to V_FWD_MIN as the turn sharpens.
        straightness = uav_utils.clamp(1.0 - abs(psi_err) / PSI_SLOW, 0.0, 1.0)
        v_forward = V_FWD_MIN + (V_CRUISE - V_FWD_MIN) * straightness

        # Strafe: center the body over the line (residual lateral offset, in meters).
        v_right = uav_utils.clamp(K_LAT * near_right, -V_STRAFE_MAX, V_STRAFE_MAX)

        _last_yaw = yaw_rate
        drone.flight.send_body_velocity(v_forward, v_right, v_up, yaw_rate)
        print(f"carrot fwd={fwd_L:+.2f}m right={right_L:+.2f}m psi={math.degrees(psi_err):+5.1f}deg "
              f"| v_fwd={v_forward:.2f} v_right={v_right:+.2f} yaw={yaw_rate:+.2f}rad/s")

    _timer += dt
    if _timer >= FOLLOW_TIME:
        _done = True
    return _done


if __name__ == "__main__":
    _drone = drone_core.create_drone()
    _launcher = neo_lab.Launcher(TARGET_HEIGHT)

    def start():
        _launcher.reset()
        reset()
        print("Metric look-ahead pursuit line follower")

    def _update():
        if not _launcher.done:        # arm + climb to target height first
            _launcher.update(_drone)
            return
        if update(_drone):
            _drone.flight.land()

    _drone.set_start_update(start, _update)
    _drone.go(not neo_lab._is_sim(_drone))
