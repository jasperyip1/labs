"""
MIT BWSI Autonomous Drone Racing Course - UAV Neo
GNU General Public License v3.0

Week 2/3 Lab — Path-Tracing Look-Ahead ("Pure Pursuit") Line Follower
An aggressive, low-tuning rewrite of velocity_line_follow.py that follows the line as a
connected PATH, so it handles sharp bends and hairpins (line enters and exits the bottom of
the frame) instead of getting confused by them. Tuned in physical units (m/s, rad/s, meters).

Why per-band centroids fail on a hairpin
----------------------------------------
The simpler approach (fit one line, or take a mean column per horizontal band) assumes each
band holds ONE piece of line. On a U-turn the lower bands cross BOTH legs, so the mean column
lands in the empty gap between them ("phantom center"), and the top band sits on the apex, so
the look-ahead shows ~zero heading error and the drone drives straight over the bend.

What this does instead: TRACE the line as a path
------------------------------------------------
1. Seed at the line cluster nearest bottom-center (the leg the drone is currently over).
2. Crawl along the line's local TANGENT with a small sliding window: step forward, take the
   window's bright centroid (weighted toward the predicted point so it stays on the current
   leg), update the direction from the last two points, repeat. Because it follows the tangent
   it ROUNDS the apex and continues down the other leg — it doesn't jump the gap or average
   the two legs together.
3. The carrot is the point a fixed arc-length AHEAD of the drone (measured from the nadir, the
   image center, since the bottom of the frame is actually BEHIND a downward camera) along that
   traced path. On a straight line the carrot is dead ahead (~0 heading error); at a hairpin it
   is partway around the bend, so the heading error grows and the drone turns to follow it.

Control (unchanged, metric): project the carrot to a ground point with the pinhole model
(downward camera, offset = pixel_offset / FOCAL_PX * altitude), steer YAW RATE by the true
heading angle to it, ease FORWARD speed as the angle grows, and on the tightest bends PIVOT
IN PLACE (forward -> 0, yaw dominates) to swing around the hairpin. A light STRAFE keeps the
body over the line. Gains are physical, so there is little to tune (start with K_PSI, V_CRUISE).

Command path: drone.flight.send_body_velocity(v_forward, v_right, v_up, yaw_rate) — the SI
body-velocity setpoint (true m/s and rad/s, no [-1,1] normalization).
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

# -- Perception: brightness + path tracer -----------------------------------
V_MIN         = 200      # brightness threshold for the glowing line (bright_mask)
MIN_TOTAL_PX  = 120      # below this many bright pixels, treat the line as lost

SEED_BAND_FRAC = 0.18    # bottom fraction of the frame used to seed the trace
SEED_MIN_PX    = 20      # bright pixels needed in the seed band
SEED_RADIUS    = 40      # px around the anchor to average into the seed point

STEP          = 22       # px advanced along the tangent each crawl iteration
WIN           = 34       # half-size (px) of the sliding search window
WIN_MIN_PX    = 12       # bright pixels a window needs to continue the crawl
MAX_STEPS     = 22       # cap on traced path length
DIR_SMOOTH    = 0.6      # 0..1 how fast the tangent turns (higher rounds sharper bends)
LOOKAHEAD_M   = 0.35     # meters of arc-length AHEAD of the drone (nadir) to place the carrot

# -- Camera / ground projection (pinhole, downward camera) -------------------
FOCAL_PX      = 320.0    # camera focal length in pixels (approx; matches module5 distance est.)
CX            = 320.0    # principal point x (640-wide image)
CY            = 240.0    # principal point y (480-tall image)
MIN_AGL_M     = 0.30     # floor on altitude used for the pixel->meter scale

# -- Steering (look-ahead yaw + light strafe), physical units ----------------
K_PSI         = 2.0      # yaw rate (rad/s) per rad of heading error to the carrot  (TUNE FIRST)
YAW_RATE_MAX  = 2.0      # rad/s cap on yaw
PSI_SLOW      = 0.70     # heading error (rad, ~40 deg) at which forward hits its minimum
PIVOT_PSI     = 1.05     # heading error (rad, ~60 deg) above which the drone pivots in place
K_LAT         = 1.2      # strafe (m/s) per meter of lateral offset under the drone
V_STRAFE_MAX  = 0.40     # m/s cap on strafe

# -- Forward speed (auto-eases in turns; no separate curvature gain) ---------
V_CRUISE      = 0.6      # m/s on straights
V_FWD_MIN     = 0.20     # m/s held through a moderate turn (0 once pivoting)

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


# -- Path tracer ------------------------------------------------------------
def _unit(vec):
    n = float(np.linalg.norm(vec))
    return vec / n if n > 1e-6 else vec


def find_seed(mask):
    """The line point nearest bottom-center — where the drone currently sits over the line.
    Returns (row, col) or None."""
    h, w = mask.shape
    r0 = int(h * (1.0 - SEED_BAND_FRAC))
    band = np.argwhere(mask[r0:h])
    if len(band) >= SEED_MIN_PX:
        gr = band[:, 0] + r0
        gc = band[:, 1]
    else:
        allpx = np.argwhere(mask)                 # fallback: search the whole frame
        if len(allpx) < SEED_MIN_PX:
            return None
        gr, gc = allpx[:, 0], allpx[:, 1]
    # Anchor on the bright pixel closest to bottom-center, then average its local cluster
    # (so the seed sits on one leg, not between two).
    i = np.argmin((gr - (h - 1)) ** 2 + (gc - CX) ** 2)
    near = (np.abs(gc - gc[i]) < SEED_RADIUS) & (np.abs(gr - gr[i]) < SEED_RADIUS)
    return (float(gr[near].mean()), float(gc[near].mean()))


def trace_line(mask):
    """Crawl along the line from the seed, following its local tangent, and return the traced
    path as a list of (row, col) points ordered from the drone outward. Rounds sharp bends and
    hairpins because the window follows the tangent instead of averaging every band."""
    h, w = mask.shape
    seed = find_seed(mask)
    if seed is None:
        return []
    pos = np.array(seed, dtype=float)
    direction = np.array([-1.0, 0.0])             # start heading up-frame (ahead of the drone)
    path = [(pos[0], pos[1])]
    for _ in range(MAX_STEPS):
        c = pos + STEP * direction                # predicted next point along the tangent
        r0, r1 = int(max(c[0] - WIN, 0)), int(min(c[0] + WIN, h))
        k0, k1 = int(max(c[1] - WIN, 0)), int(min(c[1] + WIN, w))
        if r1 <= r0 or k1 <= k0:
            break
        pts = np.argwhere(mask[r0:r1, k0:k1])
        if len(pts) < WIN_MIN_PX:
            break
        gr = pts[:, 0] + r0
        gc = pts[:, 1] + k0
        # Weight toward the predicted point c so the crawl locks onto the CONTINUING leg and
        # does not average in the other leg of a tight bend.
        wgt = np.exp(-((gr - c[0]) ** 2 + (gc - c[1]) ** 2) / (2.0 * (WIN * 0.5) ** 2))
        wsum = wgt.sum()
        if wsum < 1e-6:
            break
        new = np.array([(gr * wgt).sum() / wsum, (gc * wgt).sum() / wsum])
        step_dir = _unit(new - pos)
        if float(np.linalg.norm(new - pos)) < 1e-3:
            break
        direction = _unit((1.0 - DIR_SMOOTH) * direction + DIR_SMOOTH * step_dir)
        pos = new
        path.append((pos[0], pos[1]))
        if pos[0] <= 1 or pos[1] <= 1 or pos[0] >= h - 2 or pos[1] >= w - 2:
            break
    return path


def look_ahead(path, meters_per_px):
    """Return (carrot, near). `near` is the traced point directly under the drone (nearest the
    image center / nadir) — the line's position beneath the drone. `carrot` is the point
    LOOKAHEAD_M metres of arc-length AHEAD of `near` along the traced line, so it rounds bends
    and hairpins. Anchoring at the nadir (not the bottom-of-frame seed, which is BEHIND the
    drone) is what keeps a straight line reading as ~0 heading error. Falls back to the
    farthest-ahead traced point when the path does not reach the full look-ahead."""
    ni = min(range(len(path)), key=lambda i: abs(path[i][0] - CY))
    look_px = LOOKAHEAD_M / meters_per_px
    acc = 0.0
    for i in range(ni + 1, len(path)):
        acc += math.hypot(path[i][0] - path[i - 1][0], path[i][1] - path[i - 1][1])
        if acc >= look_px:
            return path[i], path[ni]
    return path[-1], path[ni]


def _ground_offset(row, col, meters_per_px):
    """Pinhole projection of an image point to a ground offset from directly below the drone:
    (forward_m, right_m). Top-of-frame is forward; right-of-frame is right."""
    return (CY - row) * meters_per_px, (col - CX) * meters_per_px


# -- Main loop --------------------------------------------------------------
def update(drone):
    global _timer, _done, _last_yaw
    if _done:
        return True

    dt   = drone.get_delta_time()
    v_up = neo_lab.altitude_hold_velocity(drone, TARGET_HEIGHT)   # m/s, holds height on sim & real

    frame = drone.camera.get_downward_image()
    mask  = neo_lab.bright_mask(frame, V_MIN)
    path  = trace_line(mask)
    agl   = max(neo_lab.height(drone), MIN_AGL_M)
    mpp   = agl / FOCAL_PX

    if np.count_nonzero(mask) < MIN_TOTAL_PX or len(path) < 2:
        # Lost, or too little line to know its direction: hold height, decay the last yaw so the
        # drone keeps rotating toward where the line went, and strafe to sit over the seed if any.
        _last_yaw *= LOST_YAW_DECAY
        v_right = 0.0
        if path:
            _, near_right = _ground_offset(path[0][0], path[0][1], mpp)
            v_right = uav_utils.clamp(K_LAT * near_right, -V_STRAFE_MAX, V_STRAFE_MAX)
        drone.flight.send_body_velocity(0.0, v_right, v_up, _last_yaw)
    else:
        carrot, near = look_ahead(path, mpp)                     # carrot ahead of nadir; near = under drone
        fwd_L, right_L = _ground_offset(carrot[0], carrot[1], mpp)
        _, near_right  = _ground_offset(near[0], near[1], mpp)

        # Yaw: true heading angle to the carrot along the traced path (rounds bends/hairpins).
        psi_err = math.atan2(right_L, fwd_L)                       # rad; + = carrot to the right
        yaw_rate = uav_utils.clamp(K_PSI * psi_err, -YAW_RATE_MAX, YAW_RATE_MAX)

        # Forward: cruise when aimed straight, ease to V_FWD_MIN as the turn sharpens, and pivot
        # in place (forward -> 0) once the required turn is very tight, to swing around the bend.
        if abs(psi_err) > PIVOT_PSI:
            v_forward = 0.0
        else:
            straightness = uav_utils.clamp(1.0 - abs(psi_err) / PSI_SLOW, 0.0, 1.0)
            v_forward = V_FWD_MIN + (V_CRUISE - V_FWD_MIN) * straightness

        # Strafe: center the body over the line (residual lateral offset, in meters).
        v_right = uav_utils.clamp(K_LAT * near_right, -V_STRAFE_MAX, V_STRAFE_MAX)

        _last_yaw = yaw_rate
        drone.flight.send_body_velocity(v_forward, v_right, v_up, yaw_rate)
        print(f"path={len(path):2d} carrot fwd={fwd_L:+.2f}m right={right_L:+.2f}m "
              f"psi={math.degrees(psi_err):+6.1f}deg | v_fwd={v_forward:.2f} "
              f"v_right={v_right:+.2f} yaw={yaw_rate:+.2f}rad/s{'  PIVOT' if v_forward == 0.0 else ''}")

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
        print("Path-tracing look-ahead pursuit line follower")

    def _update():
        if not _launcher.done:        # arm + climb to target height first
            _launcher.update(_drone)
            return
        if update(_drone):
            _drone.flight.land()

    _drone.set_start_update(start, _update)
    _drone.go(not neo_lab._is_sim(_drone))
