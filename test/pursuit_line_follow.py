"""
MIT BWSI Autonomous Drone Racing Course - UAV Neo
GNU General Public License v3.0

Week 2/3 Lab — Path-Tracing Look-Ahead ("Pure Pursuit") Line Follower
An aggressive, low-tuning rewrite of velocity_line_follow.py that follows the line as a
connected PATH, so it handles sharp bends and hairpins (line enters and exits the bottom of
the frame) instead of getting confused by them. Tuned in physical units (m/s, rad/s, meters).

Two things this has to get right
--------------------------------
1. CENTERLINE, not one edge. A thick line (or a line whose fill isn't bright, so only its two
   edges show up in the mask) must be followed down its MIDDLE. A least-squares line fit does
   this for free (the fit lands between the two edges); a naive path tracer instead locks onto
   ONE edge and, when both edges are in view (e.g. at altitude), flips between them and weaves.
   So at every step this samples the line's cross-section PERPENDICULAR to its own direction and
   takes the mean -> the centerline, averaging the two edges together.
2. HAIRPINS. A single line fit cannot represent a line that doubles back (enters and exits the
   bottom). So instead of fitting once, this TRACES the line: seed at the line under the drone,
   then crawl along the local tangent, re-centering on the cross-section each step. Because the
   crawl follows the tangent it rounds the apex and continues down the other leg; because the
   cross-section window is local, it averages the line's own width but NOT a far-away leg.

The carrot is the point a fixed arc-length AHEAD of the drone (measured from the nadir, the
image center, since the bottom of a downward frame is actually BEHIND the drone) along the
traced centerline. On a straight line it is dead ahead (~0 heading error); at a hairpin it is
partway around the bend, so the heading error grows and the drone turns to follow it.

Control (metric): project the carrot to a ground point with the pinhole model (offset =
pixel_offset / FOCAL_PX * altitude), steer YAW RATE by the heading angle to it, ease FORWARD
speed continuously as the angle grows, and PIVOT IN PLACE only on a sustained, well-traced tight
bend. The heading is low-pass filtered and the pivot is hysteretic and gated on how much line was
actually traced, so a small, noisy line high in the frame stays stable instead of stutter-
pivoting. A light STRAFE keeps the body over the line. Gains are physical (K_PSI, V_CRUISE).

Detection uses neo_lab.line_mask (HSV brightness), which works the same in the sim and on the
real LED-strip line. Command path: drone.flight.send_body_velocity(v_forward, v_right, v_up,
yaw_rate) -- the SI body-velocity setpoint (true m/s and rad/s, no [-1,1] normalization).
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

# -- Perception: line mask + centerline path tracer -------------------------
# neo_lab.line_mask isolates the line by HSV brightness (works in the sim and on the real LED
# strip; morphology bridges LED gaps and a thin line's two edges). The tracer below then follows
# the line's CENTERLINE, so two bright edges never make it weave.
V_MIN         = 200      # HSV Value threshold: bright LED strip (real) / bright line pixels (sim)
MIN_TOTAL_PX  = 120      # below this many line pixels, treat the line as lost

# Seed (where the drone currently sits on the line).
SEED_BAND_FRAC = 0.18    # bottom fraction of the frame used to seed the trace
SEED_MIN_PX    = 20      # line pixels needed to seed
SEED_PERP      = 55      # px: half-window that gathers the line's FULL width (both edges) into the
                         # seed centerline. Must EXCEED the line's pixel width, yet stay BELOW the
                         # spacing between separate segments so a hairpin's two legs aren't merged.

# Crawl (trace the centerline along the tangent).
STEP          = 20       # px advanced along the tangent each step
ALONG_HALF    = 12       # px: forward half-extent of the cross-section slab sampled each step
PERP_HALF     = 34       # px: half-width of the cross-section; averages the line's two edges to
                         # its centerline (same sizing rule as SEED_PERP)
MIN_SLAB_PX   = 8        # line pixels a cross-section needs to keep crawling
MAX_STEPS     = 26       # cap on traced path length
DIR_SMOOTH    = 0.55     # 0..1 how fast the tangent turns (higher rounds sharper bends)
LOOKAHEAD_M   = 0.35     # meters of arc-length AHEAD of the drone (nadir) to place the carrot
LOOKAHEAD_MIN_PX = 90    # floor on the look-ahead in PIXELS: keeps the carrot far enough ahead for
                         # a stable bearing when the line is small in the image (high altitude)

# -- Camera / ground projection (pinhole, downward camera) -------------------
FOCAL_PX      = 320.0    # camera focal length in pixels (approx; matches module5 distance est.)
CX            = 320.0    # principal point x (640-wide image)
CY            = 240.0    # principal point y (480-tall image)
MIN_AGL_M     = 0.30     # floor on altitude used for the pixel->meter scale

# -- Steering (look-ahead yaw + light strafe), physical units ----------------
K_PSI         = 0.2      # yaw rate (rad/s) per rad of heading error to the carrot  (TUNE FIRST)
YAW_RATE_MAX  = 2.0      # rad/s cap on yaw
PSI_SLOW      = 0.70     # heading error (rad, ~40 deg) at which forward eases toward its minimum
TAU_PSI       = 0.20     # s: low-pass time constant on the heading error (kills per-frame jitter)
PIVOT_ENTER   = 1.20     # rad (~63 deg): sustained heading error needed to START pivoting in place
PIVOT_EXIT    = 0.80     # rad (~46 deg): heading error to STOP pivoting (hysteresis -> no chatter)
PIVOT_MIN_ARC_PX = 140   # px: only pivot when at least this much line has been traced; a short,
                         # noisy trace (small line high up) can't be trusted to be a real hairpin
K_LAT         = 1.2      # strafe (m/s) per meter of lateral offset under the drone
V_STRAFE_MAX  = 0.40     # m/s cap on strafe

# -- Forward speed (auto-eases in turns; no separate curvature gain) ---------
V_CRUISE      = 0.25      # m/s on straights
V_FWD_MIN     = 0.15     # m/s at a moderate turn (PSI_SLOW)
V_TURN_MIN    = 0.10     # m/s floor for a sharp turn when NOT pivoting -- keep crawling forward so
                         # the drone arcs around the bend instead of dead-stopping (which oscillates)

# -- Altitude & run ----------------------------------------------------------
TARGET_HEIGHT  = 0.7        # meters above launch ground
FOLLOW_TIME    = 1000000.0   # seconds to follow before landing
LOST_YAW_DECAY = 0.6         # keep turning toward the last-seen side while the line is lost

# -- Module-level state -----------------------------------------------------
_timer    = 0.0
_done     = False
_last_yaw = 0.0          # remembered yaw rate, so a briefly-lost line keeps turning the right way
_psi_filt = 0.0          # low-pass filtered heading error (radians)
_pivoting = False        # hysteretic pivot-in-place state


def reset():
    global _timer, _done, _last_yaw, _psi_filt, _pivoting
    _timer    = 0.0
    _done     = False
    _last_yaw = 0.0
    _psi_filt = 0.0
    _pivoting = False


# -- Centerline path tracer -------------------------------------------------
def _unit(vec):
    n = float(np.linalg.norm(vec))
    return vec / n if n > 1e-6 else vec


def find_seed(mask):
    """Seed on the line's CENTERLINE nearest bottom-center — where the drone currently sits over
    the line. Gathers the full line width (both edges) around the nearest column so the seed is
    the middle of the line, not one edge. Returns (row, col) or None."""
    h, w = mask.shape
    r0 = int(h * (1.0 - SEED_BAND_FRAC))
    pts = np.argwhere(mask[r0:h])
    off = r0
    if len(pts) < SEED_MIN_PX:
        pts = np.argwhere(mask)                    # fallback: search the whole frame
        off = 0
        if len(pts) < SEED_MIN_PX:
            return None
    gr = pts[:, 0] + off
    gc = pts[:, 1]
    anchor = gc[np.argmin((gr - (h - 1)) ** 2 + (gc - CX) ** 2)]   # column nearest bottom-center
    keep = np.abs(gc - anchor) <= SEED_PERP        # the whole width around it (both edges)
    return (float(gr[keep].mean()), float(gc[keep].mean()))


def _cross_center(mask, c, d):
    """Sample the line's cross-section perpendicular to heading d at the predicted point c, and
    return the CENTERLINE point (mid-line, averaging the two bright edges), or None. Averaging
    across the width is what stops the tracer from locking onto one edge and weaving; keeping the
    slab local (ALONG_HALF x PERP_HALF) is what stops it from averaging in a far hairpin leg."""
    h, w = mask.shape
    R = int(max(ALONG_HALF, PERP_HALF)) + 2
    r0, r1 = max(int(c[0]) - R, 0), min(int(c[0]) + R + 1, h)
    k0, k1 = max(int(c[1]) - R, 0), min(int(c[1]) + R + 1, w)
    sub = np.argwhere(mask[r0:r1, k0:k1])
    if len(sub) < MIN_SLAB_PX:
        return None
    gr = (sub[:, 0] + r0) - c[0]
    gc = (sub[:, 1] + k0) - c[1]
    dr, dc = float(d[0]), float(d[1])
    along = gr * dr + gc * dc                       # distance along heading
    perp = -gr * dc + gc * dr                       # distance across heading (perp_hat = (-dc, dr))
    keep = (np.abs(along) <= ALONG_HALF) & (np.abs(perp) <= PERP_HALF)
    if np.count_nonzero(keep) < MIN_SLAB_PX:
        return None
    ma = float(along[keep].mean())
    mp = float(perp[keep].mean())
    return np.array([c[0] + ma * dr + mp * (-dc), c[1] + ma * dc + mp * dr])


def trace_line(mask):
    """Crawl along the line's centerline from the seed, following its local tangent, and return
    the path as a list of (row, col) points ordered from the drone outward. Rounds sharp bends
    and hairpins because the crawl follows the tangent; stays on the middle of the line because
    each step re-centers on the cross-section."""
    h, w = mask.shape
    seed = find_seed(mask)
    if seed is None:
        return []
    pos = np.array(seed, dtype=float)
    direction = np.array([-1.0, 0.0])              # start heading up-frame (ahead of the drone)
    path = [(pos[0], pos[1])]
    for _ in range(MAX_STEPS):
        c = pos + STEP * direction                 # predicted next point along the tangent
        new = _cross_center(mask, c, direction)
        if new is None:
            break
        if float(np.linalg.norm(new - pos)) < 1e-3:
            break
        direction = _unit((1.0 - DIR_SMOOTH) * direction + DIR_SMOOTH * _unit(new - pos))
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
    look_px = max(LOOKAHEAD_M / meters_per_px, LOOKAHEAD_MIN_PX)
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


def _path_arc(path):
    """Total traced length in pixels — a confidence measure (a longer trace = more line seen)."""
    return sum(math.hypot(path[i][0] - path[i - 1][0], path[i][1] - path[i - 1][1])
               for i in range(1, len(path)))


# -- Main loop --------------------------------------------------------------
def update(drone):
    global _timer, _done, _last_yaw, _psi_filt, _pivoting
    if _done:
        return True

    dt   = drone.get_delta_time()
    v_up = neo_lab.altitude_hold_velocity(drone, TARGET_HEIGHT)   # m/s, holds height on sim & real

    frame = drone.camera.get_downward_image()
    mask  = neo_lab.line_mask(frame, V_MIN)
    path  = trace_line(mask)
    agl   = max(neo_lab.height(drone), MIN_AGL_M)
    mpp   = agl / FOCAL_PX

    if np.count_nonzero(mask) < MIN_TOTAL_PX or len(path) < 2:
        # Lost, or too little line to know its direction: hold height, decay the last yaw so the
        # drone keeps rotating toward where the line went, and strafe to sit over the seed if any.
        _last_yaw *= LOST_YAW_DECAY
        _psi_filt *= LOST_YAW_DECAY
        _pivoting = False
        v_right = 0.0
        if path:
            _, near_right = _ground_offset(path[0][0], path[0][1], mpp)
            v_right = uav_utils.clamp(K_LAT * near_right, -V_STRAFE_MAX, V_STRAFE_MAX)
        drone.flight.send_body_velocity(0.0, v_right, v_up, _last_yaw)
    else:
        carrot, near = look_ahead(path, mpp)                     # carrot ahead of nadir; near = under drone
        fwd_L, right_L = _ground_offset(carrot[0], carrot[1], mpp)
        _, near_right  = _ground_offset(near[0], near[1], mpp)

        # Heading to the carrot, LOW-PASS FILTERED over time. Pixel-level jitter (worst when the
        # line is small in the image, i.e. high up) would otherwise fling the yaw around and keep
        # tripping the pivot; the filter turns that into a smooth, stable command.
        raw_psi = math.atan2(right_L, fwd_L)                       # rad; + = carrot to the right
        alpha = dt / (TAU_PSI + dt) if dt > 0.0 else 1.0
        _psi_filt += alpha * (raw_psi - _psi_filt)
        af = abs(_psi_filt)

        # Pivot in place only for a SUSTAINED, well-traced sharp turn: hysteresis (enter high, exit
        # low) stops chatter, and a minimum traced length stops a short/noisy trace high up from
        # ever forcing a pivot.
        arc_px = _path_arc(path)
        if _pivoting:
            _pivoting = af > PIVOT_EXIT
        elif af > PIVOT_ENTER and arc_px >= PIVOT_MIN_ARC_PX:
            _pivoting = True

        yaw_rate = uav_utils.clamp(K_PSI * _psi_filt, -YAW_RATE_MAX, YAW_RATE_MAX)

        # Forward speed: a CONTINUOUS ramp (no hard jump). Cruise when aimed straight, ease to
        # V_FWD_MIN by PSI_SLOW and down to the crawl floor V_TURN_MIN for sharp turns -- it keeps
        # moving so it ARCS around the bend. It drops to a true 0 (pivot in place) only when the
        # gated/hysteretic pivot is engaged, i.e. a real, well-traced hairpin.
        if _pivoting:
            v_forward = 0.0
        elif af <= PSI_SLOW:
            v_forward = V_CRUISE + (V_FWD_MIN - V_CRUISE) * (af / PSI_SLOW)
        else:
            t = min((af - PSI_SLOW) / (PIVOT_ENTER - PSI_SLOW), 1.0)
            v_forward = V_FWD_MIN + (V_TURN_MIN - V_FWD_MIN) * t

        # Strafe: center the body over the line (residual lateral offset, in meters).
        v_right = uav_utils.clamp(K_LAT * near_right, -V_STRAFE_MAX, V_STRAFE_MAX)

        _last_yaw = yaw_rate
        drone.flight.send_body_velocity(v_forward, v_right, v_up, yaw_rate)
        print(f"path={len(path):2d} arc={arc_px:3.0f}px psi={math.degrees(raw_psi):+6.1f}"
              f"->{math.degrees(_psi_filt):+6.1f}deg | v_fwd={v_forward:.2f} "
              f"v_right={v_right:+.2f} yaw={yaw_rate:+.2f}rad/s{'  PIVOT' if _pivoting else ''}")

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
