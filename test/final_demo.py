"""
MIT BWSI Autonomous Drone Racing Course - UAV Neo
GNU General Public License v3.0

Combined Lab — Line Following + Gate Traversal
Default: follow the line. When a gate's ArUco tags are spotted, switch to
gate-centering mode, fly through it, then resume line following.
"""

import drone_core
import drone_utils as uav_utils
import cv2
import numpy as np
import traceback

# -- Course setup: makes the shared `neo_lab` helper importable. --
import os as _os, sys as _sys
_d = _os.path.dirname(_os.path.realpath(__file__))
while _os.path.basename(_d) != "labs" and _os.path.dirname(_d) != _d:
    _d = _os.path.dirname(_d)
if _d not in _sys.path:
    _sys.path.insert(0, _d)
import neo_lab

# ============================================================
# Line-follow constants (tangent-line / polynomial fit version)
# ============================================================
V_MIN         = 200
MIN_PIXELS    = 200
MAX_ROLL      = 0.25     # strafe authority for centering
MAX_YAW       = 1.0      # yaw authority
IMAGE_CENTER  = 320       # 640-wide image -> center column

PITCH_STRAIGHT = 0.14    # fast on straights
PITCH_TURN     = 0.08    # slow through turns
CURVE_SCALE    = 100      # residual std at which you're "fully" in a turn (TUNE)

# -- PID gains --------------------------------------------------------------
YAW_KP      = 0.60
YAW_KI      = 0.0
YAW_KD      = 0.055
YAW_I_LIMIT = 0.20

ROLL_KP      = 0.24
ROLL_KI      = 0.0
ROLL_KD      = 0.02
ROLL_I_LIMIT = 0.10

D_TAU = 0.10    # derivative low-pass time constant, seconds (bigger = smoother)

# -- Line-search climb -------------------------------------------------------
MAX_CLIMB        = 3
CLIMB_THROTTLE   = 0.3
DESCEND_THROTTLE = -0.2
LOST_GRACE       = 0.4
FOUND_GRACE      = 0.5
HEIGHT_TOL       = 0.10

# -- Perception constants ----------------------------------------------------
POLY_DEGREE  = 3
IMG_W, IMG_H = 640, 480
TARGET_POINT = (IMG_W / 2, IMG_H / 2 - 80)
SAMPLE_STEP  = 2

FOLLOW_TIME = 1000000.0

# -- Debug print cadence ------------------------------------------------------
LINE_PRINT_EVERY = 5
GATE_PRINT_EVERY = 5

# ============================================================
# Gate constants (Week 2 Step 1)
# ============================================================
# -- Gate centering PID gains --
# Seeded at the old P-only values (KP=1.5, KI=0, KD=0) so behavior is
# unchanged until you tune. Tune KD in first to damp oscillation, then KI
# only if it consistently settles off-center.
GATE_ROLL_KP      = 1.0
GATE_ROLL_KI      = 0.0
GATE_ROLL_KD      = 0.0
GATE_ROLL_I_LIMIT = 0.10

GATE_ALT_KP      = 0.3
GATE_ALT_KI      = 0.0
GATE_ALT_KD      = 0.0
GATE_ALT_I_LIMIT = 0.10

# Pitch drift-brake during centering. Previously reused the search-hold
# pitch PID (SEARCH_HOLD_*), which coupled search-mode and centering-mode
# tuning together. Now dedicated, so tuning one doesn't affect the other.
# Seeded at SEARCH_HOLD's old values so behavior doesn't jump.
GATE_PITCH_KP      = 0.15
GATE_PITCH_KI      = 0.0
GATE_PITCH_KD      = 0.02
GATE_PITCH_I_LIMIT = 0.10
GATE_PITCH_LIMIT   = 0.20

# Yaw-rate brake during centering. Previously a bare P-only clamp
# (YAW_BRAKE_KP with no I/D term). Now a full PID like the other three
# axes. Seeded at the old YAW_BRAKE_KP/LIMIT values.
GATE_YAW_KP      = 0.5
GATE_YAW_KI      = 0.0
GATE_YAW_KD      = 0.0
GATE_YAW_I_LIMIT = 0.10
GATE_YAW_LIMIT   = 0.5

CENTER_TOL     = 0.05
ALT_TOL        = 0.05
CENTER_HOLD_T  = 0.5
ROLL_LIMIT     = 0.3
THROTTLE_LIMIT = 0.3
SEARCH_YAW     = 0.2
GATE_PITCH     = 0.4

GATE_CHECK_EVERY   = 5
GATE_DIST_MIN_TAGS = 1   # min tags for it to measure dist to gate
GATE_SETTLE_FRAMES = 20
GATE_CONFIRM_HITS  = 1  # min frames with tag in them for it to count as gate
GATE_MIN_TAGS_TO_ADVANCE = 3   # min tags visible before "centered" can start counting toward CENTER_HOLD_T

# -- Search-state drift hold --
# While yawing to search for a lost gate, damp any residual body-frame
# velocity (vx=right, vy=up, vz=forward) back toward zero instead of
# sending zero commands and letting momentum/wind carry the drone.
# Each axis has its own independent gains - tune separately. All seeded
# at the old shared SEARCH_HOLD_* values so behavior doesn't jump. TUNE ME.
SEARCH_ROLL_KP      = 1.5
SEARCH_ROLL_KI      = 0.0
SEARCH_ROLL_KD      = 0.02
SEARCH_ROLL_I_LIMIT = 0.10
SEARCH_ROLL_LIMIT   = 0.20

SEARCH_PITCH_KP      = 0.0
SEARCH_PITCH_KI      = 0.0
SEARCH_PITCH_KD      = 0.02
SEARCH_PITCH_I_LIMIT = 0.10
SEARCH_PITCH_LIMIT   = 0.20

SEARCH_THROTTLE_KP      = 0.0
SEARCH_THROTTLE_KI      = 0.0
SEARCH_THROTTLE_KD      = 0.02
SEARCH_THROTTLE_I_LIMIT = 0.10
SEARCH_THROTTLE_LIMIT   = 0.20

# Yaw drift-brake used ONLY during the settle sub-phase (before the active
# SEARCH_YAW sweep starts) - once actively sweeping, yaw is deliberately
# driven by the SEARCH_YAW constant, not braked. Seeded at the same values
# as centering's GATE_YAW_* so behavior is consistent, but tunable separately.
SEARCH_YAW_KP      = 0.2
SEARCH_YAW_KI      = 0.0
SEARCH_YAW_KD      = 0.0
SEARCH_YAW_I_LIMIT = 0.10
SEARCH_YAW_LIMIT   = 0.5


# -- Search start delay --
# On first entering GATE_CENTER, hold still (no yaw) for this many seconds
# before starting the SEARCH_YAW sweep, so a gate that's already in view
# has time to actually be detected instead of getting yawed away from
# immediately. TUNE ME.
GATE_SEARCH_START_DELAY = 2.0

# -- Velocity feedback filter --
# get_linear_velocity() is the only velocity source this library exposes
# (no raw optical-flow endpoint in the docs) - if a flow sensor is fused
# into the flight controller's estimator, it's already baked into this
# reading. Low-passing it here reduces jitter feeding into the search-hold
# PID, which otherwise chases every noisy sample. TUNE ME.
VEL_FILTER_TAU = 0.15   # seconds; bigger = smoother but more lag

# -- Command slew-rate limiter --
# Caps how much pitch/roll/yaw/throttle can change per second, regardless
# of which control law (search-hold vs centering) produced the new value.
# This is what actually kills the "erratic" jump at the instant a gate is
# detected, since that's a discontinuous handoff between two different
# control laws with different error signals and gains. TUNE ME.
SLEW_MAX_RATE = 1.5   # units/sec of pitch, roll, yaw, or throttle

# -- Optical flow drift estimate (Week 2 Module 7 Step 2, adapted) --
# Sparse downward-camera flow gives a direct from-imagery velocity estimate
# that doesn't depend on whatever the flight controller's EKF is/isn't
# fusing. Unlike Step 2's lab (which only kept displacement MAGNITUDE),
# this keeps the mean displacement VECTOR so it's usable for directional
# drift correction, not just a speed comparison.
#
# AXIS/SIGN CONVENTION IS UNVERIFIED: image columns are assumed to map to
# body 'right' and image rows to body 'forward', with a sign flip since the
# tracked scene moves opposite the drone's actual motion. Confirm this
# empirically (isolated small roll/pitch nudge, compare printed flow vx/vz
# against physics vx/vz) before trusting it in a real flight - the same way
# Step 2 itself validates its estimate against true velocity.
FLOW_HFOV_TAN      = 1.0     # tan(half HFOV); matches Step 2's assumption, verify against actual camera spec
FLOW_SKIP          = 2       # do the corner-tracking vision work every Nth frame
FLOW_MIN_PTS       = 20
FLOW_FEATURE_PARAMS = dict(maxCorners=80, qualityLevel=0.01, minDistance=8, blockSize=7)
FLOW_LK_PARAMS = dict(winSize=(15, 15), maxLevel=2,
                       criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
FLOW_MAX_AGE       = 0.5     # seconds; older than this, fall back to physics velocity instead

FORWARD_MARGIN    = 1.0
FORWARD_MAX_TIME  = 6.0


# ============================================================
# PID controller
# ============================================================
class PID:
    """
    Setpoint is always 0 (edge centered / line vertical / gate centered), so
    `error` is just the measurement. Includes a low-passed derivative and
    clamped integral with conditional anti-windup.
    """

    def __init__(self, kp, ki, kd, out_limit, i_limit, d_tau=D_TAU):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.out_limit = out_limit
        self.i_limit = i_limit
        self.d_tau = d_tau
        self.reset()

    def reset(self):
        self._integral = 0.0
        self._prev_error = None
        self._deriv = 0.0

    def hold(self):
        """
        Called on frames where the measurement is invalid (line/gate lost).
        Keeps the integral but drops the previous error, so the first frame
        after reacquiring doesn't produce a huge derivative spike.
        """
        self._prev_error = None

    def update(self, error, dt):
        if dt <= 0.0:
            return uav_utils.clamp(self.kp * error, -self.out_limit, self.out_limit)

        if self._prev_error is None:
            raw_deriv = 0.0
            self._deriv = 0.0
        else:
            raw_deriv = (error - self._prev_error) / dt
        alpha = dt / (self.d_tau + dt)
        self._deriv += alpha * (raw_deriv - self._deriv)
        self._prev_error = error

        integral = uav_utils.clamp(
            self._integral + error * dt, -self.i_limit, self.i_limit
        )

        output = self.kp * error + self.ki * integral + self.kd * self._deriv
        clamped = uav_utils.clamp(output, -self.out_limit, self.out_limit)

        saturated = output != clamped
        if not (saturated and output * error > 0):
            self._integral = integral

        return clamped


# ============================================================
# Module-level state
# ============================================================
LINE_FOLLOW, GATE = "LINE_FOLLOW", "GATE"
GATE_CENTER, GATE_FORWARD = "GATE_CENTER", "GATE_FORWARD"
FOLLOWING, SEARCHING, DESCENDING = "FOLLOWING", "SEARCHING", "DESCENDING"

_timer = 0.0
_done  = False
_frame = 0
_mode  = LINE_FOLLOW

# -- line-follow sub-state --
_line_state = FOLLOWING
_base_alt   = None
_visible    = True
_vis_timer  = 0.0
_settle_frame = GATE_SETTLE_FRAMES
_gate_confirm = 0
_yaw_pid  = PID(YAW_KP,  YAW_KI,  YAW_KD,  MAX_YAW,  YAW_I_LIMIT)
_roll_pid = PID(ROLL_KP, ROLL_KI, ROLL_KD, MAX_ROLL, ROLL_I_LIMIT)

# -- gate sub-state --
_gate_phase     = GATE_CENTER
_gate           = None
_hold           = 0.0
_forward_time   = 0.0
_forward_target = None
_forward_dist   = 0.0
_gate_center_time = 0.0   # time spent in the current GATE_CENTER session, for the search-start delay
_vel_filt = [0.0, 0.0, 0.0]     # low-passed (vx, vy, vz) used by the search-hold PIDs
_last_cmd = [0.0, 0.0, 0.0, 0.0]  # last (pitch, roll, yaw, throttle) actually sent, for slew limiting
_flow_prev_gray = None
_flow_prev_pts  = None
_flow_interval  = 0.0
_flow_vx = 0.0   # right, m/s - flow-estimated
_flow_vz = 0.0   # forward, m/s - flow-estimated
_flow_age = 999.0   # seconds since the last successful flow update
_gate_roll_pid = PID(GATE_ROLL_KP, GATE_ROLL_KI, GATE_ROLL_KD, ROLL_LIMIT, GATE_ROLL_I_LIMIT)
_gate_alt_pid  = PID(GATE_ALT_KP,  GATE_ALT_KI,  GATE_ALT_KD,  THROTTLE_LIMIT, GATE_ALT_I_LIMIT)
_gate_pitch_pid = PID(GATE_PITCH_KP, GATE_PITCH_KI, GATE_PITCH_KD, GATE_PITCH_LIMIT, GATE_PITCH_I_LIMIT)
_gate_yaw_pid   = PID(GATE_YAW_KP,   GATE_YAW_KI,   GATE_YAW_KD,   GATE_YAW_LIMIT,   GATE_YAW_I_LIMIT)
_search_roll_pid     = PID(SEARCH_ROLL_KP, SEARCH_ROLL_KI, SEARCH_ROLL_KD, SEARCH_ROLL_LIMIT, SEARCH_ROLL_I_LIMIT)
_search_pitch_pid    = PID(SEARCH_PITCH_KP, SEARCH_PITCH_KI, SEARCH_PITCH_KD, SEARCH_PITCH_LIMIT, SEARCH_PITCH_I_LIMIT)
_search_throttle_pid = PID(SEARCH_THROTTLE_KP, SEARCH_THROTTLE_KI, SEARCH_THROTTLE_KD, SEARCH_THROTTLE_LIMIT, SEARCH_THROTTLE_I_LIMIT)
_search_yaw_pid       = PID(SEARCH_YAW_KP, SEARCH_YAW_KI, SEARCH_YAW_KD, SEARCH_YAW_LIMIT, SEARCH_YAW_I_LIMIT)


def reset():
    global _timer, _done, _frame, _mode
    global _line_state, _base_alt, _visible, _vis_timer, _settle_frame, _gate_confirm
    global _gate_phase, _gate, _hold, _forward_time, _forward_target, _forward_dist, _gate_center_time
    global _vel_filt, _last_cmd
    global _flow_prev_gray, _flow_prev_pts, _flow_interval, _flow_vx, _flow_vz, _flow_age

    _timer = 0.0
    _done  = False
    _frame = 0
    _mode  = LINE_FOLLOW

    _line_state   = FOLLOWING
    _base_alt     = None
    _visible      = True
    _vis_timer    = 0.0
    _settle_frame = GATE_SETTLE_FRAMES
    _gate_confirm = 0
    _yaw_pid.reset()
    _roll_pid.reset()

    _gate_phase     = GATE_CENTER
    _gate           = None
    _hold           = 0.0
    _forward_time   = 0.0
    _forward_target = None
    _forward_dist   = 0.0
    _gate_center_time = 0.0
    _gate_roll_pid.reset()
    _gate_alt_pid.reset()
    _gate_pitch_pid.reset()
    _gate_yaw_pid.reset()
    _search_roll_pid.reset()
    _search_pitch_pid.reset()
    _search_throttle_pid.reset()
    _search_yaw_pid.reset()
    _vel_filt = [0.0, 0.0, 0.0]
    _last_cmd = [0.0, 0.0, 0.0, 0.0]
    _flow_prev_gray = None
    _flow_prev_pts  = None
    _flow_interval  = 0.0
    _flow_vx = 0.0
    _flow_vz = 0.0
    _flow_age = 999.0


def _reset_flow():
    """Call whenever a fresh GATE_CENTER session begins, so stale corner
    points from a much earlier time/position aren't tracked across the gap
    (e.g. across the GATE_FORWARD phase, where the downward camera isn't
    being sampled) and produce a garbage displacement on the first frame."""
    global _flow_prev_gray, _flow_prev_pts, _flow_interval, _flow_age
    _flow_prev_gray = None
    _flow_prev_pts  = None
    _flow_interval  = 0.0
    _flow_age = 999.0


def _update_optical_flow(drone, dt):
    """
    Sparse downward-camera optical flow, adapted from Week 2 Module 7 Step 2.
    Tracks corner features frame-to-frame and converts their mean pixel
    displacement to a (right, forward) m/s velocity estimate - unlike Step
    2's own lab, which collapsed displacement to a scalar magnitude, this
    keeps the vector so it's usable for directional drift correction.

    Updates the module-level _flow_vx/_flow_vz/_flow_age; does not return
    anything, since it only refreshes every FLOW_SKIP frames and callers
    should read the held state each frame regardless of whether this call
    actually refreshed it.

    Wrapped in try/except: if anything inside throws, we print the FULL
    traceback (not just the generic ">> ERROR in update()" summary) and
    fall back gracefully - callers already fall back to physics velocity
    whenever _flow_age is stale, so a swallowed exception here just means
    one more frame of fallback, not a crash.
    """
    try:
        _update_optical_flow_impl(drone, dt)
    except Exception:
        print("[gate] [optical flow ERROR] full traceback below:")
        traceback.print_exc()


def _update_optical_flow_impl(drone, dt):
    global _flow_prev_gray, _flow_prev_pts, _flow_interval, _flow_vx, _flow_vz, _flow_age

    _flow_interval += dt
    _flow_age += dt

    if _frame % FLOW_SKIP != 0:
        return

    image = drone.camera.get_downward_image()
    if image is None:
        return
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height = neo_lab.height(drone)

    if _flow_prev_gray is None or _flow_prev_pts is None or len(_flow_prev_pts) < FLOW_MIN_PTS:
        # (Re)detect corners in this frame; nothing to track against yet.
        _flow_prev_pts = cv2.goodFeaturesToTrack(gray, mask=None, **FLOW_FEATURE_PARAMS)
        _flow_prev_gray = gray
        return

    next_pts, status, _ = cv2.calcOpticalFlowPyrLK(_flow_prev_gray, gray, _flow_prev_pts, None, **FLOW_LK_PARAMS)
    keep = status.flatten() == 1
    good_new = next_pts[keep]
    good_old = _flow_prev_pts[keep]

    if len(good_new) >= FLOW_MIN_PTS and _flow_interval > 0:
        diff = (good_new - good_old).reshape(-1, 2)   # (N, 2) in cv2's (dx, dy) pixel order
        mean_dx, mean_dy = diff.mean(axis=0)
        width = drone.camera.get_width()
        meters_per_pixel = 2 * height * FLOW_HFOV_TAN / width
        # The tracked scene moves opposite the drone's own motion.
        _flow_vx = -(float(mean_dx) * meters_per_pixel) / _flow_interval   # right
        _flow_vz = -(float(mean_dy) * meters_per_pixel) / _flow_interval   # forward
        _flow_age = 0.0
        _flow_interval = 0.0

    _flow_prev_pts = good_new.reshape(-1, 1, 2) if len(good_new) > 0 else None
    _flow_prev_gray = gray


def _filtered_velocity(drone, dt):
    """Low-pass the raw (vx, vy, vz) body-frame velocity reading before it's
    used as PID feedback, to reduce jitter from a noisy raw sensor stream."""
    global _vel_filt
    vx, vy, vz = (float(v) for v in drone.physics.get_linear_velocity())
    alpha = dt / (VEL_FILTER_TAU + dt) if dt > 0 else 1.0
    _vel_filt[0] += alpha * (vx - _vel_filt[0])
    _vel_filt[1] += alpha * (vy - _vel_filt[1])
    _vel_filt[2] += alpha * (vz - _vel_filt[2])
    return tuple(_vel_filt)


def _send_pcmd_slewed(drone, pitch, roll, yaw, throttle, dt):
    """Send a pcmd, but cap how much each axis can change from the last
    command actually sent, regardless of which control law produced the
    new value. This is what prevents a discontinuous jump - e.g. the
    instant a gate is detected and control hands off from the search-hold
    PID to the centering PID - from reading as an erratic lurch."""
    global _last_cmd
    max_delta = SLEW_MAX_RATE * dt if dt > 0 else SLEW_MAX_RATE
    requested = [pitch, roll, yaw, throttle]
    out = []
    for prev, want in zip(_last_cmd, requested):
        delta = uav_utils.clamp(want - prev, -max_delta, max_delta)
        out.append(prev + delta)
    _last_cmd = out
    drone.flight.send_pcmd(*out)
    return out


def _reset_slew():
    """Call right after drone.flight.stop() so the slew limiter doesn't try
    to ramp future commands down from a stale pre-stop baseline."""
    global _last_cmd
    _last_cmd = [0.0, 0.0, 0.0, 0.0]


# ============================================================
# Line-follow perception + control (tangent-line / polynomial version)
# ============================================================
def find_edge(drone):
    camera = drone.camera.get_downward_image()
    mask = neo_lab.bright_mask_improved(camera, V_MIN)
    edges = np.argwhere(mask)
    edges = edges.astype(np.float64)

    if np.count_nonzero(edges) < MIN_PIXELS:
        return None

    ys = edges[:, 0]
    xs = edges[:, 1]

    coeffs = np.polyfit(ys, xs, POLY_DEGREE)
    poly = np.poly1d(coeffs)
    poly_deriv = poly.deriv()

    y_min, y_max = ys.min(), ys.max()
    sample_ys = np.arange(y_min, y_max + SAMPLE_STEP, SAMPLE_STEP)
    sample_xs = poly(sample_ys)

    target_x, target_y = TARGET_POINT
    dist_sq = (sample_xs - target_x) ** 2 + (sample_ys - target_y) ** 2
    best_idx = np.argmin(dist_sq)
    y0 = sample_ys[best_idx]
    x0 = sample_xs[best_idx]

    m = poly_deriv(y0)
    b = x0 - m * y0

    return ys, xs, m, b, poly, (y0, x0)


def set_yaw(m, dt):
    error = -m
    return _yaw_pid.update(error, dt)


def set_roll(xs, dt):
    edge_col = xs.mean()
    error = (edge_col - IMAGE_CENTER) / IMAGE_CENTER
    return _roll_pid.update(error, dt)


def set_pitch(ys, xs, poly):
    curviness = np.std(xs - poly(ys))
    straightness = uav_utils.clamp(1.0 - curviness / CURVE_SCALE, 0.0, 1.0)
    return PITCH_TURN + (PITCH_STRAIGHT - PITCH_TURN) * straightness


def set_throttle(drone, fit, dt):
    global _line_state, _base_alt, _visible, _vis_timer

    alt = drone.physics.get_altitude()
    if _base_alt is None:
        _base_alt = alt

    visible = fit is not None
    if visible != _visible:
        _visible   = visible
        _vis_timer = 0.0
    else:
        _vis_timer += dt

    lost_confirmed  = (not _visible) and _vis_timer >= LOST_GRACE
    found_confirmed = _visible and _vis_timer >= FOUND_GRACE

    if _line_state == FOLLOWING:
        if lost_confirmed:
            _line_state = SEARCHING
            return CLIMB_THROTTLE
        return 0.0

    if _line_state == SEARCHING:
        if found_confirmed:
            _line_state = DESCENDING
            return DESCEND_THROTTLE
        if alt - _base_alt >= MAX_CLIMB:
            return 0.0
        return CLIMB_THROTTLE

    if _line_state == DESCENDING:
        if lost_confirmed:
            _line_state = SEARCHING
            return CLIMB_THROTTLE
        if alt <= _base_alt + HEIGHT_TOL:
            _line_state = FOLLOWING
            return 0.0
        return DESCEND_THROTTLE

    return 0.0


# ============================================================
# Shared helper: check the forward camera for ArUco tags and, if found,
# return (ids, mean_z_distance_meters).
# ============================================================
def _detect_tag_distance(drone):
    img = drone.camera.get_color_image()
    if img is None:
        return None, None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = neo_lab._detect_gate_markers(gray)
    if ids is None or len(ids) < GATE_DIST_MIN_TAGS:
        return None, None

    tag_centers = np.array([c.reshape(-1, 2).mean(axis=0) for c in corners])
    depth_img = drone.camera.get_depth_image()
    if depth_img is None:
        return ids, None

    ch, cw = gray.shape[:2]
    dh, dw = depth_img.shape[:2]
    sx, sy = dw / cw, dh / ch

    dists = []
    for cx, cy in tag_centers:
        dx = int(np.clip(cx * sx, 0, dw - 1))
        dy = int(np.clip(cy * sy, 0, dh - 1))
        d = uav_utils.get_pixel_average_distance(depth_img, (dy, dx), kernel_size=5) / 100.0
        dists.append(d)

    return ids, float(np.mean(dists))


# ============================================================
# Mode: LINE_FOLLOW
# ============================================================
def _update_line_follow(drone, dt):
    global _mode, _gate_phase, _hold, _forward_time, _forward_target, _forward_dist
    global _gate_confirm, _gate_center_time

    if _frame >= _settle_frame and _frame % GATE_CHECK_EVERY == 0:
        ids, dist = _detect_tag_distance(drone)
        if ids is not None and len(ids) > 0:
            _gate_confirm += 1
            print(f'[line->gate check] tags seen (ids={ids}, dist={dist}), '
                  f'confirm={_gate_confirm}/{GATE_CONFIRM_HITS}')
        else:
            if _gate_confirm > 0:
                print('[line->gate check] tag lost, resetting confirm streak')
            _gate_confirm = 0

        if _gate_confirm >= GATE_CONFIRM_HITS:
            print(f'Gate tags confirmed (ids={ids}) - switching to GATE mode.')
            drone.flight.stop()
            _reset_slew()
            _reset_flow()
            _mode = GATE
            _gate_phase = GATE_CENTER
            _hold = 0.0
            _forward_time = 0.0
            _forward_dist = 0.0
            _gate_center_time = 0.0
            _forward_target = (dist + FORWARD_MARGIN) if dist is not None else None
            _gate_confirm = 0
            _gate_roll_pid.reset()
            _gate_alt_pid.reset()
            _gate_pitch_pid.reset()
            _gate_yaw_pid.reset()
            print(f'Tag distance={dist}, forward target={_forward_target}')
            return

    fit = find_edge(drone)
    throttle = set_throttle(drone, fit, dt)

    if fit is None:
        _yaw_pid.hold()
        _roll_pid.hold()
        # drone.flight.send_pcmd(0.0, 0.0, 0.0, throttle)   # hold level, climb -- disabled per team: untuned, works worse
        if _frame % LINE_PRINT_EVERY == 0:
            print(f'[line follow] no edge found | state={_line_state} '
                  f'(computed throttle={throttle:.3f}, NOT sent)')
    else:
        ys, xs, m, b, poly, closest_pt = fit
        pitch = set_pitch(ys, xs, poly)
        roll  = set_roll(xs, dt)
        yaw   = set_yaw(m, dt)
        # drone.flight.send_pcmd(pitch, roll, yaw, throttle)   # disabled per team: untuned altitude control, works worse
        drone.flight.send_pcmd(pitch, roll, yaw, 0.0)

        if _frame % LINE_PRINT_EVERY == 0:
            print(f'[line follow] pitch={pitch:.3f} roll={roll:.3f} yaw={yaw:.3f} '
                  f'throttle=0.0 (computed={throttle:.3f}, not sent) state={_line_state}')


# ============================================================
# Mode: GATE
# ============================================================
def _update_gate(drone, dt):
    global _mode, _gate_phase, _gate, _hold, _forward_time, _forward_dist
    global _line_state, _base_alt, _visible, _vis_timer, _settle_frame, _gate_confirm, _gate_center_time

    if _gate_phase == GATE_CENTER:
        img = drone.camera.get_color_image()
        if img is None:
            print("[gate] [Error]: Image is none!")
            return

        _gate = neo_lab.detect_gate(img)
        _gate_center_time += dt

        if _gate is None:
            _update_optical_flow(drone, dt)
            phys_vx, phys_vy, phys_vz = _filtered_velocity(drone, dt)  # low-passed right, up, forward

            if _flow_age <= FLOW_MAX_AGE:
                vx, vz, vel_src = _flow_vx, _flow_vz, 'flow'
            else:
                vx, vz, vel_src = phys_vx, phys_vz, 'phys-fallback'
            vy = phys_vy   # vertical stays on physics; flow tracker here doesn't estimate climb/descend

            hold_roll     = _search_roll_pid.update(-vx, dt)      # cancel lateral drift
            hold_throttle = _search_throttle_pid.update(-vy, dt)  # cancel vertical drift
            hold_pitch    = _search_pitch_pid.update(-vz, dt)     # cancel forward/back drift

            if _gate_center_time < GATE_SEARCH_START_DELAY:
                # Still settling in - actively brake residual yaw (same sign
                # convention as centering's yaw brake: physics angular
                # velocity is CCW-positive/RHR, send_pcmd's yaw is CW-positive,
                # so a positive coefficient on yaw_rate opposes the spin) while
                # holding position, giving detection a chance to catch a gate
                # that's already in view before we start actively sweeping.
                settle_yaw_rate = float(drone.physics.get_angular_velocity()[1])
                settle_yaw_brake = _search_yaw_pid.update(settle_yaw_rate, dt)
                _send_pcmd_slewed(drone, hold_pitch, hold_roll, settle_yaw_brake, hold_throttle, dt)
                _hold = 0.0
                _gate_roll_pid.hold()
                _gate_alt_pid.hold()
                _gate_pitch_pid.hold()
                _gate_yaw_pid.hold()
                if _frame % GATE_PRINT_EVERY == 0:
                    print(f'[gate] settling before search ({_gate_center_time:.2f}/{GATE_SEARCH_START_DELAY}s) '
                          f'| src={vel_src} vx={vx:.3f} vy={vy:.3f} vz={vz:.3f} '
                          f'| flow_vx={_flow_vx:.3f} flow_vz={_flow_vz:.3f} phys_vx={phys_vx:.3f} phys_vz={phys_vz:.3f} '
                          f'| yaw_rate={settle_yaw_rate:.3f} yaw_brake={settle_yaw_brake:.3f}')
                return

            _search_yaw_pid.hold()  # not used during active sweep - yaw is deliberately driven by SEARCH_YAW
            _send_pcmd_slewed(drone, hold_pitch, hold_roll, SEARCH_YAW, hold_throttle, dt)
            _hold = 0.0
            _gate_roll_pid.hold()
            _gate_alt_pid.hold()
            _gate_pitch_pid.hold()
            _gate_yaw_pid.hold()
            if _frame % GATE_PRINT_EVERY == 0:
                search_yaw_rate = float(drone.physics.get_angular_velocity()[1])
                print(f'[gate] no gate found - searching | src={vel_src} vx={vx:.3f} vy={vy:.3f} vz={vz:.3f} '
                      f'| flow_vx={_flow_vx:.3f} flow_vz={_flow_vz:.3f} phys_vx={phys_vx:.3f} phys_vz={phys_vz:.3f} '
                      f'| hold pitch={hold_pitch:.3f} roll={hold_roll:.3f} throttle={hold_throttle:.3f} '
                      f'| yaw_rate={search_yaw_rate:.3f} (SEARCH_YAW={SEARCH_YAW}, should be opposite sign if CCW-positive)')
            return

        width, height = drone.camera.get_width(), drone.camera.get_height()
        img_cx, img_cy = width / 2.0, height / 2.0

        _search_roll_pid.hold()
        _search_throttle_pid.hold()
        _search_pitch_pid.hold()
        _search_yaw_pid.hold()
        # All four axes now use dedicated centering PIDs (GATE_ROLL_*, GATE_ALT_*,
        # GATE_PITCH_*, GATE_YAW_*), independent from the search-hold PIDs above
        # and from each other - tune each one separately.

        # normalized error in [-1, 1]: +err_x = gate right of center
        err_x = (_gate.cx - img_cx) / (width / 2.0)
        roll = _gate_roll_pid.update(err_x, dt)

        # +err_alt = gate below center -> physically below the camera's
        # boresight -> drone needs to DESCEND to bring it to center, hence
        # the negation (positive err_alt must produce negative/descend throttle).
        err_alt = (_gate.cy - img_cy) / (height / 2.0)
        throttle = _gate_alt_pid.update(-err_alt, dt)

        # Actively brake any residual yaw rate left over from the search sweep,
        # rather than just commanding 0 and waiting for momentum to decay.
        #
        # SIGN NOTE (verified against docs): physics.get_angular_velocity()'s
        # y-axis (yaw) uses the right-hand rule -> CCW-positive when viewed
        # from above. send_pcmd's yaw argument is CW-positive (-1=CCW,
        # +1=CW per the Flight module docs). These are OPPOSITE conventions
        # for the same physical rotation, so the brake needs a POSITIVE
        # coefficient on yaw_rate, not negative - flipped from the earlier
        # version, which was actually reinforcing spin instead of killing it.
        yaw_rate = float(drone.physics.get_angular_velocity()[1])  # y-axis is yaw, CCW-positive (RHR)
        yaw_brake = _gate_yaw_pid.update(yaw_rate, dt)

        # Actively brake residual forward/backward drift too - roll and throttle
        # have dedicated pixel-based centering PIDs, but pitch had NOTHING
        # driving it (hardcoded 0) and was silently letting the drone drift
        # toward/away from the gate uncorrected the whole time centering ran.
        _update_optical_flow(drone, dt)
        _, _, phys_vz = _filtered_velocity(drone, dt)
        vz = _flow_vz if _flow_age <= FLOW_MAX_AGE else phys_vz
        pitch = _gate_pitch_pid.update(-vz, dt)

        _send_pcmd_slewed(drone, pitch, roll, yaw_brake, throttle, dt)

        if _frame % GATE_PRINT_EVERY == 0:
            print(f'[gate] centering | gate=({_gate.cx:.0f},{_gate.cy:.0f}) tags={_gate.count} '
                  f'err_x={err_x:.3f} err_alt={err_alt:.3f} yaw_rate={yaw_rate:.3f} yaw_brake={yaw_brake:.3f} '
                  f'drift_vz={vz:.3f} pitch_brake={pitch:.3f} '
                  f'roll={roll:.3f} throttle={throttle:.3f} hold={_hold:.2f}')

        enough_tags = _gate.count >= GATE_MIN_TAGS_TO_ADVANCE
        if abs(err_x) < CENTER_TOL and abs(err_alt) < ALT_TOL and enough_tags:
            _hold += dt
            if _hold >= CENTER_HOLD_T:
                drone.flight.stop()
                _reset_slew()
                print('[gate] centered - flying through.')
                _gate_phase = GATE_FORWARD
                _forward_time = 0.0
                _forward_dist = 0.0
        else:
            if abs(err_x) < CENTER_TOL and abs(err_alt) < ALT_TOL and not enough_tags:
                if _frame % GATE_PRINT_EVERY == 0:
                    print(f'[gate] position looks centered but only {_gate.count} tag(s) visible '
                          f'(need {GATE_MIN_TAGS_TO_ADVANCE}) - not counting toward hold yet.')
            _hold = 0.0

    elif _gate_phase == GATE_FORWARD:
        _forward_time += dt

        vel = drone.physics.get_linear_velocity()
        forward_speed = vel[2]
        _forward_dist += forward_speed * dt

        _send_pcmd_slewed(drone, GATE_PITCH, 0, 0, 0, dt)

        if _frame % GATE_PRINT_EVERY == 0:
            remaining = None if _forward_target is None else (_forward_target - _forward_dist)
            print(f'[gate] forward | speed={forward_speed:.3f} m/s '
                  f'dist={_forward_dist:.2f}m target={_forward_target} '
                  f'remaining={remaining} t={_forward_time:.2f}s')

        target_reached = (_forward_target is not None and _forward_dist >= _forward_target)
        time_capped     = _forward_time >= FORWARD_MAX_TIME

        if target_reached or time_capped:
            if time_capped and not target_reached:
                print(f'[gate] hit time cap ({FORWARD_MAX_TIME}s) before reaching '
                      f'target distance - check FORWARD_MAX_TIME / gate distance reading.')
            else:
                print(f'[gate] passed - traveled {_forward_dist:.2f}m - resuming line following.')
            drone.flight.stop()
            _reset_slew()
            _mode = LINE_FOLLOW
            _yaw_pid.reset()
            _roll_pid.reset()
            _line_state   = FOLLOWING
            _base_alt     = None
            _visible      = True
            _vis_timer    = 0.0
            _settle_frame = _frame + GATE_SETTLE_FRAMES
            _gate_confirm = 0


# ============================================================
# Main loop
# ============================================================
def update(drone):
    global _timer, _done, _frame
    if _done:
        return True
    dt = drone.get_delta_time()
    _timer += dt
    _frame += 1

    # if _mode == LINE_FOLLOW:
    #     _update_line_follow(drone, dt)
    # elif _mode == GATE:
    _update_gate(drone, dt)

    if _timer >= FOLLOW_TIME:
        _done = True
    return _done


if __name__ == "__main__":
    _drone = drone_core.create_drone()
    _launcher = neo_lab.Launcher(1.4)

    def start():
        reset()
        _launcher.reset()
        print("Combined lab: line following + gate traversal")

    def _update():
        if not _launcher.done:
            _launcher.update(_drone)
            return
        if update(_drone):
            _drone.flight.land()

    _drone.set_start_update(start, _update)
    _drone.go(not neo_lab._is_sim(_drone))