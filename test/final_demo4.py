"""
MIT BWSI Autonomous Drone Racing Course - UAV Neo
GNU General Public License v3.0

final_demo4 -- tangent_line_filtered line following + final_demo3 gate throttle.

This is a merge of two files:

  LINE FOLLOWING  comes from linefollower/tangent_line_filtered_line_follow.py,
                  unchanged in substance. Raw bright pixels are fitted with a
                  degree-3 polynomial x = f(y); the point on that curve nearest
                  TARGET_POINT is picked with a continuity bias so the tracked
                  point cannot hop across the curve on sharp corners; the tangent
                  SLOPE at that point is low-passed and drives yaw. Roll centres
                  the mean bright column, pitch is scheduled off the fit residual.
                  Gains (YAW_KP 0.56 / ROLL_KP 0.23, CURVE_SCALE 70) are the ones
                  tuned for that error definition and carry over as-is.

  GATE THROTTLE   comes from final_demo3.py, unchanged. ArUco tags are clustered
                  to the nearest gate, roles (top/bottom/left/right) are resolved
                  by id, the opening centre is reconstructed from the geometry
                  with a confidence, and close in the vertical target is latched
                  to an absolute altitude for the pass.

The design rule from final_demo3 is preserved exactly:

    pitch, roll and yaw ALWAYS come from line following.
    Gate detection may only ever influence THROTTLE.

WHAT THE MERGE ITSELF CHANGED (neither parent had to deal with this)

  1. set_throttle() from the line follower is split into _track_line_visibility()
     and line_throttle(). The visibility hysteresis has to be updated EVERY frame,
     including while a gate is driving the throttle -- otherwise its timers freeze
     during the pass and the FSM resumes on stale state.
  2. The two throttle sources are cross-faded (GATE_BLEND_TAU) rather than hard
     switched, so a flickering ArUco decode cannot slam the vertical axis.
  3. The line follower sent send_pcmd(..., 0.0) -- throttle was hard-wired to zero
     for bench testing and its state machine output was discarded. Throttle is now
     actually flown.
  4. _prev_y0 and _m_filt are invalidated on line loss. In the parent a loss lasted
     a frame or two; here a gate pass or a search climb can last seconds, and a
     pre-loss slope dumped into the yaw PID on the first reacquired frame is a real
     kick.
  5. dt is clamped, the altitude read coasts on its last good value, and find_edge
     degrades to None instead of raising. One dropped frame must not take down the
     flight loop.
  6. The pixel-count test counts pixels. The parent compared np.count_nonzero() of
     an Nx2 coordinate array, which is ~2N -- so its threshold of 200 was really
     ~100 pixels, and MIN_PIXELS is set to 100 here to keep the same sensitivity.

CALIBRATION
  The GATE GEOMETRY block is physical hardware data, carried over from
  final_demo3. To measure GATE_H_PER_TAG from photos, use final_demo3's replay
  mode -- the gate code here is identical, so its numbers apply directly:
      python final_demo3.py --replay <image-or-dir>
"""

import math

import cv2
import numpy as np

# -- Course setup: makes the shared `neo_lab` helper importable.
#    You don't need to read or change this block. --
import os as _os, sys as _sys
_d = _os.path.dirname(_os.path.realpath(__file__))
while _os.path.basename(_d) != "labs" and _os.path.dirname(_d) != _d:
    _d = _os.path.dirname(_d)
if _d not in _sys.path:
    _sys.path.insert(0, _d)

import drone_core
import drone_utils as uav_utils
import neo_lab


# ===========================================================================
# GENERAL
# ===========================================================================
DEBUG_PRINT        = True
DEBUG_PERIOD_S     = 0.25    # s between debug lines. The parent printed pitch/
                             # roll/yaw and curviness every frame; at 30-60 Hz
                             # that adds real jitter to dt, which then feeds every
                             # derivative term in the loop.

FOLLOW_TIME        = 1.0e6   # s of flight before landing; effectively disabled
DT_MIN, DT_MAX     = 1.0e-3, 0.1     # clamp on get_delta_time()


# ===========================================================================
# LINE PERCEPTION  (from tangent_line_filtered_line_follow.py)
# ===========================================================================
V_MIN         = 200      # HSV Value threshold for the bright line
MIN_PIXELS    = 100      # min bright PIXELS to trust a frame. NOTE: the parent
                         # compared count_nonzero of an Nx2 coordinate array,
                         # which is ~2N -- its 200 was effectively this 100.
POLY_DEGREE   = 3        # 3 or 5 both work; higher degree fits noise more easily
IMG_W, IMG_H  = 640, 480
TARGET_POINT  = (IMG_W / 2, IMG_H / 2 - 80)   # (x, y) -- "slightly higher" than
                                              # centre, i.e. ahead of the drone
SAMPLE_STEP   = 2        # px spacing when scanning the curve for the closest point

# -- Tangent-point smoothing (fixes yaw sign-flips on sharp corners) --
CONTINUITY_WEIGHT = 0.25 # how strongly to penalise the closest-point search from
                         # jumping to a different point than last frame.
                         # 0 = pure distance-to-target. TUNE.
M_TAU         = 0.12     # low-pass time constant (s) for the tangent slope m.
                         # Bigger = smoother but more lag. TUNE alongside D_TAU.


# ===========================================================================
# LINE CONTROL  (from tangent_line_filtered_line_follow.py)
# ===========================================================================
MAX_ROLL      = 0.25     # strafe authority for centring
MAX_YAW       = 1.0      # yaw authority
IMAGE_CENTER  = 320      # 640-wide image -> centre column

PITCH_STRAIGHT = 0.22    # fast on straights
PITCH_TURN     = 0.08    # slow through turns
CURVE_SCALE    = 70      # residual std at which you're "fully" in a turn (TUNE)

# -- PID gains --
# Raise KP until it oscillates, then halve it. Add KD to damp, and KI only if the
# drone consistently settles off-centre.
YAW_KP        = 0.56     # error is -m (tangent SLOPE), not an angle
YAW_KI        = 0.0
YAW_KD        = 0.055
YAW_I_LIMIT   = 0.20     # cap on the integral's contribution to yaw

ROLL_KP       = 0.23
ROLL_KI       = 0.0
ROLL_KD       = 0.02
ROLL_I_LIMIT  = 0.10

D_TAU         = 0.10     # derivative low-pass time constant, s (bigger = smoother)


# ===========================================================================
# LINE-SEARCH THROTTLE STATE MACHINE
# ===========================================================================
MAX_CLIMB        = 3.0   # m above the reference height to search
CLIMB_THROTTLE   = 0.3   # send_pcmd throttle while searching
DESCEND_THROTTLE = -0.2  # gentler than the climb on purpose
LOST_GRACE       = 0.4   # s of no-line before climbing
FOUND_GRACE      = 0.5   # s of line before committing to descend
HEIGHT_TOL       = 0.10  # m; "back at the reference height"


# ===========================================================================
# GATE GEOMETRY -- PHYSICAL CONSTANTS. CALIBRATE THESE.  (from final_demo3.py)
# ===========================================================================
# These describe the gate hardware and the camera. Pixel thresholds are DERIVED
# from them further down, so once these are right the behaviour thresholds are
# expressed in metres and stop needing to be re-tuned when the course changes.

GATE_INNER_HEIGHT_M = 1.524  # m, vertical clear opening (inside edge to inside
                             # edge). (60 in hoop -> 60 * 0.0254 = 1.524 m)
GATE_INNER_WIDTH_M  = 1.524  # m, horizontal clear opening. Hoop is circular, so
                             # width == height == diameter.
GATE_TAG_SIZE_M     = 0.2667 # m, side of the printed ArUco square, black border
                             # included. (10.5 in -> 10.5 * 0.0254 = 0.2667 m)

# The course gates carry FOUR tags at the MIDPOINT of each edge -- top, bottom,
# left, right -- forming a diamond. This is NOT the "one per corner" layout
# neo_lab's docstring describes, and the difference is what makes partial
# detections biased.
#
# GATE_H_PER_TAG converts one role-known tag into a gate centre:
#     cy_centre = cy_top + GATE_H_PER_TAG * tag_px
#
# CALIBRATE: hover facing a gate with all four tags visible and print
#     (row_of_bottom_tag - row_of_top_tag) / 2 / mean_tag_side_px
# (final_demo3.py --replay prints exactly this number for any such photo.)
# Until then it is estimated from the dimensions above, assuming the tag centre
# sits on the frame midline.
_GATE_H_PER_TAG_OVERRIDE = None
GATE_H_PER_TAG = (_GATE_H_PER_TAG_OVERRIDE if _GATE_H_PER_TAG_OVERRIDE is not None
                  else (GATE_INNER_HEIGHT_M / 2.0 + GATE_TAG_SIZE_M / 2.0) / GATE_TAG_SIZE_M)

# Hard-coded id -> role map. With fixed ids per position, role resolution is exact
# even on single-tag frames. Leave empty to learn roles online from 3+ tag views.
GATE_TAG_ROLE_BY_ID = {
    # Gate 1
    35: 'T', 0:  'L', 36: 'B', 34: 'R',
    # Gate 2
    41: 'T', 40: 'L', 44: 'B', 42: 'R',
    # Gate 3
    46: 'T', 47: 'L', 43: 'B', 45: 'R',
    # Gate 4
    39: 'T', 76: 'L', 37: 'B', 38: 'R',
}

# ArUco dictionary. neo_lab's docstring says DICT_6X6_250 but its code uses
# DICT_5X5_100 -- they cannot both be right. VERIFY THIS FIRST: if it is wrong
# nothing decodes at all and every gate behaviour below is dead code.
GATE_ARUCO_DICT = "DICT_5X5_100"

# Camera intrinsics for the FORWARD camera, used to convert pixels to metres.
GATE_CAM_HFOV_DEG   = 82.0   # horizontal field of view. CALIBRATE.
_GATE_FOCAL_PX_OVERRIDE = None   # set directly if you have a calibrated fx


# ===========================================================================
# GATE DETECTION / FILTERING
# ===========================================================================
GATE_MIN_TAG_PX      = 8.0   # ignore tags smaller than this (too far / noise)
GATE_CLUSTER_SCALE   = 2.0   # tags belong to the same gate if their side length
                             # is within this factor of the largest tag's
GATE_CLUSTER_RADIUS  = 9.0   # ...and within this many tag-widths of it
GATE_ROLE_MIN_TAGS   = 3     # tags needed before roles may be learned. With 2 the
                             # assignment is genuinely ambiguous.

# Confidence per estimate quality. These scale the PID output directly.
GATE_CONF_TB_PAIR    = 1.00  # top+bottom: exact
GATE_CONF_LR         = 0.90  # left and/or right: each individually unbiased
GATE_CONF_SINGLE_TB  = 0.55  # one top or bottom + GATE_H_PER_TAG
GATE_CONF_UNKNOWN    = 0.25  # roles unresolved: fall back to the raw mean
GATE_CONF_MIN        = 0.20  # below this the estimate is discarded entirely

GATE_MIN_FRAMES      = 2     # consecutive detections before gate control engages
GATE_LOST_HOLD_S     = 0.30  # s to keep trusting a gate after it stops decoding


# ===========================================================================
# GATE CONTROL
# ===========================================================================
GATE_ALT_KP         = 0.50   # on normalised vertical image error
GATE_ALT_KI         = 0.0
GATE_ALT_KD         = 0.05
GATE_ALT_I_LIMIT    = 0.10
GATE_THROTTLE_LIMIT = 0.30

GATE_BLEND_TAU      = 0.15   # s, cross-fade between gate and line throttle

GATE_LATCH_DIST_M   = 1.50   # m. Inside this range the vertical target is frozen
                             # to an absolute altitude, because the tags are about
                             # to leave the frame.
GATE_LATCH_MAX_S    = 4.0    # s. Hard timeout so the latch can never hang.
GATE_LATCH_EXIT_S   = 0.60   # s without any detection while latched -> assume the
                             # gate has been passed. Must exceed GATE_LOST_HOLD_S.
GATE_LATCH_MIN_CONF = 0.90   # min confidence to freeze an absolute altitude
GATE_RELATCH_S      = 1.00   # s cooldown after a latch ends, so a latch that times
                             # out on a still-visible gate cannot re-arm instantly

GATE_ALT_HOLD_KP    = 0.35   # throttle per metre of altitude error while latched
GATE_MAX_ALT_STEP_M = 1.50   # m. Cap on how far one gate may move the target
                             # altitude -- a bad estimate cannot command a huge climb.

GATE_REBASELINE     = True   # after passing a gate, treat the new altitude as the
                             # reference the line-search FSM returns to


# ===========================================================================
# DERIVED (do not edit; edit the physical constants above)
# ===========================================================================
def _focal_px(img_w):
    """Focal length in px for an image `img_w` wide, from the horizontal FOV."""
    if _GATE_FOCAL_PX_OVERRIDE is not None:
        return float(_GATE_FOCAL_PX_OVERRIDE)
    half = math.radians(GATE_CAM_HFOV_DEG) * 0.5
    t = math.tan(half)
    if t <= 1e-6:
        return float(img_w)
    return (img_w * 0.5) / t


# ===========================================================================
# PID
# ===========================================================================
class PID:
    """
    Setpoint is always 0 (line vertical / line centred / gate centred), so
    `error` IS the measurement. Low-passed derivative, clamped integral with
    conditional anti-windup.

    NOTE: with KI = 0 the integral and anti-windup paths do nothing. They are
    kept because they are correct and cost nothing, but do not tune around them
    expecting an effect until KI is non-zero.
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
        """Clear all history. Call at the start of a run."""
        self._integral = 0.0
        self._prev_error = None
        self._deriv = 0.0

    def hold(self):
        """
        Call on frames where the measurement is invalid (line lost). Keeps the
        integral but drops the previous error, so reacquiring does not produce a
        derivative spike from the stale value.
        """
        self._prev_error = None

    def update(self, error, dt):
        error = float(error)
        if not np.isfinite(error):          # a NaN here would poison the
            self.hold()                     # integral for the rest of the run
            return 0.0
        if dt <= 0.0:
            return uav_utils.clamp(self.kp * error, -self.out_limit, self.out_limit)

        # -- Derivative, low-passed. Raw d/dt of a pixel measurement is very
        #    noisy; the filter is what makes KD usable at all.
        if self._prev_error is None:
            raw_deriv = 0.0
            self._deriv = 0.0
        else:
            raw_deriv = (error - self._prev_error) / dt
        alpha = dt / (self.d_tau + dt)
        self._deriv += alpha * (raw_deriv - self._deriv)
        self._prev_error = error

        integral = uav_utils.clamp(self._integral + error * dt,
                                   -self.i_limit, self.i_limit)
        output = self.kp * error + self.ki * integral + self.kd * self._deriv
        clamped = uav_utils.clamp(output, -self.out_limit, self.out_limit)

        # Anti-windup: discard the update only if we are pinned at the limit AND
        # this error pushes further into it.
        if not (output != clamped and output * error > 0.0):
            self._integral = integral
        return clamped


# ===========================================================================
# MODULE STATE
# ===========================================================================
FOLLOWING, SEARCHING, DESCENDING = "FOLLOWING", "SEARCHING", "DESCENDING"

_timer = 0.0
_done = False

_state = FOLLOWING
_base_alt = None            # altitude the search FSM returns to
_last_alt = None            # last good altitude read, to coast a failed one
_visible = True             # line visible?
_vis_timer = 0.0

_prev_y0 = None             # last frame's tangent-point row, for continuity bias
_m_filt = 0.0               # low-passed tangent slope
_m_valid = False            # False after a loss, so the filter snaps on reacquire

_role_by_id = {}            # learned tag id -> 'T'|'B'|'L'|'R'
_gate_streak = 0            # consecutive frames with a usable gate
_gate_seen_t = 1.0e9        # s since a gate last decoded
_last_obs = None            # last good GateObs, to coast through dropped frames
_gate_blend = 0.0           # 0 = pure line throttle, 1 = pure gate throttle
_gate_latched = False
_gate_target_alt = None
_gate_latch_t = 0.0
_relatch_t = 1.0e9          # s since the last latch ended (cooldown)
_dbg_t = 0.0

_yaw_pid = PID(YAW_KP, YAW_KI, YAW_KD, MAX_YAW, YAW_I_LIMIT)
_roll_pid = PID(ROLL_KP, ROLL_KI, ROLL_KD, MAX_ROLL, ROLL_I_LIMIT)
_gate_alt_pid = PID(GATE_ALT_KP, GATE_ALT_KI, GATE_ALT_KD,
                    GATE_THROTTLE_LIMIT, GATE_ALT_I_LIMIT)


def reset():
    """Clear all run state. Roles are NOT cleared -- they are hardware facts."""
    global _timer, _done, _state, _base_alt, _visible, _vis_timer, _last_alt
    global _prev_y0, _m_filt, _m_valid
    global _gate_streak, _gate_seen_t, _gate_blend, _last_obs
    global _gate_latched, _gate_target_alt, _gate_latch_t, _relatch_t, _dbg_t
    _timer = 0.0
    _done = False
    _state = FOLLOWING
    _base_alt = None
    _last_alt = None
    _visible = True
    _vis_timer = 0.0
    _prev_y0 = None
    _m_filt = 0.0
    _m_valid = False
    _gate_streak = 0
    _gate_seen_t = 1.0e9
    _last_obs = None
    _gate_blend = 0.0
    _gate_latched = False
    _gate_target_alt = None
    _gate_latch_t = 0.0
    _relatch_t = 1.0e9
    _dbg_t = 0.0
    _yaw_pid.reset()
    _roll_pid.reset()
    _gate_alt_pid.reset()


# ===========================================================================
# LINE PERCEPTION  (from tangent_line_filtered_line_follow.py)
# ===========================================================================
def find_edge(drone, dt):
    """
    Grab the downward image, threshold it, and fit a polynomial to the bright
    pixels (column as a function of row: x = f(y)).

    Finds the point on the fitted curve closest to TARGET_POINT (image centre, or
    a bit above it), biased toward staying near last frame's point so the tracked
    point can't hop across the curve when two points are briefly equidistant from
    the target (this was the source of the yaw sign-flips on sharp corners). The
    resulting tangent slope is then low-pass filtered before being returned, since
    even a continuity-biased search can still shift a little frame to frame on a
    noisy fit.

    Returns (ys, xs, m, b, poly, closest_pt) where `poly` is the fitted np.poly1d
    (used downstream for a curvature/straightness measure that reflects the actual
    fit, not just the local tangent), m/b describe the *filtered* tangent line
    (column = m*row + b), and closest_pt is the (row, col) point used before
    filtering. Returns None if the frame can't be trusted.

    Never raises: a bad frame or a singular fit degrades to None rather than
    killing the flight loop.
    """
    global _prev_y0, _m_filt, _m_valid

    try:
        camera = drone.camera.get_downward_image()
        if camera is None or getattr(camera, "size", 0) == 0:
            return None

        mask = neo_lab.bright_mask_improved(camera, V_MIN)
        if mask is None or mask.ndim != 2:
            return None

        edges = np.argwhere(mask).astype(np.float64)

        # Count PIXELS (rows of the coordinate array). The parent counted nonzero
        # coordinate VALUES here, which is ~2x the pixel count; MIN_PIXELS is
        # halved to match.
        if edges.shape[0] < MIN_PIXELS:
            return None

        ys = edges[:, 0]
        xs = edges[:, 1]

        # Fit x = f(y) with a degree-3 (or 5) polynomial instead of a line.
        with np.errstate(all="ignore"):
            coeffs = np.polyfit(ys, xs, POLY_DEGREE)
        if not np.all(np.isfinite(coeffs)):
            return None
        poly = np.poly1d(coeffs)
        poly_deriv = poly.deriv()

        # Sample the curve over the observed row range to find the point closest
        # to the target (image centre or slightly above it).
        y_min, y_max = ys.min(), ys.max()
        sample_ys = np.arange(y_min, y_max + SAMPLE_STEP, SAMPLE_STEP)
        if sample_ys.size == 0:
            return None
        sample_xs = poly(sample_ys)

        target_x, target_y = TARGET_POINT
        dist_sq = (sample_xs - target_x) ** 2 + (sample_ys - target_y) ** 2

        # Continuity bias: penalise points far from where we were tracking last
        # frame, so the selection can't teleport to a different branch of the
        # curve just because it's momentarily closer to the target.
        if _prev_y0 is not None:
            dist_sq = dist_sq + CONTINUITY_WEIGHT * (sample_ys - _prev_y0) ** 2

        best_idx = int(np.argmin(dist_sq))
        y0 = float(sample_ys[best_idx])
        x0 = float(sample_xs[best_idx])
        _prev_y0 = y0

        # Raw tangent slope at the selected point.
        m_raw = float(poly_deriv(y0))
        if not np.isfinite(m_raw):
            return None

        # Low-pass the slope itself -- this is what actually feeds the yaw PID, so
        # filtering here (not just the point selection) catches noise that
        # continuity-biasing alone doesn't. On reacquire the filter SNAPS instead
        # of dragging a pre-loss slope back in; a gate pass or a search climb can
        # leave _m_filt stale for seconds.
        if (not _m_valid) or dt <= 0.0:
            _m_filt = m_raw
            _m_valid = True
        else:
            alpha = dt / (M_TAU + dt)
            _m_filt += alpha * (m_raw - _m_filt)
        m = _m_filt
        b = x0 - m * y0

        return ys, xs, m, b, poly, (y0, x0)
    except Exception:           # LinAlgError, camera failure, singular fit
        return None


# ===========================================================================
# LINE CONTROL AXES  (from tangent_line_filtered_line_follow.py)
# ===========================================================================
def set_yaw(m, dt):
    """Rotate to align the drone's heading with the edge's slope."""
    error = -m          # 0 when the line runs straight up the image
    return _yaw_pid.update(error, dt)


def set_roll(xs, dt):
    """Strafe to bring the average edge column back to the image centre."""
    edge_col = xs.mean()      # average column of the bright edge
    error = (edge_col - IMAGE_CENTER) / IMAGE_CENTER   # -1 (left) .. +1 (right)
    return _roll_pid.update(error, dt)


def set_pitch(ys, xs, poly):
    """
    Fly fast when the edge fits the polynomial tightly (straight/simple), slow
    when the raw pixels deviate a lot from the fit (curvy/noisy).

    Note: this measures deviation from the polynomial fit itself, not from the
    tangent line used for steering -- the tangent only approximates the curve near
    the steering point, so using it here would read as "curvy" even on a straight
    line.
    """
    curviness = float(np.std(xs - poly(ys)))
    straightness = uav_utils.clamp(1.0 - curviness / CURVE_SCALE, 0.0, 1.0)
    return PITCH_TURN + (PITCH_STRAIGHT - PITCH_TURN) * straightness


# ===========================================================================
# GATE PERCEPTION  (from final_demo3.py)
# ===========================================================================
class GateObs:
    """
    One frame's gate estimate.

      cy        image row of the gate OPENING centre (not of the tag centroid)
      conf      0..1 confidence in cy; scales control authority
      tag_px    mean tag side in px
      dist_m    range estimate, m (None if intrinsics unusable)
      roles     {id: 'T'|'B'|'L'|'R'} for the tags used
    """

    __slots__ = ("cx", "cy", "conf", "tag_px", "dist_m", "ids", "roles",
                 "count", "img_h", "img_w")

    def __init__(self, cx, cy, conf, tag_px, dist_m, ids, roles, img_h, img_w):
        self.cx = cx
        self.cy = cy
        self.conf = conf
        self.tag_px = tag_px
        self.dist_m = dist_m
        self.ids = ids
        self.roles = roles
        self.count = len(ids)
        self.img_h = img_h
        self.img_w = img_w


_aruco_cache = {"dict": None, "detector": None, "warned": False}


def _aruco_detect(gray):
    """
    Run ArUco detection across OpenCV API versions. Returns (corners, ids) or
    (None, None). Never raises -- a cv2 problem must not take down the loop.
    """
    try:
        if _aruco_cache["dict"] is None:
            aruco = cv2.aruco
            key = getattr(aruco, GATE_ARUCO_DICT)
            try:
                _aruco_cache["dict"] = aruco.getPredefinedDictionary(key)
            except AttributeError:
                _aruco_cache["dict"] = aruco.Dictionary_get(key)

        if _aruco_cache["detector"] is None:
            try:                                   # OpenCV >= 4.7
                _aruco_cache["detector"] = cv2.aruco.ArucoDetector(
                    _aruco_cache["dict"], cv2.aruco.DetectorParameters())
            except AttributeError:
                _aruco_cache["detector"] = False   # use the free-function API

        if _aruco_cache["detector"] is not False:
            corners, ids, _ = _aruco_cache["detector"].detectMarkers(gray)
        else:
            # params must come from _create() here: the direct constructor
            # segfaults detectMarkers on the old API.
            params = cv2.aruco.DetectorParameters_create()
            corners, ids, _ = cv2.aruco.detectMarkers(
                gray, _aruco_cache["dict"], parameters=params)
        return corners, ids
    except Exception as exc:
        if not _aruco_cache["warned"]:
            _aruco_cache["warned"] = True
            print(f"[gate] ArUco detection unavailable ({exc}); "
                  f"flying on line following only")
        return None, None


def _tag_side_px(quad):
    """Mean of the four edge lengths -- steadier than a single edge under skew."""
    p = np.asarray(quad, dtype=np.float64).reshape(-1, 2)
    if p.shape[0] < 4:
        return 0.0
    d = np.linalg.norm(p - np.roll(p, -1, axis=0), axis=1)
    return float(np.mean(d))


def _cluster_nearest_gate(centers, sides):
    """
    Keep only the tags belonging to the NEAREST gate.

    detect_gate() in neo_lab pools every decoded tag in the image, so with gate
    N+1 visible through gate N the "centre" it returns is a point in empty air
    between the two. Apparent tag size is a strong depth cue, so anchor on the
    largest tag and drop anything at a different scale or too far away from it.
    """
    ref = int(np.argmax(sides))
    s_ref = sides[ref]
    if s_ref <= 0.0:
        return np.zeros(len(sides), dtype=bool)
    scale_ok = (sides > s_ref / GATE_CLUSTER_SCALE) & (sides < s_ref * GATE_CLUSTER_SCALE)
    near_ok = np.linalg.norm(centers - centers[ref], axis=1) < GATE_CLUSTER_RADIUS * s_ref
    keep = scale_ok & near_ok
    keep[ref] = True
    return keep


def _learn_roles(centers, ids):
    """
    Assign T/B/L/R from each tag's offset from the cluster centroid, and cache the
    mapping by tag id.

    Only attempted with GATE_ROLE_MIN_TAGS or more: with two tags the assignment
    is genuinely ambiguous. Ids are stable per physical tag, so paying this once
    per gate buys correct roles on the sparse frames that actually need them.
    """
    if len(centers) < GATE_ROLE_MIN_TAGS:
        return
    off = centers - centers.mean(axis=0)
    guess = {}
    for i, tid in enumerate(ids):
        dx, dy = float(off[i][0]), float(off[i][1])
        if abs(dy) > abs(dx):
            guess[tid] = "B" if dy > 0.0 else "T"     # image rows increase downward
        else:
            guess[tid] = "R" if dx > 0.0 else "L"
    # Reject inconsistent assignments (two tags claiming the same position); a
    # skewed view can confuse the split, and a wrong cached role is worse than no
    # cached role.
    if len(set(guess.values())) == len(guess):
        _role_by_id.update(guess)


def _estimate_center_row(centers, sides, ids):
    """
    Reconstruct the image row of the gate OPENING from the tags on hand.

    Tags sit at edge midpoints, so:
      - top+bottom  -> exact midpoint
      - left/right  -> each sits at the gate's centre height, so either alone is
                       unbiased
      - a lone top or bottom -> offset by the fixed physical ratio GATE_H_PER_TAG
      - roles unknown -> raw mean, at low confidence

    Averaging everything biases an unpaired top/bottom by h/3 to h. Worse, the top
    tag is the first to leave frame when approaching from below, and losing it
    biases the estimate in the direction the drone was ALREADY erring -- positive
    feedback that converges onto the bottom bar.
    """
    rows = {}
    for i, tid in enumerate(ids):
        role = GATE_TAG_ROLE_BY_ID.get(tid, _role_by_id.get(tid))
        if role is not None:
            rows.setdefault(role, []).append(float(centers[i][1]))
    mean_of = lambda k: float(np.mean(rows[k]))

    if "T" in rows and "B" in rows:
        return 0.5 * (mean_of("T") + mean_of("B")), GATE_CONF_TB_PAIR

    lr = [r for k in ("L", "R") if k in rows for r in rows[k]]
    if lr:
        return float(np.mean(lr)), GATE_CONF_LR

    h_px = GATE_H_PER_TAG * float(np.mean(sides))
    if "T" in rows:
        return mean_of("T") + h_px, GATE_CONF_SINGLE_TB
    if "B" in rows:
        return mean_of("B") - h_px, GATE_CONF_SINGLE_TB

    return float(np.mean(centers[:, 1])), GATE_CONF_UNKNOWN


def detect_gate(image):
    """Locate the nearest gate's opening centre. Returns a GateObs or None."""
    if image is None or getattr(image, "size", 0) == 0:
        return None
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    except Exception:
        return None

    corners, ids = _aruco_detect(gray)
    if ids is None or len(ids) == 0 or corners is None or len(corners) == 0:
        return None

    id_list = [int(v) for v in np.asarray(ids).flatten()]
    n = min(len(id_list), len(corners))
    if n == 0:
        return None
    id_list = id_list[:n]

    centers = np.array([np.asarray(c, dtype=np.float64).reshape(-1, 2).mean(axis=0)
                        for c in corners[:n]], dtype=np.float64)
    sides = np.array([_tag_side_px(c) for c in corners[:n]], dtype=np.float64)

    big = sides >= GATE_MIN_TAG_PX
    if not np.any(big):
        return None
    centers, sides = centers[big], sides[big]
    id_list = [t for t, k in zip(id_list, big) if k]

    keep = _cluster_nearest_gate(centers, sides)
    centers, sides = centers[keep], sides[keep]
    id_list = [t for t, k in zip(id_list, keep) if k]
    if len(id_list) == 0:
        return None

    _learn_roles(centers, id_list)
    cy, conf = _estimate_center_row(centers, sides, id_list)
    cx = float(np.mean(centers[:, 0]))
    tag_px = float(np.mean(sides))

    img_h, img_w = gray.shape[:2]
    dist_m = None
    if tag_px > 1e-3 and GATE_TAG_SIZE_M > 0.0:
        dist_m = GATE_TAG_SIZE_M * _focal_px(img_w) / tag_px

    roles = {t: GATE_TAG_ROLE_BY_ID.get(t, _role_by_id.get(t)) for t in id_list}
    if not np.isfinite(cy):
        return None
    return GateObs(cx, cy, conf, tag_px, dist_m, id_list, roles, img_h, img_w)


# ===========================================================================
# THROTTLE
# ===========================================================================
def _track_line_visibility(fit, dt):
    """
    Maintain the line-visibility hysteresis timers.

    In the line follower this lived inside set_throttle. Here it is called EVERY
    frame, including while a gate owns the throttle -- otherwise the timer freezes
    during the pass and the FSM resumes afterwards on stale state, firing
    lost_confirmed immediately or never.
    """
    global _visible, _vis_timer
    visible = fit is not None
    if visible != _visible:
        _visible = visible
        _vis_timer = 0.0
    else:
        _vis_timer += dt


def line_throttle(alt):
    """
    Climb to look for a lost edge, then descend back to the reference height once
    it's reacquired. Same state machine as the line follower; it now reads the
    visibility state tracked above instead of deriving it itself.
    """
    global _state, _base_alt
    if _base_alt is None:
        _base_alt = alt

    lost_confirmed = (not _visible) and _vis_timer >= LOST_GRACE
    found_confirmed = _visible and _vis_timer >= FOUND_GRACE

    if _state == FOLLOWING:
        if lost_confirmed:
            _state = SEARCHING
            return CLIMB_THROTTLE
        return 0.0

    if _state == SEARCHING:
        if found_confirmed:
            _state = DESCENDING
            return DESCEND_THROTTLE
        if alt - _base_alt >= MAX_CLIMB:
            return 0.0                      # ceiling: hover and keep looking
        return CLIMB_THROTTLE

    if _state == DESCENDING:
        if lost_confirmed:
            _state = SEARCHING
            return CLIMB_THROTTLE
        if alt <= _base_alt + HEIGHT_TOL:
            _state = FOLLOWING
            return 0.0
        return DESCEND_THROTTLE

    return 0.0


def gate_throttle(obs, alt, dt):
    """
    Vertical command that centres the gate opening.

    Two regimes:
      FAR   -- image-space PID on the normalised vertical error, authority scaled
               by the estimate's confidence.
      CLOSE -- the target is converted to an ABSOLUTE altitude and latched. Every
               tag leaves the frame during the pass, so image-space control has
               nothing to servo on exactly when it matters; without the latch the
               drone reverts to the line FSM mid-gate.

    Returns (throttle, active).
    """
    global _gate_latched, _gate_target_alt, _gate_latch_t, _relatch_t

    # ---- Latched: fly the frozen absolute altitude.
    if _gate_latched and _gate_target_alt is not None:
        _gate_latch_t += dt
        if _gate_latch_t > GATE_LATCH_MAX_S or (
                obs is None and _gate_seen_t > GATE_LATCH_EXIT_S):
            _release_latch(alt)
            return 0.0, False
        err_m = _gate_target_alt - alt
        return uav_utils.clamp(GATE_ALT_HOLD_KP * err_m,
                               -GATE_THROTTLE_LIMIT, GATE_THROTTLE_LIMIT), True

    _relatch_t += dt

    if obs is None or obs.conf < GATE_CONF_MIN or _gate_streak < GATE_MIN_FRAMES:
        _gate_alt_pid.hold()
        return 0.0, False

    half_h = max(obs.img_h * 0.5, 1.0)
    err_px = obs.cy - half_h                    # +ve: gate low in frame -> descend

    # ---- Close enough that the tags are about to leave frame: latch.
    if (obs.dist_m is not None and obs.dist_m <= GATE_LATCH_DIST_M
            and obs.conf >= GATE_LATCH_MIN_CONF and _relatch_t >= GATE_RELATCH_S):
        f_px = _focal_px(obs.img_w)
        rise_m = -err_px * obs.dist_m / f_px    # rows increase downward
        rise_m = uav_utils.clamp(rise_m, -GATE_MAX_ALT_STEP_M, GATE_MAX_ALT_STEP_M)
        _gate_latched = True
        _gate_latch_t = 0.0
        _gate_target_alt = alt + rise_m
        _gate_alt_pid.hold()
        print(f"[gate] latched: dist={obs.dist_m:.2f}m rise={rise_m:+.2f}m "
              f"target_alt={_gate_target_alt:.2f}m conf={obs.conf:.2f}")
        err_m = _gate_target_alt - alt
        return uav_utils.clamp(GATE_ALT_HOLD_KP * err_m,
                               -GATE_THROTTLE_LIMIT, GATE_THROTTLE_LIMIT), True

    # ---- Far: image-space centring, authority scaled by confidence.
    out = _gate_alt_pid.update(-err_px / half_h, dt)
    return out * obs.conf, True


def _release_latch(alt):
    """
    End a gate pass. Re-baselines the search FSM to the altitude the gate put us
    at -- essential on a course where gates sit at different heights, otherwise a
    later line loss descends all the way back to launch height.
    """
    global _gate_latched, _gate_target_alt, _gate_latch_t, _base_alt, _relatch_t
    _gate_latched = False
    _gate_target_alt = None
    _gate_latch_t = 0.0
    _relatch_t = 0.0
    if GATE_REBASELINE:
        _base_alt = alt
        print(f"[gate] passed; reference height re-baselined to {alt:.2f} m")


# ===========================================================================
# MAIN LOOP
# ===========================================================================
def update(drone):
    global _timer, _done, _gate_streak, _gate_seen_t, _gate_blend, _dbg_t
    global _prev_y0, _m_valid, _base_alt, _last_obs, _last_alt

    if _done:
        return True

    # A dropped frame would otherwise inject a large integral step and a
    # derivative spike into every axis at once.
    dt = uav_utils.clamp(float(drone.get_delta_time()), DT_MIN, DT_MAX)

    # ---- Line perception
    fit = find_edge(drone, dt)
    if fit is None:
        # Invalidate the perception filters. A gate pass or a search climb can
        # last seconds, and a pre-loss slope fed into the yaw PID on the first
        # reacquired frame is a real kick.
        _prev_y0 = None
        _m_valid = False

    # A raising altitude read must not take down the flight loop; coast on the
    # last good value instead.
    try:
        alt = float(drone.physics.get_altitude())
        if not math.isfinite(alt):
            raise ValueError("non-finite altitude")
        _last_alt = alt
    except Exception:
        alt = _last_alt if _last_alt is not None else (_base_alt or 0.0)

    if _base_alt is None:
        _base_alt = alt        # capture BEFORE any branch, so a gate on frame 1
                               # cannot leave the FSM without a reference height

    # ---- Gate perception
    obs = None
    get_img = getattr(drone.camera, "get_color_image", None)
    if callable(get_img):
        try:
            obs = detect_gate(get_img())
        except Exception as exc:
            if not _aruco_cache["warned"]:
                _aruco_cache["warned"] = True
                print(f"[gate] detection error ({exc}); line following only")
            obs = None

    if obs is not None and obs.conf >= GATE_CONF_MIN:
        _gate_streak += 1
        _gate_seen_t = 0.0
        _last_obs = obs
    else:
        # Coast on the last good estimate for GATE_LOST_HOLD_S. ArUco decoding
        # drops frames routinely (motion blur, glare), and without this a
        # detection that alternates on/off would reset _gate_streak every other
        # frame and never reach GATE_MIN_FRAMES -- gate control would simply never
        # engage. Authority stays conf-scaled, and the estimate is discarded
        # outright once the hold expires.
        _gate_seen_t += dt
        if _last_obs is not None and _gate_seen_t <= GATE_LOST_HOLD_S:
            obs = _last_obs
        else:
            _gate_streak = 0
            _last_obs = None
            obs = None

    # ---- Throttle: BOTH sources every frame, cross-faded.
    #      The line FSM must keep running even while a gate is visible, or its
    #      timers and altitude reference go stale during the pass.
    _track_line_visibility(fit, dt)
    thr_line = line_throttle(alt)
    thr_gate, gate_active = gate_throttle(obs, alt, dt)

    a = dt / (GATE_BLEND_TAU + dt)
    _gate_blend += a * ((1.0 if gate_active else 0.0) - _gate_blend)
    throttle = uav_utils.clamp(_gate_blend * thr_gate + (1.0 - _gate_blend) * thr_line,
                               -1.0, 1.0)

    # ---- Attitude: line only, always.
    m = 0.0
    edge_col = float(IMAGE_CENTER)
    if fit is None:
        _yaw_pid.hold()
        _roll_pid.hold()
        pitch = roll = yaw = 0.0          # hold level, let throttle search
    else:
        ys, xs, m, b, poly, closest_pt = fit
        edge_col = float(xs.mean())
        pitch = set_pitch(ys, xs, poly)
        roll = set_roll(xs, dt)
        yaw = set_yaw(m, dt)

    drone.flight.send_pcmd(pitch, roll, yaw, throttle)

    # ---- Throttled debug
    _dbg_t += dt
    if DEBUG_PRINT and _dbg_t >= DEBUG_PERIOD_S:
        _dbg_t = 0.0
        if fit is None:
            line_s = "line=LOST"
        else:
            line_s = (f"m={m:+6.2f} col={edge_col:5.0f} "
                      f"pt=({closest_pt[0]:3.0f},{closest_pt[1]:3.0f})")
        if obs is None:
            gate_s = "gate=-"
        else:
            d = f"{obs.dist_m:.2f}m" if obs.dist_m is not None else "?"
            gate_s = (f"gate n={obs.count} conf={obs.conf:.2f} "
                      f"cy={obs.cy:.0f} d={d}"
                      f"{' LATCH' if _gate_latched else ''}")
        print(f"{line_s} | {gate_s} | P={pitch:+.3f} R={roll:+.3f} "
              f"Y={yaw:+.3f} T={throttle:+.3f} (blend={_gate_blend:.2f}) "
              f"{_state} alt={alt:.2f}/{_base_alt:.2f}")

    _timer += dt
    if _timer >= FOLLOW_TIME:
        _done = True
    return _done


# ===========================================================================
# ENTRY POINT
# ===========================================================================
if __name__ == "__main__":
    _drone = drone_core.create_drone()
    _launcher = neo_lab.Launcher(1.4)

    def start():
        _launcher.reset()
        reset()
        print("final_demo4: filtered-tangent line following + gate-altitude centring")
        print(f"  aruco={GATE_ARUCO_DICT}  h_per_tag={GATE_H_PER_TAG:.2f}  "
              f"focal={_focal_px(IMG_W):.0f}px")
        if not GATE_TAG_ROLE_BY_ID:
            print("  tag roles: learning online (fill GATE_TAG_ROLE_BY_ID to skip)")

    def _update():
        if not _launcher.done:            # arm + climb to a safe height first
            _launcher.update(_drone)
            return
        if update(_drone):
            _drone.flight.land()

    print("------ STARTING CODE ---------\n")
    _drone.set_start_update(start, _update)
    _drone.go(not neo_lab._is_sim(_drone))
