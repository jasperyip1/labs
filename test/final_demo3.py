"""
MIT BWSI Autonomous Drone Racing Course - UAV Neo
GNU General Public License v3.0

Week 2/3 Lab -- Step 3: Follow the Edge + gate-altitude centering (corrected)

Corrected rewrite of final_demo2.py. The original design rule is preserved:

    pitch, roll and yaw ALWAYS come from line following.
    Gate detection may only ever influence THROTTLE.

What changed, and why:

  PERCEPTION (line)
   1. The polynomial is fitted to one CENTROID PER ROW instead of every bright
      pixel. Raw-pixel fitting weights each row by how many pixels it holds, so
      wherever the line runs near-horizontal in the image a single row dominates
      the whole fit.
   2. The row coordinate is normalised to [-1, 1] before np.polyfit. With raw
      rows in [0, 480] and degree 3 the Vandermonde condition number is ~1e8 and
      the cubic coefficient is mostly noise -- which matters now that the fit is
      differentiated for the bend measure.

  CONTROL
   3. Yaw error is -arctan(m), not -m. Slope is unbounded and badly nonlinear in
      heading (45 deg -> 1.0, 80 deg -> 5.7), so the old error saturated MAX_YAW
      at roughly 61 deg of heading error and stayed pinned there. The tangent is
      also low-passed as an ANGLE now; averaging slopes near vertical is
      numerically unstable and asymmetric.
   4. Pitch scheduling measures actual bend -- the change in tangent angle across
      the visible curve -- instead of the residual to the fit. Residual is not
      curvature: a sweeping arc fits a cubic almost perfectly and used to read as
      "straight", so the drone entered real turns at full speed. Speed is now
      additionally gated on cross-track error, so it will not sprint while badly
      off-line.
   5. Roll uses cross-track error sampled at the drone's own position -- the
      image CENTRE row, since the camera looks straight down -- rather than the
      mean column of all bright pixels. The mean is dragged around by whichever
      part of the curve happens to fill the frame, and it disagreed with the
      look-ahead point yaw uses, so the two axes fought each other.

  GATE / THROTTLE
   6. Tags are clustered so that a gate further down the course cannot be
      averaged into the current one. The old code pooled every decoded tag in the
      image into a single centroid -- with two gates in view that centroid is a
      point in empty air between them.
   7. Tag ROLES (top / bottom / left / right) are resolved and cached by id, and
      the gate centre is reconstructed from the geometry instead of averaging
      whatever happened to decode. Tags sit at EDGE MIDPOINTS, so left and right
      are each individually unbiased in the vertical axis, while top and bottom
      are only unbiased as a pair. Averaging an unpaired top or bottom tag biased
      the centre by h/3 to a full h -- enough to centre on the bar instead of the
      opening.
   8. Every gate estimate carries a CONFIDENCE that scales its authority, so a
      single spurious decode can no longer seize the throttle axis.
   9. Close in, the target is LATCHED to an absolute altitude and flown on that.
      Every tag leaves the field of view during the pass, exactly when the
      reference is needed most.
  10. The line-following state machine now runs EVERY frame and the two throttle
      sources are cross-faded. Previously the FSM was skipped whenever a gate was
      visible, which froze its visibility timer and left _base_alt uncaptured.
  11. _base_alt is re-baselined after each gate, so a lost line on a high section
      no longer descends all the way back to launch height.

  ROBUSTNESS
  12. dt is clamped. One long frame used to inject a large integral step and a
      derivative spike.
  13. _line_angle and _prev_row are invalidated on line loss. They used to go
      stale across a multi-second search climb and then dump a pre-loss slope
      into the yaw PID on the first reacquired frame.
  14. The pixel-count test counts pixels. `np.count_nonzero` on an Nx2 coordinate
      array counts coordinate VALUES (~2N), not pixels.

TUNING ORDER
  Items 3 and 4 change what the controller regulates, so the inherited gains do
  not carry over. Retune YAW_KP first with the drone hand-held over a straight
  line, then BEND_SCALE_RAD, then the gate block.

CALIBRATION REQUIRED BEFORE GATES WILL WORK
  Everything in the "GATE GEOMETRY" block below is a physical property of your
  hardware and is currently a placeholder. See calibration notes there.
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

_SELFTEST = "--selftest" in _sys.argv
_REPLAY = "--replay" in _sys.argv
_OFFLINE = _SELFTEST or _REPLAY

try:
    import drone_core
    import drone_utils as uav_utils
    import neo_lab
except ImportError:
    # --selftest and --replay are meant to run on a laptop with only cv2 +
    # numpy, so the course packages are stubbed rather than required. A real
    # flight still imports them for real; the re-raise below guarantees that.
    if not _OFFLINE:
        raise
    import types as _types

    drone_core = None

    uav_utils = _types.SimpleNamespace(
        clamp=lambda v, lo, hi: lo if v < lo else (hi if v > hi else v))

    def _stub_bright_mask(image, v_min=200):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = (hsv[:, :, 2] > v_min).astype(np.uint8) * 255
        k = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if n > 1:
            big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            mask = (lab == big).astype(np.uint8) * 255
        return mask

    neo_lab = _types.SimpleNamespace(bright_mask_improved=_stub_bright_mask)
    print("[offline] course packages absent; using stubs")


# ===========================================================================
# GENERAL
# ===========================================================================
DEBUG_PRINT        = True
DEBUG_PERIOD_S     = 0.25    # s between debug lines (printing every frame at
                             # 30-60 Hz adds real jitter to dt, which then feeds
                             # every derivative term in the loop)

FOLLOW_TIME        = 1.0e6   # s of flight before landing; effectively disabled
DT_MIN, DT_MAX     = 1.0e-3, 0.1     # clamp on get_delta_time()


# ===========================================================================
# LINE PERCEPTION
# ===========================================================================
V_MIN         = 200      # HSV Value threshold for the bright line
MIN_PIXELS    = 400      # min bright PIXELS to trust a frame. NOTE: the old code
                         # compared count_nonzero of an Nx2 coordinate array,
                         # which is ~2N -- so this threshold is ~2x the old 200
                         # to keep the same effective sensitivity.
MIN_ROWS      = 12       # min distinct occupied rows before fitting
POLY_DEGREE   = 3        # 3 or 5; higher fits noise more readily
IMG_W, IMG_H  = 640, 480 # nominal downward image size (actual size is read per
                         # frame; these are only fallbacks)
SAMPLE_STEP   = 2        # px row spacing when scanning the curve

LOOKAHEAD_PX  = 80       # how far AHEAD of the drone to steer, in image rows.
                         # Image "up" (smaller row) is forward. Larger = smoother
                         # but cuts corners; smaller = twitchier but tracks
                         # tighter. This is the single most useful knob for
                         # corner behaviour.

CONTINUITY_WEIGHT = 0.25 # penalty on the closest-point search for jumping away
                         # from last frame's point. 0 = pure nearest-point.
ANGLE_TAU     = 0.12     # s, low-pass on the tangent ANGLE (not the slope)


# ===========================================================================
# LINE CONTROL
# ===========================================================================
MAX_ROLL      = 0.25
MAX_YAW       = 1.0

# Yaw: error is -arctan(m) in RADIANS, bounded to +-pi/2.
# Old error was -m (unbounded), so old gains do NOT transfer -- for the same
# physical heading error the new error is numerically smaller near vertical and
# very much smaller at large angles. Start here and raise KP until it oscillates.
YAW_KP        = 1.10
YAW_KI        = 0.0
YAW_KD        = 0.10
YAW_I_LIMIT   = 0.20

# Roll: error is normalised cross-track offset, -1 (line left) .. +1 (line right)
ROLL_KP       = 0.23
ROLL_KI       = 0.0
ROLL_KD       = 0.02
ROLL_I_LIMIT  = 0.10

D_TAU         = 0.10     # s, derivative low-pass. Raw d/dt of a pixel
                         # measurement is unusable without this.

# Pitch (open-loop speed schedule)
PITCH_STRAIGHT  = 0.20   # straights
PITCH_TURN      = 0.08   # turns
BEND_SCALE_RAD  = 1.0    # total tangent-angle change across the visible curve at
                         # which the drone is "fully" in a turn. 1.0 rad ~ 57 deg.
                         # Replaces the old CURVE_SCALE=70, which was in units of
                         # fit-residual pixels and measured line thickness rather
                         # than curvature.
PITCH_ERR_GATE  = 1.5    # how hard to back off when off-line: speed is scaled by
                         # clamp(1 - PITCH_ERR_GATE*|cross-track|, 0, 1)


# ===========================================================================
# LINE-SEARCH THROTTLE STATE MACHINE
# ===========================================================================
MAX_CLIMB        = 3.0   # m above the reference height to search
CLIMB_THROTTLE   = 0.3
DESCEND_THROTTLE = -0.2  # gentler than the climb on purpose
LOST_GRACE       = 0.4   # s of no-line before climbing
FOUND_GRACE      = 0.5   # s of line before committing to descend
HEIGHT_TOL       = 0.10  # m; "back at the reference height"


# ===========================================================================
# GATE GEOMETRY -- PHYSICAL CONSTANTS. ALL PLACEHOLDERS. CALIBRATE THESE.
# ===========================================================================
# These describe the gate hardware and the camera. Pixel thresholds are DERIVED
# from them further down, so once these are right the behaviour thresholds are
# expressed in metres and stop needing to be re-tuned when the course changes.

GATE_INNER_HEIGHT_M = 1.524  # m, vertical clear opening (inside edge to inside
                             # edge). Used for the tag<->centre offset and for
                             # the pass-clearance sanity check.
                             # (60 in diameter hoop -> 60 * 0.0254 = 1.524 m)
GATE_INNER_WIDTH_M  = 1.524  # m, horizontal clear opening. Only used for the
                             # aspect-ratio sanity check on role assignment.
                             # Hoop is circular, so width == height == diameter.
GATE_TAG_SIZE_M     = 0.2667 # m, side length of the printed ArUco square
                             # (black border included, quiet zone excluded).
                             # (10.5 in tag side -> 10.5 * 0.0254 = 0.2667 m)

# Where the tags sit on the gate. The course gates carry FOUR tags at the
# MIDPOINT of each edge -- top, bottom, left, right -- forming a diamond.
# This is NOT the "one per corner" layout neo_lab's docstring describes, and the
# difference is what makes partial detections biased.
#
# GATE_H_PER_TAG converts one role-known tag into a gate centre:
#     cy_centre = cy_top + GATE_H_PER_TAG * tag_px
# It is the ratio (vertical distance from a top/bottom tag CENTRE to the gate
# CENTRE) / (tag side), and is a fixed property of the gate.
#
# CALIBRATE: hover facing a gate with all four tags visible and print
#     (row_of_bottom_tag - row_of_top_tag) / 2 / mean_tag_side_px
# Set _GATE_H_PER_TAG_OVERRIDE to that number. Until then it is estimated from
# the dimensions above, assuming the tag centre sits on the frame midline:
#     (inner_height/2 + tag_size/2) / tag_size
_GATE_H_PER_TAG_OVERRIDE = None
GATE_H_PER_TAG = (_GATE_H_PER_TAG_OVERRIDE if _GATE_H_PER_TAG_OVERRIDE is not None
                  else (GATE_INNER_HEIGHT_M / 2.0 + GATE_TAG_SIZE_M / 2.0) / GATE_TAG_SIZE_M)

# Optional hard-coded id -> role map. If your gates use fixed ids per position,
# fill this in and role resolution becomes exact even on single-tag frames.
# Leave empty to learn roles online from 3+ tag views.
#   e.g. {0: 'T', 1: 'R', 2: 'B', 3: 'L',  4: 'T', 5: 'R', 6: 'B', 7: 'L'}
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
# nothing decodes at all and every gate behaviour below is dead code. Print
# `obs.ids` on a known-good frame to confirm.
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
GATE_ROLE_MIN_TAGS   = 3     # tags needed before roles may be learned. With 2
                             # the assignment is genuinely ambiguous (a top+left
                             # pair on a wide gate misreads top as horizontal).

# Confidence per estimate quality. These scale the PID output directly.
GATE_CONF_TB_PAIR    = 1.00  # top+bottom: exact
GATE_CONF_LR         = 0.90  # left and/or right: each individually unbiased
GATE_CONF_SINGLE_TB  = 0.55  # one top or bottom + GATE_H_PER_TAG
GATE_CONF_UNKNOWN    = 0.25  # roles unresolved: fall back to the raw mean
GATE_CONF_MIN        = 0.20  # below this the estimate is discarded entirely

GATE_MIN_FRAMES      = 2     # consecutive detections before gate control engages
                             # (rejects isolated spurious decodes)
GATE_LOST_HOLD_S     = 0.30  # s to keep trusting a gate after it stops decoding


# ===========================================================================
# GATE CONTROL
# ===========================================================================
GATE_ALT_KP         = 0.50   # on normalised vertical image error
GATE_ALT_KI         = 0.0
GATE_ALT_KD         = 0.05
GATE_ALT_I_LIMIT    = 0.10
GATE_THROTTLE_LIMIT = 0.30

GATE_BLEND_TAU      = 0.15   # s, cross-fade between gate and line throttle.
                             # Prevents a flickering detection from hard-switching
                             # between two very different throttle sources.

GATE_LATCH_DIST_M   = 1.50   # m. Inside this range the vertical target is frozen
                             # to an absolute altitude and flown on that, because
                             # the tags are about to leave the frame.
GATE_LATCH_MAX_S    = 4.0    # s. Hard timeout on the latch so it can never hang.
GATE_LATCH_EXIT_S   = 0.60   # s without any detection while latched -> assume the
                             # gate has been passed. Must exceed GATE_LOST_HOLD_S
                             # or the coast cache would end the latch early.
GATE_LATCH_MIN_CONF = 0.90   # min confidence to freeze an absolute altitude.
                             # Defaults to the left/right tier: a single-tag
                             # estimate leans entirely on GATE_H_PER_TAG, which
                             # is too much trust to place in one calibration
                             # constant for a committed manoeuvre.
GATE_RELATCH_S      = 1.00   # s cooldown after a latch ends. Without it, a latch
                             # that hits GATE_LATCH_MAX_S while the gate is still
                             # decoding re-arms on the same gate next frame.

GATE_ALT_HOLD_KP    = 0.35   # throttle per metre of altitude error while latched
GATE_MAX_ALT_STEP_M = 1.50   # m. Cap on how far one gate may move the target
                             # altitude -- a bad estimate cannot command a huge
                             # climb.

GATE_REBASELINE     = True   # after passing a gate, treat the new altitude as the
                             # reference the line-search FSM returns to. Required
                             # for a course where gates sit at different heights.


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
        self._integral = 0.0
        self._prev_error = None
        self._deriv = 0.0

    def hold(self):
        """
        Call on frames where the measurement is invalid. Keeps the integral but
        drops the previous error, so reacquiring does not produce a derivative
        spike from the stale value.
        """
        self._prev_error = None

    def update(self, error, dt):
        error = float(error)
        if not np.isfinite(error):          # a NaN here would poison the
            self.hold()                     # integral for the rest of the run
            return 0.0
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
# LINE PERCEPTION
# ===========================================================================
class LineFit:
    """
    A fitted line for one frame.

      angle       filtered tangent angle at the look-ahead point, radians.
                  0 = line runs straight up the image. atan(dcol/drow).
      cross       normalised cross-track error at the drone, -1 .. +1.
                  Positive = line is to the RIGHT of the drone.
      bend        |tangent angle change| across the visible curve, radians.
      point       (row, col) of the look-ahead point, for debug/overlay.
    """

    __slots__ = ("angle", "cross", "bend", "point", "n_px", "n_rows")

    def __init__(self, angle, cross, bend, point, n_px, n_rows):
        self.angle = angle
        self.cross = cross
        self.bend = bend
        self.point = point
        self.n_px = n_px
        self.n_rows = n_rows


def _row_centroids(mask):
    """
    Collapse a binary mask to one column-centroid per occupied row.

    Fitting raw pixels weights each row by its pixel count, so a near-horizontal
    stretch of line dominates the fit. One point per row removes that bias and
    drops the fit from ~15k points to <= img_h.

    Returns (rows, cols, weights, n_px) or None.
    """
    h = mask.shape[0]
    rows_i, cols_i = np.nonzero(mask)
    n_px = int(rows_i.size)
    if n_px < MIN_PIXELS:
        return None

    cnt = np.bincount(rows_i, minlength=h).astype(np.float64)
    tot = np.bincount(rows_i, weights=cols_i.astype(np.float64),
                      minlength=h).astype(np.float64)
    occ = cnt > 0.0
    n_rows = int(np.count_nonzero(occ))
    if n_rows < MIN_ROWS:
        return None

    rows = np.nonzero(occ)[0].astype(np.float64)
    cols = tot[occ] / cnt[occ]
    weights = np.sqrt(cnt[occ])         # rows with more pixels are more certain
    return rows, cols, weights, n_px


def find_line(drone, dt):
    """
    Threshold the downward image and fit column-as-a-function-of-row.

    Returns a LineFit, or None when the frame cannot be trusted. Never raises:
    a bad frame or a singular fit degrades to None rather than killing the
    flight loop.
    """
    try:
        image = drone.camera.get_downward_image()
    except Exception:
        return None
    return fit_line_image(image, dt)


def fit_line_image(image, dt, debug=None):
    """
    The perception core, on a bare image. Split out of find_line so --replay can
    run the exact same code path on a still frame -- a replay that exercised a
    parallel implementation would prove nothing about flight behaviour.

    Pass a dict as `debug` to receive the fit internals (poly, sample points,
    mask) for overlay drawing.
    """
    global _prev_row, _line_angle, _angle_valid

    if image is None or getattr(image, "size", 0) == 0:
        return None

    try:
        mask = neo_lab.bright_mask_improved(image, V_MIN)
    except Exception:
        return None
    if mask is None or mask.ndim != 2:
        return None

    img_h, img_w = mask.shape[:2]
    got = _row_centroids(mask)
    if got is None:
        return None
    rows, cols, weights, n_px = got

    # -- Normalise the row coordinate to [-1, 1] before fitting. Raw rows in
    #    [0, 480] at degree 3 give a Vandermonde condition number ~1e8, and the
    #    cubic term is then noise -- which matters because it is differentiated
    #    below for the bend measure.
    r_lo, r_hi = float(rows[0]), float(rows[-1])     # nonzero() output is sorted
    r_mid = 0.5 * (r_lo + r_hi)
    r_half = max(0.5 * (r_hi - r_lo), 1.0)
    u = (rows - r_mid) / r_half

    deg = int(min(POLY_DEGREE, rows.size - 1))
    if deg < 1:
        return None
    try:
        with np.errstate(all="ignore"):
            coeffs = np.polyfit(u, cols, deg, w=weights)
    except Exception:                    # LinAlgError, singular fit, all-equal u
        return None
    if not np.all(np.isfinite(coeffs)):
        return None

    poly = np.poly1d(coeffs)
    dpoly = poly.deriv()

    def slope_at(uu):
        """dcol/drow at normalised row uu (chain rule through the normalisation)."""
        return float(dpoly(uu)) / r_half

    # ---- Look-ahead point: the point on the curve nearest the steering target,
    #      biased toward last frame's choice so the tracked point cannot hop to
    #      another branch of the curve when two points are briefly equidistant.
    target_col = img_w * 0.5
    target_row = img_h * 0.5 - LOOKAHEAD_PX

    s_rows = np.arange(r_lo, r_hi + SAMPLE_STEP, SAMPLE_STEP, dtype=np.float64)
    if s_rows.size == 0:
        s_rows = np.array([r_mid], dtype=np.float64)
    s_u = np.clip((s_rows - r_mid) / r_half, -1.0, 1.0)
    s_cols = poly(s_u)

    dist_sq = (s_cols - target_col) ** 2 + (s_rows - target_row) ** 2
    if _prev_row is not None:
        dist_sq = dist_sq + CONTINUITY_WEIGHT * (s_rows - _prev_row) ** 2

    best = int(np.argmin(dist_sq))
    row0 = float(s_rows[best])
    col0 = float(s_cols[best])
    _prev_row = row0

    # ---- Tangent ANGLE, low-passed.
    #      Filtering the angle rather than the slope matters: near vertical the
    #      slope is huge and its mean is dominated by whichever frame happened to
    #      be steepest, and the average of +5.7 and -5.7 is 0 even though both
    #      describe an almost-vertical line. Angles average sanely.
    angle_raw = math.atan(slope_at(np.clip((row0 - r_mid) / r_half, -1.0, 1.0)))
    if (not _angle_valid) or dt <= 0.0:
        _line_angle = angle_raw          # reacquiring: snap, do not drag the
        _angle_valid = True              # stale pre-loss angle back in
    else:
        a = dt / (ANGLE_TAU + dt)
        _line_angle += a * (angle_raw - _line_angle)

    # ---- Cross-track error AT THE DRONE.
    #      The camera looks straight down, so the drone projects to the principal
    #      point -- the image CENTRE, not the bottom edge. Clamped to the observed
    #      row range so the cubic is never extrapolated.
    u_veh = float(np.clip((img_h * 0.5 - r_mid) / r_half, -1.0, 1.0))
    col_veh = float(poly(u_veh))
    half_w = max(img_w * 0.5, 1.0)
    cross = uav_utils.clamp((col_veh - half_w) / half_w, -1.0, 1.0)

    # ---- Bend: how much the tangent angle turns across the visible curve.
    #      This is curvature. The old measure -- std of the residual to the fit --
    #      is not: an arc fits a cubic almost perfectly, so real sweeping turns
    #      read as perfectly straight and were entered at full speed.
    bend = abs(math.atan(slope_at(-1.0)) - math.atan(slope_at(1.0)))

    if debug is not None:
        debug.update(mask=mask, rows=rows, cols=cols, poly=poly,
                     r_mid=r_mid, r_half=r_half, s_rows=s_rows, s_cols=s_cols,
                     col_veh=col_veh, angle_raw=angle_raw,
                     target=(target_row, target_col))

    return LineFit(_line_angle, cross, bend, (row0, col0), n_px, int(rows.size))


# ===========================================================================
# LINE CONTROL AXES
# ===========================================================================
def set_yaw(fit, dt):
    """Rotate until the line runs straight up the image."""
    return _yaw_pid.update(-fit.angle, dt)


def set_roll(fit, dt):
    """Strafe until the line passes under the drone."""
    return _roll_pid.update(fit.cross, dt)


def set_pitch(fit):
    """
    Fast on straights, slow through bends, and slower still when badly off-line
    (no point sprinting down a path that is not under the drone).
    """
    straightness = uav_utils.clamp(1.0 - fit.bend / BEND_SCALE_RAD, 0.0, 1.0)
    pitch = PITCH_TURN + (PITCH_STRAIGHT - PITCH_TURN) * straightness
    quality = uav_utils.clamp(1.0 - PITCH_ERR_GATE * abs(fit.cross), 0.0, 1.0)
    return pitch * quality


# ===========================================================================
# GATE PERCEPTION
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
    Assign T/B/L/R from each tag's offset from the cluster centroid, and cache
    the mapping by tag id.

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
    # Reject inconsistent assignments (two tags claiming the same position);
    # a skewed view can confuse the split, and a wrong cached role is worse
    # than no cached role.
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

    Averaging everything (the old behaviour) biases an unpaired top/bottom by
    h/3 to h. Worse, the top tag is the first to leave frame when approaching
    from below, and losing it biases the estimate in the direction the drone was
    ALREADY erring -- positive feedback that converges onto the bottom bar.
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

_prev_row = None            # last look-ahead row, for continuity biasing
_line_angle = 0.0           # low-passed tangent angle
_angle_valid = False        # False after a loss, so the filter snaps on reacquire

_role_by_id = {}            # learned tag id -> 'T'|'B'|'L'|'R'
_gate_streak = 0            # consecutive frames with a usable gate
_gate_seen_t = 0.0          # s since a gate last decoded
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
    global _prev_row, _line_angle, _angle_valid
    global _gate_streak, _gate_seen_t, _gate_blend, _last_obs
    global _gate_latched, _gate_target_alt, _gate_latch_t, _relatch_t, _dbg_t
    _timer = 0.0
    _done = False
    _state = FOLLOWING
    _base_alt = None
    _last_alt = None
    _visible = True
    _vis_timer = 0.0
    _prev_row = None
    _line_angle = 0.0
    _angle_valid = False
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
# THROTTLE
# ===========================================================================
def _track_line_visibility(fit, dt):
    """
    Maintain the line-visibility hysteresis timers.

    Split out of set_throttle and called EVERY frame. In the original this lived
    inside set_throttle, which was skipped whenever a gate was visible -- so the
    timer froze and the FSM resumed afterwards on stale state, firing
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
    Climb to look for a lost line, then descend back to the reference height once
    it is reacquired. Same state machine as the original; it now reads the
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
    global _prev_row, _angle_valid, _base_alt, _last_obs, _last_alt

    if _done:
        return True

    # A dropped frame used to inject a large integral step and a derivative
    # spike into every axis at once.
    dt = uav_utils.clamp(float(drone.get_delta_time()), DT_MIN, DT_MAX)

    fit = find_line(drone, dt)
    if fit is None:
        # Invalidate the perception filters. They used to survive a multi-second
        # search climb and then feed a pre-loss angle straight into the yaw PID
        # on the first reacquired frame.
        _prev_row = None
        _angle_valid = False

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
        # frame and never reach GATE_MIN_FRAMES -- gate control would simply
        # never engage. Authority stays conf-scaled, and the estimate is
        # discarded outright once the hold expires.
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
    if fit is None:
        _yaw_pid.hold()
        _roll_pid.hold()
        pitch = roll = yaw = 0.0          # hold level, let throttle search
    else:
        pitch = set_pitch(fit)
        roll = set_roll(fit, dt)
        yaw = set_yaw(fit, dt)

    drone.flight.send_pcmd(pitch, roll, yaw, throttle)

    # ---- Throttled debug
    _dbg_t += dt
    if DEBUG_PRINT and _dbg_t >= DEBUG_PERIOD_S:
        _dbg_t = 0.0
        if fit is None:
            line_s = "line=LOST"
        else:
            line_s = (f"ang={math.degrees(fit.angle):+6.1f}d "
                      f"cross={fit.cross:+.2f} bend={math.degrees(fit.bend):5.1f}d")
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
# REPLAY -- `python final_demo3.py --replay <image-or-dir> [--down] [--save]`
# ===========================================================================
# Runs the REAL perception path on a still frame, so gate geometry and the line
# fit can be validated from a photo without flying. Default is forward/gate
# analysis; --down analyses the downward line camera instead.
#
# The point of this mode is CALIBRATION: it prints the measured
# GATE_H_PER_TAG for any frame showing top+bottom tags, which is the one number
# the single-tag path depends on and the one you cannot get from the sim.
def _replay_gate(image, name):
    """Analyse one forward frame. Returns the measured h_per_tag, or None."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    img_h, img_w = gray.shape[:2]
    corners, ids = _aruco_detect(gray)

    print(f"\n=== {name}  ({img_w}x{img_h}) ===")
    if ids is None or len(ids) == 0:
        print("  no tags decoded")
        print(f"  -> if tags ARE present, GATE_ARUCO_DICT={GATE_ARUCO_DICT} is wrong.")
        print("     neo_lab's docstring says DICT_6X6_250 but its code uses")
        print("     DICT_5X5_100 -- try the other one.")
        return None

    id_list = [int(v) for v in np.asarray(ids).flatten()]
    centers = np.array([np.asarray(c, np.float64).reshape(-1, 2).mean(axis=0)
                        for c in corners], np.float64)
    sides = np.array([_tag_side_px(c) for c in corners], np.float64)
    print(f"  decoded {len(id_list)} tag(s): {id_list}")
    for t, c, s in zip(id_list, centers, sides):
        print(f"    id={t:<4} at ({c[0]:6.1f},{c[1]:6.1f})  side={s:5.1f}px")

    keep = _cluster_nearest_gate(centers, sides)
    if int(np.sum(keep)) < len(id_list):
        dropped = [t for t, k in zip(id_list, keep) if not k]
        print(f"  clustering dropped {dropped} (different gate / scale)")
    centers, sides = centers[keep], sides[keep]
    id_list = [t for t, k in zip(id_list, keep) if k]

    _learn_roles(centers, id_list)
    roles = {t: GATE_TAG_ROLE_BY_ID.get(t, _role_by_id.get(t)) for t in id_list}
    print(f"  roles: {roles}"
          f"{'  (need 3+ tags to learn)' if len(id_list) < GATE_ROLE_MIN_TAGS else ''}")

    cy, conf = _estimate_center_row(centers, sides, id_list)
    tag_px = float(np.mean(sides))
    dist = GATE_TAG_SIZE_M * _focal_px(img_w) / tag_px if tag_px > 1e-3 else float("nan")
    err_px = cy - img_h * 0.5
    print(f"  centre row = {cy:.1f}  (image centre {img_h/2:.0f}, "
          f"err {err_px:+.1f}px)   conf={conf:.2f}")
    print(f"  tag={tag_px:.1f}px -> range~{dist:.2f}m "
          f"(assumes GATE_TAG_SIZE_M={GATE_TAG_SIZE_M})")

    naive = float(np.mean(centers[:, 1]))
    if abs(naive - cy) > 1.0:
        print(f"  old code would have said {naive:.1f} -- off by {naive-cy:+.1f}px")

    # What the controller would actually do with this frame.
    reset()
    globals()["_gate_streak"] = GATE_MIN_FRAMES
    obs = GateObs(float(np.mean(centers[:, 0])), cy, conf, tag_px,
                  dist if math.isfinite(dist) else None, id_list, roles, img_h, img_w)
    thr, act = gate_throttle(obs, 1.4, 1 / 30.0)
    verb = "climb" if thr > 0.01 else ("descend" if thr < -0.01 else "hold")
    print(f"  -> throttle {thr:+.3f} ({verb}), active={act}"
          f"{', LATCHED' if _gate_latched else ''}")

    # THE CALIBRATION NUMBER.
    measured = None
    rows_by_role = {}
    for t, c in zip(id_list, centers):
        r = roles.get(t)
        if r:
            rows_by_role.setdefault(r, []).append(float(c[1]))
    if "T" in rows_by_role and "B" in rows_by_role:
        half_px = abs(np.mean(rows_by_role["B"]) - np.mean(rows_by_role["T"])) / 2.0
        measured = half_px / tag_px
        print(f"\n  ** MEASURED GATE_H_PER_TAG = {measured:.3f} **")
        print(f"     (currently {GATE_H_PER_TAG:.3f}; "
              f"set _GATE_H_PER_TAG_OVERRIDE = {measured:.3f})")
        if abs(measured - GATE_H_PER_TAG) > 0.15 * max(GATE_H_PER_TAG, 1e-6):
            print("     WARNING: differs from the configured value by >15%. "
                  "The single-tag path would be biased.")
    else:
        print("\n  (need BOTH top and bottom tags in one frame to measure "
              "GATE_H_PER_TAG)")
    return measured


def _replay_line(image, name):
    """Analyse one downward frame through the real line-perception path."""
    print(f"\n=== {name} (downward) ===")
    reset()
    dbg = {}
    fit = fit_line_image(image, 1 / 30.0, debug=dbg)
    if fit is None:
        print("  no usable line "
              f"(need >={MIN_PIXELS}px over >={MIN_ROWS} rows; check V_MIN={V_MIN})")
        if image.ndim == 3:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            v = hsv[:, :, 2]
            print(f"  V channel: max={v.max()} mean={v.mean():.0f}, "
                  f"{int((v > V_MIN).sum())}px above threshold")
        return None
    print(f"  {fit.n_px} px over {fit.n_rows} rows")
    print(f"  angle = {math.degrees(fit.angle):+.1f} deg   "
          f"cross = {fit.cross:+.3f}   bend = {math.degrees(fit.bend):.1f} deg")
    print(f"  look-ahead point (row,col) = "
          f"({fit.point[0]:.0f}, {fit.point[1]:.0f})")
    print(f"  -> pitch {set_pitch(fit):+.3f}  roll {set_roll(fit, 1/30):+.3f}  "
          f"yaw {set_yaw(fit, 1/30):+.3f}")
    return dbg


def _replay_overlay(image, dbg, out_path):
    """Draw the fit, look-ahead point and vehicle sample onto a copy."""
    vis = image.copy() if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    h, w = vis.shape[:2]
    mask = dbg.get("mask")
    if mask is not None:
        vis[mask > 0] = (0.45 * vis[mask > 0] + 0.55 * np.array([0, 255, 0])).astype(np.uint8)
    s_rows, s_cols = dbg.get("s_rows"), dbg.get("s_cols")
    if s_rows is not None:
        pts = np.stack([s_cols, s_rows], 1).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(vis, [pts], False, (0, 0, 255), 2)
    cv2.line(vis, (w // 2, 0), (w // 2, h), (128, 128, 128), 1)
    cv2.line(vis, (0, h // 2), (w, h // 2), (128, 128, 128), 1)
    tr, tc = dbg["target"]
    cv2.drawMarker(vis, (int(tc), int(tr)), (255, 255, 0), cv2.MARKER_CROSS, 18, 2)
    cv2.circle(vis, (int(dbg["col_veh"]), h // 2), 6, (0, 255, 255), -1)
    cv2.imwrite(out_path, vis)
    print(f"  overlay written: {out_path}")


def _replay(paths, downward=False, save=False):
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
    files = []
    for p in paths:
        if _os.path.isdir(p):
            files += [_os.path.join(p, f) for f in sorted(_os.listdir(p))
                      if f.lower().endswith(exts)]
        else:
            files.append(p)
    if not files:
        print("no images found")
        return 1

    measured = []
    for f in files:
        image = cv2.imread(f)
        if image is None:
            print(f"\n=== {f} ===\n  cannot read")
            continue
        if downward:
            dbg = _replay_line(image, _os.path.basename(f))
            if save and dbg:
                _replay_overlay(image, dbg, _os.path.splitext(f)[0] + "_overlay.png")
        else:
            m = _replay_gate(image, _os.path.basename(f))
            if m is not None:
                measured.append(m)

    if len(measured) > 1:
        arr = np.array(measured)
        print(f"\n{'='*52}")
        print(f"GATE_H_PER_TAG over {len(arr)} frames: "
              f"mean={arr.mean():.3f} sd={arr.std():.3f} "
              f"min={arr.min():.3f} max={arr.max():.3f}")
        print(f"  -> set _GATE_H_PER_TAG_OVERRIDE = {arr.mean():.3f}")
        if arr.std() > 0.1:
            print("  NOTE: high spread. Tags at an angle bias the estimate; "
                  "prefer frames shot square-on to the gate.")
    return 0


# ===========================================================================
# SELF TEST -- `python final_demo3.py --selftest`
# ===========================================================================
# Exercises perception and control against synthetic frames with fake hardware.
# Needs only cv2 + numpy, so it runs on a laptop without the sim. It verifies
# LOGIC, not flight behaviour: gains still have to be tuned on the real thing.
def _selftest():
    ok = [0]
    bad = []

    def check(name, cond, extra=""):
        if cond:
            ok[0] += 1
            print(f"  PASS  {name}")
        else:
            bad.append(name)
            print(f"  FAIL  {name} {extra}")

    def line_image(w=IMG_W, h=IMG_H, col=None, slope=0.0, curve=0.0, width=14):
        """White line on black; col = f(row), rows normalised about the centre."""
        img = np.zeros((h, w, 3), np.uint8)
        col = w * 0.5 if col is None else col
        for r in range(h):
            t = (r - h * 0.5) / (h * 0.5)
            c = int(round(col + slope * (r - h * 0.5) + curve * t * t * w * 0.25))
            if 0 <= c < w:
                cv2.circle(img, (c, r), width // 2, (255, 255, 255), -1)
        return img

    class FakeCam:
        def __init__(self):
            self.down = line_image()
            self.color = np.zeros((IMG_H, IMG_W, 3), np.uint8)
        def get_downward_image(self):
            return self.down
        def get_color_image(self):
            return self.color

    class FakePhysics:
        def __init__(self):
            self.alt = 1.4
        def get_altitude(self):
            return self.alt

    class FakeFlight:
        def __init__(self):
            self.cmd = (0.0, 0.0, 0.0, 0.0)
        def send_pcmd(self, p, r, y, t):
            for v in (p, r, y, t):
                assert isinstance(v, float) and math.isfinite(v), f"bad cmd {(p,r,y,t)}"
                assert -1.0 <= v <= 1.0, f"cmd out of range {(p,r,y,t)}"
            self.cmd = (p, r, y, t)
        def land(self):
            pass

    class FakeDrone:
        def __init__(self):
            self.camera = FakeCam()
            self.physics = FakePhysics()
            self.flight = FakeFlight()
            self.dt = 1.0 / 30.0
        def get_delta_time(self):
            return self.dt

    d = FakeDrone()

    print("\n-- perception --")
    reset()
    d.camera.down = line_image(slope=0.0)
    f = find_line(d, 1 / 30)
    check("straight line detected", f is not None)
    check("straight -> angle ~0", f and abs(math.degrees(f.angle)) < 5,
          f"got {math.degrees(f.angle):.1f}d" if f else "")
    check("centred -> cross ~0", f and abs(f.cross) < 0.05,
          f"got {f.cross:.3f}" if f else "")
    check("straight -> bend ~0", f and math.degrees(f.bend) < 5,
          f"got {math.degrees(f.bend):.1f}d" if f else "")

    reset()
    d.camera.down = line_image(col=IMG_W * 0.5 + 120)
    f = find_line(d, 1 / 30)
    check("line right -> cross > 0", f and f.cross > 0.25, f"got {f.cross:.3f}" if f else "")
    check("line right -> roll > 0 (strafe right)", f and set_roll(f, 1 / 30) > 0)

    reset()
    d.camera.down = line_image(slope=0.5)          # col grows with row
    f = find_line(d, 1 / 30)
    check("tilted -> angle > 0", f and f.angle > 0.1, f"got {f.angle:.3f}" if f else "")
    check("tilted -> yaw opposes", f and set_yaw(f, 1 / 30) < 0)
    check("yaw bounded by MAX_YAW", f and abs(set_yaw(f, 1 / 30)) <= MAX_YAW)

    # The old -m error saturated near-vertical lines; arctan must not.
    reset()
    d.camera.down = line_image(slope=6.0)          # ~80 deg
    f = find_line(d, 1 / 30)
    check("steep line -> angle < pi/2", f and abs(f.angle) < math.pi / 2)

    reset()
    d.camera.down = line_image(curve=1.2)
    f = find_line(d, 1 / 30)
    check("curve -> bend > straight", f and f.bend > 0.15, f"got {f.bend:.3f}" if f else "")
    check("curve -> pitch < straight pitch", f and set_pitch(f) < PITCH_STRAIGHT)

    reset()
    d.camera.down = np.zeros((IMG_H, IMG_W, 3), np.uint8)
    check("blank frame -> None", find_line(d, 1 / 30) is None)
    d.camera.down = np.full((IMG_H, IMG_W, 3), 255, np.uint8)
    _wf = find_line(d, 1 / 30)                 # degenerate, but must not crash
    check("all-white frame handled",
          _wf is None or (math.isfinite(_wf.angle) and math.isfinite(_wf.cross)))

    print("\n-- gate geometry --")
    cy_true, h_px, cx_true = 200.0, 90.0, 320.0

    def fake_tags(roles):
        pos = {"T": (cx_true, cy_true - h_px), "B": (cx_true, cy_true + h_px),
               "L": (cx_true - h_px, cy_true), "R": (cx_true + h_px, cy_true)}
        s = 20.0
        centers, ids = [], []
        for i, r in enumerate(("T", "B", "L", "R")):
            if r in roles:
                x, y = pos[r]
                centers.append([x, y])
                ids.append(i)
        return np.array(centers, np.float64), np.full(len(ids), s), ids

    _role_by_id.clear()
    c, s, i = fake_tags("TBLR")
    _learn_roles(c, i)
    check("roles learned from 4 tags",
          all(_role_by_id.get(k) == v for k, v in zip((0, 1, 2, 3), "TBLR")),
          str(_role_by_id))

    for subset, tol, label in (
        ("TBLR", 1.0, "all four"),
        ("TB",   1.0, "top+bottom"),
        ("LR",   1.0, "left+right"),
        ("L",    1.0, "left only"),
        ("R",    1.0, "right only"),
        ("BLR",  1.0, "bottom+left+right"),
        ("TLR",  1.0, "top+left+right"),
    ):
        c, s, i = fake_tags(subset)
        cy, conf = _estimate_center_row(c, s, i)
        check(f"cy unbiased: {label}", abs(cy - cy_true) <= tol,
              f"got {cy:.1f} want {cy_true:.1f} (err {cy-cy_true:+.1f})")

    # Single top/bottom leans on GATE_H_PER_TAG; check the SIGN is right, since
    # a sign error here aims straight at the bar.
    for subset in ("T", "B"):
        c, s, i = fake_tags(subset)
        cy, conf = _estimate_center_row(c, s, i)
        check(f"single '{subset}' corrects toward centre",
              abs(cy - cy_true) < h_px, f"got {cy:.1f} want {cy_true:.1f}")
        check(f"single '{subset}' low confidence", conf <= GATE_CONF_SINGLE_TB)

    # What the ORIGINAL code did, for contrast.
    c, s, i = fake_tags("BLR")
    naive = float(np.mean(c[:, 1]))
    check("naive mean IS biased (old bug)", abs(naive - cy_true) > 20.0,
          f"naive={naive:.1f} true={cy_true:.1f}")

    print("\n-- gate clustering --")
    near = np.array([[300., 200.], [340., 200.]])
    far = np.array([[500., 210.], [510., 210.]])
    centers = np.vstack([near, far])
    sides = np.array([20., 20., 5., 5.])
    keep = _cluster_nearest_gate(centers, sides)
    check("far gate rejected", keep.tolist() == [True, True, False, False], str(keep))

    print("\n-- control loop --")
    reset()
    d.camera.down = line_image()
    for _ in range(60):
        update(d)
    check("loop runs, cmd finite", all(math.isfinite(v) for v in d.flight.cmd))
    check("no gate -> blend ~0", _gate_blend < 0.05, f"blend={_gate_blend:.3f}")
    check("base_alt captured", _base_alt is not None)

    reset()                                     # line loss -> climb
    d.camera.down = np.zeros((IMG_H, IMG_W, 3), np.uint8)
    for _ in range(40):
        update(d)
    check("line lost -> SEARCHING", _state == SEARCHING, _state)
    check("line lost -> climbing", d.flight.cmd[3] > 0, str(d.flight.cmd))
    check("line lost -> level attitude", d.flight.cmd[:3] == (0.0, 0.0, 0.0))

    d.camera.down = line_image()                # reacquire -> descend
    for _ in range(30):
        update(d)
    check("line found -> DESCENDING/FOLLOWING", _state in (DESCENDING, FOLLOWING), _state)

    reset()                                     # dt spikes must not explode
    d.camera.down = line_image(slope=0.3)
    for k in range(30):
        d.dt = 5.0 if k == 10 else 1 / 30.0
        update(d)
    d.dt = 1 / 30.0
    check("dt spike survived", all(abs(v) <= 1.0 for v in d.flight.cmd), str(d.flight.cmd))

    reset()                                     # missing color camera
    d.camera.down = line_image()
    saved = d.camera.get_color_image
    d.camera.get_color_image = None
    for _ in range(5):
        update(d)
    d.camera.get_color_image = saved
    check("no colour camera -> still flies", all(math.isfinite(v) for v in d.flight.cmd))

    print("\n-- gate control --")
    reset()
    obs = GateObs(320.0, 400.0, 1.0, 20.0, 5.0, [0, 1], {}, IMG_H, IMG_W)
    globals()["_gate_streak"] = GATE_MIN_FRAMES
    t1, act = gate_throttle(obs, 1.4, 1 / 30)
    check("gate low in frame -> descend", t1 < 0 and act, f"thr={t1:.3f}")
    obs.cy = 80.0
    globals()["_gate_streak"] = GATE_MIN_FRAMES
    t2, act = gate_throttle(obs, 1.4, 1 / 30)
    check("gate high in frame -> climb", t2 > 0 and act, f"thr={t2:.3f}")
    check("gate throttle clamped", abs(t2) <= GATE_THROTTLE_LIMIT + 1e-9)

    reset()                                      # low confidence -> low authority
    globals()["_gate_streak"] = GATE_MIN_FRAMES
    lo = GateObs(320.0, 80.0, GATE_CONF_UNKNOWN, 20.0, 5.0, [0], {}, IMG_H, IMG_W)
    t_lo, _ = gate_throttle(lo, 1.4, 1 / 30)
    reset()
    globals()["_gate_streak"] = GATE_MIN_FRAMES
    hi = GateObs(320.0, 80.0, 1.0, 20.0, 5.0, [0, 1], {}, IMG_H, IMG_W)
    t_hi, _ = gate_throttle(hi, 1.4, 1 / 30)
    check("low confidence -> less authority", abs(t_lo) < abs(t_hi),
          f"lo={t_lo:.3f} hi={t_hi:.3f}")

    reset()                                      # latch near the gate
    globals()["_gate_streak"] = GATE_MIN_FRAMES
    globals()["_relatch_t"] = 1.0e9
    near_obs = GateObs(320.0, 140.0, 1.0, 90.0, 1.0, [0, 1], {}, IMG_H, IMG_W)
    _t, act = gate_throttle(near_obs, 1.4, 1 / 30)
    check("close gate -> latched", _gate_latched and act)
    check("latch target above current alt", _gate_target_alt > 1.4,
          f"target={_gate_target_alt}")
    check("latch step bounded", abs(_gate_target_alt - 1.4) <= GATE_MAX_ALT_STEP_M + 1e-9)
    globals()["_gate_seen_t"] = GATE_LATCH_EXIT_S + 1.0
    _t, act = gate_throttle(None, 2.0, 1 / 30)
    check("latch releases after pass", not _gate_latched and not act)
    check("re-baselined to new altitude", abs(_base_alt - 2.0) < 1e-6, f"base={_base_alt}")

    print("\n-- PID --")
    p = PID(1.0, 0.1, 0.05, 0.5, 0.2)
    check("PID clamps output", abs(p.update(100.0, 1 / 30)) <= 0.5)
    check("PID handles NaN", p.update(float("nan"), 1 / 30) == 0.0)
    check("PID handles dt=0", math.isfinite(p.update(0.5, 0.0)))
    p.reset()
    p.hold()
    check("PID hold -> no deriv spike", abs(p.update(10.0, 1 / 30)) <= 0.5)

    print(f"\n{'='*52}\n{ok[0]} passed, {len(bad)} failed")
    if bad:
        for b in bad:
            print(f"  FAILED: {b}")
        return 1
    print("All self-tests passed.")
    return 0


if __name__ == "__main__" and _SELFTEST:
    _sys.exit(_selftest())

if __name__ == "__main__" and "--replay" in _sys.argv:
    _i = _sys.argv.index("--replay")
    _args = [a for a in _sys.argv[_i + 1:] if not a.startswith("--")]
    if not _args:
        print("usage: python final_demo3.py --replay <image-or-dir> [--down] [--save]")
        _sys.exit(2)
    _sys.exit(_replay(_args, downward="--down" in _sys.argv,
                      save="--save" in _sys.argv))


if __name__ == "__main__":
    _drone = drone_core.create_drone()
    _launcher = neo_lab.Launcher(1.4)

    def start():
        _launcher.reset()
        reset()
        print("Step 3: Follow the Edge + gate-altitude centring (corrected)")
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
