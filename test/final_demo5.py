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

  GATE THROTTLE   comes from final_demo3.py. ArUco tags are clustered to the
                  nearest gate, roles (top/bottom/left/right) are resolved by id,
                  and the opening centre is reconstructed from the geometry with a
                  confidence that scales control authority.

DIFFERENCE FROM final_demo4
  final_demo4 flies EVERY gate blind. This version flies only the FIRST gate
  blind; every other gate is crossed with line following still running. Which
  gates get the blind treatment is set by BLIND_PASS_TAG_IDS below -- it holds
  gate 1's four tag ids by default, so identification is by which gate is in
  frame, not by how many have been counted. A miscount or a spurious early
  detection therefore cannot shift the blind pass onto the wrong gate.

THREE REGIMES

  APPROACH   pitch, roll and yaw come from line following; the gate may only ever
             influence THROTTLE. This is final_demo3's rule, and it holds whenever
             the drone is further than GATE_COMMIT_DIST_M from a gate.

  BLIND      gate 1 only. Inside GATE_COMMIT_DIST_M (1.8 m) line following is
             switched OFF and the gate owns every axis. The gate structure is
             WHITE, so from close range it dominates the downward camera's bright
             mask and the line fit starts tracking the gate itself. There is no
             filtering fix for that -- the gate really is a big bright blob. So
             the drone freezes its heading, holds the altitude the gate implied,
             and flies forward on a fixed pitch until it is GATE_PASS_CLEAR_M
             (1 m) past the gate plane, then re-acquires the line.

  LATCHED    every other gate, flown EXACTLY as final_demo3 flies it. Line
             following keeps pitch/roll/yaw the whole way through; only throttle
             is taken over. gate_throttle is demo3's function unchanged, latch and
             all: image-space vertical centring on approach, then an absolute
             altitude frozen at GATE_LATCH_DIST_M and released on demo3's
             timeout/lost-detection conditions.

  Altitude is adjusted for EVERY gate on EVERY lap regardless of which regime
  applies -- that is gate_throttle's far-field PID, which runs on approach to all
  four gates. The regimes differ only in what happens in the last ~1.5 m.

  The blind crossing dead-reckons progress by integrating forward speed, since
  this airframe has no position source, and is hard-capped at GATE_PASS_SECONDS.

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
  mode -- the perception is the same, so its numbers apply directly:
      python final_demo3.py --replay <image-or-dir>

  Two numbers matter more now that a blind manoeuvre depends on them:
    GATE_TAG_SIZE_M      scales the frozen altitude linearly. Measure the printed
                         tag with a ruler.
    GATE_CAM_HFOV_DEG    sets obs.dist_m, and therefore both WHERE the commit
                         fires and how far the drone thinks it must fly to be
                         clear. Watch the d= field in the debug line against a
                         tape measure before trusting the 1.8 m trigger.

  See the GATE_PASS_PITCH note about how far 7 s of blind flight actually carries
  the real drone -- the default pitch does not clear the gate within the timeout.
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

# The same ratio on the horizontal axis, for turning a lone left or right tag into
# a gate centre column. On a circular hoop this equals GATE_H_PER_TAG; it is kept
# separate so a rectangular gate stays correct.
_GATE_W_PER_TAG_OVERRIDE = None
GATE_W_PER_TAG = (_GATE_W_PER_TAG_OVERRIDE if _GATE_W_PER_TAG_OVERRIDE is not None
                  else (GATE_INNER_WIDTH_M / 2.0 + GATE_TAG_SIZE_M / 2.0) / GATE_TAG_SIZE_M)

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

GATE_ALT_HOLD_KP    = 0.35   # throttle per metre of altitude error during a pass
GATE_MAX_ALT_STEP_M = 1.50   # m. Cap on how far one gate may move the target
                             # altitude -- a bad estimate cannot command a huge climb.

GATE_LATCH_DIST_M   = 1.50   # m. Inside this range the vertical target is frozen
                             # to an absolute altitude, because the tags are about
                             # to leave the frame.
GATE_LATCH_MAX_S    = 4.0    # s. Hard timeout so the latch can never hang.
GATE_LATCH_EXIT_S   = 0.60   # s without any detection while latched -> assume the
                             # gate has been passed. Must exceed GATE_LOST_HOLD_S.
GATE_LATCH_MIN_CONF = 0.90   # min confidence to freeze an absolute altitude
GATE_RELATCH_S      = 1.00   # s cooldown after a latch or a blind crossing ends,
                             # so the gate just flown cannot immediately re-arm

GATE_REBASELINE     = True   # after passing a gate, treat the new altitude as the
                             # reference the line-search FSM returns to


# ===========================================================================
# WHICH GATES ARE FLOWN BLIND
# ===========================================================================
# Tag ids belonging to the gates that get line following switched OFF for the
# crossing. Any other gate is crossed with line following still flying
# pitch/roll/yaw, and only its throttle taken over (see LATCHED above).
#
# Defaults to gate 1's four ids, taken from the "# Gate 1" row of
# GATE_TAG_ROLE_BY_ID above. Keep the two in step if you renumber the course.
# Add another gate's ids here to fly that one blind too; empty the set to keep
# line following through every gate.
#
# Identification is by tag id rather than by counting crossings on purpose: a
# spurious early detection would shift a counter onto the wrong gate, and the
# consequence of being wrong is flying blind at a gate that was never surveyed.
BLIND_PASS_TAG_IDS = {35, 0, 36, 34}     # Gate 1: T, L, B, R


# ===========================================================================
# GATE PASS-THROUGH -- the crossing commit
# ===========================================================================
# The gate structure is WHITE, so from close range it lights up the downward
# camera's bright mask and the line fit starts tracking the gate instead of the
# line. There is no filtering fix for that: the gate genuinely is a bright blob
# and it genuinely is bigger than the line. So inside GATE_COMMIT_DIST_M the line
# is abandoned outright -- the drone freezes its heading, flies forward on a fixed
# pitch at the altitude the gate implied, and only re-opens its eyes once it is
# clear on the far side.
#
# This applies ONLY to the gates in BLIND_PASS_TAG_IDS. Every other gate is flown
# exactly as final_demo3 flies it -- line following the whole way, with the
# altitude latch above taking over the vertical axis on approach. The blind commit
# fires at 1.80 m, before the latch's 1.50 m, so at a blind gate the latch simply
# never engages and there is no ambiguity about who owns the vertical axis.

GATE_COMMIT_DIST_M   = 1.80  # m. Range at which line following is abandoned.
                             # NOTE: obs.dist_m is derived from tag pixel size via
                             # _focal_px, so GATE_CAM_HFOV_DEG directly sets where
                             # this fires. Watch the d= field in the debug line
                             # against a tape measure before trusting it.
GATE_COMMIT_MIN_CONF = 0.90  # min gate confidence to commit. Defaults to the
                             # left/right tier: a single-tag estimate rests
                             # entirely on GATE_H_PER_TAG, which is too much trust
                             # for one uncalibrated constant on a blind manoeuvre.
GATE_PASS_CLEAR_M    = 1.00  # m past the gate plane before line following resumes

GATE_PASS_PITCH      = PITCH_STRAIGHT   # forward command held through the pass.
                             # SPEED ARITHMETIC, real drone: send_pcmd pitch maps
                             # to pitch * MAX_SPEED m/s and MAX_SPEED is 1.0
                             # (flight_real.py:29). At 0.22 that is 0.22 m/s, so
                             # clearing 1.80 + 1.00 = 2.80 m takes ~12.7 s -- far
                             # longer than GATE_PASS_SECONDS below, which means the
                             # timeout would end the pass INSIDE the gate. To clear
                             # 2.80 m within 7 s this needs to be ~0.40. The sim's
                             # send_pcmd is a tilt command and flies faster, so it
                             # exits on distance long before the timeout. See the
                             # note in _end_pass about which exit actually fired.
GATE_PASS_SECONDS    = 7.0   # s. Hard cap on the blind period. The pass ends at
                             # whichever comes first: the integrated distance below
                             # reaching target, or this.
GATE_PASS_MIN_S      = 0.30  # s. Floor, so a momentary bad velocity read cannot
                             # end the pass the frame after it starts.
GATE_PASS_FALLBACK_MPS = 0.25 # m/s assumed if get_linear_velocity() is unusable

# Lateral trim during the pass. The drone strafes to keep the gate opening
# centred while tags are still decoding, then holds zero once they leave frame.
# Set GATE_PASS_ALIGN_KP = 0.0 to fly the pass on pure dead reckoning.
GATE_PASS_ALIGN_KP       = 0.35  # roll per unit of normalised horizontal error
GATE_PASS_MAX_ROLL       = 0.20  # cap on that roll
GATE_PASS_ALIGN_MIN_CONF = 0.55  # below this the horizontal estimate is ignored


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
_gate_latched = False       # final_demo3's altitude latch (non-blind gates)
_gate_target_alt = None     # absolute altitude frozen by a latch or a commit
_gate_latch_t = 0.0         # s since the latch engaged
_relatch_t = 1.0e9          # s since the last latch/crossing ended (cooldown)

_passing = False            # flying a BLIND crossing: line following is OFF
_pass_t = 0.0               # s since the commit
_pass_dist = 0.0            # m travelled since the commit (velocity-integrated)
_pass_need_m = 0.0          # m of travel required to be clear
_pass_by_vel = False        # True if get_linear_velocity() is usable this pass

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
    global _passing, _pass_t, _pass_dist, _pass_need_m, _pass_by_vel
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
    _passing = False
    _pass_t = 0.0
    _pass_dist = 0.0
    _pass_need_m = 0.0
    _pass_by_vel = False
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
      cx        image column of the gate OPENING centre
      conf_x    0..1 confidence in cx; scales the lateral trim during a pass
      tag_px    mean tag side in px
      dist_m    range estimate, m (None if intrinsics unusable)
      roles     {id: 'T'|'B'|'L'|'R'} for the tags used
    """

    __slots__ = ("cx", "cy", "conf", "conf_x", "tag_px", "dist_m", "ids", "roles",
                 "count", "img_h", "img_w")

    def __init__(self, cx, cy, conf, tag_px, dist_m, ids, roles, img_h, img_w,
                 conf_x=GATE_CONF_UNKNOWN):
        self.cx = cx
        self.cy = cy
        self.conf = conf
        self.conf_x = conf_x
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


def _estimate_center_col(centers, sides, ids):
    """
    Horizontal twin of _estimate_center_row: the image COLUMN of the opening.

    Same edge-midpoint geometry, transposed. The TOP and BOTTOM tags each sit on
    the gate's vertical centreline, so either one alone is unbiased in x, while
    left and right are only unbiased as a pair.

    This matters more than the vertical case. During a blind pass the drone
    STRAFES on this number, and a lone left tag read naively as "the centre" says
    the gate is to the left -- steering the drone straight into the left edge, at
    exactly the moment it has no line to fall back on.
    """
    cols = {}
    for i, tid in enumerate(ids):
        role = GATE_TAG_ROLE_BY_ID.get(tid, _role_by_id.get(tid))
        if role is not None:
            cols.setdefault(role, []).append(float(centers[i][0]))
    mean_of = lambda k: float(np.mean(cols[k]))

    if "L" in cols and "R" in cols:
        return 0.5 * (mean_of("L") + mean_of("R")), GATE_CONF_TB_PAIR

    tb = [c for k in ("T", "B") if k in cols for c in cols[k]]
    if tb:
        return float(np.mean(tb)), GATE_CONF_LR

    w_px = GATE_W_PER_TAG * float(np.mean(sides))
    if "L" in cols:
        return mean_of("L") + w_px, GATE_CONF_SINGLE_TB
    if "R" in cols:
        return mean_of("R") - w_px, GATE_CONF_SINGLE_TB

    return float(np.mean(centers[:, 0])), GATE_CONF_UNKNOWN


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
    cx, conf_x = _estimate_center_col(centers, sides, id_list)
    tag_px = float(np.mean(sides))

    img_h, img_w = gray.shape[:2]
    dist_m = None
    if tag_px > 1e-3 and GATE_TAG_SIZE_M > 0.0:
        dist_m = GATE_TAG_SIZE_M * _focal_px(img_w) / tag_px

    roles = {t: GATE_TAG_ROLE_BY_ID.get(t, _role_by_id.get(t)) for t in id_list}
    if not np.isfinite(cy):
        return None
    if not np.isfinite(cx):
        cx, conf_x = float(img_w) * 0.5, GATE_CONF_UNKNOWN
    return GateObs(cx, cy, conf, tag_px, dist_m, id_list, roles, img_h, img_w,
                   conf_x=conf_x)


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
    Vertical command that centres the gate opening. final_demo3's version,
    unchanged -- this is what flies every gate that is not in BLIND_PASS_TAG_IDS.

    Two regimes:
      FAR   -- image-space PID on the normalised vertical error, authority scaled
               by the estimate's confidence. This is what raises or lowers the
               drone onto each gate's height on approach, every gate, every lap.
      CLOSE -- the target is converted to an ABSOLUTE altitude and latched. Every
               tag leaves the frame during the pass, so image-space control has
               nothing to servo on exactly when it matters; without the latch the
               drone reverts to the line FSM mid-gate.

    At a blind gate this is never reached: the commit fires at 1.80 m, outside the
    latch's 1.50 m, and update() stops calling this once _passing is set.

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
    End a latched gate pass. Re-baselines the search FSM to the altitude the gate
    put us at -- essential on a course where gates sit at different heights,
    otherwise a later line loss descends all the way back to launch height.
    """
    global _gate_latched, _gate_target_alt, _gate_latch_t, _base_alt, _relatch_t
    _gate_latched = False
    _gate_target_alt = None
    _gate_latch_t = 0.0
    _relatch_t = 0.0
    if GATE_REBASELINE:
        _base_alt = alt
        print(f"[gate] passed; reference height re-baselined to {alt:.2f} m")


# ---------------------------------------------------------------------------
# The blind pass
# ---------------------------------------------------------------------------
def _gate_is_blind(obs):
    """
    True when the gate in frame is one of the gates flown with line following OFF.

    Any decoded tag from a listed gate is enough. Requiring all four would fail at
    exactly the range where the commit happens -- the outer tags are the first to
    leave the frame on approach, which is what makes the crossing blind in the
    first place.
    """
    return bool(BLIND_PASS_TAG_IDS) and any(t in BLIND_PASS_TAG_IDS for t in obs.ids)


def _should_commit(obs):
    """
    True when a BLIND gate is close and trustworthy enough to commit to.

    _gate_is_blind is what keeps every other gate on final_demo3's path: without
    a match here the commit never fires, _passing stays False, and the gate is
    flown by line following with gate_throttle's latch, exactly as in demo3.
    """
    return (not _passing
            and obs is not None
            and _gate_is_blind(obs)
            and obs.dist_m is not None
            and obs.dist_m <= GATE_COMMIT_DIST_M
            and obs.conf >= GATE_COMMIT_MIN_CONF
            and _gate_streak >= GATE_MIN_FRAMES
            and _relatch_t >= GATE_RELATCH_S)


def _commit_pass(drone, obs, alt):
    """
    Abandon line following and fly blind through a gate. Only ever reached for a
    gate in BLIND_PASS_TAG_IDS.

    Freezes the altitude the gate implies -- the same computation the latch in
    gate_throttle does, just triggered 0.30 m earlier -- and records how far the
    drone has to travel to be clear: the measured range to the gate plane plus
    GATE_PASS_CLEAR_M.

    Note the altitude step is independent of the camera FOV. rise_m is
    -err_px * dist_m / f_px and dist_m is TAG_SIZE * f_px / tag_px, so f_px
    cancels -- it depends only on GATE_TAG_SIZE_M and the measured tag size. A
    wrong GATE_CAM_HFOV_DEG shifts WHERE the commit fires, not the height it aims
    for.
    """
    global _passing, _pass_t, _pass_dist, _pass_need_m, _pass_by_vel
    global _gate_target_alt, _gate_latched

    f_px = _focal_px(obs.img_w)
    err_px = obs.cy - max(obs.img_h * 0.5, 1.0)
    rise_m = uav_utils.clamp(-err_px * obs.dist_m / f_px,
                             -GATE_MAX_ALT_STEP_M, GATE_MAX_ALT_STEP_M)
    _gate_target_alt = alt + rise_m
    _gate_latched = False       # the blind crossing owns the vertical axis now
    _gate_alt_pid.hold()

    _passing = True
    _pass_t = 0.0
    _pass_dist = 0.0
    _pass_need_m = obs.dist_m + GATE_PASS_CLEAR_M

    # Decide the progress source ONCE, here. Switching mid-pass would make the
    # travelled distance jump.
    _pass_by_vel = False
    try:
        v = drone.physics.get_linear_velocity()
        _pass_by_vel = math.isfinite(float(v[2]))
    except Exception:
        pass

    src = "velocity" if _pass_by_vel else f"dead reckoning @{GATE_PASS_FALLBACK_MPS}m/s"
    print(f"[gate] BLIND COMMIT at {obs.dist_m:.2f}m conf={obs.conf:.2f} "
          f"ids={obs.ids}: line following OFF, crossing {_pass_need_m:.2f}m on "
          f"{src}, alt target {_gate_target_alt:.2f}m (rise {rise_m:+.2f}m)")


def _pass_progress(drone, dt):
    """
    Metres travelled since the commit, by integrating forward speed.

    There is no position source on this airframe, so this is dead reckoning and it
    drifts. That is what GATE_PASS_SECONDS is for -- it bounds the blind period
    regardless of what the integral says.
    """
    global _pass_dist
    if _pass_by_vel:
        try:
            v_fwd = float(drone.physics.get_linear_velocity()[2])  # (x, y, z) = (right, up, fwd)
            if math.isfinite(v_fwd):
                _pass_dist += max(v_fwd, 0.0) * dt      # backwards drift is not progress
                return _pass_dist
        except Exception:
            pass
    _pass_dist += GATE_PASS_FALLBACK_MPS * dt
    return _pass_dist


def _pass_throttle(alt):
    """Hold the altitude frozen at the commit. Used by BOTH crossing modes."""
    if _gate_target_alt is None:
        return 0.0
    return uav_utils.clamp(GATE_ALT_HOLD_KP * (_gate_target_alt - alt),
                           -GATE_THROTTLE_LIMIT, GATE_THROTTLE_LIMIT)


def _pass_attitude(obs):
    """
    The blind attitude: fixed forward pitch, frozen heading, and a lateral trim
    onto the opening while tags still decode.

    Yaw is held at zero deliberately. Rotating mid-crossing would change where
    "forward" points, and dead reckoning down a straight line is the only thing
    keeping track of where the drone is.
    """
    roll = 0.0
    if obs is not None and GATE_PASS_ALIGN_KP > 0.0 and obs.conf_x >= GATE_PASS_ALIGN_MIN_CONF:
        half_w = max(obs.img_w * 0.5, 1.0)
        err = uav_utils.clamp((obs.cx - half_w) / half_w, -1.0, 1.0)
        roll = uav_utils.clamp(GATE_PASS_ALIGN_KP * err * obs.conf_x,
                               -GATE_PASS_MAX_ROLL, GATE_PASS_MAX_ROLL)
    return GATE_PASS_PITCH, roll, 0.0


def _end_pass(alt, why):
    """
    Re-open the drone's eyes on the far side of a blind gate.

    The line has been ignored for several metres, so every piece of line state is
    started clean: a stale tangent point or slope from before the gate would be
    meaningless now, and the visibility timers must not resume mid-count and fire
    a search climb the instant following resumes.

    Re-baselines the search FSM to the altitude the gate put us at, the same as
    _release_latch does for a non-blind gate.
    """
    global _passing, _pass_t, _pass_dist, _pass_need_m
    global _gate_target_alt, _relatch_t, _base_alt, _state, _visible, _vis_timer
    global _prev_y0, _m_valid

    _passing = False
    _pass_t = 0.0
    _pass_dist = 0.0
    _pass_need_m = 0.0
    _gate_target_alt = None
    _relatch_t = 0.0
    _gate_alt_pid.hold()

    _prev_y0 = None
    _m_valid = False
    _yaw_pid.hold()
    _roll_pid.hold()
    _state = FOLLOWING
    _visible = True
    _vis_timer = 0.0

    if GATE_REBASELINE:
        _base_alt = alt
    print(f"[gate] blind crossing complete ({why}); line following ON at {alt:.2f} m")


# ===========================================================================
# MAIN LOOP
# ===========================================================================
def update(drone):
    global _timer, _done, _gate_streak, _gate_seen_t, _gate_blend, _dbg_t
    global _prev_y0, _m_valid, _base_alt, _last_obs, _last_alt
    global _pass_t

    if _done:
        return True

    # A dropped frame would otherwise inject a large integral step and a
    # derivative spike into every axis at once.
    dt = uav_utils.clamp(float(drone.get_delta_time()), DT_MIN, DT_MAX)

    # ---- Line perception. Skipped only during a blind crossing: there the white
    #      gate fills the bright mask, so the fit would be tracking the gate, and
    #      every filter it touched would be thrown away at the far side anyway.
    fit = None
    if not _passing:
        fit = find_edge(drone, dt)
        if fit is None:
            # Invalidate the perception filters. A search climb can last seconds,
            # and a pre-loss slope fed into the yaw PID on the first reacquired
            # frame is a real kick.
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

    # ---- Pass-through state machine. Runs before the controllers, because a
    #      commit changes who owns every axis this frame.
    was_passing = _passing
    if _passing:
        _pass_t += dt
        travelled = _pass_progress(drone, dt)
        if _pass_t >= GATE_PASS_MIN_S and travelled >= _pass_need_m:
            _end_pass(alt, f"{travelled:.2f}m travelled")
        elif _pass_t >= GATE_PASS_SECONDS:
            # Hit the clock before the integrator said we were clear. On the real
            # drone at PITCH 0.22 this is the NORMAL exit, and it means the pass
            # ended after ~1.5m of travel -- see the GATE_PASS_PITCH note.
            _end_pass(alt, f"{GATE_PASS_SECONDS:.1f}s timeout at {travelled:.2f}m "
                           f"of {_pass_need_m:.2f}m")
    elif _should_commit(obs):
        # Only ever true for a gate in BLIND_PASS_TAG_IDS. Every other gate falls
        # through to the demo3 path below, where gate_throttle's latch handles the
        # crossing and line following never stops. (_relatch_t is advanced inside
        # gate_throttle, as in demo3.)
        _commit_pass(drone, obs, alt)

    if was_passing and not _passing:
        # A blind crossing ended on THIS frame. Perception was skipped at the top
        # of the frame, so take a real look now -- otherwise the visibility tracker
        # below reads "we didn't look" as "the line is gone" and starts counting
        # down toward a search climb on the one frame it should be reacquiring.
        fit = find_edge(drone, dt)
        if fit is None:
            _prev_y0 = None
            _m_valid = False

    # ---- Command
    m = 0.0
    edge_col = float(IMAGE_CENTER)
    if _passing:
        # Blind crossing (gate 1 only): the gate owns EVERY axis. The line is not
        # trustworthy this close to a white gate, so it is not consulted at all.
        # The line FSM is frozen too -- _track_line_visibility is not called, so
        # its timers cannot run down and strand the drone in SEARCHING on the far
        # side.
        pitch, roll, yaw = _pass_attitude(obs)
        throttle = _pass_throttle(alt)
        _gate_blend = 1.0
    else:
        # ---- Throttle: BOTH sources every frame, cross-faded.
        #      The line FSM must keep running even while a gate is visible on
        #      approach, or its timers and altitude reference go stale.
        _track_line_visibility(fit, dt)
        thr_line = line_throttle(alt)
        thr_gate, gate_active = gate_throttle(obs, alt, dt)

        a = dt / (GATE_BLEND_TAU + dt)
        _gate_blend += a * ((1.0 if gate_active else 0.0) - _gate_blend)
        throttle = uav_utils.clamp(
            _gate_blend * thr_gate + (1.0 - _gate_blend) * thr_line, -1.0, 1.0)

        # ---- Attitude: line only.
        if fit is None:
            _yaw_pid.hold()
            _roll_pid.hold()
            pitch = roll = yaw = 0.0      # hold level, let throttle search
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
        if _passing:
            line_s = "line=OFF(blind)"
        elif fit is None:
            line_s = "line=LOST"
        else:
            line_s = (f"m={m:+6.2f} col={edge_col:5.0f} "
                      f"pt=({closest_pt[0]:3.0f},{closest_pt[1]:3.0f})")
        if obs is None:
            gate_s = "gate=-"
        else:
            d = f"{obs.dist_m:.2f}m" if obs.dist_m is not None else "?"
            gate_s = (f"gate n={obs.count} conf={obs.conf:.2f}/{obs.conf_x:.2f} "
                      f"cy={obs.cy:.0f} cx={obs.cx:.0f} d={d}"
                      f"{' LATCH' if _gate_latched else ''}")
        if _passing:
            gate_s += (f" BLIND {_pass_t:.1f}/{GATE_PASS_SECONDS:.1f}s "
                       f"{_pass_dist:.2f}/{_pass_need_m:.2f}m")
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
        print("final_demo5: line following + blind crossing at gate 1 only")
        print(f"  aruco={GATE_ARUCO_DICT}  h_per_tag={GATE_H_PER_TAG:.2f}  "
              f"focal={_focal_px(IMG_W):.0f}px")
        print(f"  commit at {GATE_COMMIT_DIST_M:.2f}m -> cross "
              f"{GATE_COMMIT_DIST_M + GATE_PASS_CLEAR_M:.2f}m at pitch "
              f"{GATE_PASS_PITCH:.2f}, capped at {GATE_PASS_SECONDS:.1f}s")
        print(f"  blind (line following OFF) at tag ids "
              f"{sorted(BLIND_PASS_TAG_IDS) if BLIND_PASS_TAG_IDS else '(none)'}; "
              f"every other gate keeps line following")
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
