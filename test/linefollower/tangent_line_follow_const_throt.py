"""
MIT BWSI Autonomous Drone Racing Course - UAV Neo
GNU General Public License v3.0

Week 2/3 Lab — Step 3: Follow the Edge
Steer the drone to keep the bright edge centered while flying forward.
"""

import drone_core
import drone_utils as uav_utils
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
import neo_lab

# -- Constants --------------------------------------------------------------
V_MIN         = 200
MIN_PIXELS    = 200
MAX_ROLL      = 0.25     # strafe authority for centering
MAX_YAW       = 1.0     # yaw authority
FOLLOW_TIME   = 1000000.0     # seconds to follow before landing
IMAGE_CENTER  = 320      # 640-wide image -> center column

PITCH_STRAIGHT = 0.14    # fast on straights
PITCH_TURN     = 0.08    # slow through turns
CURVE_SCALE    = 100   # residual std at which you're "fully" in a turn (TUNE)

# -- PID gains --------------------------------------------------------------
# Start by increasing the proportional gain (KP) incrementally to a point where it
# oscillates, then reduce by 50%. Then add derivative gain (KD) to dampen oscillations,
# and finally add integral gain (KI) only if the drone consistently settles off-center.
# The integral term can help correct for steady-state errors, but it can also introduce
# overshoot if not tuned carefully.

YAW_KP      = 0.60       # Adjust first
YAW_KI      = 0.0
YAW_KD      = 0.055
YAW_I_LIMIT = 0.20      # cap on the integral's contribution to yaw

ROLL_KP      = 0.24      # Adjust first
ROLL_KI      = 0.0
ROLL_KD      = 0.02
ROLL_I_LIMIT = 0.10

D_TAU = 0.10    # derivative low-pass time constant, seconds (bigger = smoother)

# -- Altitude hold ------------------------------------------------------
# Straight P controller for now: KI and KD are set to 0.0 on purpose.
# Tune ALT_KP first (same procedure as the other axes: raise until it
# oscillates, back off ~50%), then reintroduce KD to damp overshoot, and
# only add KI if the drone settles at a steady offset above/below base_alt.
ALT_KP      = 0.6
ALT_KI      = 0.0
ALT_KD      = 0.0
ALT_I_LIMIT = 0.10
MAX_THROTTLE = 0.3      # cap on send_pcmd throttle magnitude

# -- Perception Constants ----------------------------------------------------
POLY_DEGREE = 3          # 3 or 5 both work; higher degree fits noise more easily
IMG_W, IMG_H = 640, 480
TARGET_POINT = (IMG_W / 2, IMG_H / 2 - 80)   # (x, y) — "slightly higher" than center
SAMPLE_STEP = 2           # px spacing when scanning the curve for the closest point


# -- PID controller ---------------------------------------------------------
class PID:
    """
    Setpoint is always 0 (edge centered / line vertical), so `error` is just
    the measurement. Includes a low-passed derivative and clamped integral
    with conditional anti-windup.
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
        Called on frames where the measurement is invalid (edge lost).
        Keeps the integral but drops the previous error, so the first frame
        after reacquiring the line doesn't produce a huge derivative spike.
        """
        self._prev_error = None

    def update(self, error, dt):
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

        # -- Integral, provisional until we know we aren't saturated.
        integral = uav_utils.clamp(
            self._integral + error * dt, -self.i_limit, self.i_limit
        )

        output = self.kp * error + self.ki * integral + self.kd * self._deriv
        clamped = uav_utils.clamp(output, -self.out_limit, self.out_limit)

        # -- Anti-windup: if we're already pinned at the limit and this error
        #    would push us further into the limit, throw the update away.
        saturated = output != clamped
        if not (saturated and output * error > 0):
            self._integral = integral

        return clamped


# -- Module-level state -----------------------------------------------------
_timer = 0.0
_done  = False
_base_alt = None

_yaw_pid  = PID(YAW_KP,  YAW_KI,  YAW_KD,  MAX_YAW,  YAW_I_LIMIT)
_roll_pid = PID(ROLL_KP, ROLL_KI, ROLL_KD, MAX_ROLL, ROLL_I_LIMIT)
_alt_pid  = PID(ALT_KP,  ALT_KI,  ALT_KD,  MAX_THROTTLE, ALT_I_LIMIT)


def reset():
    global _timer, _done, _base_alt
    _timer = 0.0
    _done  = False
    _base_alt = None
    _yaw_pid.reset()
    _roll_pid.reset()
    _alt_pid.reset()


# -- Perception -------------------------------------------------------------
def find_edge(drone):
    """
    Grab the downward image, threshold it, and fit a polynomial to the bright
    pixels (column as a function of row: x = f(y)).

    Finds the point on the fitted curve closest to TARGET_POINT (image center,
    or a bit above it), then returns the tangent line to the curve at that
    point in the same (m, b) format as before: column = m * row + b.

    Returns (ys, xs, m, b, poly, closest_pt) where `poly` is the fitted
    np.poly1d (used downstream for a curvature/straightness measure that
    reflects the actual fit, not just the local tangent), and closest_pt is
    the (row, col) point used for the tangent. Returns None if there aren't
    enough bright pixels to trust.
    """
    camera = drone.camera.get_downward_image()
    mask = neo_lab.bright_mask_improved(camera, V_MIN)
    edges = np.argwhere(mask)
    edges = edges.astype(np.float64)

    if np.count_nonzero(edges) < MIN_PIXELS:
        return None

    ys = edges[:, 0]
    xs = edges[:, 1]

    # Fit x = f(y) with a degree-3 (or 5) polynomial instead of a line.
    coeffs = np.polyfit(ys, xs, POLY_DEGREE)
    poly = np.poly1d(coeffs)
    poly_deriv = poly.deriv()

    # Sample the curve over the observed row range to find the point closest
    # to the target (image center or slightly above it).
    y_min, y_max = ys.min(), ys.max()
    sample_ys = np.arange(y_min, y_max + SAMPLE_STEP, SAMPLE_STEP)
    sample_xs = poly(sample_ys)

    target_x, target_y = TARGET_POINT
    dist_sq = (sample_xs - target_x) ** 2 + (sample_ys - target_y) ** 2
    best_idx = np.argmin(dist_sq)
    y0 = sample_ys[best_idx]
    x0 = sample_xs[best_idx]

    # Tangent line to the curve at (y0, x0): column = m * row + b
    m = poly_deriv(y0)
    b = x0 - m * y0

    return ys, xs, m, b, poly, (y0, x0)


# -- Control axes -----------------------------------------------------------
def set_yaw(m, dt):
    """Rotate to align the drone's heading with the edge's slope."""
    error = -m          # 0 when the line runs straight up the image
    return _yaw_pid.update(error, dt)


def set_roll(xs, dt):
    """Strafe to bring the average edge column back to the image center."""
    edge_col = xs.mean()      # average column of the bright edge
    error = (edge_col - IMAGE_CENTER) / IMAGE_CENTER   # -1 (left) .. +1 (right)
    return _roll_pid.update(error, dt)


def set_pitch(ys, xs, poly):
    """
    Fly fast when the edge fits the polynomial tightly (straight/simple),
    slow when the raw pixels deviate a lot from the fit (curvy/noisy).

    Note: this measures deviation from the polynomial fit itself, not from
    the tangent line used for steering — the tangent only approximates the
    curve near the steering point, so using it here would read as "curvy"
    even on a straight line.
    """
    curviness = np.std(xs - poly(ys))
    print(f'Curviness: {curviness}')
    straightness = uav_utils.clamp(1.0 - curviness / CURVE_SCALE, 0.0, 1.0)
    return PITCH_TURN + (PITCH_STRAIGHT - PITCH_TURN) * straightness


def set_throttle(drone, dt):
    """
    Altitude-hold P controller. Locks in the launch altitude the first time
    this is called, then drives throttle proportionally to how far off that
    altitude the drone currently is. KI and KD are 0 for now (pure P), so
    this will hold roughly steady but may settle with a small steady-state
    offset — that's expected until KI gets tuned in.
    """
    global _base_alt

    alt = drone.physics.get_altitude()
    if _base_alt is None:
        _base_alt = alt

    error = _base_alt - alt   # positive when below target -> climb
    return _alt_pid.update(error, dt)


# -- Main loop --------------------------------------------------------------
def update(drone):
    global _timer, _done
    # print('Update running')
    if _done:
        return True
    dt  = drone.get_delta_time()
    fit = find_edge(drone)
    throttle = set_throttle(drone, dt)

    if fit is None:
        _yaw_pid.hold()
        _roll_pid.hold()
        drone.flight.send_pcmd(0.0, 0.0, 0.0, throttle)   # hold level, hold altitude
    else:
        ys, xs, m, b, poly, closest_pt = fit
        pitch = set_pitch(ys, xs, poly)
        roll  = set_roll(xs, dt)
        yaw   = set_yaw(m, dt)
        print(f'Pitch = {pitch}, Roll = {roll}, Yaw = {yaw}, Throttle = {throttle}')
        drone.flight.send_pcmd(pitch, roll, yaw, throttle)

    _timer += dt
    if _timer >= FOLLOW_TIME:
        _done = True
    return _done

    ###### END PUT CODE HERE #########
    ##################################


if __name__ == "__main__":
    _drone = drone_core.create_drone()
    _launcher = neo_lab.Launcher(1.4)

    def start():
        _launcher.reset()
        reset()
        print("Step 3: Follow the Edge")

    def _update():
        if not _launcher.done:        # arm + climb to a safe height first
            _launcher.update(_drone)
            return
        if update(_drone):
            _drone.flight.land()

    print(f'------ STARTING CODE ---------\n\n')
    _drone.set_start_update(start, _update)
    _drone.go(not neo_lab._is_sim(_drone))