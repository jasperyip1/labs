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
MAX_ROLL      = 0.5     # strafe authority for centering
MAX_YAW       = 1.0     # yaw authority
FOLLOW_TIME   = 1000000.0     # seconds to follow before landing
IMAGE_CENTER  = 320      # 640-wide image -> center column

PITCH_STRAIGHT = 0.8    # fast on straights
PITCH_TURN     = 0.2    # slow through turns
CURVE_SCALE    = 160   # residual std at which you're "fully" in a turn (TUNE)

# -- PID gains --------------------------------------------------------------
# KP values below reproduce your original proportional-only behavior exactly
# when KI and KD are 0. Tune KD up first (damps the weave), then KI only if
# the drone settles consistently off-center.
YAW_KP      = 1.0
YAW_KI      = 0.0
YAW_KD      = 0.1
YAW_I_LIMIT = 0.30      # cap on the integral's contribution to yaw

ROLL_KP      = 0.2      # was MAX_ROLL used as the gain
ROLL_KI      = 0.0
ROLL_KD      = 0.15
ROLL_I_LIMIT = 0.10

D_TAU = 0.10    # derivative low-pass time constant, seconds (bigger = smoother)

# -- Line-search climb -------------------------------------------------------
MAX_CLIMB        = 3    # meters above launch height to search (TUNE)
CLIMB_THROTTLE   = 0.3    # send_pcmd throttle while searching
DESCEND_THROTTLE = -0.2   # gentler than the climb on purpose
LOST_GRACE       = 0.4    # seconds of no-line before climbing
FOUND_GRACE      = 0.5    # seconds of line before committing to descend
HEIGHT_TOL       = 0.10   # meters; "back at base height"


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
FOLLOWING, SEARCHING, DESCENDING = "FOLLOWING", "SEARCHING", "DESCENDING"

_state     = FOLLOWING
_base_alt  = None
_visible   = True
_vis_timer = 0.0

_yaw_pid  = PID(YAW_KP,  YAW_KI,  YAW_KD,  MAX_YAW,  YAW_I_LIMIT)
_roll_pid = PID(ROLL_KP, ROLL_KI, ROLL_KD, MAX_ROLL, ROLL_I_LIMIT)


def reset():
    global _timer, _done, _state, _base_alt, _visible, _vis_timer
    _timer = 0.0
    _done  = False
    _state     = FOLLOWING
    _base_alt  = None
    _visible   = True
    _vis_timer = 0.0
    _yaw_pid.reset()
    _roll_pid.reset()


# -- Perception -------------------------------------------------------------
def find_edge(drone):
    """
    Grab the downward image, threshold it, and fit a line to the bright pixels.

    Returns (ys, xs, m, b) where the fit is column = m * row + b,
    or None if there aren't enough bright pixels to trust.
    """
    camera = drone.camera.get_downward_image()
    mask = neo_lab.bright_mask(camera, V_MIN)
    edges = np.argwhere(mask)
    edges = edges.astype(np.float64)

    if np.count_nonzero(edges) < MIN_PIXELS:
        return None

    ys = edges[:, 0]
    xs = edges[:, 1]
    m, b = np.polyfit(ys, xs, 1)
    return ys, xs, m, b


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


def set_pitch(ys, xs, m, b):
    """Fly fast when the edge fits a straight line, slow when it curves."""
    curviness = np.std(xs - (m * ys + b))
    print(curviness)
    straightness = uav_utils.clamp(1.0 - curviness / CURVE_SCALE, 0.0, 1.0)
    return PITCH_TURN + (PITCH_STRAIGHT - PITCH_TURN) * straightness


def set_throttle(drone, fit, dt):
    """
    Climb to look for a lost edge, then descend back to the launch height
    once it's reacquired. Returns a send_pcmd throttle value.
    """
    global _state, _base_alt, _visible, _vis_timer

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


# -- Main loop --------------------------------------------------------------
def update(drone):
    global _timer, _done
    if _done:
        return True
    dt  = drone.get_delta_time()
    fit = find_edge(drone)
    throttle = set_throttle(drone, fit, dt)

    if fit is None:
        _yaw_pid.hold()
        _roll_pid.hold()
        drone.flight.send_pcmd(0.0, 0.0, 0.0, throttle)   # hold level, climb
    else:
        ys, xs, m, b = fit
        pitch = set_pitch(ys, xs, m, b)
        roll  = set_roll(xs, dt)
        yaw   = set_yaw(m, dt)
        drone.flight.send_pcmd(pitch, roll, yaw, throttle)

    _timer += dt
    if _timer >= FOLLOW_TIME:
        _done = True
    return _done

    ###### END PUT CODE HERE #########
    ##################################


if __name__ == "__main__":
    _drone = drone_core.create_drone()
    _launcher = neo_lab.Launcher(1)

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

    _drone.set_start_update(start, _update)
    _drone.go(not neo_lab._is_sim(_drone))