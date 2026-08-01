"""
MIT BWSI Autonomous Drone Racing Course - UAV Neo
GNU General Public License v3.0

Merged final_demo3.py with tangent_line_filtered_line_follow.py line tracking.
"""

import math
import cv2
import numpy as np

# -- Course setup: makes the shared `neo_lab` helper importable. --
import os as _os, sys as _sys
_d = _os.path.dirname(_os.path.realpath(__file__))
while _os.path.basename(_d) != "labs" and _os.path.dirname(_d) != _d:
    _d = _os.path.dirname(_d)
if _d not in _sys.path:
    _sys.path.insert(0, _d)

try:
    import drone_core
    import drone_utils as uav_utils
    import neo_lab
except ImportError:
    pass

# ===========================================================================
# GENERAL & PERCEPTION CONSTANTS
# ===========================================================================
DEBUG_PRINT        = True
DEBUG_PERIOD_S     = 0.25
FOLLOW_TIME        = 1.0e6
DT_MIN, DT_MAX     = 1.0e-3, 0.1

V_MIN         = 200
MIN_PIXELS    = 200
POLY_DEGREE   = 3          
IMG_W, IMG_H  = 640, 480
TARGET_POINT  = (IMG_W / 2, IMG_H / 2 - 80)   
SAMPLE_STEP   = 2           
CONTINUITY_WEIGHT = 0.25   
M_TAU = 0.12              

# ===========================================================================
# LINE CONTROL CONSTANTS
# ===========================================================================
MAX_ROLL      = 0.25     
MAX_YAW       = 1.0     
IMAGE_CENTER  = 320      

PITCH_STRAIGHT = 0.15    
PITCH_TURN     = 0.08    
CURVE_SCALE    = 70   

YAW_KP      = 0.56       
YAW_KI      = 0.0
YAW_KD      = 0.055
YAW_I_LIMIT = 0.20      

ROLL_KP      = 0.23      
ROLL_KI      = 0.0
ROLL_KD      = 0.02
ROLL_I_LIMIT = 0.10

D_TAU = 0.10    

# ===========================================================================
# LINE-SEARCH THROTTLE STATE MACHINE CONSTANTS
# ===========================================================================
MAX_CLIMB        = 3.0   
CLIMB_THROTTLE   = 0.3
DESCEND_THROTTLE = -0.2  
LOST_GRACE       = 0.4   
FOUND_GRACE      = 0.5   
HEIGHT_TOL       = 0.10  

# ===========================================================================
# GATE GEOMETRY & DETECTION (Kept for final_demo3 structure)
# ===========================================================================
GATE_INNER_HEIGHT_M = 1.524  
GATE_TAG_SIZE_M     = 0.2667 
GATE_ARUCO_DICT     = "DICT_5X5_100"

# ===========================================================================
# PID CONTROLLER 
# ===========================================================================
class PID:
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
        self._prev_error = None

    def update(self, error, dt):
        error = float(error)
        if not np.isfinite(error):          
            self.hold()                     
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

        if not (output != clamped and output * error > 0.0):
            self._integral = integral
        return clamped


# ===========================================================================
# GLOBALS FOR FILTERS AND STATE MACHINE
# ===========================================================================
_timer = 0.0
_done = False

_prev_y0 = None 
_m_filt = 0.0 

FOLLOWING, SEARCHING, DESCENDING = "FOLLOWING", "SEARCHING", "DESCENDING"
_state = FOLLOWING
_base_alt = None
_visible = True
_vis_timer = 0.0

_yaw_pid = PID(YAW_KP, YAW_KI, YAW_KD, MAX_YAW, YAW_I_LIMIT)
_roll_pid = PID(ROLL_KP, ROLL_KI, ROLL_KD, MAX_ROLL, ROLL_I_LIMIT)

# ===========================================================================
# PERCEPTION
# ===========================================================================
def find_edge(drone, dt):
    """
    Grab the downward image, threshold it, and fit a polynomial to the bright
    pixels (column as a function of row: x = f(y)).
    """
    global _prev_y0, _m_filt

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

    # Continuity bias: penalize points far from where we were tracking last
    # frame, so the selection can't teleport to a different branch of the
    # curve just because it's momentarily closer to the target.
    if _prev_y0 is not None:
        dist_sq = dist_sq + CONTINUITY_WEIGHT * (sample_ys - _prev_y0) ** 2

    best_idx = np.argmin(dist_sq)
    y0 = sample_ys[best_idx]
    x0 = sample_xs[best_idx]
    _prev_y0 = y0

    # Raw tangent slope at the selected point.
    m_raw = poly_deriv(y0)

    # Low-pass the slope itself — this is what actually feeds the yaw PID,
    # so filtering here (not just the point selection) catches noise that
    # continuity-biasing alone doesn't.
    if dt > 0.0:
        alpha = dt / (M_TAU + dt)
        _m_filt += alpha * (m_raw - _m_filt)
    else:
        _m_filt = m_raw
    m = _m_filt
    b = x0 - m * y0

    return ys, xs, m, b, poly, (y0, x0)


# ===========================================================================
# CONTROL AXES
# ===========================================================================
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
    """
    curviness = np.std(xs - poly(ys))
    # print(f'Curviness: {curviness}')
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


# ===========================================================================
# UPDATE LOOP (Execution)
# ===========================================================================
def update(drone):
    global _timer, _done

    if _done:
        return True

    # Get delta time, clamp for stability
    dt_raw = drone.get_delta_time()
    dt = uav_utils.clamp(dt_raw, DT_MIN, DT_MAX)
    
    # 1. Perception
    fit = find_edge(drone, dt)

    # 2. Altitude Control State Machine
    throttle = set_throttle(drone, fit, dt)

    # 3. Flight Control Update
    if fit is None:
        _yaw_pid.hold()
        _roll_pid.hold()
        # Lost line: use the throttle state machine to climb/search, hold pitch/roll/yaw at 0
        drone.flight.send_pcmd(0.0, 0.0, 0.0, throttle)
    else:
        ys, xs, m, b, poly, closest_pt = fit
        
        pitch = set_pitch(ys, xs, poly)
        roll  = set_roll(xs, dt)
        yaw   = set_yaw(m, dt)
        
        # print(f'Pitch = {pitch:.3f}, Roll = {roll:.3f}, Yaw = {yaw:.3f}, Throttle = {throttle:.3f}')
        drone.flight.send_pcmd(pitch, roll, yaw, throttle)

    # 4. Timer
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
        print("Starting flight with filtered tangent logic...")

    _drone.set_start_callback(start)
    _drone.set_update_callback(update)
    _drone.run()