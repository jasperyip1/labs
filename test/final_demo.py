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
GATE_ROLL_KP      = 1.5
GATE_ROLL_KI      = 0.0
GATE_ROLL_KD      = 0.0
GATE_ROLL_I_LIMIT = 0.10

GATE_ALT_KP      = 0.3
GATE_ALT_KI      = 0.0
GATE_ALT_KD      = 0.0
GATE_ALT_I_LIMIT = 0.10

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
_gate_roll_pid = PID(GATE_ROLL_KP, GATE_ROLL_KI, GATE_ROLL_KD, ROLL_LIMIT, GATE_ROLL_I_LIMIT)
_gate_alt_pid  = PID(GATE_ALT_KP,  GATE_ALT_KI,  GATE_ALT_KD,  THROTTLE_LIMIT, GATE_ALT_I_LIMIT)


def reset():
    global _timer, _done, _frame, _mode
    global _line_state, _base_alt, _visible, _vis_timer, _settle_frame, _gate_confirm
    global _gate_phase, _gate, _hold, _forward_time, _forward_target, _forward_dist

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
    _gate_roll_pid.reset()
    _gate_alt_pid.reset()


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
    global _gate_confirm

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
            _mode = GATE
            _gate_phase = GATE_CENTER
            _hold = 0.0
            _forward_time = 0.0
            _forward_dist = 0.0
            _forward_target = (dist + FORWARD_MARGIN) if dist is not None else None
            _gate_confirm = 0
            _gate_roll_pid.reset()
            _gate_alt_pid.reset()
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
    global _line_state, _base_alt, _visible, _vis_timer, _settle_frame, _gate_confirm

    if _gate_phase == GATE_CENTER:
        img = drone.camera.get_color_image()
        if img is None:
            print("[gate] [Error]: Image is none!")
            return

        _gate = neo_lab.detect_gate(img)

        if _gate is None:
            drone.flight.stop()
            drone.flight.send_pcmd(0, 0, SEARCH_YAW, 0)
            _hold = 0.0
            _gate_roll_pid.hold()
            _gate_alt_pid.hold()
            if _frame % GATE_PRINT_EVERY == 0:
                print('[gate] no gate found - searching...')
            return

        width, height = drone.camera.get_width(), drone.camera.get_height()
        img_cx, img_cy = width / 2.0, height / 2.0

        # normalized error in [-1, 1]: +err_x = gate right of center
        err_x = (_gate.cx - img_cx) / (width / 2.0)
        roll = _gate_roll_pid.update(err_x, dt)

        # +err_alt = gate below center -> need to climb, so negate for throttle
        err_alt = (_gate.cy - img_cy) / (height / 2.0)
        throttle = _gate_alt_pid.update(-err_alt, dt)

        drone.flight.send_pcmd(0, roll, 0, throttle)

        if _frame % GATE_PRINT_EVERY == 0:
            print(f'[gate] centering | gate=({_gate.cx:.0f},{_gate.cy:.0f}) tags={_gate.count} '
                  f'err_x={err_x:.3f} err_alt={err_alt:.3f} '
                  f'roll={roll:.3f} throttle={throttle:.3f} hold={_hold:.2f}')

        enough_tags = _gate.count >= GATE_MIN_TAGS_TO_ADVANCE
        if abs(err_x) < CENTER_TOL and abs(err_alt) < ALT_TOL and enough_tags:
            _hold += dt
            if _hold >= CENTER_HOLD_T:
                drone.flight.stop()
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

        drone.flight.send_pcmd(GATE_PITCH, 0, 0, 0)

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