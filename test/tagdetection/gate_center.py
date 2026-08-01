"""
MIT BWSI Autonomous Drone Racing Course - UAV Neo
GNU General Public License v3.0

Gate Centering — single-tag capable.

  no tags seen  -> yaw at SEARCH_YAW until tags appear
  ONE tag seen  -> solvePnP: rvec aligns yaw, tvec centers on the GATE center
  2+ tags seen  -> same pixel-centroid logic as final_demo.py

The single-tag case works because each tag's 4 image corners are paired with their
true 3D positions in the GATE frame (tag center offset by the gate geometry, then
+/- half a tag). So solvePnP returns the pose of the *gate*, not the tag -- and
projecting the gate origin back into the image gives the true gate center even
when only one corner tag is visible.

Centering only. No fly-through (add GATE_FORWARD from final_demo.py once this works).
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
# ArUco detector
# ============================================================
# These gates carry 5x5 tags. NOTE: neo_lab._detect_gate_markers() and
# neo_lab.detect_gate() are bound to the course's DICT_6X6_250, so they return
# *nothing* on 5x5 tags -- silently, as "no gate" rather than as an error. That's
# why this script runs its own detector instead of the course helper.
# If you ever point this at the sim's 6x6 gates, flip this one constant.
ARUCO_DICT = cv2.aruco.DICT_5X5_250

try:                                  # OpenCV >= 4.7 (yours is 4.8.1)
    _adict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    _aparams = cv2.aruco.DetectorParameters()
    _detector = cv2.aruco.ArucoDetector(_adict, _aparams)

    def _detect_markers(gray):
        return _detector.detectMarkers(gray)
except AttributeError:                # older OpenCV fallback
    _adict = cv2.aruco.Dictionary_get(ARUCO_DICT)
    _aparams = cv2.aruco.DetectorParameters_create()

    def _detect_markers(gray):
        return cv2.aruco.detectMarkers(gray, _adict, parameters=_aparams)


# ============================================================
# Gate geometry
# ============================================================
# Diamond layout: one tag at each of top / bottom / left / right.
ID_TOP, ID_BOTTOM, ID_LEFT, ID_RIGHT = 32, 34, 35, 33

# [CONFIRM] 1.8 m is taken as the TAG-CENTER-to-TAG-CENTER span (top tag center to
# bottom tag center, and left to right). If 1.8 is instead the gate's open aperture,
# the tag centers sit slightly outside that -- adjust here.
GATE_W = 1.8
GATE_H = 1.8

# [MEASURE THIS] Printed side length of one ArUco tag, in METERS -- the black border
# square, edge to edge. This sets the metric SCALE of the single-tag solution: if it's
# wrong, tvec (and therefore the reported distance) is wrong by the same ratio.
# Centering direction still works, but the distance print will lie.
TAG_SIZE = 0.15

# Gate-frame tag centers: +X right, +Y down, Z = 0 (the gate plane), origin = gate center.
ID_TO_CENTER = {
    ID_TOP:    (0.0,        -GATE_H / 2.0),
    ID_BOTTOM: (0.0,        +GATE_H / 2.0),
    ID_LEFT:   (-GATE_W / 2.0, 0.0),
    ID_RIGHT:  (+GATE_W / 2.0, 0.0),
}

# ============================================================
# Camera intrinsics (nominal, derived from FOV + resolution)
# ============================================================
# The library exposes no get_intrinsics(), so we derive a nominal pinhole matrix.
# SIM and REAL are NOT the same camera model:
#   sim   -- Unity renders square pixels at vertical FOV 42.5 deg -> fx == fy
#   real  -- D435 color spec FOV is 69.4 x 42.5 deg -> fx != fy
SIM_HFOV_DEG,  SIM_VFOV_DEG  = 54.8, 42.5   # Unity: 42.5 vertical, 4:3 -> 54.8 horizontal
REAL_HFOV_DEG, REAL_VFOV_DEG = 69.4, 42.5   # Intel D435 color sensor spec

_K = None
_D = np.zeros((5, 1), dtype=np.float64)     # both are rectified/pinhole -> no distortion


def _init_intrinsics(drone):
    """Build the camera matrix once, from the library's own reported resolution."""
    global _K
    w, h = drone.camera.get_width(), drone.camera.get_height()
    if neo_lab._is_sim(drone):
        hfov, vfov, tag = SIM_HFOV_DEG, SIM_VFOV_DEG, "sim"
    else:
        hfov, vfov, tag = REAL_HFOV_DEG, REAL_VFOV_DEG, "real"
    fx = (w / 2.0) / np.tan(np.radians(hfov) / 2.0)
    fy = (h / 2.0) / np.tan(np.radians(vfov) / 2.0)
    _K = np.array([[fx, 0.0, w / 2.0],
                   [0.0, fy, h / 2.0],
                   [0.0, 0.0, 1.0]], dtype=np.float64)
    print(f"[intrinsics] {tag} {w}x{h}  fx={fx:.1f} fy={fy:.1f} "
          f"cx={w/2:.0f} cy={h/2:.0f}")


# ============================================================
# Control constants  (seeded from final_demo.py)
# ============================================================
GATE_ROLL_KP, GATE_ROLL_KI, GATE_ROLL_KD = 1.5, 0.0, 0.0
GATE_ROLL_I_LIMIT = 0.10
GATE_ALT_KP,  GATE_ALT_KI,  GATE_ALT_KD  = 0.3, 0.0, 0.0
GATE_ALT_I_LIMIT  = 0.10

ROLL_LIMIT     = 0.3
THROTTLE_LIMIT = 0.3

CENTER_TOL    = 0.05     # normalized image error
ALT_TOL       = 0.05
YAW_TOL       = 0.12     # rad (~7 deg), single-tag alignment only
CENTER_HOLD_T = 0.5      # seconds held centered before declaring done

SEARCH_YAW              = 0.2
GATE_SEARCH_START_DELAY = 1.5    # settle before sweeping, so an in-view gate isn't yawed away

# Single-tag yaw alignment (from rvec).
YAW_ALIGN_KP    = 0.8
YAW_ALIGN_LIMIT = 0.4

# Multi-tag: brake residual yaw rate instead of just commanding 0 (final_demo behavior).
YAW_BRAKE_KP    = 0.5
YAW_BRAKE_LIMIT = 0.5

# Search-state drift hold (final_demo behavior).
SEARCH_HOLD_KP, SEARCH_HOLD_KI, SEARCH_HOLD_KD = 0.15, 0.0, 0.02
SEARCH_HOLD_I_LIMIT = 0.10
SEARCH_HOLD_LIMIT   = 0.20

D_TAU = 0.10
PRINT_EVERY = 5


# ============================================================
# PID (same as final_demo.py)
# ============================================================
class PID:
    def __init__(self, kp, ki, kd, out_limit, i_limit, d_tau=D_TAU):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.out_limit, self.i_limit, self.d_tau = out_limit, i_limit, d_tau
        self.reset()

    def reset(self):
        self._integral = 0.0
        self._prev_error = None
        self._deriv = 0.0

    def hold(self):
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
        integral = uav_utils.clamp(self._integral + error * dt, -self.i_limit, self.i_limit)
        output = self.kp * error + self.ki * integral + self.kd * self._deriv
        clamped = uav_utils.clamp(output, -self.out_limit, self.out_limit)
        if not ((output != clamped) and output * error > 0):
            self._integral = integral
        return clamped


# ============================================================
# Module state
# ============================================================
_done = False
_frame = 0
_hold = 0.0
_center_time = 0.0

_roll_pid = PID(GATE_ROLL_KP, GATE_ROLL_KI, GATE_ROLL_KD, ROLL_LIMIT, GATE_ROLL_I_LIMIT)
_alt_pid  = PID(GATE_ALT_KP,  GATE_ALT_KI,  GATE_ALT_KD,  THROTTLE_LIMIT, GATE_ALT_I_LIMIT)
_search_roll_pid     = PID(SEARCH_HOLD_KP, SEARCH_HOLD_KI, SEARCH_HOLD_KD,
                           SEARCH_HOLD_LIMIT, SEARCH_HOLD_I_LIMIT)
_search_pitch_pid    = PID(SEARCH_HOLD_KP, SEARCH_HOLD_KI, SEARCH_HOLD_KD,
                           SEARCH_HOLD_LIMIT, SEARCH_HOLD_I_LIMIT)
_search_throttle_pid = PID(SEARCH_HOLD_KP, SEARCH_HOLD_KI, SEARCH_HOLD_KD,
                           SEARCH_HOLD_LIMIT, SEARCH_HOLD_I_LIMIT)


def reset():
    global _done, _frame, _hold, _center_time
    _done = False
    _frame = 0
    _hold = 0.0
    _center_time = 0.0
    _roll_pid.reset()
    _alt_pid.reset()
    _search_roll_pid.reset()
    _search_pitch_pid.reset()
    _search_throttle_pid.reset()


# ============================================================
# Perception
# ============================================================
def _tag_object_points(tag_id):
    """The 4 corners of this tag, in GATE-frame meters, in detectMarkers order
    (TL, TR, BR, BL). Tags assumed mounted upright at the diamond points."""
    cx, cy = ID_TO_CENTER[tag_id]
    s = TAG_SIZE / 2.0
    return np.array([
        [cx - s, cy - s, 0.0],
        [cx + s, cy - s, 0.0],
        [cx + s, cy + s, 0.0],
        [cx - s, cy + s, 0.0],
    ], dtype=np.float64)


def detect(drone):
    """Returns (tag_centers, ids, corners) for gate tags only, or (None, None, None)."""
    img = drone.camera.get_color_image()
    if img is None:
        return None, None, None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = _detect_markers(gray)
    if ids is None or len(ids) == 0:
        return None, None, None

    keep_c, keep_i = [], []
    for c, i in zip(corners, ids.flatten()):
        if int(i) in ID_TO_CENTER:
            keep_c.append(c.reshape(-1, 2))
            keep_i.append(int(i))
    if not keep_i:
        return None, None, None

    centers = np.array([c.mean(axis=0) for c in keep_c])
    return centers, keep_i, keep_c


def solve_single_tag(tag_corners, tag_id):
    """solvePnP on ONE tag -> (gate_center_px, yaw_err_rad, distance_m) or None.

    The object points are in the GATE frame, so the solution is the gate's pose.
    Projecting the gate origin back into the image gives the true gate center --
    which is generally NOT where the tag is.
    """
    obj = _tag_object_points(tag_id)
    img = np.array(tag_corners, dtype=np.float64)

    ok, rvec, tvec = cv2.solvePnP(obj, img, _K, _D, flags=cv2.SOLVEPNP_IPPE)
    if not ok:
        return None

    # Gate center (gate-frame origin) projected into the image.
    proj, _ = cv2.projectPoints(np.zeros((1, 3)), rvec, tvec, _K, _D)
    gate_px = proj.reshape(2)

    # Heading error: gate's X axis expressed in camera frame. Square-on -> ~0.
    R, _ = cv2.Rodrigues(rvec)
    yaw_err = float(np.arctan2(R[2, 0], R[0, 0]))

    return gate_px, yaw_err, float(tvec[2])


# ============================================================
# Main update
# ============================================================
def update(drone):
    global _done, _frame, _hold, _center_time
    if _done:
        return True
    if _K is None:
        _init_intrinsics(drone)

    dt = drone.get_delta_time()
    _frame += 1

    width, height = drone.camera.get_width(), drone.camera.get_height()
    img_cx, img_cy = width / 2.0, height / 2.0

    centers, ids, corners = detect(drone)

    # ---------- no tags: hold position and sweep ----------
    if ids is None:
        _center_time += dt
        vx, vy, vz = (float(v) for v in drone.physics.get_linear_velocity())  # right, up, forward
        hold_roll     = _search_roll_pid.update(-vx, dt)
        hold_throttle = _search_throttle_pid.update(-vy, dt)
        hold_pitch    = _search_pitch_pid.update(-vz, dt)

        yaw = 0.0 if _center_time < GATE_SEARCH_START_DELAY else SEARCH_YAW
        drone.flight.send_pcmd(hold_pitch, hold_roll, yaw, hold_throttle)

        _hold = 0.0
        _roll_pid.hold()
        _alt_pid.hold()
        if _frame % PRINT_EVERY == 0:
            phase = "settling" if yaw == 0.0 else "searching"
            print(f"[gate] no tags - {phase} | drift vx={vx:.2f} vy={vy:.2f} vz={vz:.2f}")
        return False

    _search_roll_pid.hold()
    _search_pitch_pid.hold()
    _search_throttle_pid.hold()

    n = len(ids)
    yaw_err = None
    dist = None

    # ---------- ONE tag: solvePnP ----------
    if n == 1:
        sol = solve_single_tag(corners[0], ids[0])
        if sol is None:
            drone.flight.send_pcmd(0, 0, 0, 0)
            if _frame % PRINT_EVERY == 0:
                print(f"[gate] 1 tag (id={ids[0]}) but solvePnP failed")
            return False
        gate_px, yaw_err, dist = sol
        gx, gy = float(gate_px[0]), float(gate_px[1])
        yaw_cmd = uav_utils.clamp(YAW_ALIGN_KP * yaw_err, -YAW_ALIGN_LIMIT, YAW_ALIGN_LIMIT)
        mode = f"pnp id={ids[0]}"

    # ---------- 2+ tags: pixel centroid (final_demo logic) ----------
    else:
        gx, gy = float(centers[:, 0].mean()), float(centers[:, 1].mean())
        yaw_rate = float(drone.physics.get_angular_velocity()[1])   # index 1 = yaw
        yaw_cmd = uav_utils.clamp(-YAW_BRAKE_KP * yaw_rate, -YAW_BRAKE_LIMIT, YAW_BRAKE_LIMIT)
        mode = f"centroid n={n}"

    # ---------- shared centering control ----------
    err_x   = (gx - img_cx) / (width / 2.0)     # + => gate is right of center
    err_alt = (gy - img_cy) / (height / 2.0)    # + => gate is below center

    roll     = _roll_pid.update(err_x, dt)
    throttle = _alt_pid.update(-err_alt, dt)    # gate below -> climb

    drone.flight.send_pcmd(0, roll, yaw_cmd, throttle)

    if _frame % PRINT_EVERY == 0:
        extra = ""
        if yaw_err is not None:
            extra = f" yaw_err={np.degrees(yaw_err):+.1f}deg dist={dist:.2f}m"
        print(f"[gate] {mode} | center=({gx:.0f},{gy:.0f}) "
              f"err_x={err_x:+.3f} err_alt={err_alt:+.3f} "
              f"roll={roll:+.3f} yaw={yaw_cmd:+.3f} thr={throttle:+.3f} "
              f"hold={_hold:.2f}{extra}")

    # ---------- centered? ----------
    aligned = True if yaw_err is None else abs(yaw_err) < YAW_TOL
    if abs(err_x) < CENTER_TOL and abs(err_alt) < ALT_TOL and aligned:
        _hold += dt
        if _hold >= CENTER_HOLD_T:
            drone.flight.stop()
            print(f"[gate] CENTERED ({mode})"
                  + (f" at {dist:.2f}m" if dist is not None else ""))
            _done = True
    else:
        _hold = 0.0

    return _done


if __name__ == "__main__":
    _drone = drone_core.create_drone()
    _launcher = neo_lab.Launcher(1.4)

    def start():
        reset()
        _launcher.reset()
        print("Gate centering (single-tag solvePnP capable)")

    def _update():
        if not _launcher.done:
            _launcher.update(_drone)
            return
        if update(_drone):
            _drone.flight.land()

    _drone.set_start_update(start, _update)
    _drone.go(not neo_lab._is_sim(_drone))