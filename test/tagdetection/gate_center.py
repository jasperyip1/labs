#!/usr/bin/env python3
"""
gate_center.py -- ArUco (5x5) gate centering + fly-through for the UAV Neo Final Challenge.

A 5x5 ArUco tag is mounted at each of the four points of the gate -- top, bottom, left,
and right (a DIAMOND layout, not corners). Given the gate geometry (which ID is at which
point, the gate size, and the tag size) we can build a 3D model of the gate. Each
*visible* tag contributes 4 image<->3D corner correspondences,
which is enough for cv2.solvePnP(SOLVEPNP_IPPE) -- so ONE tag is sufficient to recover
the full gate pose. solvePnP returns tvec = gate CENTER in camera coordinates, so the
drone can center on the true gate center even when it only sees a single corner tag.

State machine:
  ALIGN  -- P-control roll (lateral), throttle (vertical), yaw (heading) to drive the
            gate center onto the optical axis. Pitch held at 0.
  COMMIT -- once centered for N consecutive frames, dead-reckon forward by integrating
            get_linear_velocity() until we've traveled the last-seen distance + margin.
  DONE   -- stop.

Camera convention (OpenCV): +X right, +Y down, +Z forward (into scene).
Flight convention (send_pcmd): pitch>0 fwd, roll>0 ?, yaw normalized (*MAX_YAW_RATE), throttle raw.
Signs marked [SIGN?] must be confirmed on the field -- flip the sign if it drives the wrong way.
"""

import time
import numpy as np
import cv2

# ---- your course framework ----
import drone_core            # noqa: F401  (adjust imports to match your other scripts)
import neo_lab                # noqa: F401
import d435_intrinsics        # nominal D435 color-camera intrinsics (no pyrealsense2)

# =============================================================================
# CONFIG -- edit these to match the physical gate and your camera
# =============================================================================

# 5x5 ArUco dictionary the tags were printed from.
ARUCO_DICT = cv2.aruco.DICT_5X5_250

# Which marker ID sits at which point of the diamond, as seen by an upright drone
# approaching the front face. Change the IDs to your actual tags.
GATE_ID_TOP    = 0
GATE_ID_BOTTOM = 1
GATE_ID_LEFT   = 2
GATE_ID_RIGHT  = 3

# Gate geometry, in METERS.
#   GATE_W = horizontal distance between the LEFT and RIGHT tag centers (full width).
#   GATE_H = vertical distance between the TOP and BOTTOM tag centers (full height).
#   TAG_SIZE = printed marker side length.
GATE_W   = 1.20
GATE_H   = 1.20
TAG_SIZE = 0.15

# In-plane rotation of each tag about its own center, in DEGREES. Leave at 0 if the
# tags are mounted upright (readable) at the diamond points. Set to 45 only if the
# tags themselves are physically rotated into diamonds.
TAG_INPLANE_ROT_DEG = 0.0

# Forward-camera (D435 COLOR stream) intrinsics. Derived from the D435's published spec
# FOV + the library's own get_width()/get_height() -- see d435_intrinsics.py. No
# pyrealsense2 needed: drone.camera already gives us color + depth directly.
# CAMERA_MATRIX / DIST_COEFFS below are placeholders; run() overwrites them at startup
# via d435_intrinsics.get_intrinsics(drone.camera.get_width(), drone.camera.get_height()).
CAMERA_MATRIX = np.array([[600.0,   0.0, 320.0],
                          [  0.0, 600.0, 240.0],
                          [  0.0,   0.0,   1.0]], dtype=np.float64)
DIST_COEFFS   = np.zeros((5, 1), dtype=np.float64)   # D435 color stream is pre-rectified

# ---- depth-based ranging (D435, via drone.camera.get_depth_image()) ----
# The fly-through distance comes from the depth image sampled at the TAG pixels (real
# surfaces on the gate plane), not the gate center (an opening -> invalid/background).
# This gives an absolute gate-plane distance independent of solvePnP scale. Falls back
# to solvePnP tz if no valid depth is available.
# NOTE: get_depth_image() returns cm (per the library docs); everything below is in
# METERS, so get_frame() converts once at the source.
USE_DEPTH_RANGE = True
DEPTH_PATCH     = 5      # px window (odd) median-sampled at each tag pixel
DEPTH_MIN_M     = 0.20   # valid depth clamp (m); D435 min range ~0.2 m
DEPTH_MAX_M     = None   # set at runtime from drone.camera.get_max_range() (cm -> m)



# ---- control gains / limits (normalized command units, like your line follower) ----
KP_ROLL   = 0.60     # roll per meter of lateral error
KP_THR    = 0.40     # throttle per meter of vertical error
KP_YAW    = 0.80     # normalized yaw per radian of heading error
THR_HOVER = 0.0      # hover throttle offset for send_pcmd (match your altitude scripts)

MAX_ROLL  = 0.25
MAX_YAW   = 1.0      # normalized; actual rad/s = MAX_YAW * MAX_YAW_RATE in flight_real
MAX_THR   = 0.40
PITCH_CRUISE = 0.15  # forward pitch during the fly-through (your line-follower cruise)

# ---- "centered" thresholds ----
TOL_XY   = 0.10      # meters, lateral + vertical
TOL_YAW  = 0.12      # radians (~7 deg)
CENTERED_FRAMES = 8  # consecutive good frames required before committing

# ---- fly-through ----
GATE_PASS_MARGIN = 0.60   # meters past the gate plane to guarantee clearance
VEL_FWD_INDEX    = 0      # [CONFIRM] index of forward axis in get_linear_velocity()
FLYTHROUGH_TIMEOUT = 6.0  # s, hard escape hatch so we never dead-reckon forever

SUMMARY_PERIOD = 0.5      # s between status prints (no per-frame spam)


# =============================================================================
# GATE MODEL  (3D tag centers in the gate frame; +X right, +Y down, Z=0 plane)
# Diamond layout: tags at the top, bottom, left, and right points. Center = origin.
# =============================================================================
ID_TO_CENTER = {
    GATE_ID_TOP:    (0.0,          -GATE_H / 2.0, 0.0),
    GATE_ID_BOTTOM: (0.0,          +GATE_H / 2.0, 0.0),
    GATE_ID_LEFT:   (-GATE_W / 2.0, 0.0,          0.0),
    GATE_ID_RIGHT:  (+GATE_W / 2.0, 0.0,          0.0),
}


def _tag_object_corners(cx, cy):
    """4 corners of a tag centered at (cx,cy) in the gate plane, in detectMarkers
    order: top-left, top-right, bottom-right, bottom-left. Honors TAG_INPLANE_ROT_DEG."""
    s = TAG_SIZE / 2.0
    base = [(-s, -s), (+s, -s), (+s, +s), (-s, +s)]   # TL, TR, BR, BL
    th = np.radians(TAG_INPLANE_ROT_DEG)
    c, sn = np.cos(th), np.sin(th)
    out = []
    for dx, dy in base:
        rx = c * dx - sn * dy
        ry = sn * dx + c * dy
        out.append((cx + rx, cy + ry, 0.0))
    return out


# =============================================================================
# PERCEPTION
# =============================================================================
_aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
_aruco_params = cv2.aruco.DetectorParameters()
_detector = cv2.aruco.ArucoDetector(_aruco_dict, _aruco_params)


def _sample_depth_m(depth_m, u, v):
    """Median valid depth (meters) in a small patch around pixel (u,v). None if invalid.
    depth_m must be aligned-to-color and already in METERS."""
    if depth_m is None:
        return None
    h, w = depth_m.shape[:2]
    u, v = int(round(u)), int(round(v))
    r = DEPTH_PATCH // 2
    u0, u1 = max(0, u - r), min(w, u + r + 1)
    v0, v1 = max(0, v - r), min(h, v + r + 1)
    if u0 >= u1 or v0 >= v1:
        return None
    win = depth_m[v0:v1, u0:u1]
    vals = win[(win > DEPTH_MIN_M) & (win < DEPTH_MAX_M)]
    if vals.size < 3:
        return None
    return float(np.median(vals))


def estimate_gate_pose(gray, depth_m=None):
    """Detect gate tags and solve for the gate pose.

    Returns (tx, ty, tz, yaw_err, n_tags, depth_range) or None if no usable tag seen.
      tx  gate center offset right of optical axis (m)
      ty  gate center offset below optical axis (m)   [camera Y is down]
      tz  gate center forward distance from solvePnP (m)
      yaw_err  heading misalignment of the gate plane (rad), 0 when square-on
      depth_range  gate-plane distance from the DEPTH stream at the tag pixels (m),
                   or None if depth unavailable/invalid. Prefer this over tz.
    """
    corners, ids, _ = _detector.detectMarkers(gray)
    if ids is None:
        return None

    obj_pts, img_pts, tag_pixels, n_tags = [], [], [], 0
    for marker_corners, mid in zip(corners, ids.flatten()):
        if int(mid) not in ID_TO_CENTER:
            continue
        cx, cy, _ = ID_TO_CENTER[int(mid)]
        obj_pts.extend(_tag_object_corners(cx, cy))
        c4 = marker_corners.reshape(-1, 2)
        img_pts.extend(c4)                     # TL,TR,BR,BL image order
        tag_pixels.append(c4.mean(axis=0))     # this tag's center pixel (a real surface)
        n_tags += 1

    if len(obj_pts) < 4:
        return None

    obj = np.array(obj_pts, dtype=np.float64)
    img = np.array(img_pts, dtype=np.float64)

    ok, rvec, tvec = cv2.solvePnP(
        obj, img, CAMERA_MATRIX, DIST_COEFFS, flags=cv2.SOLVEPNP_IPPE
    )
    if not ok:
        return None

    tx, ty, tz = tvec.flatten()

    # Heading error: gate's X axis (right edge) expressed in camera frame.
    # Square-on -> gate X aligns with camera X -> yaw_err ~ 0.
    R, _ = cv2.Rodrigues(rvec)
    yaw_err = float(np.arctan2(R[2, 0], R[0, 0]))

    # Depth range: sample the depth stream at each detected tag pixel (real gate-plane
    # surfaces) and take the median. Absolute, independent of solvePnP scale.
    depth_range = None
    if USE_DEPTH_RANGE and depth_m is not None:
        ds = [d for (u, v) in tag_pixels if (d := _sample_depth_m(depth_m, u, v)) is not None]
        if ds:
            depth_range = float(np.median(ds))

    return float(tx), float(ty), float(tz), yaw_err, n_tags, depth_range


# =============================================================================
# CONTROL
# =============================================================================
def _clamp(v, lim):
    return max(-lim, min(lim, v))


def alignment_command(tx, ty, yaw_err):
    """P-control mapping the gate-center offset to a send_pcmd command.
    Returns (pitch, roll, yaw, throttle). Pitch is 0 during alignment."""
    roll     = _clamp( KP_ROLL * tx,      MAX_ROLL)          # [SIGN?] gate right -> roll right
    throttle = THR_HOVER + _clamp(KP_THR * (-ty), MAX_THR)   # [SIGN?] gate above (ty<0) -> climb
    yaw      = _clamp( KP_YAW * yaw_err,  MAX_YAW)            # [SIGN?] rotate to face gate
    pitch    = 0.0
    return pitch, roll, yaw, throttle


def is_centered(tx, ty, yaw_err):
    return abs(tx) < TOL_XY and abs(ty) < TOL_XY and abs(yaw_err) < TOL_YAW


# =============================================================================
# MAIN
# =============================================================================
def get_frame(drone):
    """Grab a grayscale color frame + depth (meters) via drone.camera (uav-neo-library).

    drone.camera.get_color_image()  -> BGR forward image
    drone.camera.get_depth_image()  -> per-pixel distance in CM, same forward camera,
                                        already pixel-aligned with the color image
    No pyrealsense2 needed -- the library wraps the D435 directly.
    """
    color = drone.camera.get_color_image()
    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY) if color.ndim == 3 else color

    depth_cm = drone.camera.get_depth_image()
    depth_m = depth_cm.astype(np.float32) / 100.0 if depth_cm is not None else None

    return gray, depth_m


def init_intrinsics(drone):
    """Fetch stream dimensions from drone.camera and derive nominal D435 intrinsics
    (spec FOV + resolution -- see d435_intrinsics.py). Also sets DEPTH_MAX_M from
    the library's own get_max_range() (cm -> m) instead of a guessed constant."""
    global CAMERA_MATRIX, DIST_COEFFS, DEPTH_MAX_M
    w = drone.camera.get_width()
    h = drone.camera.get_height()
    CAMERA_MATRIX, DIST_COEFFS = d435_intrinsics.get_intrinsics(w, h)
    DEPTH_MAX_M = drone.camera.get_max_range() / 100.0
    print(f"[gate_center] intrinsics set for {w}x{h}, depth max range {DEPTH_MAX_M:.1f}m")




def run(drone):
    init_intrinsics(drone)   # derive CAMERA_MATRIX/DIST_COEFFS + DEPTH_MAX_M from drone.camera

    state = "ALIGN"
    centered_streak = 0
    last_range = None        # best available gate-plane distance (depth preferred)
    last_summary = 0.0

    # fly-through dead-reckoning accumulators
    dist_traveled = 0.0
    t_prev = None
    t_commit = None

    print("[gate_center] ALIGN: searching for gate tags...")

    while True:
        now = time.time()
        gray, depth_m = get_frame(drone)
        pose = estimate_gate_pose(gray, depth_m)

        if state == "ALIGN":
            if pose is None:
                # No tag this frame -- hold level, keep looking. (No search sweep; minimal.)
                drone.flight.send_pcmd(0.0, 0.0, 0.0, THR_HOVER)
                centered_streak = 0
            else:
                tx, ty, tz, yaw_err, n, depth_range = pose
                # Prefer the absolute depth measurement; fall back to solvePnP tz.
                last_range = depth_range if depth_range is not None else tz
                src = "depth" if depth_range is not None else "pnp"
                pitch, roll, yaw, throttle = alignment_command(tx, ty, yaw_err)
                drone.flight.send_pcmd(pitch, roll, yaw, throttle)

                centered_streak = centered_streak + 1 if is_centered(tx, ty, yaw_err) else 0
                if centered_streak >= CENTERED_FRAMES:
                    state = "COMMIT"
                    t_commit = now
                    t_prev = now
                    dist_traveled = 0.0
                    target = (last_range or 0.0) + GATE_PASS_MARGIN
                    print(f"[gate_center] CENTERED (tags={n}, range={last_range:.2f}m "
                          f"[{src}]). COMMIT: flying through {target:.2f}m.")

                if now - last_summary >= SUMMARY_PERIOD:
                    print(f"[ALIGN] tags={n} tx={tx:+.2f} ty={ty:+.2f} "
                          f"range={last_range:.2f}[{src}] tz={tz:.2f} "
                          f"yaw={np.degrees(yaw_err):+5.1f}deg streak={centered_streak}")
                    last_summary = now

        elif state == "COMMIT":
            # Dead-reckon forward through the gate: pitch and integrate forward velocity
            # until we've covered the gate-plane distance + margin. Distance seed comes
            # from the depth stream (absolute) when available, else solvePnP.
            drone.flight.send_pcmd(PITCH_CRUISE, 0.0, 0.0, THR_HOVER)

            vel = drone.get_linear_velocity()
            v_fwd = float(vel[VEL_FWD_INDEX])
            dt = now - t_prev
            t_prev = now
            dist_traveled += max(0.0, v_fwd) * dt

            target = (last_range or 0.0) + GATE_PASS_MARGIN
            if now - last_summary >= SUMMARY_PERIOD:
                print(f"[COMMIT] traveled={dist_traveled:.2f}/{target:.2f}m v_fwd={v_fwd:+.2f}")
                last_summary = now

            if dist_traveled >= target or (now - t_commit) >= FLYTHROUGH_TIMEOUT:
                reason = "distance" if dist_traveled >= target else "timeout"
                print(f"[gate_center] DONE ({reason}). Stopping.")
                drone.flight.stop()
                state = "DONE"

        if state == "DONE":
            break

        time.sleep(0.01)   # let the loop breathe; tune to your camera frame rate


if __name__ == "__main__":
    # Bring up the drone the same way your other field scripts do, then run().
    #   drone = drone_core.Drone(...)      # <-- match your launcher / flight_real setup
    #   neo_lab.Launcher(...) ...
    #   run(drone)
    raise SystemExit(
        "Wire up the drone object (see your line-follower launch) and call run(drone)."
    )
