"""
MIT BWSI Autonomous Drone Racing Course - UAV Neo
Gate-detection GROUND TEST -- NO TAKEOFF, NO PROPS, PRINT ONLY.

Purpose
  Walk the drone by hand through the course and watch, in real time, whether
  detect_gate() finds the ArUco tags and reconstructs a sane gate centre --
  before ever putting props on.

Why this can't drift out of sync with final_demo3.py
  This script does not reimplement detection. It imports final_demo3 as a
  module and calls its REAL detect_gate() function directly, using the SAME
  GATE_ARUCO_DICT / GATE_TAG_SIZE_M / GATE_TAG_ROLE_BY_ID / clustering /
  role-resolution code final_demo3 will fly with. If this script says a gate
  looks good, final_demo3 will see the same thing on the same frame.

Why this is print-only
  cv2.imshow() opens an X11/Qt window, and the companion computer here has no
  display server on :0 (headless jupyter_ws session) -- that plugin failure
  is what crashed the previous version ("qt.qpa.xcb: could not connect to
  display", "Aborted (core dumped)"). final_demo3.py never calls cv2.imshow,
  cv2.namedWindow, or cv2.waitKey anywhere (grep the file -- it isn't there),
  so that failure mode does not exist for the real flight script. It was
  only ever a problem in this ground-test script, and removing imshow here
  removes it entirely.

Safety
  This script NEVER calls anything under drone.flight (no arm, no launch, no
  land, no setpoints). It only reads drone.get_delta_time() and
  drone.camera.get_color_image(). Props should still be off per your own
  safety process -- this script does not command motors, but it also doesn't
  verify your hardware state for you.

Usage
    python gate_check_live.py

Before this
  Run `python final_demo3.py --selftest` first. That checks the perception
  and control LOGIC (role learning, clustering, centre-row math) against
  synthetic frames with no hardware at all. This script is the next step:
  same code, real camera, real gates, no motors, console output only.
"""

import os as _os
import sys as _sys
import time as _time

_sys.path.insert(0, _os.path.dirname(_os.path.realpath(__file__)))
import final_demo3 as fd3  # noqa: E402  (real detect_gate(), real constants)

_frame_n = 0
_t_last_print = 0.0
PRINT_PERIOD_S = 0.2   # console spam control; independent of detection rate


def start():
    print("=" * 64)
    print("GATE DETECTION GROUND TEST -- props OFF, no flight commands sent")
    print(f"  aruco dict     = {fd3.GATE_ARUCO_DICT}")
    print(f"  tag size       = {fd3.GATE_TAG_SIZE_M:.4f} m")
    print(f"  gate opening   = {fd3.GATE_INNER_HEIGHT_M:.3f} m "
          f"x {fd3.GATE_INNER_WIDTH_M:.3f} m")
    print(f"  hardcoded ids  = {fd3.GATE_TAG_ROLE_BY_ID}")
    print("Walk the drone through the course. Ctrl+C to stop.")
    print("=" * 64)


def _update():
    global _frame_n, _t_last_print

    # Only reads used by real flight are get_delta_time() and the colour
    # camera -- exactly what final_demo3.update() reads before doing anything
    # with a gate estimate. Nothing here ever touches drone.flight.
    _drone.get_delta_time()

    image = _drone.camera.get_color_image()
    if image is None:
        return False

    obs = fd3.detect_gate(image)
    _frame_n += 1
    now = _time.time()

    if now - _t_last_print > PRINT_PERIOD_S:
        if obs is not None and obs.conf >= fd3.GATE_CONF_MIN:
            err = obs.cy - obs.img_h * 0.5
            dist_str = f"{obs.dist_m:.2f}m" if obs.dist_m is not None else "n/a"
            print(f"[{_frame_n:6d}] ids={obs.ids}  roles={obs.roles}  "
                  f"cx={obs.cx:6.1f} cy={obs.cy:6.1f}  err={err:+6.1f}px  "
                  f"conf={obs.conf:.2f}  dist={dist_str}")
        elif obs is not None:
            print(f"[{_frame_n:6d}] tags seen {obs.ids} but conf "
                  f"{obs.conf:.2f} < GATE_CONF_MIN={fd3.GATE_CONF_MIN} "
                  f"(too little to trust yet)")
        else:
            print(f"[{_frame_n:6d}] no gate detected")
        _t_last_print = now

    return False   # never signal "done" -- there is nothing to land


if __name__ == "__main__":
    _drone = fd3.drone_core.create_drone()
    _drone.set_start_update(start, _update)
    try:
        _drone.go(not fd3.neo_lab._is_sim(_drone))
    except KeyboardInterrupt:
        pass
    finally:
        print("\nstopped. no flight command was ever sent.")
