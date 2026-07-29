"""
MIT BWSI Autonomous Drone Racing Course - UAV Neo
GNU General Public License v3.0

Week 2 Lab — Step 1: Detect Gate Markers
Creep forward until the gate's ArUco corner tags decode, then report them.
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
SEARCH_PITCH   = 0.1        # creep forward; ArUco tags only resolve up close
SEARCH_TIMEOUT = 15.0       # give up if no gate decodes in this many seconds
PHASES = ['detect_tag', 'detect_gate', 'p_center', 'forward']
SET_PHASE = 0

CENTER_KP_X    = 1.5     # roll gain: normalized pixel error -> roll command
CENTER_KP_Y    = 1.5     # throttle gain: normalized pixel error -> throttle command
CENTER_TOL     = 0.05    # normalized error under this counts as "centered"
CENTER_HOLD_T  = 0.5     # seconds error must stay within tolerance before done
ROLL_LIMIT     = 0.3
THROTTLE_LIMIT = 0.3

# -- Module-level state -----------------------------------------------------
_timer = 0.0
_done  = False
_frame = 0
_phase = SET_PHASE
_gate = None
_hold = 0.0

def reset():
    global _timer, _done, _frame, _phase, _gate, _hold
    _timer = 0.0
    _done  = False
    _frame = 0
    _phase = SET_PHASE
    _gate = None
    _hold = 0.0

def update(drone):
    global _timer, _done, _frame, _phase, _gate, _hold
    if _done:
        return True
    _timer += drone.get_delta_time()
    _frame += 1
    ##################################
    #### START PUT CODE HERE #########

    if _phase == 0: # detect tags
        img = drone.camera.get_color_image()
        if img is None:
            print("[Error]: Image is none!")
        else:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = neo_lab._detect_gate_markers(gray)

            if ids is None or len(ids) == 0:
                drone.flight.send_pcmd(SEARCH_PITCH, 0, 0, 0)
                if _frame % 5 == 0:
                    print(f'No tags found!')

            else:
                tag_centers = np.array([c.reshape(-1, 2).mean(axis=0) for c in corners])
                depth_img = drone.camera.get_depth_image()
                dists = np.array([
                    uav_utils.get_pixel_average_distance(depth_img, (int(cx), int(cy)), kernel_size=5) / 100.0
                    for cx, cy in tag_centers
                ])  # cm -> meters

                print(f'{len(ids)} tags found! Distances: {dists}')

    if _phase == 1:  # detect gate
        img = drone.camera.get_color_image()
        if img is None:
            print("[Error]: Image is none!")
        else:
            _gate = neo_lab.detect_gate(img)

            if _gate is None:
                if _frame % 5 == 0:
                    print(f'No gates found!')

            else:
                print(f'Gate detected! cx, cy = {_gate.cx, _gate.cy}')

    if _phase == 2:  # P controller: center on the gate
        img = drone.camera.get_color_image()
        if img is None:
            print("[Error]: Image is none!")
        else:
            _gate = neo_lab.detect_gate(img)

            if _gate is None:
                drone.flight.stop()
                _hold = 0.0
                if _frame % 5 == 0:
                    print(f'No gates found!')
            else:
                width, height = drone.camera.get_width(), drone.camera.get_height()
                img_cx, img_cy = width / 2.0, height / 2.0

                # normalized error in [-1, 1]: +err_x = gate right of center, +err_y = gate below center
                err_x = (_gate.cx - img_cx) / (width / 2.0)
                err_y = (_gate.cy - img_cy) / (height / 2.0)

                roll     = uav_utils.clamp(CENTER_KP_X * err_x, -ROLL_LIMIT, ROLL_LIMIT)
                throttle = uav_utils.clamp(-CENTER_KP_Y * err_y, -THROTTLE_LIMIT, THROTTLE_LIMIT)

                drone.flight.send_pcmd(0, roll, 0, throttle)

                if _frame % 5 == 0:
                    print(f'Centering... err_x={err_x:.3f}, err_y={err_y:.3f}, hold={_hold:.2f}')

                if abs(err_x) < CENTER_TOL and abs(err_y) < CENTER_TOL:
                    _hold += drone.get_delta_time()
                    if _hold >= CENTER_HOLD_T:
                        drone.flight.stop()
                        print('Gate centered!')
                        _done = True
                else:
                    _hold = 0.0

    ###### END PUT CODE HERE #########
    ##################################
    return _done


if __name__ == "__main__":
    _drone = drone_core.create_drone()
    # _launcher = neo_lab.Launcher()

    def start():
        # _launcher.reset()
        # reset()
        print(f"===== ARUCO MARKER TEST CODE. Phase {PHASES[_phase]} =====")

    def _update():
        # if not _launcher.done:        # arm + climb to a safe height first
        #     _launcher.update(_drone)
        #     return
        if update(_drone):
            _drone.flight.land()
            return

    _drone.set_start_update(start, _update)
    _drone.go(not neo_lab._is_sim(_drone))