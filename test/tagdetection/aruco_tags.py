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
PITCH = 0.4
SEARCH_YAW = 0.1
SEARCH_TIMEOUT = 15.0       # give up if no gate decodes in this many seconds
PHASES = ['detect_tag', 'detect_gate', 'p_center', 'forward']
SET_PHASE = 1

CENTER_KP_X    = 1.5     # roll gain: normalized pixel error -> roll command
ALT_KP         = 1.5     # altitude gain: normalized vertical pixel error -> throttle command
CENTER_TOL     = 0.05    # normalized horizontal error under this counts as "centered"
ALT_TOL        = 0.05    # normalized vertical error under this counts as "height matched"
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
_pass_gate = 0.0

def reset():
    global _timer, _done, _frame, _phase, _gate, _hold, _pass_gate
    _timer = 0.0
    _done  = False
    _frame = 0
    _phase = SET_PHASE
    _gate = None
    _hold = 0.0
    _pass_gate = 0.0

def update(drone):
    global _timer, _done, _frame, _phase, _gate, _hold, _pass_gate
    if _done:
        return True
    dt = drone.get_delta_time()
    _timer += dt
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
                # drone.flight.send_pcmd(SEARCH_PITCH, 0, 0, 0)
                if _frame % 5 == 0:
                    print(f'No tags found!')

            else:
                tag_centers = np.array([c.reshape(-1, 2).mean(axis=0) for c in corners])
                depth_img = drone.camera.get_depth_image()

                if depth_img is None:
                    print("[Error]: Depth image is none!")
                else:
                    ch, cw = gray.shape[:2]          # color image height, width
                    dh, dw = depth_img.shape[:2]     # depth image height, width
                    sx, sy = dw / cw, dh / ch        # scale factors, color -> depth space

                    dists = []
                    for cx, cy in tag_centers:
                        dx = int(np.clip(cx * sx, 0, dw - 1))   # column -> clipped to width
                        dy = int(np.clip(cy * sy, 0, dh - 1))   # row    -> clipped to height
                        d = uav_utils.get_pixel_average_distance(depth_img, (dy, dx), kernel_size=5) / 100.0
                        dists.append(d)
                    dists = np.array(dists)

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

    if _phase == 2:  # P controllers: center on the gate (roll) and match its height (altitude)
        img = drone.camera.get_color_image()
        if img is None:
            print("[Error]: Image is none!")
        else:
            _gate = neo_lab.detect_gate(img)

            if _gate is None:
                drone.flight.stop()
                drone.flight.send_pcmd(0, 0, SEARCH_YAW, 0)
                _hold = 0.0
                if _frame % 5 == 0:
                    print(f'No gates found!')
            else:
                width, height = drone.camera.get_width(), drone.camera.get_height()
                img_cx, img_cy = width / 2.0, height / 2.0

                # --- Horizontal centering P controller (roll) ---
                # normalized error in [-1, 1]: +err_x = gate right of center
                err_x = (_gate.cx - img_cx) / (width / 2.0)
                roll = uav_utils.clamp(CENTER_KP_X * err_x, -ROLL_LIMIT, ROLL_LIMIT)

                # --- Altitude P controller (throttle) ---
                # normalized error in [-1, 1]: +err_alt = gate below center -> need to climb
                err_alt = (_gate.cy - img_cy) / (height / 2.0)
                throttle = uav_utils.clamp(-ALT_KP * err_alt, -THROTTLE_LIMIT, THROTTLE_LIMIT)

                drone.flight.send_pcmd(0, roll, 0, throttle)

                if _frame % 5 == 0:
                    print(f'Centering... err_x={err_x:.3f}, err_alt={err_alt:.3f}, hold={_hold:.2f}')

                if abs(err_x) < CENTER_TOL and abs(err_alt) < ALT_TOL:
                    _hold += dt
                    if _hold >= CENTER_HOLD_T:
                        drone.flight.stop()
                        print('Gate centered!')
                        _phase = 3
                else:
                    _hold = 0.0

    if _phase == 3:  # forward
        _pass_gate += dt
        drone.flight.send_pcmd(PITCH, 0, 0, 0)

        if _pass_gate > 3.0:
            _done = True

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