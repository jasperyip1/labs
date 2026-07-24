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
MAX_ROLL      = 0.3     # strafe authority for centering
FOLLOW_TIME   = 1000000.0     # seconds to follow before landing
IMAGE_CENTER  = 320      # 640-wide image -> center column
# add near your other constants
PITCH_STRAIGHT = 0.5    # fast on straights
PITCH_TURN     = 0.2    # slow through turns
CURVE_SCALE    = 50    # residual std at which you're "fully" in a turn (TUNE)

# -- Module-level state -----------------------------------------------------
_timer = 0.0
_done  = False

def reset():
    global _timer, _done
    _timer = 0.0
    _done  = False


def update(drone):
    global _timer, _done
    if _done:
        return True
    ##################################
    #### START PUT CODE HERE #########

    camera = drone.camera.get_downward_image()
    mask = neo_lab.bright_mask(camera, V_MIN)
    edges = np.argwhere(mask)

    edges = edges.astype(np.float64)

    

    if (np.count_nonzero(edges) < MIN_PIXELS):  
        drone.flight.stop() 
    else:
        ys = edges[:, 0]
        xs = edges[:, 1]
        m, b = np.polyfit(ys, xs, 1)
        yaw = uav_utils.clamp(-m, -1, 1)

        curviness = np.std(xs - (m * ys + b))
        straightness = uav_utils.clamp(1.0 - curviness / CURVE_SCALE, 0.0, 1.0)
        pitch = PITCH_TURN + (PITCH_STRAIGHT - PITCH_TURN) * straightness

        edge_col = edges[:, 1].mean()      # average column of the bright edge
        offset = (edge_col - IMAGE_CENTER) / IMAGE_CENTER   # -1 (left) .. +1 (right)
        roll = uav_utils.clamp(offset * MAX_ROLL, -MAX_ROLL, MAX_ROLL)

       # throttle = uav_utils.clamp(2 - neo_lab.height(drone),-1,1)
        throttle = 0

        drone.flight.send_pcmd(pitch,roll,yaw,throttle)
        print("pitch: ",pitch,"roll: ",roll,"yaw: ",yaw)

    _timer += drone.get_delta_time()         
    if _timer >= FOLLOW_TIME:  
            _done = True
    return _done 


    # GOAL: fly forward at FORWARD_PITCH while strafing (roll) to keep the bright
    # edge under the middle of the downward camera.
    #
    # Tools: drone.camera.get_downward_image(); neo_lab.bright_mask(image, V_MIN);
    #        np.argwhere(mask) -> bright pixel (row, col); uav_utils.clamp(...);
    #        drone.flight.send_pcmd(pitch, roll, yaw, throttle).
    #
    # The average column of the bright pixels tells you how far off-center the edge
    # is. Turn that pixel offset into a roll command (clamped to MAX_ROLL): an edge
    # right of center means roll right to chase it. If you see too few bright pixels,
    # hold position rather than steering on noise -- but keep the timer running every
    # frame and finish after FOLLOW_TIME regardless, so losing the edge never hangs.

    ###### END PUT CODE HERE #########
    ##################################
    return _done


if __name__ == "__main__":
    _drone = drone_core.create_drone()
    _launcher = neo_lab.Launcher(1.0)

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
