"""
MIT BWSI Autonomous Drone Racing Course - UAV Neo
GNU General Public License v3.0

Week 2/3 Lab — Step 1: Detect the Bright Edge Pixels
Find the glowing gate-edge pixels in the downward camera.
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
V_MIN      = 200
HOVER_TIME = 3.0

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
    drone.flight.stop()   # hover in place
    ##################################
    #### START PUT CODE HERE #########
    camera = drone.camera.get_downward_image()
    mask = neo_lab.bright_mask(camera, V_MIN)
    edges = np.argwhere(mask)
    pix = np.count_nonzero(edges)

    _timer += drone.get_delta_time()         
    if _timer >= HOVER_TIME:  
            _done = True
            print("num pixels: ",pix)
    return _done 
    # Gate edges glow bright, so threshold by brightness (HSV Value): neo_lab.bright_mask(
    # image, V_MIN) gives a mask of the bright pixels. Count them, and after HOVER_TIME
    # print the count and set _done. See the README (Key terms).

    ###### END PUT CODE HERE #########
    ##################################
    return _done


if __name__ == "__main__":
    _drone = drone_core.create_drone()
    _launcher = neo_lab.Launcher(3.0)

    def start():
        _launcher.reset()
        reset()
        print("Step 1: Detect the Bright Edge Pixels")

    def _update():
        if not _launcher.done:        # arm + climb to a safe height first
            _launcher.update(_drone)
            return
        if update(_drone):
            _drone.flight.land()

    _drone.set_start_update(start, _update)
    _drone.go(not neo_lab._is_sim(_drone))
