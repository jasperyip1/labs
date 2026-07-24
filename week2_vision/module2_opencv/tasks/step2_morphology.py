"""
MIT BWSI Autonomous Drone Racing Course - UAV Neo
GNU General Public License v3.0

Week 2/3 Lab — Step 2: Morphology (Opening)
Clean a binary mask with erosion followed by dilation (an 'opening').
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
THRESHOLD_VALUE = 80
KERNEL_SIZE     = 5
HOVER_TIME      = 3.0

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

    # Opening (erode then dilate) removes small speckles but keeps big shapes. Build a
    # binary mask like Step 1, then open it with a KERNEL_SIZE square kernel and compare
    # the white-pixel count before and after to see what was removed. Advance _timer and
    # finish once it reaches HOVER_TIME. See the README (Key terms) for morphology.

    _timer += drone.get_delta_time()
    image = drone.camera.get_downward_image()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, THRESHOLD_VALUE, 255, cv2.THRESH_BINARY)
    original_white_pixel = np.count_nonzero(binary)

    if _timer >= HOVER_TIME:
        kernel = np.ones((KERNEL_SIZE, KERNEL_SIZE), np.uint8)
        eroded = cv2.erode(binary, kernel, iterations=1)
        opened = cv2.dilate(eroded, kernel, iterations=1)
        opened_white_pixel = np.count_nonzero(opened)
        print(f"[Step 2] Opening with {KERNEL_SIZE}x{KERNEL_SIZE} kernel: "
              f"Removed {original_white_pixel - opened_white_pixel} white pixels ({original_white_pixel} -> {opened_white_pixel})")
        _done = True

    ###### END PUT CODE HERE #########
    ##################################
    return _done


if __name__ == "__main__":
    _drone = drone_core.create_drone()
    _launcher = neo_lab.Launcher(3.0)

    def start():
        _launcher.reset()
        reset()
        print("Step 2: Morphology (Opening)")

    def _update():
        if not _launcher.done:        # arm + climb to a safe height first
            _launcher.update(_drone)
            return
        if update(_drone):
            _drone.flight.land()

    _drone.set_start_update(start, _update)
    _drone.go()