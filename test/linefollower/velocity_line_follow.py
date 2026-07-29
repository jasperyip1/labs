"""
MIT BWSI Autonomous Drone Racing Course - UAV Neo
GNU General Public License v3.0

Week 2/3 Lab — Step 3: Follow the Edge (Velocity Controlled - High Speed)
Steer the drone using body-frame velocity setpoints to keep the bright 
edge centered while flying forward.
"""

import drone_core
import drone_utils as uav_utils
import cv2
import numpy as np

# -- Course setup: makes the shared `neo_lab` helper importable.
import os as _os, sys as _sys
_d = _os.path.dirname(_os.path.realpath(__file__))
while _os.path.basename(_d) != "labs" and _os.path.dirname(_d) != _d:
    _d = _os.path.dirname(_d)
if _d not in _sys.path:
    _sys.path.insert(0, _d)
import neo_lab

# -- Constants --------------------------------------------------------------
V_MIN              = 200
MIN_PIXELS         = 200
TARGET_HEIGHT      = 0.75        # meters above ground
FOLLOW_TIME        = 1000000.0   # seconds to follow before landing
IMAGE_CENTER       = 320         # 640-wide image -> center column

# Velocity limits and gains (Scaled up for full pitch/roll responsiveness)
V_FORWARD_STRAIGHT = 0.3         # Fast on straights (m/s)
V_FORWARD_TURN     = 0.15         # Slow through turns (m/s)
MAX_V_RIGHT        = 0.5         # Max strafe velocity (m/s)
CURVE_SCALE        = 50.0        # Residual std threshold for curvature
KP_YAW             = 2.0         # Stronger yaw rate response

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

    # 1. Maintain consistent altitude via vertical velocity (v_up)
    v_up = neo_lab.altitude_hold_velocity(drone, TARGET_HEIGHT)

    # 2. Capture downward camera image and mask glowing edge
    camera = drone.camera.get_downward_image()
    mask = neo_lab.bright_mask(camera, V_MIN)
    edges = np.argwhere(mask).astype(np.float64)

    # 3. Fail-safe: if the line drops out, stop horizontal motion but maintain altitude hold
    if np.count_nonzero(edges) < MIN_PIXELS:
        neo_lab.send_velocity(drone, 0.0, v_up, 0.0, 0.0)
    else:
        ys = edges[:, 0]
        xs = edges[:, 1]

        # Fit linear line through pixel coordinates
        m, b = np.polyfit(ys, xs, 1)

        # Yaw rate: align rotation with line slope
        yaw_rate = uav_utils.clamp(-m * KP_YAW, -1.0, 1.0)

        # Adaptive forward speed based on curve sharpness
        curviness = np.std(xs - (m * ys + b))
        straightness = uav_utils.clamp(1.0 - (curviness / CURVE_SCALE), 0.0, 1.0)
        v_forward = V_FORWARD_TURN + (V_FORWARD_STRAIGHT - V_FORWARD_TURN) * straightness

        # Strafe velocity (v_right): proportional offset to center line
        edge_col = xs.mean()
        offset = (edge_col - IMAGE_CENTER) / IMAGE_CENTER
        v_right = uav_utils.clamp(offset * MAX_V_RIGHT, -MAX_V_RIGHT, MAX_V_RIGHT)

        # Send full body-frame velocity setpoint: (v_right, v_up, v_forward, yaw_rate)
        neo_lab.send_velocity(drone, v_right, v_up, v_forward, yaw_rate)
        print(f"v_forward: {v_forward:.2f} | v_right: {v_right:.2f} | yaw_rate: {yaw_rate:.2f} | v_up: {v_up:.2f}")

    # Track time and handle termination condition
    _timer += drone.get_delta_time()
    if _timer >= FOLLOW_TIME:
        _done = True

    ###### END PUT CODE HERE #########
    ##################################
    return _done


if __name__ == "__main__":
    _drone = drone_core.create_drone()
    _launcher = neo_lab.Launcher(TARGET_HEIGHT)

    def start():
        _launcher.reset()
        reset()
        print("Step 3: Follow the Edge (Velocity Control - High Speed)")

    def _update():
        if not _launcher.done:        # arm + climb to target height first
            _launcher.update(_drone)
            return
        if update(_drone):
            _drone.flight.land()

    _drone.set_start_update(start, _update)
    _drone.go(not neo_lab._is_sim(_drone))