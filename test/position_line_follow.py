"""
MIT BWSI Autonomous Drone Racing Course - UAV Neo
GNU General Public License v3.0

Position-Based Line Follower with Periodic Debug Logging via update_slow()
"""

import drone_core
import drone_utils as uav_utils
import cv2
import numpy as np
import os as _os, sys as _sys

# -- Course setup --
_d = _os.path.dirname(_os.path.realpath(__file__))
while _os.path.basename(_d) != "labs" and _os.path.dirname(_d) != _d:
    _d = _os.path.dirname(_d)
if _d not in _sys.path:
    _sys.path.insert(0, _d)
import neo_lab

# -- Constants --------------------------------------------------------------
V_MIN             = 200
MIN_PIXELS        = 200
FOLLOW_TIME       = 1000000.0   # seconds to follow before landing
IMAGE_CENTER      = 320         # 640-wide image center column

# Lookahead & Speed tuning
LOOKAHEAD_ROW     = 100         # Target row in image space (0=top, 480=bottom)
STEP_STRAIGHT     = 0.55        # Forward step size on straightaways (meters)
STEP_TURN         = 0.20        # Forward step size in tight curves (meters)
CURVE_SCALE       = 50.0        # Residual standard deviation threshold for turns

# Lateral control bounds
LATERAL_GAIN      = 0.35        # Lateral correction multiplier
MAX_LATERAL_ERR   = 0.60        # Maximum lateral target offset limit (meters)

# -- Module-level state -----------------------------------------------------
_timer = 0.0
_done  = False

# Shared telemetry state for update_slow() logging
_last_target_east  = 0.0
_last_target_up    = 0.0
_last_target_north = 0.0
_last_curr_east    = 0.0
_last_curr_up      = 0.0
_last_curr_north   = 0.0
_last_straightness = 0.0
_last_step_forward = 0.0
_last_offset       = 0.0
_has_line          = False


def reset():
    global _timer, _done
    global _last_target_east, _last_target_up, _last_target_north
    global _last_curr_east, _last_curr_up, _last_curr_north
    global _last_straightness, _last_step_forward, _last_offset, _has_line
    
    _timer = 0.0
    _done  = False
    _last_target_east  = 0.0
    _last_target_up    = 0.0
    _last_target_north = 0.0
    _last_curr_east    = 0.0
    _last_curr_up      = 0.0
    _last_curr_north   = 0.0
    _last_straightness = 0.0
    _last_step_forward = 0.0
    _last_offset       = 0.0
    _has_line          = False


def update(drone):
    global _timer, _done
    global _last_target_east, _last_target_up, _last_target_north
    global _last_curr_east, _last_curr_up, _last_curr_north
    global _last_straightness, _last_step_forward, _last_offset, _has_line

    if _done:
        return True

    # 1. Capture downward image and extract bright pixels
    camera = drone.camera.get_downward_image()
    mask = neo_lab.bright_mask(camera, V_MIN)
    edges = np.argwhere(mask).astype(np.float64)

    # 2. Get current drone world position
    curr_east, curr_up, curr_north = drone.physics.get_position()
    _, _, yaw_deg = drone.physics.get_attitude()
    yaw_rad = np.radians(yaw_deg)

    # Store current position for debug logging
    _last_curr_east, _last_curr_up, _last_curr_north = curr_east, curr_up, curr_north

    if np.count_nonzero(edges) < MIN_PIXELS:
        _has_line = False
        target_east, target_up, target_north = curr_east, curr_up, curr_north
        drone.flight.goto_position(target_east, target_up, target_north)
    else:
        _has_line = True
        ys = edges[:, 0]
        xs = edges[:, 1]

        # 3. Fit line x = m * y + b
        m, b = np.polyfit(ys, xs, 1)

        # 4. Compute curviness & straightness ratio
        curviness = np.std(xs - (m * ys + b))
        straightness = uav_utils.clamp(1.0 - curviness / CURVE_SCALE, 0.0, 1.0)
        step_forward = STEP_TURN + (STEP_STRAIGHT - STEP_TURN) * straightness

        # 5. Lookahead offset
        x_lookahead = m * LOOKAHEAD_ROW + b
        offset = (x_lookahead - IMAGE_CENTER) / IMAGE_CENTER

        # 6. Body-frame displacements
        body_forward = step_forward
        body_right = uav_utils.clamp(offset * LATERAL_GAIN, -MAX_LATERAL_ERR, MAX_LATERAL_ERR)
        body_right += uav_utils.clamp(-m * 0.15, -0.25, 0.25)

        # 7. World-frame position calculation
        delta_east  = body_forward * np.sin(yaw_rad) + body_right * np.cos(yaw_rad)
        delta_north = body_forward * np.cos(yaw_rad) - body_right * np.sin(yaw_rad)

        target_east  = curr_east + delta_east
        target_north = curr_north + delta_north
        target_up    = curr_up

        # 8. Send target position command
        drone.flight.goto_position(target_east, target_up, target_north)

        # Save values for update_slow()
        _last_straightness = straightness
        _last_step_forward = step_forward
        _last_offset       = offset

    # Store target position commands
    _last_target_east  = target_east
    _last_target_up    = target_up
    _last_target_north = target_north

    _timer += drone.get_delta_time()
    if _timer >= FOLLOW_TIME:
        _done = True

    return _done


def update_slow(drone):
    """
    Called periodically (e.g. every 0.2 seconds) to print debug information
    without spamming the terminal every frame.
    """
    if not _has_line:
        print("[SLOW DEBUG] Line LOST -> Holding Position at East: {:.2f}, Up: {:.2f}, North: {:.2f}".format(
            _last_curr_east, _last_curr_up, _last_curr_north
        ))
    else:
        print("[SLOW DEBUG]")
        print("  |-- Current Pose : East={:.2f}m, Up={:.2f}m, North={:.2f}m".format(
            _last_curr_east, _last_curr_up, _last_curr_north
        ))
        print("  |-- SENT COMMANDS: Target East={:.2f}m, Target Up={:.2f}m, Target North={:.2f}m".format(
            _last_target_east, _last_target_up, _last_target_north
        ))
        print("  \\-- Line Metrics : Offset={:.2f} | Straightness={:.2f} | Step={:.2f}m".format(
            _last_offset, _last_straightness, _last_step_forward
        ))


if __name__ == "__main__":
    _drone = drone_core.create_drone()
    _launcher = neo_lab.Launcher(0.85)

    # Set update_slow interval to 0.2s (5 Hz) for clean terminal debugs
    _drone.set_update_slow_time(0.2)

    def start():
        _launcher.reset()
        reset()
        print("Step 3: Advanced Position Line Follower with update_slow Debugging")

    def _update():
        if not _launcher.done:        # Arm and reach flight altitude
            _launcher.update(_drone)
            return
        if update(_drone):
            _drone.flight.land()

    def _update_slow():
        if _launcher.done:            # Only print debug info once airborne
            update_slow(_drone)

    # Register start, frame update, and slow periodic update callbacks
    _drone.set_start_update(start, _update, _update_slow)
    _drone.go(not neo_lab._is_sim(_drone))