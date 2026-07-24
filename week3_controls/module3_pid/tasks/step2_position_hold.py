"""
MIT BWSI Autonomous Drone Racing Course - UAV Neo
GNU General Public License v3.0

Week 2/3 Lab — Step 2: Fly a Distance (PID on Position)
Integrate forward velocity into position and PID to a target distance,
while a proportional term keeps altitude.
"""

import drone_core
import drone_utils as uav_utils

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
TARGET_DIST = 15.0    # meters forward
TARGET_HEIGHT = 7.0  # hold launch height
KP = 0.18
KI = 0.05
KD = 0.4    # strong velocity damping to avoid overshoot
PITCH_LIMIT = 0.25
ALT_KP = 0.10
THROTTLE_LIMIT = 0.5
MIN_TRAVEL = 5.0   # fly at least this long before checking 'arrived'
SETTLE_SPEED = 0.25  # must slow below this to count as arrived
HOLD_TIME = 1.5

# -- Module-level state -----------------------------------------------------
_pos = 0.0
_err_int = 0.0
_prev_err = 0.0
_t = 0.0
_hold = 0.0
_done = False

def pid_control(err, err_int, err_dot, kp, ki, kd):
    """Return the PID controller output from the three gain terms (see README, Key terms)."""
    ##################################
    #### START PUT CODE HERE #########
    output = err * kp + err_int * ki + err_dot * kd
    ###### END PUT CODE HERE #########
    ##################################
    return output

def reset():
    global _pos, _err_int, _prev_err, _t, _hold, _done
    _pos = 0.0
    _err_int = 0.0
    _prev_err = 0.0
    _t = 0.0
    _hold = 0.0
    _done = False


def update(drone):
    global _pos, _err_int, _prev_err, _t, _hold, _done
    if _done:
        return True
    ##################################
    #### START PUT CODE HERE #########

    # There is no direct (x, z) readout, so estimate forward distance by dead reckoning:
    # integrate the forward component of drone.physics.get_linear_velocity() over time.
    # PID that distance to TARGET_DIST for the pitch command (clamped to PITCH_LIMIT), and
    # use a proportional term (ALT_KP) on height to hold TARGET_HEIGHT. Count as arrived
    # only after MIN_TRAVEL, once speed drops below SETTLE_SPEED for HOLD_TIME. See the
    # README (Key terms) for dead reckoning and the PID law.

    vel = drone.physics.get_linear_velocity()
    dt = drone.get_delta_time()
    _t += dt
    _pos += vel[2] * dt
    err = TARGET_DIST - _pos
    _err_int += err * dt
    err_int = uav_utils.clamp(_err_int, -PITCH_LIMIT, PITCH_LIMIT)
    err_dot = (err - _prev_err) / dt
    _prev_err = err
    pitch = pid_control(err, err_int, err_dot, KP, KI, KD)
    clamped_pitch = uav_utils.clamp(pitch, -PITCH_LIMIT, PITCH_LIMIT)
    height = drone.physics.get_altitude()
    alt_err = TARGET_HEIGHT - height
    throttle = uav_utils.clamp(alt_err * ALT_KP, -THROTTLE_LIMIT, THROTTLE_LIMIT)
    drone.flight.send_pcmd(clamped_pitch, 0.0, 0.0, throttle)
    if _t >= MIN_TRAVEL and abs(vel[2]) < SETTLE_SPEED and abs(err) < 0.5:
        _hold += dt
    if _hold >= HOLD_TIME:
        print(f"Arrived at destination after traveling {_pos} m")
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
        print("Step 2: Fly a Distance (PID on Position)")

    def _update():
        if not _launcher.done:        # arm + climb to a safe height first
            _launcher.update(_drone)
            return
        if update(_drone):
            _drone.flight.land()

    _drone.set_start_update(start, _update)
    _drone.go()