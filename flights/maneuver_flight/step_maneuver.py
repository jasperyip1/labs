"""
MIT BWSI Autonomous Drone Racing Course - UAV Neo
GNU General Public License v3.0

Flight B - Step: Body-velocity maneuver.

Runs a timed sequence of body-frame velocity commands through neo_lab.send_velocity,
which is portable: on the real drone it maps straight to the velocity setpoint, and in
the simulator an inner loop turns the velocity into tilt. No position feedback, so the
maneuver is defined purely by speeds and durations.
"""

# -- Course setup: makes the shared `neo_lab` helper importable.
#    You don't need to read or change this block. --
import os as _os, sys as _sys
_d = _os.path.dirname(_os.path.realpath(__file__))
while _os.path.basename(_d) != "labs" and _os.path.dirname(_d) != _d:
    _d = _os.path.dirname(_d)
if _d not in _sys.path:
    _sys.path.insert(0, _d)
import neo_lab

# -- Maneuver definition ----------------------------------------------------
# Each segment is (v_right, v_up, v_forward, yaw_rate, seconds). Speeds are m/s;
# yaw_rate is a normalized [-1, 1] command. Forward, then a banking arc (forward
# while yawing), then a sideways slide, then a brief stop.
FORWARD_SPEED = 0.4
ARC_YAW = 0.3
STRAFE_SPEED = 0.4
SEGMENTS = [
    (0.0, 0.0, FORWARD_SPEED, 0.0, 2.0),
    (0.0, 0.0, FORWARD_SPEED, ARC_YAW, 3.0),
    (STRAFE_SPEED, 0.0, 0.0, 0.0, 2.0),
    (0.0, 0.0, 0.0, 0.0, 0.5),
]
_TOTAL_S = sum(seg[4] for seg in SEGMENTS)

# -- Module-level state -----------------------------------------------------
_clock = 0.0
_done = False


def reset():
    global _clock, _done
    _clock = 0.0
    _done = False


def _segment_at(t):
    elapsed = 0.0
    for v_right, v_up, v_forward, yaw_rate, seconds in SEGMENTS:
        elapsed += seconds
        if t < elapsed:
            return v_right, v_up, v_forward, yaw_rate
    return 0.0, 0.0, 0.0, 0.0


def update(drone):
    global _clock, _done
    if _done:
        return True

    _clock += drone.get_delta_time()
    if _clock >= _TOTAL_S:
        drone.flight.stop()
        print("[maneuver] complete")
        _done = True
        return True

    v_right, v_up, v_forward, yaw_rate = _segment_at(_clock)
    neo_lab.send_velocity(drone, v_right, v_up, v_forward, yaw_rate)
    return False
