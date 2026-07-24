"""
MIT BWSI Autonomous Drone Racing Course - UAV Neo
GNU General Public License v3.0

Flight A - Step: Go To a Waypoint.

Flies to a point offset from where the climb finished, using drone.flight.goto_position.
That call is the same in the simulator and on the real drone: on real the flight
controller closes the position loop; in the simulator an internal controller drives
there with velocity commands. The target is built as an offset from a position
captured at flight time, because the real flight controller's world origin is wherever
its EKF initialized, not a fixed point.
"""

# -- Course setup: makes the shared `neo_lab` helper importable.
#    You don't need to read or change this block. --
import os as _os, sys as _sys
_d = _os.path.dirname(_os.path.realpath(__file__))
while _os.path.basename(_d) != "labs" and _os.path.dirname(_d) != _d:
    _d = _os.path.dirname(_d)
if _d not in _sys.path:
    _sys.path.insert(0, _d)

# -- Constants --------------------------------------------------------------
WAYPOINT_EAST_M = 1.0     # meters east of the takeoff point
WAYPOINT_NORTH_M = 1.0    # meters north of the takeoff point
ARRIVE_TOL_M = 0.5        # within this distance counts as arrived
SETTLE_TIME_S = 1.5       # must stay inside the tolerance this long

# -- Module-level state -----------------------------------------------------
_target = None            # (east, up, north) captured relative to takeoff
_hold = 0.0
_done = False


def reset():
    global _target, _hold, _done
    _target = None
    _hold = 0.0
    _done = False


def update(drone):
    global _target, _hold, _done
    if _done:
        return True

    east, up, north = drone.physics.get_position()

    # Capture the target on the first frame, relative to the current position, so
    # it works regardless of where the flight controller's origin sits.
    if _target is None:
        _target = (east + WAYPOINT_EAST_M, up, north + WAYPOINT_NORTH_M)
        print(f"[waypoint] target east={_target[0]:.2f} up={_target[1]:.2f} "
              f"north={_target[2]:.2f}")

    drone.flight.goto_position(_target[0], _target[1], _target[2])

    d_east = _target[0] - east
    d_up = _target[1] - up
    d_north = _target[2] - north
    distance = (d_east * d_east + d_up * d_up + d_north * d_north) ** 0.5

    _hold = _hold + drone.get_delta_time() if distance < ARRIVE_TOL_M else 0.0
    if _hold >= SETTLE_TIME_S:
        print(f"[waypoint] arrived (distance={distance:.2f} m)")
        _done = True
    return _done
