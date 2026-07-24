"""
MIT BWSI Autonomous Drone Racing Course - UAV Neo
GNU General Public License v3.0

Shape flight - Step: trace a repeating shape.

Same code in the sim and on the real drone. Two modes:
  open_loop   - trace a continuous pattern (circle, spiral, figure8, wave) by
                commanding body velocity with neo_lab.send_velocity. Dead-reckoned,
                so it drifts - which is the point of comparing it to closed_loop.
  closed_loop - fly a polygon of waypoints (square, triangle, circle) with
                drone.flight.goto_position. Each corner is held by the position
                controller, so the shape stays anchored.

Pick the shape and mode with the constants below. This is the portable counterpart
to uav_neo_ros2_driver/shape_node.py, which draws the same shapes by talking to
MAVROS directly (a ROS 2 learning example).
"""

# -- Course setup: makes the shared `neo_lab` helper and this folder importable. --
import os as _os, sys as _sys
_here = _os.path.dirname(_os.path.realpath(__file__))
if _here not in _sys.path:
    _sys.path.insert(0, _here)
_d = _here
while _os.path.basename(_d) != "labs" and _os.path.dirname(_d) != _d:
    _d = _os.path.dirname(_d)
if _d not in _sys.path:
    _sys.path.insert(0, _d)
import neo_lab

import shapes

# -- Pick the shape --------------------------------------------------------
MODE = 'closed_loop'        # 'open_loop' or 'closed_loop'
SHAPE = 'square'          # open_loop: circle/spiral/figure8/wave; closed_loop: square/triangle/circle
CYCLES = 2                # how many times to repeat the shape

# -- Tuning ----------------------------------------------------------------
SPEED = 0.5               # m/s applied to the normalized open-loop pattern
ARRIVE_TOL_M = 0.4        # closed_loop: within this distance a corner counts as reached
CORNER_SETTLE_S = 0.4     # closed_loop: hold inside the tolerance this long per corner
LED_COLOR = (0, 200, 255)  # RGB the strip lights while tracing (real drone only)

# -- Module-level state -----------------------------------------------------
_clock = 0.0
_start = None             # (east, up, north) captured at takeoff (closed_loop)
_corner = 0
_hold = 0.0
_done = False
_led_lit = False


def reset():
    global _clock, _start, _corner, _hold, _done, _led_lit
    _clock = 0.0
    _start = None
    _corner = 0
    _hold = 0.0
    _done = False
    _led_lit = False
    control = "velocity control" if MODE == 'open_loop' else "position control"
    print(f"[shape] tracing '{SHAPE}' via {MODE} ({control}), {CYCLES} cycle(s)")


def _update_open_loop(drone):
    global _clock, _done
    period = shapes.PATTERN_PERIOD_S[SHAPE]
    _clock += drone.get_delta_time()
    if _clock >= CYCLES * period:
        drone.flight.stop()
        drone.led.off()
        print(f"[shape] {SHAPE} open_loop complete ({CYCLES} cycles)")
        _done = True
        return True
    fwd, left, up, yaw = shapes.VELOCITY_PATTERNS[SHAPE](_clock)
    # Pattern is (forward, left, up, yaw); send_velocity wants (right, up, forward)
    # in m/s plus a yaw rate. left -> -right, and pattern yaw (CCW+) -> send yaw (CW+).
    neo_lab.send_velocity(drone, -left * SPEED, up * SPEED, fwd * SPEED, -yaw)
    return False


def _update_closed_loop(drone):
    global _start, _corner, _hold, _done
    east, up, north = drone.physics.get_position()
    if _start is None:
        _start = (east, up, north)

    offsets = shapes.WAYPOINT_SHAPES[SHAPE]
    total = CYCLES * len(offsets)
    if _corner >= total:
        drone.flight.stop()
        drone.led.off()
        print(f"[shape] {SHAPE} closed_loop complete ({CYCLES} cycles)")
        _done = True
        return True

    d_east, d_north, d_up = offsets[_corner % len(offsets)]
    target = (_start[0] + d_east, _start[1] + d_up, _start[2] + d_north)
    drone.flight.goto_position(target[0], target[1], target[2])

    dist = ((target[0] - east) ** 2 + (target[1] - up) ** 2
            + (target[2] - north) ** 2) ** 0.5
    _hold = _hold + drone.get_delta_time() if dist < ARRIVE_TOL_M else 0.0
    if _hold >= CORNER_SETTLE_S:
        _corner += 1
        _hold = 0.0
        if _corner < total:
            print(f"[shape] reached corner {_corner} of {total}")
    return False


def update(drone):
    global _led_lit
    if _done:
        return True
    if not _led_lit:
        drone.led.fill(*LED_COLOR)   # light the strip for the long exposure
        _led_lit = True
    if MODE == 'open_loop':
        return _update_open_loop(drone)
    return _update_closed_loop(drone)
