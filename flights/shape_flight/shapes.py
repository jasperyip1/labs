"""
MIT BWSI Autonomous Drone Racing Course - UAV Neo
GNU General Public License v3.0

Shape geometry for the shape flight (pure math, no ROS / no drone).

Two families, matching the two ways to fly a shape:
  VELOCITY_PATTERNS  - f(t_seconds) -> normalized (forward, left, up, yaw), for
                       smooth continuous curves flown with body velocity.
  WAYPOINT_SHAPES    - a list of (east, north, up) offsets in meters from the
                       takeoff point, for exact corners flown as positions.

Sizes are tuned to fit a ~3 x 6 m (10 x 20 ft) space; the 3 m width is the
binding dimension.
"""

import math

# Continuous velocity patterns: one full loop takes *_PERIOD_S seconds; the traced
# size scales with the speed the flight applies.
CIRCLE_PERIOD_S = 9.0
SPIRAL_PERIOD_S = 20.0
SPIRAL_TURNS = 4
FIGURE8_PERIOD_S = 8.0
FIGURE8_LEFT_SCALE = 0.8
WAVE_PERIOD_S = 5.0
WAVE_FORWARD_NORM = 0.25
WAVE_SIDE_NORM = 0.6

# Waypoint polygon sizing.
SHAPE_RADIUS_M = 0.75
CIRCLE_WAYPOINT_SIDES = 12


def _circle_pattern(t):
    """Constant-speed velocity whose direction rotates: a smooth circle."""
    w = 2.0 * math.pi / CIRCLE_PERIOD_S
    return (math.cos(w * t), math.sin(w * t), 0.0, 0.0)


def _spiral_pattern(t):
    """Rotating velocity whose magnitude ramps up then down: spiral out and in."""
    phase = (t % SPIRAL_PERIOD_S) / SPIRAL_PERIOD_S
    envelope = 1.0 - abs(2.0 * phase - 1.0)
    w = 2.0 * math.pi * SPIRAL_TURNS / SPIRAL_PERIOD_S
    return (envelope * math.cos(w * t), envelope * math.sin(w * t), 0.0, 0.0)


def _figure8_pattern(t):
    """Lissajous velocity (2:1 side-to-forward) that traces a figure eight."""
    w = 2.0 * math.pi / FIGURE8_PERIOD_S
    return (math.cos(w * t),
            FIGURE8_LEFT_SCALE * math.cos(2.0 * w * t), 0.0, 0.0)


def _wave_pattern(t):
    """Steady forward speed with a sinusoidal sideways weave: a snaking line."""
    w = 2.0 * math.pi / WAVE_PERIOD_S
    return (WAVE_FORWARD_NORM, WAVE_SIDE_NORM * math.sin(w * t), 0.0, 0.0)


def _regular_polygon_waypoints(sides, radius, altitude, start_angle=0.0):
    """Corner offsets (east, north, up) of a regular polygon around the takeoff
    point, traced at `altitude` meters above takeoff."""
    points = []
    for k in range(sides):
        angle = start_angle + k * (2.0 * math.pi / sides)
        points.append((radius * math.cos(angle), radius * math.sin(angle), altitude))
    return points


# name -> f(t_seconds) returning normalized (forward, left, up, yaw)
VELOCITY_PATTERNS = {
    'circle': _circle_pattern,
    'spiral': _spiral_pattern,
    'figure8': _figure8_pattern,
    'wave': _wave_pattern,
}

# One full loop of each pattern, in seconds (used to run a whole number of cycles).
PATTERN_PERIOD_S = {
    'circle': CIRCLE_PERIOD_S,
    'spiral': SPIRAL_PERIOD_S,
    'figure8': FIGURE8_PERIOD_S,
    'wave': WAVE_PERIOD_S,
}

# name -> list of (east, north, up) offsets in meters from the takeoff point.
# `up` is the height above the takeoff point (the flight adds it to the launch alt).
WAYPOINT_SHAPES = {
    'square': _regular_polygon_waypoints(4, SHAPE_RADIUS_M, 0.0,
                                         start_angle=math.pi / 4.0),
    'triangle': _regular_polygon_waypoints(3, SHAPE_RADIUS_M, 0.0,
                                           start_angle=math.pi / 2.0),
    'circle': _regular_polygon_waypoints(CIRCLE_WAYPOINT_SIDES, SHAPE_RADIUS_M, 0.0),
}
