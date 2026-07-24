"""
MIT BWSI Autonomous Drone Racing Course - UAV Neo
GNU General Public License v3.0

Week 4 · Module 3 — Trajectory Tracking (SOLUTION) — Main orchestrator

Runs every step in sequence against the simulator:
    drone sim module3_trajectory/main_solution.py
Run a single step directly instead:
    drone sim solutions/<step_file>.py
"""

# -- Course setup: makes the shared `neo_lab` helper importable (don't edit). --
import os as _os, sys as _sys
_d = _os.path.dirname(_os.path.realpath(__file__))
while _os.path.basename(_d) != "labs" and _os.path.dirname(_d) != _d:
    _d = _os.path.dirname(_d)
if _d not in _sys.path:
    _sys.path.insert(0, _d)
import neo_lab

from solutions import (
    step1_timed_segment,
    step2_smooth_course,
    step3_orbit,
)

neo_lab.run_module("Week 4 · Module 3 — Trajectory Tracking (SOLUTION)", [
    ("Step 1: Track a Timed Segment", step1_timed_segment),
    ("Step 2: Fly a Smooth Course", step2_smooth_course),
    ("Step 3: Orbit a Point (Geometric Outer Loop)", step3_orbit),
])
