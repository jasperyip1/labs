"""
MIT BWSI Autonomous Drone Racing Course - UAV Neo
GNU General Public License v3.0

Week 3 · Module 2 — Proportional Control (Altitude Hold) (SOLUTION) — Main orchestrator

Runs every step in sequence against the simulator:
    drone sim module2_feedback_control/main_solution.py
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
    step1_altitude_hold,
    step2_altitude_steps,
)

neo_lab.run_module("Week 3 · Module 2 — Proportional Control (Altitude Hold) (SOLUTION)", [
    ("Step 1: Proportional Altitude Hold", step1_altitude_hold),
    ("Step 2: Altitude Setpoint Sequence", step2_altitude_steps),
])
