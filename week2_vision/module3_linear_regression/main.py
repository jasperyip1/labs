"""
MIT BWSI Autonomous Drone Racing Course - UAV Neo
GNU General Public License v3.0

Week 2 · Module 3 — Linear Regression (Line Following) — Main orchestrator

Runs every step in sequence against the simulator:
    drone sim module3_linear_regression/main.py
Run a single step directly instead:
    drone sim tasks/<step_file>.py
"""

# -- Course setup: makes the shared `neo_lab` helper importable (don't edit). --
import os as _os, sys as _sys
_d = _os.path.dirname(_os.path.realpath(__file__))
while _os.path.basename(_d) != "labs" and _os.path.dirname(_d) != _d:
    _d = _os.path.dirname(_d)
if _d not in _sys.path:
    _sys.path.insert(0, _d)
import neo_lab

from tasks import (
    step1_detect_line,
    step2_fit_line,
    step3_follow_line,
)

neo_lab.run_module("Week 2 · Module 3 — Linear Regression (Line Following)", [
    ("Step 1: Detect the Line Pixels", step1_detect_line),
    ("Step 2: Fit a Line (Least Squares)", step2_fit_line),
    ("Step 3: Follow the Line", step3_follow_line),
])
