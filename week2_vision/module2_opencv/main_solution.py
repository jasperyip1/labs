"""
MIT BWSI Autonomous Drone Racing Course - UAV Neo
GNU General Public License v3.0

Week 2 · Module 2 — OpenCV (Thresholding & Morphology) (SOLUTION) — Main orchestrator

Runs every step in sequence against the simulator:
    drone sim module2_opencv/main_solution.py
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
    step1_threshold,
    step2_morphology,
    step3_blur_edges,
)

neo_lab.run_module("Week 2 · Module 2 — OpenCV (Thresholding & Morphology) (SOLUTION)", [
    ("Step 1: Grayscale Thresholding", step1_threshold),
    ("Step 2: Morphology (Opening)", step2_morphology),
    ("Step 3: Blur & Edge Detection", step3_blur_edges),
])
