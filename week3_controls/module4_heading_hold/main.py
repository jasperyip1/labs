"""
MIT BWSI Autonomous Drone Racing Course - UAV Neo
GNU General Public License v3.0

Week 3 · Module 4 — Heading Hold — Main orchestrator

Runs every step in sequence against the simulator:
    drone sim module4_heading_hold/main.py
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
    step1_read_heading,
    step2_hold_heading,
)

neo_lab.run_module("Week 3 · Module 4 — Heading Hold", [
    ("Step 1: Read the Heading", step1_read_heading),
    ("Step 2: Hold a Heading", step2_hold_heading),
])
