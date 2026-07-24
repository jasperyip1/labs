"""
MIT BWSI Autonomous Drone Racing Course - UAV Neo
GNU General Public License v3.0

Week 3 · Module 3 — PID Control — Main orchestrator

Runs every step in sequence against the simulator:
    drone sim module3_pid/main.py
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
    step1_pid_altitude,
    step2_position_hold,
    step3_visual_servo,
    step4_track_reference,
)

neo_lab.run_module("Week 3 · Module 3 — PID Control", [
    ("Step 1: PID Altitude Hold", step1_pid_altitude),
    ("Step 2: Fly a Distance (PID on Position)", step2_position_hold),
    ("Step 3: Visual Servoing (Vision + PID)", step3_visual_servo),
    ("Step 4: Track a Moving Reference (Feedforward)", step4_track_reference),
])
