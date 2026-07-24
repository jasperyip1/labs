# Week 3 · Module 2 — Proportional Control (Altitude Hold)

Your first closed-loop controller: proportional control. Warm up on a 1-D math model, then hold the real drone at a target height.

## What you'll learn

- **The proportional law** — a command proportional to the error, `command = Kp · error`.
- **Steady-state error** — why a P-only controller settles a little short of the target.
- **Throttle from altitude error** — closing the loop on the real drone's height.

## How it works

This is your first **closed-loop controller**: instead of commanding a throttle and hoping,
you measure the height, compare it to the target, and correct — every frame, forever.

**Error drives everything.** The **error** is `target − measured` (here `TARGET_HEIGHT −
current_height`). It is positive when you are too low, negative when too high, zero on the
**setpoint**. A controller is just a rule for turning that error into a command.

**Proportional control.** The simplest rule: make the command proportional to the error,
`command = Kp · error`. Far below the target → a big throttle up; almost there → a gentle one;
on target → nothing. The **gain** `Kp` is the one knob: too small and the drone crawls to the
target, too large and it overshoots and oscillates. Because throttle here sets vertical
*speed*, `Kp` has to stay small.

**The catch: steady-state error.** As the drone nears the target the error shrinks, so the
command shrinks — until it is too weak to close the final gap against gravity (and the
throttle deadband). The drone parks slightly low and stays there. That permanent droop is
**steady-state error** — not a bug in your code but the fundamental limitation of P-only
control. Module 3's integral term is the fix.

**Step response.** Step 2 chases a sequence of heights. Watching how the drone reacts when the
target suddenly jumps — the **step response** — is how control engineers judge a controller:
how fast it rises, whether it overshoots, how long it takes to settle.

Why it matters: proportional control is the atom of feedback. Every loop you build after this —
PID altitude, heading hold, waypoints — starts from `Kp · error` and adds to it.

## Key terms

- **Closed-loop control** — measure the actual state, compare it to the target, and correct. (Open-loop would just command and hope.)
- **Error** — `target − measured`. Here, `target_height − current_height`.
- **Setpoint** — the value you want to reach (the target height).
- **Proportional control (P)** — command proportional to the error: `command = Kp · error`. Far from target → big push; close → gentle.
- **Gain (Kp)** — the tuning knob multiplying the error. Too small is sluggish; too large overshoots or oscillates.
- **Steady-state error** — the small leftover gap a P-only controller settles into: as error shrinks, the command shrinks, until it's too weak to close the last bit (made worse here by the throttle deadband). The integral term in Module 3 fixes this.
- **Step response** — how the system reacts when the target suddenly jumps to a new value.

## How to run

```bash
python3 tasks/p_control.py        # your work (prints PASS/FAIL)
python3 solutions/p_control.py    # reference
```

Then on the simulator:
```bash
drone open_sim                          # launch the sim once
drone sim course/week3_controls/module2_feedback_control/main.py            # all steps, your code
drone sim course/week3_controls/module2_feedback_control/main_solution.py   # reference flight
```

Press **Enter** in the simulator window to start.

## Steps

1. **`step1_altitude_hold.py`** — proportional throttle to hold a target height above ground
2. **`step2_altitude_steps.py`** — chase a sequence of target heights (a step response)

## What to expect

`p_control.py` self-checks with `python3`. The sim steps arm, climb, and hold each target height (settling a little short — that's the P-control lesson).

## You're done when

- `p_control.py` prints all `[PASS]`.
- Step 1: the drone climbs to roughly `TARGET_HEIGHT` (within `TOL`), holds for `HOLD_TIME`, and lands. Settling slightly below the target is expected and correct.
- Step 2: the drone visibly steps through each height in `SETPOINTS` in order, holding each before moving on.

## If it doesn't work

| Symptom | Fix |
|---------|-----|
| Drone shoots up and oscillates | `KP` too high, or you forgot to clamp the throttle to `THROTTLE_LIMIT`. |
| Drone barely moves | `KP` too low, or the error is below the throttle deadband (~0.05). |
| Step never completes | You aren't accumulating `_hold` while within `TOL`, or never reset it when you leave `TOL`. |
| Step 2 skips heights | Reset `_hold = 0.0` when you advance `_index`. |

## Going further (optional)

- Measure the steady-state error: print `TARGET_HEIGHT − height` once settled. How does it change if you double `KP`?
- Add a feed-forward term: a constant throttle that roughly cancels gravity's pull, so P only handles the remainder. Does the steady-state error shrink?
- Plot (or log) height vs. time for a step and identify the rise time and any overshoot.

---

Fill in the blanks in `tasks/`; completed references are in `solutions/` (try it yourself first!).
