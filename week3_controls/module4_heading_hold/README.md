# Week 3 · Module 4 — Heading Hold

Control the drone's compass heading with the IMU instead of the camera. This is the
first controller in the course that closes a loop on an angle, which forces you to
handle wrap-around (359 deg and 1 deg are 2 deg apart, not 358).

## What you'll learn

- **Reading attitude** — getting pitch/roll/yaw from the IMU with `get_attitude()`.
- **Proportional yaw control** — turning to and holding a compass heading.
- **Angle wrap-around** — computing the *shortest* turn between two headings.

## How it works

So far your controllers acted on distances. Heading hold acts on an **angle**, and angles wrap
around — which quietly breaks the naive error you have used until now.

**Sense orientation, not position.** The drone's **attitude** (pitch, roll, yaw) comes from the
**IMU**, an inertial sensor independent of the camera. `get_attitude()` returns all three;
**yaw** is the compass **heading**, the direction the nose points, in degrees `[0, 360)`.

**The wrap-around trap.** To hold a heading you want the error `target − current`. But if the
target is 10° and you are at 350°, plain subtraction says the error is −340° and the drone
spins almost all the way around — when the real shortest turn is +20°. The fix is to wrap the
error into `−180..180`, which always picks the short way: `((target − current + 180) % 360) −
180`. Any time you subtract two angles, you have to do this.

**Then it's just proportional control.** With a correctly wrapped error, hold heading exactly
as you held altitude: command a yaw rate proportional to the error (gain `KP_YAW`, clamped so
it cannot spin too fast), and count the time spent within tolerance before declaring success.
If the drone turns the wrong way, the sign is flipped.

Why it matters: wrap-around shows up wherever a robot reasons about angles — heading, joint
limits, the bearing to a gate. Getting the "shortest angle" right once, in a reusable
`heading_error`, spares you a whole class of spin-the-wrong-way bugs.

## Key terms

- **Attitude** — the drone's orientation: pitch (nose up/down), roll (tilt left/right), yaw (compass heading).
- **IMU** — the inertial measurement unit, the sensor that reports attitude. It is independent of the camera.
- **Yaw / heading** — the direction the nose points, here in degrees `[0, 360)`.
- **Angle wrap-around** — angles repeat every 360 deg, so a raw subtraction can give a 350 deg "error" when the true shortest turn is 10 deg. Wrapping into `-180..180` picks the short way around.
- **Setpoint** — the heading you want to hold (`TARGET_HEADING`).

## How to run

```bash
drone open_sim                          # launch the sim once
drone sim course/week3_controls/module4_heading_hold/main.py            # all steps, your code
drone sim course/week3_controls/module4_heading_hold/main_solution.py   # reference flight
```

Press **Enter** in the simulator window to start.

## Steps

1. **`step1_read_heading.py`** — read and print the current yaw heading while turning slowly
2. **`step2_hold_heading.py`** — turn to a target heading and hold it (with wrap-around)

## What to expect

Step 1 turns the drone slowly and prints its heading. Step 2 rotates to
`TARGET_HEADING`, holds it within `TOL` for `HOLD_TIME`, then lands.

## You're done when

- Step 1 prints a yaw value in degrees.
- Step 2 rotates to about `TARGET_HEADING` and reports it is holding, without spinning the long way around.

## If it doesn't work

| Symptom | Fix |
|---------|-----|
| Spins the long way to the target | Your error isn't wrapped — use `((target - current + 180) % 360) - 180`. |
| Spins continuously, never settles | The yaw command sign is inverted; negate it. |
| Oscillates around the target | Lower `KP_YAW`, or add a small derivative term on the heading rate. |
| Never reports holding | `_hold` must accumulate only while `abs(error) < TOL` and reset otherwise. |

## Going further (optional)

- Make `heading_error` a reusable helper and use it anywhere you compare angles.
- Add a derivative term (rate of heading change) to kill the oscillation, turning this into PD heading hold.
- Combine with Module 3's forward controller: hold a heading *while* flying forward a set distance, so the drone tracks a straight line over the ground.

---

Fill in the blanks in `tasks/`; completed references are in `solutions/` (try it yourself first!).
