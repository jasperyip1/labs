# Week 4 · Module 3 — Trajectory Tracking

Fly a *path*, not a pin. In Module 1 you flew to a waypoint and stopped; in Module 2 you
chained waypoints into a square, braking at every corner. A race drone never stops — it carries
a smooth line through the whole course. This module builds that line as a **trajectory** (a
position for every instant in time) and a controller that rides it by **commanding velocity**.

This is the drone-racing lab from MIT's VNAV course, rebuilt for the UAV Neo sim: generate a
smooth trajectory through a set of waypoints, then track it. The heavy trajectory-optimization
math of the original is replaced with a cubic **spline** you can build by hand.

## What you'll learn

- **Trajectory vs waypoint** — why "be here by time *t*" is a stronger command than "reach here
  eventually," and what that buys a racing drone.
- **Commanding velocity** — driving the drone with a velocity setpoint (the way the real drone's
  flight controller is driven), instead of a raw tilt.
- **Feedforward + position correction** — following a *moving* target by commanding its velocity
  and adding a nudge to close any position gap.
- **Time-scaling (smoothstep)** — starting and stopping a segment at rest so there is no jerk.
- **Cubic splines (Hermite)** — stitching waypoints into one continuous curve with matched slopes,
  so the path never kinks.
- **The racing line** — why flying *through* waypoints without stopping is both smoother and faster.
- **The geometric controller** — why holding a *curved* path needs constant inward acceleration,
  and where that acceleration comes from when all you command is velocity.

## How it works

**A waypoint is a place; a trajectory is a place *for every instant*.** Write it as `p(t)`: at
`t = 0` you are at the start, at `t = DURATION` you are at the goal, and in between `p(t)` names
exactly where you should be right now. Because `p(t)` is a function of time you can also read off
its **velocity** `p'(t)` — how fast the target itself is moving. Those two numbers are everything
the controller needs.

**Commanding velocity (Step 1).** Instead of computing a tilt, you compute a **velocity** each
frame and hand it to `neo_lab.send_velocity(drone, v_right, v_up, v_forward)`. For each axis:

```
v_command = desired velocity  +  KP_POS · (desired position − your position)
```

The first term is **feedforward**: the target is already moving at `desired velocity`, so command
that speed directly instead of waiting for an error to build. The second is a **position
correction** — a gentle pull that closes any gap between where you are and where the trajectory
says you should be. `KP_POS` (units of 1/s) sets how hard that pull is. There is *no* damping or
derivative term here: unlike a tilt command (which sets acceleration and needs damping), a velocity
command is first-order, and the flight controller underneath handles the damping.

Why velocity? Because that is how the **real drone** is flown. The ROS2 driver takes a velocity
setpoint (`/mux/cmd_vel`); the autopilot turns it into motor thrusts. `send_velocity` maps your
command straight to that setpoint on hardware, and on the sim it runs a small inner loop that turns
the velocity error into tilt — so the exact same controller code flies both.

**Smoothstep: start and stop at rest (Step 1).** If `p(t)` moved at constant speed, the target
would jump from standstill to full speed instantly — a jerk. Instead the segment uses a
**time-scaling** `s(u)`, with `u = t/DURATION` going 0→1, shaped so its *slope* is zero at both
ends: `s(u) = 3u² − 2u³`. Position is `start + s·(goal − start)`, so the drone eases out of the
start and eases into the goal. This one cubic is the whole trajectory for a single segment.

**Splines: one smooth curve through many waypoints (Step 2).** A course is a list of waypoints.
Draw a straight line between each pair and the path kinks at every one — the drone would have to
stop and turn. A **cubic Hermite spline** fixes this: on each segment (fraction `s` from 0 to 1
between `p0` and `p1`) the position blends the two endpoints *and* a chosen slope (tangent) at each
end:

```
p(s) = h00(s)·p0 + h10(s)·m0 + h01(s)·p1 + h11(s)·m1

h00 = 2s³ − 3s² + 1     (weight on the start point)
h10 = s³ − 2s² + s      (weight on the start tangent m0)
h01 = −2s³ + 3s²        (weight on the end point)
h11 = s³ − s²           (weight on the end tangent m1)
```

At `s=0` this is exactly `p0`; at `s=1` exactly `p1`; in between it curves according to the
tangents. Pick the tangent at each waypoint to point from the *previous* one to the *next*
(a **Catmull-Rom** tangent, `m_i = ½(p_{i+1} − p_{i−1})`), and neighbouring segments leave and
enter each waypoint at the same slope — so the full path is smooth, with no kinks. You implement
the `hermite` blend; the tangents, the velocity (from the path), and the bookkeeping are provided.
The tracking controller is the same velocity law as Step 1.

**Why not stop at each waypoint?** Stopping means decelerating to zero and re-accelerating — slow,
and it wastes the drone's momentum. Carrying a smooth line through keeps speed up and control gentle.
That trade — hold the line, don't stop — is the core idea of drone racing.

**Orbiting a point: the geometric controller (Step 3).** A circle is the same velocity law with a
twist: the feedforward velocity **rotates**. On a circle of radius `R` at speed `v`, the velocity
vector is always tangent to the circle and keeps turning, so each frame you command a slightly
different direction. You write the *same* code as Step 1 — `v = feedforward + KP_POS · position
error` — and the rotating feedforward carries you around.

But holding a circle takes constant **centripetal** acceleration `v²/R`, pointed at the center. You
never command that acceleration; the thing that produces it is the **velocity controller** — the
inner loop that makes the drone actually reach the velocity you asked for. On the real drone that is
the autopilot; on the sim it is the tilt loop inside `send_velocity`. That inner loop *is* a
**geometric controller**: it turns a desired velocity (and, in the full VNAV version, a desired
acceleration and orientation) into the thrust vector — the tilt — the drone should hold. So the
geometric controller hasn't gone away; standardizing on velocity moved it below your code, exactly
where it lives on real hardware.

**Easing in (spin-up).** The drone starts at rest, but a circle at full speed demands an instant
sideways velocity. Jumping straight to it makes the drone lurch and swing wide on the first quarter
lap. So the orbit **spins up**: the angular rate eases from zero to full over the first lap, so the
commanded velocity grows smoothly from zero. There is still a hard ceiling on orbit speed — the
centripetal acceleration `v²/R` can only come from tilt, and the velocity controller's tilt is
limited, so past some speed the drone cannot hold the radius. Shrinking `PERIOD` runs into that wall.

## Key terms

- **Trajectory** — a target position given for every instant, `p(t)`, together with its velocity `p'(t)`. Contrast a waypoint, which is a single fixed point.
- **Velocity command** — driving the drone with a desired velocity (m/s), via `neo_lab.send_velocity`. This is the real drone's native interface; the sim converts it to tilt internally.
- **Feedforward** — commanding the trajectory's own velocity so the controller keeps pace with a moving target instead of lagging behind it.
- **Position gain** — `KP_POS` (1/s): how strongly a position error is turned into corrective velocity. Raise it for tighter tracking; too high and it gets twitchy.
- **Smoothstep (time-scaling)** — `s(u) = 3u² − 2u³`; zero slope at both ends, so a segment begins and ends at rest without a jerk.
- **Cubic Hermite spline** — a curve on each segment built from two endpoints and two tangents via the four basis functions `h00, h10, h01, h11`.
- **Catmull-Rom tangent** — the slope at a waypoint chosen as `½(next − previous)`, which makes adjacent spline segments join smoothly.
- **Racing line** — a single continuous trajectory carried through all the waypoints without stopping.
- **Centripetal acceleration** — the inward acceleration `v²/R` needed to travel a circle of radius `R` at speed `v`. The velocity controller produces it; you only command the (rotating) velocity.
- **Geometric controller** — the inner loop that turns a desired velocity (and, fully, acceleration and orientation) into thrust. On the real drone it is the autopilot; on the sim it is the tilt loop inside `send_velocity`.
- **Spin-up** — easing the orbit's angular rate from zero to full over the first lap so the commanded velocity grows from rest instead of jumping.

## How to run

```bash
drone open_sim                          # launch the sim once
drone sim course/week4_integration/module3_trajectory/main.py            # all steps, your code
drone sim course/week4_integration/module3_trajectory/main_solution.py   # reference flight
```

Press **Enter** in the simulator window to start.

## Steps

1. **`step1_timed_segment.py`** — track one smooth timed segment by commanding velocity (write the velocity controller)
2. **`step2_smooth_course.py`** — build a cubic spline through five waypoints and fly the whole course without stopping (write the `hermite` blend)
3. **`step3_orbit.py`** — circle a fixed point by commanding a rotating velocity (write the velocity controller for the circle)

## What to expect

Runs the three steps in order: glide smoothly out to a point and settle, sweep through a
five-waypoint slalom in one continuous line, then orbit a point for two laps, and land.

## You're done when

- Step 1: the drone eases from the start to `(GOAL_RIGHT, GOAL_FWD)` over `DURATION` seconds — starting and stopping gently, not lurching — and reports a small **max tracking error**.
- Step 2: the drone flies through all `len(WAYPOINTS) − 1` segments in one smooth line (each `Reached waypoint k` prints in order) and finishes near the last waypoint with a small max tracking error.
- Step 3: the drone eases into a circle and holds it for `REVOLUTIONS` laps, reporting a small **max radius error** (how far its distance from the center strays from `RADIUS`).

## If it doesn't work

| Symptom | Fix |
|---------|-----|
| Step 1 always trails behind the target | You dropped the feedforward — the commanded velocity must start with `vel_r`/`vel_f` (the trajectory's own velocity), then add the position correction. |
| Step 1 barely moves / drifts slowly | Check you are passing `neo_lab.send_velocity(drone, v_right, v_up, v_forward)` — note the axis order (right, up, forward), and that `v_up` comes from the altitude error. |
| Step 2 flies to the first waypoint and stops / path has corners | Check your `hermite` blend: at `s=0` it must return `p0` and at `s=1` return `p1`. If it returns only `p0`, every segment collapses to a point. |
| Step 2 overshoots wildly between waypoints | The tangents scale with waypoint spacing; keep them within a few meters of each other, or lengthen `SEG_TIME` to slow the pass. |
| Drifts off the course near the end | Position is dead-reckoned from velocity and drifts (same as Modules 1–2). See "Going further." |
| Step 3 spirals outward into a bigger circle | Your feedforward velocity isn't rotating — make sure you read `vel_r, vel_f` from `trajectory(_t)` each frame rather than reusing a fixed heading. |
| Step 3 can't hold the radius | The orbit is too fast: the centripetal acceleration `v²/R` exceeds what the velocity controller's tilt can supply. Raise `PERIOD` to slow it down. |

## Going further (optional)

- Position here is dead-reckoned (`position += velocity · dt`) and drifts. `neo_lab.world_position(drone)` returns the sim's true position on a new enough build — swap it in and watch the tracking error shrink. On the real drone, the ROS2 odometry replaces dead reckoning entirely.
- Add a third axis: make the waypoints `(right, up, forward)` and spline the height too, so the course climbs and dives.
- Tune for speed: shorten `SEG_TIME` until the drone can no longer hold the line. Where does it break — the controller, or the dead-reckoning drift?
- The real VNAV lab replaces this hand-built spline with a **minimum-snap** trajectory: instead of picking tangents by a rule, it solves for the polynomial through all waypoints that minimizes total *snap* (the fourth derivative of position), which is what keeps a quadrotor's motors from saturating. Same idea — a smooth timed path — found by optimization instead of by hand.
- Orbit (Step 3): shrink `PERIOD` toward a faster orbit until the radius blows up, and read `v²/R` against the speed the velocity controller can hold to see why there is a ceiling.
- Make the orbit point a *camera* target: yaw the drone to face the center as it circles (pass a `yaw_rate` to `send_velocity` and reuse the heading control from Module 4), turning the orbit into an inspection sweep of the point.

---

Fill in the blanks in `tasks/`; completed references are in `solutions/` (try it yourself first!).
