"""
MIT BWSI Autonomous Drone Racing Course - UAV Neo

bench_line — hand-held perception diagnostic, now binned by HEIGHT and TURNING.

You hold the drone (powered + connected, sensors streaming) and move it over a
line that has straight and curvy sections, RAISING and LOWERING it through the
0-2 m range as you go. Nothing flies. Every frame it logs perception numbers
alongside:
  height   -> physics.get_altitude()            (m above ground)
  turning  -> |physics.get_angular_velocity()[1]| > TURN_THRESH   (yaw rate)

The summary then bins everything by height band AND by straight-vs-turning, so
you can read directly:
  - how the pixel count DROPS as you raise the drone (sets MIN_PIXELS per height)
  - where the V_MIN cliff sits (does it move with height?)
  - straight-line curviness floor vs. turning curviness (sets CURVE_SCALE)

Because you're hand-holding, "turning" means YOU rotated the drone. So: hold
STEADY over straight bits, rotate SMOOTHLY following the line over curvy bits,
and raise/lower through the whole height range during both.

Read the summary, not the scrolling frames. Everything also goes to CSV.

Run:
    drone sim bench_line.py
    drone bench_line.py        (real drone, camera + sensors only)
"""

import cv2
import drone_core
import numpy as np

# -- Course setup: makes the shared `neo_lab` helper importable. --
import os as _os, sys as _sys
_d = _os.path.dirname(_os.path.realpath(__file__))
while _os.path.basename(_d) != "labs" and _os.path.dirname(_d) != _d:
    _d = _os.path.dirname(_d)
if _d not in _sys.path:
    _sys.path.insert(0, _d)
import neo_lab

# -- Settings ---------------------------------------------------------------
DOWNWARD     = True        # the line-following camera
TEST_SECONDS = 60.0        # run length, then it summarizes
REPORT_HZ    = 3.0         # per-frame print rate

TURN_THRESH  = 0.15        # rad/s; above this |yaw rate| a frame counts as TURNING
HEIGHT_BINS  = [0.0, 0.5, 1.0, 1.5, 2.0, 99.0]   # band edges (m), for 0-2 m range

# Candidate thresholds swept every frame, for each channel.
V_SWEEP = [120, 150, 180, 200, 220, 235, 250]   # brightness (Value)
S_SWEEP = [ 40,  60,  80, 100, 120, 150, 180]   # saturation

V_MIN = 200
S_MIN = 100

LOG_PATH = "bench_line_log.csv"


# -- Module state -----------------------------------------------------------
_t           = 0.0
_next_report = 0.0
_done        = False
_log         = None
_rows        = []          # in-memory rows for the binned summary


def reset():
    global _t, _next_report, _done, _log, _rows
    _t = 0.0
    _next_report = 0.0
    _done = False
    _rows = []
    _log = open(LOG_PATH, "w")
    _log.write("t,height,yaw_rate,turning,"
               "v_full,v_p99,angle_full,curv_full,count_full\n")


# -- Perception -------------------------------------------------------------
def get_image(drone):
    if DOWNWARD:
        return drone.camera.get_downward_image()
    return drone.camera.get_color_image()


def fit_line(mask):
    """Returns (angle_deg, curviness, count) or None."""
    pts = np.argwhere(mask).astype(np.float64)
    count = len(pts)
    if count < 2:
        return None
    ys, xs = pts[:, 0], pts[:, 1]
    xy = np.column_stack([xs, ys]).astype(np.float32)
    vx, vy, x0, y0 = cv2.fitLine(xy, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
    if vy > 0:
        vx, vy = -vx, -vy
    angle = float(np.degrees(np.arctan2(vx, -vy)))
    resid = (xs - x0) * vy - (ys - y0) * vx
    return angle, float(np.std(resid)), int(count)


def height_band(h):
    for i in range(len(HEIGHT_BINS) - 1):
        if HEIGHT_BINS[i] <= h < HEIGHT_BINS[i + 1]:
            return f"{HEIGHT_BINS[i]:.1f}-{HEIGHT_BINS[i+1]:.1f}m"
    return ">2.0m"


# -- Main loop --------------------------------------------------------------
def update(drone):
    global _t, _next_report, _done

    if _done:
        return True

    dt = drone.get_delta_time()
    _t += dt

    image = get_image(drone)
    if image is None or image.size == 0:
        print("!! empty frame from the camera", flush=True)
        return False

    # -- Sensors.
    height = float(drone.physics.get_altitude())
    yaw_rate = float(drone.physics.get_angular_velocity()[1])
    turning = abs(yaw_rate) > TURN_THRESH

    # -- Perception.
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    v_chan = hsv[:, :, 2]
    v_p99 = float(np.percentile(v_chan, 99))

    v_mask = neo_lab.bright_mask(image, V_MIN)
    s_mask = neo_lab.saturated_mask(image, S_MIN)
    v_full = int(np.count_nonzero(v_mask))
    s_full = int(np.count_nonzero(s_mask))

    fit = fit_line(v_mask)
    if fit is not None:
        angle, curv, count = fit
    else:
        angle = curv = float("nan")
        count = 0

    # -- Sweeps (for the V_MIN/S_MIN cliff), stored per row for binning.
    v_sweep = {vt: int(np.count_nonzero(v_chan > vt)) for vt in V_SWEEP}
    s_chan = hsv[:, :, 1]
    s_sweep = {st: int(np.count_nonzero(s_chan > st)) for st in S_SWEEP}

    row = {
        "t": _t, "height": height, "yaw_rate": yaw_rate, "turning": turning,
        "band": height_band(height),
        "v_full": v_full, "s_full": s_full, "v_p99": v_p99,
        "angle": angle, "curv": curv, "count": count,
        "v_sweep": v_sweep, "s_sweep": s_sweep,
    }
    _rows.append(row)
    _log.write(f"{_t:.3f},{height:.3f},{yaw_rate:.4f},{int(turning)},"
               f"{v_full},{v_p99:.0f},{angle:.3f},{curv:.3f},{count}\n")

    if _t >= _next_report:
        _next_report = _t + 1.0 / REPORT_HZ
        state = "TURN " if turning else "strt "
        print(f"[t={_t:5.1f}] h={height:4.2f}m {state} "
              f"yawrate={yaw_rate:+.2f}  V_p99={v_p99:3.0f}  "
              f"px={v_full:6d}  angle={angle:+6.1f}  curv={curv:6.1f}",
              flush=True)

    if _t >= TEST_SECONDS:
        _summarize()
        _done = True
        if _log:
            _log.close()
    return _done


# -- Summary ----------------------------------------------------------------
def _summarize():
    print("\n" + "=" * 74)
    print("bench_line summary   (binned by HEIGHT and by STRAIGHT/TURNING)")
    print("=" * 74)

    if not _rows:
        print("  no frames captured")
        return

    valid = [r for r in _rows if r["count"] > 0]
    print(f"  frames: {len(_rows)} total, {len(valid)} with a line detected")
    hs = [r["height"] for r in _rows]
    print(f"  height range seen: {min(hs):.2f} - {max(hs):.2f} m")

    # -- Pixel count and curviness by height band.
    print("\n  PIXEL COUNT & CURVINESS by height (Value mask @V_MIN=%d):" % V_MIN)
    print("    band        frames   px(min/med)      straight-curv   turning-curv")
    bands = []
    for i in range(len(HEIGHT_BINS) - 1):
        label = f"{HEIGHT_BINS[i]:.1f}-{HEIGHT_BINS[i+1]:.1f}m"
        bands.append(label)
    for band in bands:
        br = [r for r in valid if r["band"] == band]
        if not br:
            continue
        px = np.array([r["v_full"] for r in br])
        straight_c = [r["curv"] for r in br if not r["turning"] and r["curv"] == r["curv"]]
        turn_c     = [r["curv"] for r in br if r["turning"] and r["curv"] == r["curv"]]
        sc = f"{np.median(straight_c):6.1f}" if straight_c else "   -- "
        tc = f"{np.median(turn_c):6.1f}" if turn_c else "   -- "
        print(f"    {band:10s}  {len(br):5d}   {px.min():5d}/{np.median(px):6.0f}"
              f"      {sc}          {tc}")

    # -- V_MIN cliff, overall and does it move with height.
    print("\n  V_MIN SWEEP (median count | p10 count), all heights:")
    prev = None
    for vt in V_SWEEP:
        counts = np.array([r["v_sweep"][vt] for r in _rows])
        med = np.median(counts)
        p10 = np.percentile(counts, 10)
        drop = ""
        if prev is not None and prev > 0 and (1 - med / prev) > 0.5:
            drop = f"   <-- {100*(1-med/prev):.0f}% drop: cliff"
        print(f"    >{vt:>3d}   {med:>8.0f} | {p10:>7.0f}{drop}")
        prev = med

    # -- Saturation, just to confirm it's dead.
    s_med = np.median([r["s_full"] for r in _rows])
    print(f"\n  SATURATION @S_MIN={S_MIN}: median {s_med:.0f} px "
          f"(if this doesn't track the line, use bright_mask)")

    # -- Straight vs turning curviness, all heights pooled.
    straight = [r["curv"] for r in valid if not r["turning"] and r["curv"] == r["curv"]]
    turning  = [r["curv"] for r in valid if r["turning"] and r["curv"] == r["curv"]]
    print("\n  CURVE_SCALE inputs (all heights pooled):")
    if straight:
        print(f"    straight curviness: median {np.median(straight):.1f}, "
              f"p90 {np.percentile(straight, 90):.1f}   <- the 'no turn' floor")
    if turning:
        print(f"    turning  curviness: median {np.median(turning):.1f}, "
              f"p90 {np.percentile(turning, 90):.1f}   <- set CURVE_SCALE near here")
    if straight and turning:
        print(f"    -> CURVE_SCALE between {np.median(straight):.0f} and "
              f"{np.median(turning):.0f}; nearer the turning value brakes only "
              f"for real bends")

    # -- MIN_PIXELS guidance, per height (this is the height-dependent one).
    print("\n  MIN_PIXELS guidance (per height band = ~40% of that band's min count):")
    for band in bands:
        br = [r for r in valid if r["band"] == band]
        if not br:
            continue
        mn = min(r["v_full"] for r in br)
        print(f"    at {band:10s}: min count {mn:6d}  ->  MIN_PIXELS ~ {int(mn*0.4)}")
    print("    Use the value for the HEIGHT YOU'LL ACTUALLY FLY AT.")

    print(f"\n  full per-frame log written to {LOG_PATH}")
    print("=" * 74)


# -- Entry point ------------------------------------------------------------
if __name__ == "__main__":
    _drone = drone_core.create_drone()

    def start():
        reset()
        print("bench_line — hand-held, binned by height + turning\n"
              "  RAISE and LOWER the drone (0-2 m) over straight AND curvy line.\n"
              "  Hold steady on straights, rotate smoothly on curves.\n"
              f"  {TEST_SECONDS:.0f} s, then it summarizes.\n")

    def _update():
        if update(_drone):
            pass   # camera + sensors only; nothing to land

    _drone.set_start_update(start, _update)
    _drone.go(not neo_lab._is_sim(_drone))