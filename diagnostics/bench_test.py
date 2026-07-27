"""
MIT BWSI Autonomous Drone Racing Course - UAV Neo

bench_line — hand-held perception diagnostic for the LED-strip bench.

You hold the drone and move it over a line that has both straight and curvy
sections. Nothing flies. The drone only needs to be ON and CONNECTED so the
camera streams. Every frame it prints, and at the end it summarizes:

  THRESHOLD  — for BOTH mask channels (Value/brightness and Saturation),
               a sweep of candidate thresholds and the resulting pixel counts,
               so you can see which channel actually separates the line from
               the floor, and pick V_MIN / S_MIN at the cliff.
  LINE FIT   — angle (deg), magnitude (line length in px), and curviness,
               computed on the FULL frame and on the TOP and BOTTOM thirds,
               so you can see whether a band tracks better than the whole
               image on the curvy section (or confirm whole-frame is fine).

Read the summary, not the scrolling frames. The per-frame lines are just so
you can watch it respond as you move over straight vs. curved bits.

Bench tips that make the numbers trustworthy:
  - keep the drone at a roughly CONSTANT height above the strip (mimic flight)
  - keep it LEVEL — don't tilt as you sweep
  - move it through the CURVY part at a realistic speed, so the p10 (worst-
    frame) counts reflect motion blur you'll actually see in flight

Run:
    drone sim bench_line.py         (sim, to check it works)
    drone bench_line.py             (real drone, camera only — the real tuning)
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
TEST_SECONDS = 40.0        # run length, then it summarizes
REPORT_HZ    = 3.0         # per-frame print rate (raw is too fast to read)
SAVE_FRAMES  = 4           # dump this many frame+mask PNGs to /tmp

# -- CURVE_SCALE calibration (two-pass) -------------------------------------
# The script cannot tell a straight line from a curve on its own, so DON'T
# trust a CURVE_SCALE guessed from one mixed pass. Instead run it TWICE:
#   1. set LABEL = "STRAIGHT", hold the drone over a STRAIGHT section only
#   2. set LABEL = "CURVE",    hold it over a CURVY section only
# Each run prints its own curviness median for that label. Then:
#   CURVE_SCALE  ~  the CURVE run's median  (curviness at 'fully in a turn')
#   sanity check:   the STRAIGHT run's median should be much lower
# Set CURVE_SCALE near the curve value to brake only for real bends, or a bit
# lower to start slowing earlier. Final tuning happens in flight.
LABEL = "STRAIGHT"         # "STRAIGHT" or "CURVE" (or "MIXED" for a rough guess)

# Candidate thresholds swept every frame, for each channel.
V_SWEEP = [120, 150, 180, 200, 220, 235, 250]   # brightness (Value)
S_SWEEP = [ 40,  60,  80, 100, 120, 150, 180]   # saturation

# The threshold each channel's live pixel count is reported at.
V_MIN = 200
S_MIN = 100


# -- Module state -----------------------------------------------------------
_t          = 0.0
_next_report = 0.0
_saved      = 0
_done       = False

# Per-channel full-frame count history, and per-threshold sweep history.
_hist = {
    "V": {"full": [], "sweep": {v: [] for v in V_SWEEP}},
    "S": {"full": [], "sweep": {s: [] for s in S_SWEEP}},
}
# Line-fit history per region.
_fit_hist = {"full": [], "top": [], "bottom": []}


def reset():
    global _t, _next_report, _saved, _done, _hist, _fit_hist
    _t = 0.0
    _next_report = 0.0
    _saved = 0
    _done = False
    _hist = {
        "V": {"full": [], "sweep": {v: [] for v in V_SWEEP}},
        "S": {"full": [], "sweep": {s: [] for s in S_SWEEP}},
    }
    _fit_hist = {"full": [], "top": [], "bottom": []}


# -- Perception -------------------------------------------------------------
def get_image(drone):
    if DOWNWARD:
        return drone.camera.get_downward_image()
    return drone.camera.get_color_image()


def fit_region(mask):
    """
    Fit a line to the bright pixels of a mask region.
    Returns (angle_deg, magnitude_px, curviness, count) or None if too few.
      angle      0 = line runs straight up the image, + leaning right going up
      magnitude  length of the point spread along the line (proxy for how much
                 line is visible)
      curviness  std of perpendicular residuals about the fitted line
    """
    pts = np.argwhere(mask).astype(np.float64)      # rows of (row, col)
    count = len(pts)
    if count < 2:
        return None

    ys, xs = pts[:, 0], pts[:, 1]
    xy = np.column_stack([xs, ys]).astype(np.float32)   # (x, y) for OpenCV
    vx, vy, x0, y0 = cv2.fitLine(xy, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
    if vy > 0:                          # keep direction pointing up the image
        vx, vy = -vx, -vy

    angle = float(np.degrees(np.arctan2(vx, -vy)))

    # Project points onto the line direction -> spread = magnitude.
    proj = (xs - x0) * vx + (ys - y0) * vy
    magnitude = float(proj.max() - proj.min())

    # Perpendicular residuals -> curviness.
    resid = (xs - x0) * vy - (ys - y0) * vx
    curviness = float(np.std(resid))

    return angle, magnitude, curviness, count


def regions(mask):
    """Full frame, top third, bottom third."""
    h = mask.shape[0]
    third = h // 3
    return {
        "full":   mask,
        "top":    mask[:third, :],
        "bottom": mask[h - third:, :],
    }


# -- Main loop --------------------------------------------------------------
def update(drone):
    global _t, _next_report, _saved, _done

    if _done:
        return True

    dt = drone.get_delta_time()
    _t += dt

    image = get_image(drone)
    if image is None or image.size == 0:
        print("!! empty frame from the camera", flush=True)
        return False

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    v_chan = hsv[:, :, 2]
    s_chan = hsv[:, :, 1]

    # -- Threshold histories, both channels.
    v_mask = neo_lab.bright_mask(image, V_MIN)
    s_mask = neo_lab.saturated_mask(image, S_MIN)
    _hist["V"]["full"].append(int(np.count_nonzero(v_mask)))
    _hist["S"]["full"].append(int(np.count_nonzero(s_mask)))
    for vt in V_SWEEP:
        _hist["V"]["sweep"][vt].append(int(np.count_nonzero(v_chan > vt)))
    for st in S_SWEEP:
        _hist["S"]["sweep"][st].append(int(np.count_nonzero(s_chan > st)))

    # -- Line fit per region, on the Value mask (the line-following channel).
    fits = {}
    for name, region in regions(v_mask).items():
        f = fit_region(region)
        fits[name] = f
        if f is not None:
            _fit_hist[name].append(f)

    # -- Save a few PNGs for eyeballing.
    if _saved < SAVE_FRAMES and _t > 1.0 * (_saved + 1):
        cv2.imwrite(f"/tmp/bench_cam_{_saved}.png", image)
        cv2.imwrite(f"/tmp/bench_vmask_{_saved}.png", v_mask)
        cv2.imwrite(f"/tmp/bench_smask_{_saved}.png", s_mask)
        print(f"   [saved /tmp/bench_cam_{_saved}.png + v/s masks]", flush=True)
        _saved += 1

    if _t >= _next_report:
        _next_report = _t + 1.0 / REPORT_HZ
        _report(v_chan, s_chan, fits)

    if _t >= TEST_SECONDS:
        _summarize()
        _done = True
    return _done


def _report(v_chan, s_chan, fits):
    vp = np.percentile(v_chan, [50, 99])
    sp = np.percentile(s_chan, [50, 99])
    vfull = _hist["V"]["full"][-1]
    sfull = _hist["S"]["full"][-1]

    print(f"\n[t={_t:5.1f}s]  V median={vp[0]:3.0f} p99={vp[1]:3.0f} "
          f"-> {vfull:6d}px @V_MIN={V_MIN}    "
          f"S median={sp[0]:3.0f} p99={sp[1]:3.0f} -> {sfull:6d}px @S_MIN={S_MIN}")

    for name in ("full", "top", "bottom"):
        f = fits[name]
        if f is None:
            print(f"   {name:6s}  --")
        else:
            angle, mag, curv, cnt = f
            print(f"   {name:6s}  angle={angle:+7.2f} deg  mag={mag:6.1f}px  "
                  f"curviness={curv:6.2f}  count={cnt:6d}")


def _summarize():
    print("\n" + "=" * 72)
    print(f"bench_line summary   [LABEL = {LABEL}]")
    print("=" * 72)

    # -- Threshold, per channel.
    for ch, sweep, label, cur in (
        ("V", V_SWEEP, "VALUE / brightness (bright_mask)", V_MIN),
        ("S", S_SWEEP, "SATURATION (saturated_mask)", S_MIN),
    ):
        full = np.array(_hist[ch]["full"])
        if len(full) == 0:
            continue
        print(f"\n  {label}")
        print(f"    full-frame @cur={cur}:  min={full.min()}  "
              f"p10={np.percentile(full, 10):.0f}  median={np.median(full):.0f}  "
              f"max={full.max()}")
        print(f"    sweep (median count | p10 count):")
        prev_med = None
        for t in sweep:
            counts = np.array(_hist[ch]["sweep"][t])
            med = np.median(counts)
            p10 = np.percentile(counts, 10)
            drop = ""
            if prev_med is not None and prev_med > 0:
                frac = 1.0 - med / prev_med
                if frac > 0.5:
                    drop = f"   <-- big drop ({100 * frac:.0f}%) : cliff near here"
            print(f"      >{t:>3d}   {med:>9.0f} | {p10:>8.0f}{drop}")
            prev_med = med

    # -- Line fit, per region.
    print(f"\n  LINE FIT (Value mask)")
    print(f"    region   curviness (min / median / p90)     angle spread")
    for name in ("full", "top", "bottom"):
        hist = _fit_hist[name]
        if not hist:
            print(f"    {name:6s}   no valid fits")
            continue
        curv = np.array([h[2] for h in hist])
        ang = np.array([h[0] for h in hist])
        print(f"    {name:6s}   {curv.min():6.1f} / {np.median(curv):6.1f} / "
              f"{np.percentile(curv, 90):6.1f}          "
              f"{ang.min():+.0f}..{ang.max():+.0f} deg")

    # -- Suggestions.
    vfull = np.array(_hist["V"]["full"])
    print("\n  SUGGESTIONS")
    if len(vfull):
        p10 = np.percentile(vfull, 10)
        print(f"    MIN_PIXELS ~ {max(int(p10 * 0.5), 20)}  "
              f"(half the worst-frame Value count; survives motion blur)")
    print("    V_MIN / S_MIN: pick the threshold just BELOW the cliff row above")
    print("      for whichever channel shows the cleanest cliff. That channel")
    print("      is the one that separates line from floor on YOUR strip.")

    full_hist = _fit_hist["full"]
    if full_hist:
        curv = np.array([h[2] for h in full_hist])
        med = np.median(curv)
        if LABEL.upper() == "CURVE":
            print(f"    CURVE run: full-frame curviness median = {med:.0f}")
            print(f"      -> set CURVE_SCALE near {med:.0f} (curviness at a real bend).")
            print(f"      Confirm your STRAIGHT run's median is well below this.")
        elif LABEL.upper() == "STRAIGHT":
            print(f"    STRAIGHT run: full-frame curviness median = {med:.0f}")
            print(f"      -> this is your 'no turn' floor. CURVE_SCALE must be ABOVE it.")
            print(f"      Now rerun with LABEL = \"CURVE\" over a curvy section.")
        else:  # MIXED / anything else -> fall back to the distribution guess
            print(f"    CURVE_SCALE ~ {np.percentile(curv, 90):.0f}  "
                  f"(p90 GUESS from a mixed pass — do the two-pass for a real number)")

    # Band-vs-full guidance.
    if _fit_hist["full"] and _fit_hist["bottom"]:
        cf = np.median([h[2] for h in _fit_hist["full"]])
        cb = np.median([h[2] for h in _fit_hist["bottom"]])
        if cf > 1e-6:
            ratio = cb / cf
            if ratio < 0.6:
                print(f"    BAND: bottom-third curviness is much lower than full "
                      f"({cb:.0f} vs {cf:.0f}) -> a near-band crop tracks straighter;")
                print("      worth using a band instead of the whole frame.")
            else:
                print(f"    BAND: bottom-third ~ full frame ({cb:.0f} vs {cf:.0f}) "
                      f"-> whole-frame is fine, don't bother cropping.")

    print("\n    Check /tmp/bench_vmask_*.png and _smask_*.png: confirm the mask")
    print("    is the LINE, not glare or floor. A fat blob inflates every count.")
    print("=" * 72)


# -- Entry point ------------------------------------------------------------
if __name__ == "__main__":
    _drone = drone_core.create_drone()

    def start():
        reset()
        section = {"STRAIGHT": "a STRAIGHT section only",
                   "CURVE": "a CURVY section only"}.get(
                       LABEL.upper(), "straight AND curvy line")
        print(f"bench_line — hand-held perception diagnostic   [LABEL = {LABEL}]\n"
              f"  move the drone over {section}, ~constant height, level.\n"
              f"  {TEST_SECONDS:.0f} s, then it summarizes.\n")

    def _update():
        if update(_drone):
            pass   # camera-only; nothing to land

    _drone.set_start_update(start, _update)
    _drone.go(not neo_lab._is_sim(_drone))
