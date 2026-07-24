"""
MIT BWSI Autonomous Drone Racing Course - UAV Neo

camera_test — Bright-mask diagnostics for tuning V_MIN and MIN_PIXELS.

Flies nothing by default. Set FLY = False and you can hold the drone over the
line by hand (or leave it on the floor) while it streams mask statistics, which
is the fastest way to tune the real camera. Set FLY = True to hover instead.

Each report line gives you, for the CURRENT V_MIN:
    V-channel percentiles  — where the line's brightness sits vs the background
    full / near counts     — bright pixels in the whole frame vs the near band
    angle, vector          — cv2.fitLine direction, and the heading error it implies
    curviness              — std of perpendicular residuals (this is CURVE_SCALE)

It also sweeps V_MIN across V_SWEEP every frame so you can see the whole
threshold curve at once instead of re-running with one value at a time.

At the end it prints percentiles across the run and suggests both constants.

Run:
    drone sim camera_test.py
"""

import time

import cv2
import drone_core
import drone_utils as uav_utils
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
FLY           = False    # False = never arm, just stream (bench / hand-held)
LAUNCH_HEIGHT = 1.0      # only used when FLY is True

V_MIN         = 200      # the value you are currently testing
NEAR_FRACTION = 0.4      # bottom fraction of the image used for the fit
MIN_PIXELS    = 200      # the threshold you are trying to validate

V_SWEEP = [120, 150, 180, 200, 220, 240, 250]

REPORT_HZ    = 2.0       # report lines per second (raw frame rate is too fast)
TEST_SECONDS = 30.0      # run length, then summarize
SAVE_FRAMES  = 3         # dump this many frame+mask PNGs to /tmp for eyeballing

DOWNWARD = True          # False -> use the forward color camera instead


# -- Module state -----------------------------------------------------------
_t           = 0.0
_next_report = 0.0
_saved       = 0
_done        = False
_near_hist  = []         # near-band pixel counts, every frame
_full_hist  = []
_sweep_hist = {v: [] for v in V_SWEEP}
_curv_hist  = []


def reset():
    global _t, _next_report, _saved, _done
    global _near_hist, _full_hist, _sweep_hist, _curv_hist
    _t = 0.0
    _next_report = 0.0
    _saved = 0
    _done = False
    _near_hist = []
    _full_hist = []
    _sweep_hist = {v: [] for v in V_SWEEP}
    _curv_hist = []


# -- Perception -------------------------------------------------------------
def get_image(drone):
    if DOWNWARD:
        return drone.camera.get_downward_image()
    return drone.camera.get_color_image()


def near_band(mask):
    """Bottom NEAR_FRACTION of the mask — the locally straight part of the line."""
    h = mask.shape[0]
    return mask[int(h * (1.0 - NEAR_FRACTION)):, :]


def fit_line(ys, xs):
    """
    Returns (angle_rad, curviness, vx, vy).
      angle     0 when the line runs straight up the image, + leaning right
      curviness std of perpendicular distances from the fitted line
    """
    pts = np.column_stack([xs, ys]).astype(np.float32)
    vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
    if vy > 0:                       # keep the direction pointing up the image
        vx, vy = -vx, -vy
    angle = float(np.arctan2(vx, -vy))
    resid = (xs - x0) * vy - (ys - y0) * vx
    return angle, float(np.std(resid)), float(vx), float(vy)


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
    v = hsv[:, :, 2]

    mask = neo_lab.bright_mask(image, V_MIN)
    near = near_band(mask)

    full_count = int(np.count_nonzero(mask))
    near_count = int(np.count_nonzero(near))
    _full_hist.append(full_count)
    _near_hist.append(near_count)

    for vt in V_SWEEP:
        _sweep_hist[vt].append(int(np.count_nonzero(v > vt)))

    # -- Fit, if there is anything to fit.
    pts = np.argwhere(near).astype(np.float64)
    if len(pts) >= 2:
        ys, xs = pts[:, 0], pts[:, 1]
        angle, curviness, vx, vy = fit_line(ys, xs)
        centroid = float(xs.mean())
        _curv_hist.append(curviness)
    else:
        angle = curviness = vx = vy = centroid = float("nan")

    # -- Save a few PNGs for visual confirmation.
    if _saved < SAVE_FRAMES and _t > 1.0 * (_saved + 1):
        cv2.imwrite(f"/tmp/cam_{_saved}.png", image)
        cv2.imwrite(f"/tmp/mask_{_saved}.png", mask)
        print(f"   [saved /tmp/cam_{_saved}.png and /tmp/mask_{_saved}.png]",
              flush=True)
        _saved += 1

    # -- Rate-limited report.
    if _t >= _next_report:
        _next_report = _t + 1.0 / REPORT_HZ
        _report(image, v, full_count, near_count,
                angle, curviness, vx, vy, centroid)

    if FLY:
        drone.flight.send_pcmd(0.0, 0.0, 0.0, 0.0)

    if _t >= TEST_SECONDS:
        _summarize()
        _done = True
    return _done


def _report(image, v, full_count, near_count, angle, curviness, vx, vy, centroid):
    h, w = v.shape
    p = np.percentile(v, [50, 90, 99])
    verdict = "PASS" if near_count >= MIN_PIXELS else "fail"

    print(f"\n[t={_t:5.1f}s] {w}x{h}  V-channel  median={p[0]:5.1f}  "
          f"p90={p[1]:5.1f}  p99={p[2]:5.1f}  max={v.max():3d}")
    print(f"   V_MIN={V_MIN}:  full={full_count:7d} px "
          f"({100.0 * full_count / v.size:5.2f}%)   "
          f"near={near_count:6d} px  vs MIN_PIXELS={MIN_PIXELS} -> {verdict}")

    if near_count >= 2:
        print(f"   vector=({vx:+.4f}, {vy:+.4f})  "
              f"angle={np.degrees(angle):+7.2f} deg  "
              f"curviness={curviness:7.2f}  centroid_col={centroid:6.1f}")
    else:
        print("   vector=--  (not enough pixels in the near band to fit)")

    sweep = "  ".join(f"{vt}:{_sweep_hist[vt][-1]:>7d}" for vt in V_SWEEP)
    print(f"   sweep  {sweep}")


def _summarize():
    print("\n" + "=" * 70)
    print("camera_test summary")
    print("=" * 70)

    if not _near_hist:
        print("  no frames captured")
        return

    near = np.array(_near_hist)
    full = np.array(_full_hist)
    print(f"  frames                {len(near)}")
    print(f"  near-band pixels      min={near.min()}  p10={np.percentile(near, 10):.0f}  "
          f"median={np.median(near):.0f}  max={near.max()}")
    print(f"  full-frame pixels     min={full.min()}  median={np.median(full):.0f}  "
          f"max={full.max()}")

    print(f"\n  V_MIN sweep (median full-frame count):")
    for vt in V_SWEEP:
        counts = np.array(_sweep_hist[vt])
        print(f"    v>{vt:>3d}   median={np.median(counts):>9.0f} px"
              f"   ({np.percentile(counts, 10):>8.0f} at p10)")

    if _curv_hist:
        c = np.array(_curv_hist)
        print(f"\n  curviness             min={c.min():.2f}  "
              f"median={np.median(c):.2f}  p90={np.percentile(c, 90):.2f}  "
              f"max={c.max():.2f}")
        print(f"    -> CURVE_SCALE near {np.percentile(c, 90):.0f} "
              f"(the value that reads as 'fully in a turn')")

    p10 = np.percentile(near, 10)
    print(f"\n  suggested MIN_PIXELS  {max(int(p10 * 0.5), 20)}  "
          f"(half the 10th-percentile near-band count)")
    print("    low enough to survive the worst frames you actually saw,")
    print("    high enough to reject noise blobs")

    print(f"\n  for V_MIN: pick the HIGHEST sweep value whose p10 count is still")
    print("    comfortably above your MIN_PIXELS. Higher rejects more background.")
    print("    Check /tmp/mask_*.png to confirm the mask is the line and not glare.")
    print("=" * 70)


# -- Entry point ------------------------------------------------------------
if __name__ == "__main__":
    _drone = drone_core.create_drone()
    _launcher = neo_lab.Launcher(LAUNCH_HEIGHT) if FLY else None

    def start():
        if _launcher is not None:
            _launcher.reset()
        reset()
        mode = f"hover at {LAUNCH_HEIGHT} m" if FLY else "grounded, camera only"
        print(f"camera_test — {mode}, {TEST_SECONDS:.0f} s\n"
              f"  V_MIN={V_MIN}  MIN_PIXELS={MIN_PIXELS}  "
              f"NEAR_FRACTION={NEAR_FRACTION}\n")

    def _update():
        if _launcher is not None and not _launcher.done:
            _launcher.update(_drone)
            return
        if update(_drone):
            if FLY:
                _drone.flight.land()

    _drone.set_start_update(start, _update)
    _drone.go(not neo_lab._is_sim(_drone))
