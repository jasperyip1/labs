"""
MIT BWSI Autonomous Drone Racing Course - UAV Neo
GNU General Public License v3.0

Handheld sanity check — Step 3 prerequisite
Just looks at the downward camera and tells you whether it sees the LED
strip on the ground. No flight commands are ever called, so it's safe to
run while holding the drone in your hand.

Run with `-d` to also pop up a window showing the mask, e.g.:
    python test_line_detection.py -d
"""

import drone_core
import cv2
import numpy as np
import warnings

# -- Constants --------------------------------------------------------------
V_MIN      = 200   # brightness threshold for the LED strip mask (tune this)
MIN_PIXELS = 200   # need at least this many bright pixels to call it "seen"


def bright_mask(image, v_min):
    """Boolean mask of pixels whose HSV 'Value' (brightness) is >= v_min.
    Self-contained replacement for neo_lab.bright_mask, which isn't
    available in this environment."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    v_channel = hsv[:, :, 2]
    return v_channel >= v_min

# -- Module-level state -------------------------------------------------
_frame_count = 0
_latest_num_pixels = 0
_latest_centroid = (0, 0)
_latest_eq = None


def reset():
    global _frame_count, _latest_num_pixels, _latest_centroid, _latest_eq
    _frame_count = 0
    _latest_num_pixels = 0
    _latest_centroid = (0, 0)
    _latest_eq = None


def update(drone):
    """Fast loop (~20-30Hz): Handles camera fetching, masking, and math."""
    global _frame_count, _latest_num_pixels, _latest_centroid, _latest_eq
    _frame_count += 1

    image = drone.camera.get_downward_image()
    mask = bright_mask(image, V_MIN)
    edges = np.argwhere(mask)
    _latest_num_pixels = edges.shape[0]

    if _latest_num_pixels >= MIN_PIXELS:
        ys = edges[:, 0]
        xs = edges[:, 1]
        centroid_row = ys.mean()
        centroid_col = xs.mean()
        _latest_centroid = (centroid_row, centroid_col)
        
        # Suppress RankWarning for perfectly vertical lines
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', np.RankWarning)
            try:
                # Perform linear regression to get slope (m) and intercept (b)
                m, b = np.polyfit(xs, ys, 1)
                _latest_eq = (m, b)
            except Exception:
                _latest_eq = None
    else:
        _latest_eq = None

    # Optional visual check if the drone was started with the -d flag.
    if drone.display is not None:
        overlay = image.copy()
        overlay[mask] = [0, 0, 255]  # paint bright-mask pixels red (BGR)
        
        # Draw the regression line in green if we found one
        if _latest_eq is not None:
            m, b = _latest_eq
            h, w = overlay.shape[:2]
            # calculate y points at the extreme left (x=0) and right (x=w)
            x1, x2 = 0, w
            y1, y2 = int(m * x1 + b), int(m * x2 + b)
            cv2.line(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
        drone.display.show_color_image(overlay)

    return False  # never signal "done" -- just keep looking every frame


def update_slow(drone):
    """Slow loop (~1-2Hz): Handles printing to prevent terminal spam."""
    global _frame_count, _latest_num_pixels, _latest_centroid, _latest_eq
    
    if _latest_num_pixels >= MIN_PIXELS:
        row, col = _latest_centroid
        
        if _latest_eq is not None:
            m, b = _latest_eq
            eq_str = f"y = {m:.2f}x + {b:.2f}"
        else:
            eq_str = "undefined (vertical)"
            
        print(f"[frame {_frame_count}] LINE DETECTED  "
              f"pixels={_latest_num_pixels}  centroid=(row={row:.0f}, col={col:.0f})  "
              f"Equation: {eq_str}")
    else:
        print(f"[frame {_frame_count}] no line  (bright pixels={_latest_num_pixels}, need {MIN_PIXELS})")
        
    return False


if __name__ == "__main__":
    _drone = drone_core.create_drone()

    def start():
        reset()
        print("Handheld line-detection test — no motors will move.")
        print(f"Looking for pixels brighter than V_MIN={V_MIN}...")

    def _update():
        update(_drone)
        
    def _update_slow():
        update_slow(_drone)

    # Register start, fast loop, and slow loop
    _drone.set_start_update(start, _update, _update_slow)
    _drone.go(autostart=True)