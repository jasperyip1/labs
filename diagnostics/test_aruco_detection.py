"""
MIT BWSI Autonomous Drone Racing Course - UAV Neo
GNU General Public License v3.0

Handheld sanity check — ArUco tag detection (Robust Auto-Scanning Version)
Fixes empty camera frame crashes and applies aggressive vision preprocessing 
(Subpixel corner refinement, CLAHE contrast boost, multi-dictionary search).

Run with `-d` to pop up a visual window:
    python test_aruco_detection.py -d
"""

import drone_core
import cv2

# -- Course setup --
import os as _os, sys as _sys
_d = _os.path.dirname(_os.path.realpath(__file__))
while _os.path.basename(_d) != "labs" and _os.path.dirname(_d) != _d:
    _d = _os.path.dirname(_d)
if _d not in _sys.path:
    _sys.path.insert(0, _d)
import neo_lab  # noqa: F401

# -- Dictionaries to Auto-Scan ------------------------------------------
DICT_TO_TRY = {
    "ArUco 4x4_50": cv2.aruco.DICT_4X4_50,
    "ArUco 4x4_100": cv2.aruco.DICT_4X4_100,
    "AprilTag 16h5": cv2.aruco.DICT_APRILTAG_16h5,
    "ArUco 4x4_250": cv2.aruco.DICT_4X4_250,
    "ArUco 5x5_100": cv2.aruco.DICT_5X5_100,
}

# -- Configure Preprocessor & Detector Parameters ------------------------
_has_new_api = hasattr(cv2.aruco, "ArucoDetector")
_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

def _build_params():
    if _has_new_api:
        params = cv2.aruco.DetectorParameters()
    else:
        params = cv2.aruco.DetectorParameters_create()
        
    # High-sensitivity thresholding settings
    params.adaptiveThreshWinSizeMin = 3
    params.adaptiveThreshWinSizeMax = 23
    params.adaptiveThreshWinSizeStep = 10
    
    # Expand size range (allows tags close-up or far away)
    params.minMarkerPerimeterRate = 0.02
    params.maxMarkerPerimeterRate = 4.0
    
    # Subpixel corner refinement for better marker edge finding
    if hasattr(cv2.aruco, "CORNER_REFINE_SUBPIX"):
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        
    return params

_params = _build_params()
_detectors = {}

if _has_new_api:
    for name, dict_id in DICT_TO_TRY.items():
        _dictionary = cv2.aruco.getPredefinedDictionary(dict_id)
        _detectors[name] = cv2.aruco.ArucoDetector(_dictionary, _params)
        
    def _detect_single_pass(gray, name, detector):
        corners, ids, _ = detector.detectMarkers(gray)
        return corners, ids
else:
    for name, dict_id in DICT_TO_TRY.items():
        _detectors[name] = cv2.aruco.getPredefinedDictionary(dict_id)
        
    def _detect_single_pass(gray, name, dictionary):
        corners, ids, _ = cv2.aruco.detectMarkers(gray, dictionary, parameters=_params)
        return corners, ids

def _detect_all(gray):
    """Scans all dictionaries on raw grayscale first, then attempts CLAHE if empty."""
    # Pass 1: Raw Image
    for name, det in _detectors.items():
        corners, ids = _detect_single_pass(gray, name, det)
        if ids is not None and len(ids) > 0:
            return corners, ids, name

    # Pass 2: Contrast Boosted Image (CLAHE) for dark/glare conditions
    gray_enhanced = _clahe.apply(gray)
    for name, det in _detectors.items():
        corners, ids = _detect_single_pass(gray_enhanced, name, det)
        if ids is not None and len(ids) > 0:
            return corners, ids, f"{name} (Enhanced)"

    return None, None, None

# -- Module-level state -------------------------------------------------
_frame_count = 0
_latest_ids = None
_latest_centers = []
_latest_dict_name = None


def reset():
    global _frame_count, _latest_ids, _latest_centers, _latest_dict_name
    _frame_count = 0
    _latest_ids = None
    _latest_centers = []
    _latest_dict_name = None


def update(drone):
    """Fast loop (~20-30Hz): Safely captures frames and runs detection."""
    global _frame_count, _latest_ids, _latest_centers, _latest_dict_name
    _frame_count += 1

    image = drone.camera.get_downward_image()

    # GUARD: Fixes OpenCV `!_src.empty()` assertion error when camera warms up
    if image is None or image.size == 0:
        return False

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners, ids, dict_name = _detect_all(gray)
    
    _latest_ids = ids
    _latest_centers = []
    _latest_dict_name = dict_name

    if ids is not None and len(ids) > 0:
        for c in corners:
            pts = c.reshape(4, 2)
            cx, cy = pts.mean(axis=0)
            _latest_centers.append((round(float(cx)), round(float(cy))))

    if drone.display is not None:
        overlay = image.copy()
        if ids is not None and len(ids) > 0:
            cv2.aruco.drawDetectedMarkers(overlay, corners, ids)
            cv2.putText(overlay, f"Format: {dict_name}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        drone.display.show_color_image(overlay)

    return False


def update_slow(drone):
    """Slow loop (~1-2Hz): Handles terminal printing."""
    global _frame_count, _latest_ids, _latest_centers, _latest_dict_name
    
    if _latest_ids is not None and len(_latest_ids) > 0:
        id_list = _latest_ids.flatten().tolist()
        print(f"[frame {_frame_count}] TAG DETECTED ids={id_list} format={_latest_dict_name} centers={_latest_centers}")
    else:
        print(f"[frame {_frame_count}] scanning (no tags detected)")
        
    return False


if __name__ == "__main__":
    _drone = drone_core.create_drone()

    def start():
        reset()
        print("Handheld detection test — Auto-scanning with enhanced vision filters.")

    def _update():
        update(_drone)
        
    def _update_slow():
        update_slow(_drone)

    _drone.set_start_update(start, _update, _update_slow)
    _drone.go(autostart=True)