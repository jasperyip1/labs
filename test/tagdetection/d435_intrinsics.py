#!/usr/bin/env python3
"""
d435_intrinsics.py -- CAMERA_MATRIX / DIST_COEFFS for the D435 forward color camera,
using ONLY the uav-neo-library (drone.camera) -- no pyrealsense2 required.

The uav-neo-library's Camera module (get_color_image, get_depth_image, get_width,
get_height, get_max_range) does not expose intrinsics directly. But the D435 color
sensor's field of view is a published, fixed spec: 69.4 deg (H) x 42.5 deg (V).
That's a pinhole-camera assumption baked in at the factory ISP level (the color stream
is already rectified), so we can derive a nominal intrinsic matrix from FOV + resolution
without touching the RealSense SDK.

  fx = (width  / 2) / tan(hFOV / 2)
  fy = (height / 2) / tan(vFOV / 2)
  cx = width / 2,  cy = height / 2

This is a nominal/spec-sheet matrix, not a per-unit factory calibration -- expect it to
be correct to within a few percent, which is what centering + depth-based ranging need.
If you later want the exact per-unit calibration, that requires pyrealsense2 querying
the device directly (or a checkerboard) -- not necessary for this application.
"""

import numpy as np

# Intel RealSense D435 color sensor published spec (Intel datasheet).
D435_HFOV_DEG = 69.4
D435_VFOV_DEG = 42.5


def get_intrinsics(width, height):
    """Nominal (CAMERA_MATRIX, DIST_COEFFS) for the D435 color stream at width x height.
    Call with drone.camera.get_width() / get_height() so it always matches your stream."""
    hfov = np.radians(D435_HFOV_DEG)
    vfov = np.radians(D435_VFOV_DEG)

    fx = (width / 2.0) / np.tan(hfov / 2.0)
    fy = (height / 2.0) / np.tan(vfov / 2.0)
    cx, cy = width / 2.0, height / 2.0

    K = np.array([[fx, 0.0, cx],
                  [0.0, fy, cy],
                  [0.0, 0.0, 1.0]], dtype=np.float64)
    # D435 color stream is already rectified by the sensor ISP -> distortion ~0.
    dist = np.zeros((5, 1), dtype=np.float64)
    return K, dist


if __name__ == "__main__":
    # Example: python3 d435_intrinsics.py 640 480
    import sys
    w = int(sys.argv[1]) if len(sys.argv) > 1 else 640
    h = int(sys.argv[2]) if len(sys.argv) > 2 else 480
    K, dist = get_intrinsics(w, h)
    np.set_printoptions(suppress=True, precision=4)
    print(f"\n# nominal D435 color intrinsics at {w}x{h} (spec FOV {D435_HFOV_DEG}x{D435_VFOV_DEG} deg)")
    print("CAMERA_MATRIX = np.array([")
    for r in K:
        print(f"    [{r[0]:.4f}, {r[1]:.4f}, {r[2]:.4f}],")
    print("], dtype=np.float64)")
    print("DIST_COEFFS = np.zeros((5, 1), dtype=np.float64)\n")
