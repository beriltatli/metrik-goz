"""
Reference object detection — the thing that puts scale into the scene.

Mathematically, the only way to get an absolute measurement out of a single
photo is to have something of known size in the scene. However good the camera
is, "this is 40 cm" cannot be extracted from a reference-free image — the scale
stays free.

Two routes are supported:
  ArUco marker  — the most reliable; corner positions are sub-pixel accurate, the
                  id is known, detection is automatic. Print it and put it on the
                  counter or in the field.
  Known object  — giving the four corners of something with standard dimensions
                  (a credit card, a sheet of A4) by hand. It saves you when no
                  marker is around, but corner reading is noisier (~1.5 px).
"""

from __future__ import annotations

import numpy as np

# Standard-sized references you can find anywhere (mm)
KNOWN_OBJECTS: dict[str, tuple[float, float]] = {
    "credit_card": (85.60, 53.98),   # ISO/IEC 7810 ID-1
    "a4": (297.0, 210.0),
    "a5": (210.0, 148.0),
    "cd": (120.0, 120.0),
    "post_it": (76.0, 76.0),
}

# Things carrying a single known LENGTH that everyone has in a pocket or a drawer.
# A coin is the handiest: being round, it does not matter which way you measure it
# — whatever angle it sits at in the photo, its diameter is its diameter.
#
# name -> (length_mm, human-readable description)
KNOWN_LENGTHS: dict[str, tuple[float, str]] = {
    # Turkish lira coins, diameter (Central Bank 2009 series)
    "1_tl":            (26.15, "1 TL — diameter"),
    "50_kurus":        (23.85, "50 kurus — diameter"),
    "25_kurus":        (20.50, "25 kurus — diameter"),
    "10_kurus":        (18.50, "10 kurus — diameter"),
    "5_kurus":         (17.50, "5 kurus — diameter"),
    "1_kurus":         (16.50, "1 kurus — diameter"),
    # Other common coins
    "2_euro":          (25.75, "2 euro — diameter"),
    "1_euro":          (23.25, "1 euro — diameter"),
    "50_cent":         (24.25, "50 euro cent — diameter"),
    "us_quarter":      (24.26, "US quarter — diameter"),
    # Standard objects
    "credit_card_long": (85.60, "credit card — long edge"),
    "credit_card_short": (53.98, "credit card — short edge"),
    "a4_long":         (297.0, "A4 sheet — long edge"),
    "a4_short":        (210.0, "A4 sheet — short edge"),
    "cd":              (120.0, "CD/DVD — diameter"),
    "aa_battery":      (50.5,  "AA battery — length"),
}

# Empirical values for corner reading noise (pixels, std)
TYPICAL_SIGMA_PX: dict[str, float] = {
    "aruco": 0.4,
    "manual": 1.5,
}


def square_world(edge_mm: float, center_mm=(0.0, 0.0)) -> np.ndarray:
    """Corners of a square of edge `edge_mm` around the given center (clockwise from top-left)."""
    half = edge_mm / 2.0
    cx, cy = center_mm
    return np.array([
        [cx - half, cy - half],
        [cx + half, cy - half],
        [cx + half, cy + half],
        [cx - half, cy + half],
    ])


def rectangle_world(width_mm: float, height_mm: float, center_mm=(0.0, 0.0)) -> np.ndarray:
    hx, hy = width_mm / 2.0, height_mm / 2.0
    cx, cy = center_mm
    return np.array([
        [cx - hx, cy - hy],
        [cx + hx, cy - hy],
        [cx + hx, cy + hy],
        [cx - hx, cy + hy],
    ])


def known_object(name: str, center_mm=(0.0, 0.0)) -> np.ndarray:
    """World corners of a standard object from the table."""
    if name not in KNOWN_OBJECTS:
        raise KeyError(f"Unknown reference '{name}'. Options: {sorted(KNOWN_OBJECTS)}")
    w, h = KNOWN_OBJECTS[name]
    return rectangle_world(w, h, center_mm)


def find_aruco(image, edge_mm: float, *, dictionary: str = "DICT_4X4_50",
               marker_id: int | None = None):
    """
    Finds the ArUco marker in the image.

    Returns: (world_mm, image_px, marker_id)
    The corner order of ArUco itself (top-left, top-right, bottom-right,
    bottom-left) is kept identical to the order of `square_world`.
    """
    import cv2  # needed only on this path; the core math runs without cv2

    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    dictionary_object = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary))
    parameters = cv2.aruco.DetectorParameters()
    # Sub-pixel corner refinement: our uncertainty claim rests on it
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.ArucoDetector(dictionary_object, parameters)

    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None or len(ids) == 0:
        raise ValueError("No ArUco marker was found in the image.")

    ids = ids.ravel()
    if marker_id is None:
        chosen = 0
    else:
        match = np.nonzero(ids == marker_id)[0]
        if len(match) == 0:
            raise ValueError(f"No marker with id {marker_id}. Found: {ids.tolist()}")
        chosen = int(match[0])

    image_px = corners[chosen].reshape(4, 2).astype(float)
    return square_world(edge_mm), image_px, int(ids[chosen])
