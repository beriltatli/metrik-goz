"""
Sample scene generator — for the panel's "I don't have a photo" case.

You should be able to see what the system does without dragging a photo into the
panel. The images generated here are synthetic: we place the camera, the
reference and the thing to be measured, so we KNOW THE RIGHT ANSWER. The panel
prints that answer next to the measurement — you see at a glance that the
confidence interval really holds.

None of this is possible with a real photo; there you only have the measurement
and its error bar. Sample scenes exist precisely to supply that missing
reference.
"""

from __future__ import annotations

import numpy as np

from .synthetic import build_scene

# Same palette as the panel (shared with validation.py)
_GROUND = (238, 236, 232)         # BGR
_TABLE = (206, 214, 224)
_GRID = (188, 196, 206)
_OBSTACLE = (86, 84, 82)
_FREE = (214, 222, 214)
_TARGET = (52, 104, 235)          # BGR ≈ the inverse of #eb6834: orange
_TARGET2 = (122, 175, 27)


def _world_map(H: np.ndarray, size_px: tuple[int, int]):
    """
    The world coordinate on the plane for every pixel.

    Pixels falling behind the camera (where the projective scale changes sign)
    are marked invalid; painting there would fill the scene with the wrong half
    of the plane.
    """
    height, width = size_px
    yy, xx = np.mgrid[0:height, 0:width]
    px = np.stack([xx.ravel(), yy.ravel(), np.ones(height * width)], axis=1).astype(float)
    hn = px @ np.linalg.inv(H).T
    w = hn[:, 2]

    # The origin is in front of the camera; the sign there means "in front".
    front_sign = np.sign(np.array([0.0, 0.0, 1.0]) @ np.linalg.inv(H).T[:, 2] or 1.0)
    valid = (np.abs(w) > 1e-9) & (np.sign(w) == front_sign)

    w = np.where(np.abs(w) < 1e-9, 1e-9, w)
    X = (hn[:, 0] / w).reshape(height, width)
    Y = (hn[:, 1] / w).reshape(height, width)
    return X, Y, valid.reshape(height, width)


def _project_world(H: np.ndarray, world) -> np.ndarray:
    d = np.atleast_2d(np.asarray(world, dtype=float))
    h = np.hstack([d, np.ones((len(d), 1))]) @ H.T
    return h[:, :2] / h[:, 2:3]


def _signed_area(p: np.ndarray) -> float:
    x, y = p[:, 0], p[:, 1]
    return float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _draw_aruco(canvas: np.ndarray, H: np.ndarray, edge_mm: float,
                marker_id: int = 7, dictionary: str = "DICT_4X4_50") -> None:
    """
    Pastes an ArUco marker onto the plane with the right perspective.

    There are two subtleties:

    1) Because the camera looks down at the plane, the world -> image transform
       FLIPS the orientation (it contains a reflection). If we map the corners as
       they are, the marker lands mirrored in the image and no detector will
       recognize it — a mirrored code is not in the dictionary. We look at the
       signed area of the target quadrilateral and flip the mapping when needed.
       The world frame then ends up mirrored relative to the `square_world`
       order; since a reflection is an isometry, the measured lengths, areas and
       widths are unaffected.

    2) We leave a white quiet zone around the marker, like a printed sheet.
       ArUco's corner detection relies on the contrast around the outer black
       border; if the ground gets darker, detection silently degrades.
    """
    import cv2

    s = 400
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary))
    marker = cv2.cvtColor(cv2.aruco.generateImageMarker(d, marker_id, s),
                          cv2.COLOR_GRAY2BGR)

    half = edge_mm / 2.0
    height, width = canvas.shape[:2]
    source = np.array([[0, 0], [s, 0], [s, s], [0, s]], dtype=np.float32)

    def paste(source_img, corner_half):
        # ArUco corner order: top-left, top-right, bottom-right, bottom-left.
        world = np.array([[-corner_half, -corner_half], [corner_half, -corner_half],
                          [corner_half, corner_half], [-corner_half, corner_half]])
        target = _project_world(H, world).astype(np.float32)
        if _signed_area(target) * _signed_area(source) < 0:
            target = target[[1, 0, 3, 2]]
        M = cv2.getPerspectiveTransform(source, target)
        warp = cv2.warpPerspective(source_img, M, (width, height), flags=cv2.INTER_LINEAR)
        mask = cv2.warpPerspective(np.full(source_img.shape[:2], 255, np.uint8), M,
                                   (width, height), flags=cv2.INTER_NEAREST)
        canvas[mask > 127] = warp[mask > 127]

    paste(np.full((s, s, 3), 250, np.uint8), half * 1.35)   # quiet zone
    paste(marker, half)


def _draw_disk(canvas, H, center_mm, radius_mm, color, label=None, *, crosshair=False):
    """A circular object on the plane (an ellipse once projected)."""
    import cv2

    angle = np.linspace(0, 2 * np.pi, 96, endpoint=False)
    rim = np.asarray(center_mm) + radius_mm * np.column_stack([np.cos(angle), np.sin(angle)])
    h = np.hstack([rim, np.ones((len(rim), 1))]) @ H.T
    px = (h[:, :2] / h[:, 2:3]).astype(np.int32)
    cv2.fillPoly(canvas, [px], color, lineType=cv2.LINE_AA)

    h0 = np.append(center_mm, 1.0) @ H.T
    m = (h0[:2] / h0[2]).astype(int)
    if crosshair:
        cv2.drawMarker(canvas, tuple(m), (255, 255, 255), cv2.MARKER_CROSS, 18, 2,
                       cv2.LINE_AA)
    if label:
        cv2.putText(canvas, label, (m[0] + 14, m[1] - 12), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return m.astype(float)


# The phone: the object to be measured. Its dimensions are NOT in the reference
# list — the whole point of the demo is that the user measures something they
# genuinely do not know.
PHONE_WIDTH, PHONE_HEIGHT = 146.7, 71.5
COIN_MM = 26.15                      # 1 TL, diameter


def _widest_diameter_ends(H: np.ndarray, center_mm, radius_mm: float):
    """
    The two ends of the WIDEST part of the circle in the image.

    The projection of a round reference under perspective is an ellipse, so which
    diameter you measure matters. The widest part (the ellipse's major axis) is
    the diameter that suffered no foreshortening, i.e. the only one that gives
    the right answer. A user naturally measures there too; that is where we put
    the hint points.
    """
    angle = np.linspace(0, np.pi, 720, endpoint=False)
    direction = np.column_stack([np.cos(angle), np.sin(angle)])
    a = _project_world(H, np.asarray(center_mm) + radius_mm * direction)
    b = _project_world(H, np.asarray(center_mm) - radius_mm * direction)
    k = int(np.argmax(np.hypot(*(a - b).T)))
    return np.array([a[k], b[k]])


def _table(tilt_deg: float, name: str, description: str):
    """A phone on a table with a 1 TL coin beside it — the panel's main scenario."""
    import cv2

    size = (900, 1300)
    scene = build_scene(reference_size_mm=COIN_MM, focal_px=1500.0, size_px=size,
                        distance_mm=520.0, tilt_deg=tilt_deg, azimuth_deg=0.0)
    H = scene.H_true
    X, Y, valid = _world_map(H, size)

    canvas = np.full((*size, 3), _GROUND, np.uint8)
    table = valid & (np.abs(X) < 420) & (np.abs(Y) < 300)
    canvas[table] = _TABLE
    texture = table & (np.minimum(Y % 90, 90 - Y % 90) < 1.2)      # wood grain lines
    canvas[texture] = _GRID

    # The phone: sharp corners, clickable
    center = np.array([-40.0, 0.0])
    corner_mm = np.array([[-PHONE_WIDTH / 2, -PHONE_HEIGHT / 2],
                          [PHONE_WIDTH / 2, -PHONE_HEIGHT / 2],
                          [PHONE_WIDTH / 2, PHONE_HEIGHT / 2],
                          [-PHONE_WIDTH / 2, PHONE_HEIGHT / 2]]) + center
    corner_px = _project_world(H, corner_mm)
    cv2.fillPoly(canvas, [corner_px.astype(np.int32)], (46, 44, 42), cv2.LINE_AA)
    screen = _project_world(H, (corner_mm - center) * 0.9 + center)
    cv2.fillPoly(canvas, [screen.astype(np.int32)], (28, 26, 25), cv2.LINE_AA)

    # 1 TL: right next to the phone, on the table
    coin_center = np.array([110.0, 0.0])
    _draw_disk(canvas, H, coin_center, COIN_MM / 2, (86, 158, 196))
    _draw_disk(canvas, H, coin_center, COIN_MM / 2 * 0.72, (104, 178, 214))
    coin_ends = _widest_diameter_ends(H, coin_center, COIN_MM / 2)

    return dict(
        image=canvas,
        name=f"example-{name}.png",
        description=description,
        reference=dict(type="scale", name="1_tl", length_mm=COIN_MM),
        default_measurement="box",
        truth={
            "width": dict(value=PHONE_WIDTH, unit="mm", description="the phone's long edge"),
            "height": dict(value=PHONE_HEIGHT, unit="mm", description="the phone's short edge"),
            "area": dict(value=PHONE_WIDTH * PHONE_HEIGHT / 100.0, unit="cm²",
                         description="the face of the phone"),
        },
        hint=dict(reference=coin_ends.tolist(), box=corner_px.tolist()),
    )


def _flat(seed: int):
    return _table(1.5, "flat",
                  "A phone on a table with a 1 TL coin beside it. The photo was taken "
                  "almost straight down — this is the case where the simple scale "
                  "model works correctly.")


def _tilted(seed: int):
    return _table(26.0, "tilted",
                  "The same scene, but shot at an angle. A scale built from a single "
                  "length cannot correct perspective; watch whether the system "
                  "notices that.")


def _passage(seed: int):
    """Rubble scenario: a free corridor that narrows in the middle."""
    size = (900, 1400)
    scene = build_scene(reference_size_mm=200.0, focal_px=1100.0, size_px=size,
                        distance_mm=2600.0, tilt_deg=18.0, azimuth_deg=0.0)
    H = scene.H_true
    X, Y, valid = _world_map(H, size)

    NARROW_MM, WIDE_MM = 520.0, 900.0
    half = np.where(np.abs(X) < 420.0, NARROW_MM / 2.0, WIDE_MM / 2.0)
    corridor = valid & (np.abs(Y) < half) & (np.abs(X) < 1500.0)
    field = valid & (np.abs(X) < 1800.0) & (np.abs(Y) < 1300.0)

    canvas = np.full((*size, 3), _GROUND, np.uint8)
    canvas[field] = _OBSTACLE
    canvas[corridor] = _FREE

    _draw_aruco(canvas, H, 200.0, marker_id=3)

    # Corners of the free corridor: the "draw the polygon here" hint
    polygon_mm = np.array([
        [-1400, -WIDE_MM / 2], [-420, -WIDE_MM / 2], [-420, -NARROW_MM / 2],
        [420, -NARROW_MM / 2], [420, -WIDE_MM / 2], [1400, -WIDE_MM / 2],
        [1400, WIDE_MM / 2], [420, WIDE_MM / 2], [420, NARROW_MM / 2],
        [-420, NARROW_MM / 2], [-420, WIDE_MM / 2], [-1400, WIDE_MM / 2],
    ], dtype=float)
    h = np.hstack([polygon_mm, np.ones((len(polygon_mm), 1))]) @ H.T
    polygon_px = h[:, :2] / h[:, 2:3]

    return dict(
        image=canvas,
        name="example-passage.png",
        description="A rubble corridor: a 200 mm ArUco reference, free space narrowing "
                    "in the middle. Draw a polygon around the free space.",
        reference=dict(type="aruco", edge_mm=200.0),
        default_measurement="passage",
        truth={
            "passage": dict(value=NARROW_MM, unit="mm",
                            description="the narrowest point of the corridor"),
        },
        hint=dict(passage=polygon_px.tolist()),
        footprint_mm=480.0,
    )


SCENES = {"flat": _flat, "tilted": _tilted, "passage": _passage}

# The samples the panel offers: all of them fit the "size of an object" flow.
PANEL_SCENES = ["flat", "tilted"]


def sample_scene(name: str = "flat", *, seed: int = 0) -> dict:
    """Generates the named sample scene; returns the true answer alongside it."""
    if name not in SCENES:
        raise KeyError(f"Unknown sample '{name}'. Options: {sorted(SCENES)}")
    return SCENES[name](seed)
