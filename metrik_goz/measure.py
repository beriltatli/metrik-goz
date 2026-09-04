"""
The measurement layer: given a homography, turn clicked pixels into millimeters.

A projective transform maps lines to lines, so pushing the polygon corners onto
the world plane and measuring there is exact. No need to sample along the edges.
The one exception is `narrowest_passage`, where the free space itself can have
curved boundaries, so that one scans on the world plane.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .homography import Homography


@dataclass
class Passage:
    width_mm: float
    position_mm: np.ndarray     # midpoint of the passage in world coordinates
    axis: np.ndarray            # direction of travel, unit vector
    profile_mm: np.ndarray      # free width at each station
    stations_mm: np.ndarray
    edge_margin_mm: float = 0.0  # how much we skipped at each end, see below

    def fits(self, footprint_mm: float, clearance_mm: float = 0.0) -> bool:
        return self.width_mm >= footprint_mm + clearance_mm


@dataclass
class Box:
    """Plane dimensions of an object whose four corners were marked."""

    width_mm: float             # mean of edges 1 and 3, in drawing order
    height_mm: float            # mean of edges 2 and 4
    area_mm2: float
    edges_mm: np.ndarray
    corners_mm: np.ndarray
    rectangularity: float       # how far opposite edges disagree; 0 is perfect

    @property
    def diagonal_mm(self) -> float:
        c = self.corners_mm
        return float((np.hypot(*(c[2] - c[0])) + np.hypot(*(c[3] - c[1]))) / 2.0)


def distance(homography: Homography, p1_px, p2_px) -> float:
    """Real distance between two pixels, in mm."""
    d = homography.to_world(np.array([p1_px, p2_px], dtype=float))
    return float(np.hypot(*(d[1] - d[0])))


def length(homography: Homography, polyline_px) -> float:
    """Total length of a polyline, in mm."""
    d = homography.to_world(np.asarray(polyline_px, dtype=float))
    return float(np.sum(np.hypot(*(np.diff(d, axis=0).T))))


def box(homography: Homography, four_corners_px) -> Box:
    """
    Width, height and area of a four-cornered object.

    Opposite edges get averaged rather than picking one of them: nobody clicks a
    corner exactly, and averaging halves the slip.

    `rectangularity` gives back what that average hides, i.e. how differently the
    two opposite edges came out. Anything away from zero means one of two things,
    either the object isn't on the reference plane, or perspective was never
    corrected (under a similarity model an oblique shot skews the object). Both
    wreck the measurement, and both are invisible unless we report this.
    """
    d = homography.to_world(np.asarray(four_corners_px, dtype=float))
    if len(d) != 4:
        raise ValueError(f"A box needs exactly 4 corners, {len(d)} given.")

    edges = np.array([float(np.hypot(*(d[(i + 1) % 4] - d[i]))) for i in range(4)])
    width = float((edges[0] + edges[2]) / 2.0)
    height = float((edges[1] + edges[3]) / 2.0)

    x, y = d[:, 0], d[:, 1]
    area_mm2 = float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))

    deviation = max(abs(edges[0] - edges[2]) / max(width, 1e-9),
                    abs(edges[1] - edges[3]) / max(height, 1e-9))
    return Box(width_mm=width, height_mm=height, area_mm2=area_mm2, edges_mm=edges,
               corners_mm=d, rectangularity=float(deviation))


def area(homography: Homography, polygon_px) -> float:
    """Polygon area in mm^2, shoelace."""
    d = homography.to_world(np.asarray(polygon_px, dtype=float))
    x, y = d[:, 0], d[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def narrowest_passage(
    homography: Homography,
    free_mask: np.ndarray,
    *,
    axis=None,
    step_mm: float = 20.0,
    sample_mm: float = 5.0,
    edge_margin_mm: float | None = None,
    max_samples: int = 4_000_000,
) -> Passage:
    """
    Scan the world plane for the tightest point of the free space.

    free_mask is a bool array the size of the image, True where the ground is
    traversable. axis is the direction of travel in world coordinates; leave it
    out and we take the principal axis of the free space. step_mm is how often we
    cut a cross-section, sample_mm the spacing within one, which is effectively
    the resolution of the answer. edge_margin_mm is how much to skip at each end,
    defaulting to 5% of the corridor.

    That margin isn't cosmetic. Hand-drawn masks tend to taper to a point, and
    the outermost cross-section then comes out nearly zero wide, so the minimum
    locks onto it. That zero is where the mask ended, not how narrow the passage
    is. We drop the ends and report how much we dropped in `edge_margin_mm`.

    Each cross-section is cut perpendicular to the axis and its width is the
    longest uninterrupted free run in it, not the total free length: in a
    corridor split by an island, the sum would flatter the passage badly. The
    minimum over all stations is the answer.

    On the AEON side this answers "will the robot get through here", and in the
    kitchen "how many boxes fit on this shelf", off the same code.
    """
    if step_mm <= 0 or sample_mm <= 0:
        raise ValueError("step_mm and sample_mm must be positive.")
    if free_mask.dtype != bool:
        free_mask = free_mask.astype(bool)
    if free_mask.ndim != 2:
        raise ValueError("The free mask must be two-dimensional.")
    height, width = free_mask.shape

    ys, xs = np.nonzero(_boundary_pixels(free_mask))
    if len(xs) < 10:
        raise ValueError("The free mask is nearly empty; there is no passage to measure.")

    # Only the boundary of the mask goes through the transform, not the interior.
    # The extremes that set the scan range are on the boundary by definition and
    # interior pixels just reproduce the same numbers. For a typical corridor
    # that turns a half-million-point transform into a few thousand, and since
    # Monte Carlo runs this hundreds of times over it dominates the runtime.
    world = homography.to_world(np.column_stack([xs, ys]).astype(float))

    if axis is None:
        # PCA over the boundary points. The lengthwise direction agrees with what
        # area-based PCA would give, the elongation dominates either way.
        centered = world - world.mean(axis=0)
        _, _, Vt = np.linalg.svd(centered, full_matrices=False)
        axis = Vt[0]
    axis = np.asarray(axis, dtype=float).reshape(2)
    norm = np.linalg.norm(axis)
    if norm < 1e-12:
        raise ValueError("The axis cannot be the zero vector.")
    axis = axis / norm
    normal = np.array([-axis[1], axis[0]])

    center = world.mean(axis=0)
    t = (world - center) @ axis          # along the axis
    s = (world - center) @ normal        # across it

    t0, t1 = float(t.min()), float(t.max())
    s_half = float(np.abs(s).max()) + 2 * sample_mm

    if edge_margin_mm is None:
        edge_margin_mm = 0.05 * (t1 - t0)
    t0 += edge_margin_mm
    t1 -= edge_margin_mm
    if t1 <= t0:
        raise ValueError("The edge margin eats up the whole free space.")

    stations = np.arange(t0, t1 + 1e-9, step_mm)
    offsets = np.arange(-s_half, s_half + 1e-9, sample_mm)
    if stations.size * offsets.size > max_samples:
        raise ValueError(
            f"The scan grid is too large ({stations.size}×{offsets.size}). "
            f"Increase step_mm or sample_mm.")

    # Every cross-section goes to pixel space in one transform. Doing one
    # homography call per station used to be the single most expensive part of a
    # measurement, and Monte Carlo repeats the whole thing hundreds of times.
    tracks = center + stations[:, None, None] * axis + offsets[None, :, None] * normal
    section_px = homography.to_image(tracks.reshape(-1, 2))
    u = np.rint(section_px[:, 0]).astype(np.int64)
    v = np.rint(section_px[:, 1]).astype(np.int64)
    valid = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    free = np.zeros(u.shape, dtype=bool)
    free[valid] = free_mask[v[valid], u[valid]]
    free = free.reshape(len(stations), len(offsets))

    widths = np.zeros(len(stations))
    midpoints = np.zeros((len(stations), 2))
    for i, ti in enumerate(stations):
        start, run = _longest_run(free[i])
        widths[i] = run * sample_mm
        middle = (offsets[start] + (run - 1) * sample_mm / 2.0) if run > 0 else 0.0
        midpoints[i] = center + ti * axis + middle * normal

    narrowest = int(np.argmin(widths))
    return Passage(
        width_mm=float(widths[narrowest]),
        position_mm=midpoints[narrowest],
        axis=axis,
        profile_mm=widths,
        stations_mm=stations - stations[0],
        edge_margin_mm=float(edge_margin_mm),
    )


def _boundary_pixels(mask: np.ndarray) -> np.ndarray:
    """
    Free pixels that have at least one of their four neighbours not free. Pixels
    leaning on the image border count as boundary too.
    """
    if mask.shape[0] < 3 or mask.shape[1] < 3:
        return mask
    interior = np.zeros_like(mask)
    interior[1:-1, 1:-1] = (mask[1:-1, 1:-1] & mask[:-2, 1:-1] & mask[2:, 1:-1]
                            & mask[1:-1, :-2] & mask[1:-1, 2:])
    boundary = mask & ~interior
    return boundary if boundary.sum() >= 10 else mask


def _longest_run(flags: np.ndarray) -> tuple[int, int]:
    """(start index, length) of the longest run of True."""
    if not flags.any():
        return 0, 0
    # Pad both ends with zeros so every run has a clean transition to find.
    d = np.diff(np.concatenate([[0], flags.view(np.int8), [0]]))
    starts = np.nonzero(d == 1)[0]
    ends = np.nonzero(d == -1)[0]
    lengths = ends - starts
    k = int(np.argmax(lengths))
    return int(starts[k]), int(lengths[k])
