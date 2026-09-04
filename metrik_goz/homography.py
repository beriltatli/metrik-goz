"""
Plane homography: the mapping between image pixels and the world plane (mm).

The assumption — and the most important limit of this package: everything you
measure must lie on the SAME PLANE as the reference object. You can measure the
tomato on the counter with the card you put on the counter; you cannot measure
the box sitting on the shelf. When this assumption breaks the error grows
silently, which is why `Homography.off_plane_warning` reports how far out you
are extrapolating.

The fit has two stages:
  1) DLT — a closed-form starting solution with Hartley normalization.
  2) LM  — a refinement that minimizes the geometric reprojection error.

Why two stages: DLT minimizes the algebraic error, which under noise is NOT the
geometrically best solution. To make a claim about measurement error we have to
minimize the geometric one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .lm import solve


# ------------------------------------------------------------------ helpers
def _normalization_matrix(points: np.ndarray) -> np.ndarray:
    """
    Hartley normalization: move the centroid to the origin, make the mean
    distance sqrt(2). The numerical conditioning of DLT depends on this; skip it
    and the squared pixel values (on the order of 10^6) wreck the design matrix.
    """
    center = points.mean(axis=0)
    shifted = points - center
    mean_distance = np.sqrt((shifted ** 2).sum(axis=1)).mean()
    if mean_distance < 1e-12:
        scale = 1.0
    else:
        scale = np.sqrt(2.0) / mean_distance
    return np.array([
        [scale, 0.0, -scale * center[0]],
        [0.0, scale, -scale * center[1]],
        [0.0, 0.0, 1.0],
    ])


# The floor we allow w (the projective scale) to drop to.
_W_FLOOR = 1e-12


def _homogeneous(points: np.ndarray) -> np.ndarray:
    return np.hstack([points, np.ones((len(points), 1))])


def _apply(H: np.ndarray, points: np.ndarray) -> np.ndarray:
    """
    Projective transform by H; (N,2) -> (N,2).

    When w (the third component) approaches zero the point is on the horizon
    line; there is no real answer there. To keep the division from blowing up we
    clamp w to the floor while PRESERVING its sign — if the sign is lost the
    point lands on the wrong half of the plane and the error spreads silently.
    """
    points = np.atleast_2d(np.asarray(points, dtype=float))
    hn = _homogeneous(points) @ H.T
    w = hn[:, 2:3]
    small = np.abs(w) < _W_FLOOR
    if small.any():
        w = np.where(small, np.where(w < 0.0, -_W_FLOOR, _W_FLOOR), w)
    return hn[:, :2] / w


# ------------------------------------------------------------------ DLT
def dlt(world: np.ndarray, image: np.ndarray) -> np.ndarray:
    """
    Direct linear transform: world (mm) -> image (px) homography.
    At least 4 points are needed, no three of them collinear.
    """
    world = np.asarray(world, dtype=float)
    image = np.asarray(image, dtype=float)
    if len(world) < 4:
        raise ValueError("A homography needs at least 4 points.")
    if len(world) != len(image):
        raise ValueError("World and image point counts do not match.")

    T_w = _normalization_matrix(world)
    T_i = _normalization_matrix(image)
    w = _apply(T_w, world)
    i = _apply(T_i, image)

    A = []
    for (X, Y), (u, v) in zip(w, i):
        A.append([-X, -Y, -1, 0, 0, 0, u * X, u * Y, u])
        A.append([0, 0, 0, -X, -Y, -1, v * X, v * Y, v])
    A = np.asarray(A)

    _, _, Vt = np.linalg.svd(A)
    H_norm = Vt[-1].reshape(3, 3)

    H = np.linalg.inv(T_i) @ H_norm @ T_w
    if abs(H[2, 2]) < 1e-12:
        raise ValueError("Degenerate homography: h33 is too close to zero.")
    return H / H[2, 2]


# ------------------------------------------------------------------ LM refinement
def _residuals(p: np.ndarray, world: np.ndarray, image: np.ndarray) -> np.ndarray:
    """Reprojection error in pixels, flattened."""
    H = np.append(p, 1.0).reshape(3, 3)
    return (_apply(H, world) - image).ravel()


def _jacobian(p: np.ndarray, world: np.ndarray, image: np.ndarray) -> np.ndarray:
    """
    Analytic derivative of the residuals with respect to the 8 parameters.

    u = a/w,  a = h11*X + h12*Y + h13
    v = b/w,  b = h21*X + h22*Y + h23
              w = h31*X + h32*Y + 1
    """
    h11, h12, h13, h21, h22, h23, h31, h32 = p
    X = world[:, 0]
    Y = world[:, 1]
    a = h11 * X + h12 * Y + h13
    b = h21 * X + h22 * Y + h23
    w = h31 * X + h32 * Y + 1.0
    w = np.where(np.abs(w) < 1e-12, 1e-12, w)

    n = len(X)
    J = np.zeros((2 * n, 8))
    # du rows (even indices)
    J[0::2, 0] = X / w
    J[0::2, 1] = Y / w
    J[0::2, 2] = 1.0 / w
    J[0::2, 6] = -a * X / w ** 2
    J[0::2, 7] = -a * Y / w ** 2
    # dv rows (odd indices)
    J[1::2, 3] = X / w
    J[1::2, 4] = Y / w
    J[1::2, 5] = 1.0 / w
    J[1::2, 6] = -b * X / w ** 2
    J[1::2, 7] = -b * Y / w ** 2
    return J


# ------------------------------------------------------------------ main class
@dataclass
class Homography:
    """
    The mapping between the world plane (mm) and the image (px), plus its quality.

    H             : world -> image
    H_inv         : image -> world (measurement uses this direction)
    rms_px        : RMS of the reprojection error, pixels
    covariance    : covariance of the 8 homography parameters (from LM)
    reference_box : bounds of the reference in world coordinates — the
                    extrapolation warning is computed against it
    """

    H: np.ndarray
    H_inv: np.ndarray
    rms_px: float
    covariance: np.ndarray | None
    reference_box: tuple[float, float, float, float]
    converged: bool
    model: str = "projective"     # "projective" | "similarity" — explained below

    # -------------------------------------------------------------- constructor
    @classmethod
    def fit(cls, world_mm, image_px, *, refine: bool = True,
            covariance: bool = True) -> "Homography":
        """
        `covariance=False`: skips LM's post-solve covariance estimate.
        Monte Carlo builds this thousands of times and never looks at the
        covariance there — skipping saves one Jacobian and one matrix inverse
        per sample.
        """
        world = np.asarray(world_mm, dtype=float)
        image = np.asarray(image_px, dtype=float)

        H = dlt(world, image)
        rms = float(np.sqrt(np.mean((_apply(H, world) - image) ** 2)))
        cov = None
        converged = True

        # 4 points with 8 parameters: no degrees of freedom, so LM has nothing
        # to refine — the DLT solution already fits exactly.
        if refine and len(world) >= 5:
            p0 = (H / H[2, 2]).ravel()[:8]
            result = solve(
                lambda p: _residuals(p, world, image),
                p0,
                lambda p: _jacobian(p, world, image),
                compute_covariance=covariance,
            )
            H = np.append(result.p, 1.0).reshape(3, 3)
            rms = result.rms
            cov = result.covariance
            converged = result.converged

        try:
            H_inv = np.linalg.inv(H)
        except np.linalg.LinAlgError as error:
            raise ValueError("The homography is not invertible: the reference "
                             "corners may be collinear or coincident.") from error

        return cls(
            H=H,
            H_inv=H_inv,
            rms_px=rms,
            covariance=cov,
            reference_box=(
                float(world[:, 0].min()), float(world[:, 1].min()),
                float(world[:, 0].max()), float(world[:, 1].max()),
            ),
            converged=converged,
        )

    # -------------------------------------------------------------- similarity
    @classmethod
    def from_length(cls, p1_px, p2_px, length_mm: float) -> "Homography":
        """
        A SIMILARITY homography from a single known length (the diameter of a
        coin, the long edge of a card): scale + rotation + translation.

        Why a separate constructor and why not "projective": two points and a
        length carry three numbers, while a projective transform has eight
        degrees of freedom. Instead of inventing the missing information we build
        a narrower model — perspective is NOT corrected, only the scale is known.

        Where this model is valid: the camera looks straight down at the plane
        and the thing you measure is at the same depth as the reference. Under an
        oblique view the error grows silently; that is why the `model` field is
        marked "similarity" and the measurement layer can catch and warn about
        the tilt (through the rectangularity deviation).

        To actually correct perspective you need four points: the corners of a
        rectangular reference (card, A4) or an ArUco marker — see `fit`.
        """
        p1 = np.asarray(p1_px, dtype=float).reshape(2)
        p2 = np.asarray(p2_px, dtype=float).reshape(2)
        if length_mm <= 0:
            raise ValueError("The reference length must be positive.")
        v = p2 - p1
        length_px = float(np.hypot(*v))
        if length_px < 1e-6:
            raise ValueError("The two ends of the reference coincide; they must be apart.")

        direction = v / length_px
        normal = np.array([-direction[1], direction[0]])
        middle = (p1 + p2) / 2.0
        half = length_px / 2.0

        # A synthetic square: its corners are placed so that the two observed
        # points become the midpoints of opposite edges. That makes the four-point
        # correspondence exactly a similarity transform, and `reference_box`
        # really does sit around the reference.
        image = np.array([
            middle - half * direction - half * normal,
            middle + half * direction - half * normal,
            middle + half * direction + half * normal,
            middle - half * direction + half * normal,
        ])
        half_mm = length_mm / 2.0
        world = np.array([
            [-half_mm, -half_mm], [half_mm, -half_mm],
            [half_mm, half_mm], [-half_mm, half_mm],
        ])
        h = cls.fit(world, image, refine=False)
        h.model = "similarity"
        return h

    # -------------------------------------------------------------- transforms
    def to_world(self, image_points) -> np.ndarray:
        """Pixels -> mm (on the plane)."""
        return _apply(self.H_inv, image_points)

    def to_image(self, world_points) -> np.ndarray:
        """mm -> pixels."""
        return _apply(self.H, world_points)

    # -------------------------------------------------------------- quality
    def scale_mm_px(self, image_point) -> float:
        """
        Local scale (mm / pixel) around the given pixel.

        Because the homography is projective the scale is not constant across the
        image — a distant pixel covers more millimeters. This function returns the
        geometric mean of the singular values of the local Jacobian.
        """
        point = np.asarray(image_point, dtype=float).reshape(2)
        h = 0.5
        center = self.to_world(point)[0]
        dx = self.to_world(point + [h, 0])[0] - center
        dy = self.to_world(point + [0, h])[0] - center
        J = np.column_stack([dx / h, dy / h])
        return float(np.sqrt(abs(np.linalg.det(J))))

    def off_plane_warning(self, image_point) -> float:
        """
        How many box-widths outside the reference box the measured point falls.

        0 means the point is inside the reference; 1 means one box width outside.
        Empirical threshold: above 2, don't trust the measurement — move the
        reference closer to what you are measuring.
        """
        d = self.to_world(image_point)[0]
        x0, y0, x1, y1 = self.reference_box
        width = max(x1 - x0, 1e-9)
        height = max(y1 - y0, 1e-9)
        dx = max(x0 - d[0], d[0] - x1, 0.0) / width
        dy = max(y0 - d[1], d[1] - y1, 0.0) / height
        return float(np.hypot(dx, dy))
