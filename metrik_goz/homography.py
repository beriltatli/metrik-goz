"""
Plane homography: the map between image pixels and the world plane in mm.

The assumption, and the sharpest limit of this whole package: whatever you
measure has to lie on the same plane as the reference. Measure the tomato on the
counter with the card you put on the counter; you cannot measure the box on the
shelf with it. When that assumption breaks the error grows without any sign of
it, so `Homography.off_plane_warning` at least reports how far out you are.

Fitting happens in two stages. DLT gives a closed-form starting point (with
Hartley normalization), then LM refines it against the geometric reprojection
error. Two stages because DLT minimizes the algebraic error, which under noise
is not the geometrically best answer, and every error claim downstream assumes
we minimized the geometric one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .lm import solve


def _normalization_matrix(points: np.ndarray) -> np.ndarray:
    """
    Hartley normalization: centroid to the origin, mean distance to sqrt(2).

    Skip this and DLT falls apart numerically, because the design matrix ends up
    full of squared pixel coordinates on the order of 1e6.
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


# How small we let w (the projective scale) get before clamping it.
_W_FLOOR = 1e-12


def _homogeneous(points: np.ndarray) -> np.ndarray:
    return np.hstack([points, np.ones((len(points), 1))])


def _apply(H: np.ndarray, points: np.ndarray) -> np.ndarray:
    """
    Projective transform, (N,2) -> (N,2).

    w near zero means the point is on the horizon line and there is no sensible
    answer for it. We clamp w to keep the division finite, but we keep its sign:
    lose that and the point quietly lands on the wrong half of the plane.
    """
    points = np.atleast_2d(np.asarray(points, dtype=float))
    hn = _homogeneous(points) @ H.T
    w = hn[:, 2:3]
    small = np.abs(w) < _W_FLOOR
    if small.any():
        w = np.where(small, np.where(w < 0.0, -_W_FLOOR, _W_FLOOR), w)
    return hn[:, :2] / w


def dlt(world: np.ndarray, image: np.ndarray) -> np.ndarray:
    """
    Direct linear transform, world (mm) -> image (px).

    Needs at least 4 points, no three of them collinear.
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


def _residuals(p: np.ndarray, world: np.ndarray, image: np.ndarray) -> np.ndarray:
    """Reprojection error in pixels, flattened."""
    H = np.append(p, 1.0).reshape(3, 3)
    return (_apply(H, world) - image).ravel()


def _jacobian(p: np.ndarray, world: np.ndarray, image: np.ndarray) -> np.ndarray:
    """
    Derivative of the residuals w.r.t. the 8 free parameters, done on paper:

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
    # du rows sit at the even indices, dv rows at the odd ones
    J[0::2, 0] = X / w
    J[0::2, 1] = Y / w
    J[0::2, 2] = 1.0 / w
    J[0::2, 6] = -a * X / w ** 2
    J[0::2, 7] = -a * Y / w ** 2
    J[1::2, 3] = X / w
    J[1::2, 4] = Y / w
    J[1::2, 5] = 1.0 / w
    J[1::2, 6] = -b * X / w ** 2
    J[1::2, 7] = -b * Y / w ** 2
    return J


@dataclass
class Homography:
    """
    World plane (mm) <-> image (px), together with how good the fit was.

    H is world -> image and H_inv the other way round, which is the direction
    measurement actually uses. `reference_box` holds the bounds of the reference
    in world coordinates; the extrapolation warning is measured against it.
    """

    H: np.ndarray
    H_inv: np.ndarray
    rms_px: float
    covariance: np.ndarray | None
    reference_box: tuple[float, float, float, float]
    converged: bool
    model: str = "projective"     # or "similarity", see from_length below

    @classmethod
    def fit(cls, world_mm, image_px, *, refine: bool = True,
            covariance: bool = True) -> "Homography":
        """
        Pass covariance=False to skip LM's post-solve covariance estimate. Monte
        Carlo builds thousands of these and never looks at it, and skipping saves
        a Jacobian plus a matrix inverse per sample.
        """
        world = np.asarray(world_mm, dtype=float)
        image = np.asarray(image_px, dtype=float)

        H = dlt(world, image)
        rms = float(np.sqrt(np.mean((_apply(H, world) - image) ** 2)))
        cov = None
        converged = True

        # 4 points against 8 parameters leaves no degrees of freedom, so there is
        # nothing for LM to refine: DLT already fits those exactly.
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

    @classmethod
    def from_length(cls, p1_px, p2_px, length_mm: float) -> "Homography":
        """
        Similarity homography from one known length: the diameter of a coin, the
        long edge of a card. Scale, rotation, translation, nothing else.

        Two points and a length carry three numbers; a projective transform has
        eight degrees of freedom. Rather than inventing the five we don't have,
        we fit a narrower model. Perspective is not corrected here, only scale.

        That makes it valid when the camera looks straight down at the plane and
        the thing you measure sits at the same depth as the reference. Off that,
        the error grows with nothing to show for it, which is why `model` is
        tagged "similarity": the measurement layer can then spot the tilt through
        the rectangularity deviation and warn.

        For actual perspective correction you need four points, so a rectangular
        reference (card, A4) or an ArUco marker. See `fit`.
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

        # Build a square whose corners put the two observed points at the
        # midpoints of opposite edges. The four-point correspondence is then
        # exactly a similarity, and reference_box really does sit on the
        # reference rather than somewhere arbitrary.
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

    def to_world(self, image_points) -> np.ndarray:
        """Pixels to mm on the plane."""
        return _apply(self.H_inv, image_points)

    def to_image(self, world_points) -> np.ndarray:
        return _apply(self.H, world_points)

    def scale_mm_px(self, image_point) -> float:
        """
        Local mm-per-pixel around a given pixel.

        A projective map has no single scale: a pixel further away covers more
        millimeters. This returns the geometric mean of the singular values of
        the local Jacobian.
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
        How many reference-box widths outside the reference the point falls. 0 is
        inside, 1 is one box width out. Past about 2 the measurement isn't worth
        much; move the reference closer to whatever you're measuring.
        """
        d = self.to_world(image_point)[0]
        x0, y0, x1, y1 = self.reference_box
        width = max(x1 - x0, 1e-9)
        height = max(y1 - y0, 1e-9)
        dx = max(x0 - d[0], d[0] - x1, 0.0) / width
        dy = max(y0 - d[1], d[1] - y1, 0.0) / height
        return float(np.hypot(dx, dy))
