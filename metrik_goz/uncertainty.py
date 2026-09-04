"""
Uncertainty propagation. This is the part the package actually stands on.

"412 mm" on its own isn't information. "412 ± 9 mm, 95% confidence" is.

The error comes from two independent places and both have to be counted, or the
interval comes out too narrow:

  1. pixel noise on the reference corners, which builds the wrong homography
  2. pixel noise on the points you clicked, which is the right homography read at
     the wrong spot

Most single-photo measurement code counts only the first, or neither, which is
how you end up with something labelled 95% that holds about 60% of the time.
tests/test_coverage.py exists to check we didn't do that.

Two ways to get there. Monte Carlo samples both noise sources, rebuilds the
homography and re-measures; slow, but it assumes nothing, so it's the reference.
Analytic is first-order propagation: fast, and validated against MC.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import NormalDist

import numpy as np

from .homography import Homography, _jacobian   # same Jacobian the fit uses


@dataclass
class Measurement:
    value: float
    std: float
    low: float
    high: float
    confidence: float
    method: str
    unit: str = "mm"

    def __str__(self) -> str:
        return (f"{self.value:.1f} ± {self.std:.1f} {self.unit} "
                f"({self.confidence * 100:.0f}%: {self.low:.1f}–{self.high:.1f})")

    @property
    def relative_error(self) -> float:
        return float(self.std / abs(self.value)) if self.value else float("inf")


def parameter_covariance(world_mm, image_px, H: np.ndarray, sigma_px: float) -> np.ndarray:
    """
    Covariance of the 8 homography parameters: sigma_px^2 * (J^T J)^-1.

    Note this uses the corner noise handed in from outside, not the residuals. A
    4-corner reference has 8 residuals and 8 parameters, so zero degrees of
    freedom: the residuals are near zero by construction and any uncertainty read
    off them would be meaninglessly small. Corner noise, on the other hand, is
    something you can actually measure (0.3 to 0.7 px for an ArUco corner).
    """
    world = np.asarray(world_mm, dtype=float)
    image = np.asarray(image_px, dtype=float)
    p = (H / H[2, 2]).ravel()[:8]
    J = _jacobian(p, world, image)
    JtJ = J.T @ J
    try:
        inverse = np.linalg.inv(JtJ)
    except np.linalg.LinAlgError:
        inverse = np.linalg.pinv(JtJ)
    return (sigma_px ** 2) * inverse


def analytic(
    homography: Homography,
    world_mm,
    image_px,
    measure_fn,
    points_px=None,
    *,
    sigma_px: float = 0.5,
    sigma_point_px: float | None = None,
    confidence: float = 0.95,
    unit: str = "mm",
) -> Measurement:
    """
    First-order propagation:

        var = grad_p^T Cov_p grad_p  +  sigma_point^2 * ||grad_x||^2

    measure_fn(homography, points_px) returns a float. points_px=None means the
    measurement doesn't depend on clicked points (a mask-based passage, say), so
    only the homography's own uncertainty gets carried through.
    """
    if sigma_point_px is None:
        sigma_point_px = sigma_px

    cov = parameter_covariance(world_mm, image_px, homography.H, sigma_px)
    p0 = (homography.H / homography.H[2, 2]).ravel()[:8]
    points = None if points_px is None else np.asarray(points_px, dtype=float)
    g0 = float(measure_fn(homography, points))

    grad_p = np.zeros(8)
    for i in range(8):
        h = 1e-6 * max(1.0, abs(p0[i]))
        forward, backward = p0.copy(), p0.copy()
        forward[i] += h
        backward[i] -= h
        grad_p[i] = (measure_fn(_from_params(forward, homography), points)
                     - measure_fn(_from_params(backward, homography), points)) / (2 * h)
    var = float(grad_p @ cov @ grad_p)

    if points is not None and sigma_point_px > 0:
        h = 1e-4
        squares = 0.0
        flat = points.ravel()
        for i in range(flat.size):
            forward, backward = flat.copy(), flat.copy()
            forward[i] += h
            backward[i] -= h
            derivative = (measure_fn(homography, forward.reshape(points.shape))
                          - measure_fn(homography, backward.reshape(points.shape))) / (2 * h)
            squares += derivative ** 2
        var += (sigma_point_px ** 2) * squares

    std = float(np.sqrt(max(var, 0.0)))
    z = _z_value(confidence)
    return Measurement(g0, std, g0 - z * std, g0 + z * std, confidence, "analytic", unit)


def monte_carlo(
    world_mm,
    image_px,
    measure_fn,
    points_px=None,
    *,
    sigma_px: float = 0.5,
    sigma_point_px: float | None = None,
    n: int = 400,
    confidence: float = 0.95,
    seed: int | None = 0,
    unit: str = "mm",
    fit_fn=None,
) -> Measurement:
    """
    Perturb the reference observations and the measured points, rebuild the
    homography each time, re-measure. What comes out is the actual sampling
    distribution of the measurement.

    The interval comes from percentiles rather than value ± z*std, so it stays
    honest when the distribution is skewed, which it is once you extrapolate far
    from the reference.

    fit_fn(perturbed_image_px) -> Homography lets the caller supply the reference
    model. The default is a plain four-point correspondence, but the similarity
    model from `Homography.from_length` observes only two points and derives the
    other two corners from them. Perturbing derived corners independently would
    invent noise that doesn't exist, and the model is the only thing that knows
    which numbers are real observations, so it does its own fitting.

    world_mm may be None when fit_fn is given.
    """
    if sigma_point_px is None:
        sigma_point_px = sigma_px

    if n < 2:
        raise ValueError("Monte Carlo needs at least 2 samples.")

    world = None if world_mm is None else np.asarray(world_mm, dtype=float)
    image = np.asarray(image_px, dtype=float)
    points = None if points_px is None else np.asarray(points_px, dtype=float)
    rng = np.random.default_rng(seed)

    if fit_fn is None:
        if world is None:
            raise ValueError("world_mm is required when fit_fn is not given.")
        refine = len(world) >= 5
        fit_fn = lambda observation: Homography.fit(world, observation, refine=refine,
                                                    covariance=False)

    # Draw all the noise up front instead of once per iteration. Same numbers,
    # without paying to re-enter the generator n times.
    ref_noise = rng.normal(0.0, sigma_px, size=(n, *image.shape))
    point_noise = (None if points is None or sigma_point_px <= 0
                   else rng.normal(0.0, sigma_point_px, size=(n, *points.shape)))

    samples = np.empty(n)
    succeeded = 0
    for i in range(n):
        noisy_ref = image + ref_noise[i]
        if points is None:
            noisy_points = None
        elif point_noise is None:
            noisy_points = points
        else:
            noisy_points = points + point_noise[i]
        try:
            samples[succeeded] = float(measure_fn(fit_fn(noisy_ref), noisy_points))
            succeeded += 1
        except (ValueError, np.linalg.LinAlgError):
            continue

    if succeeded < max(20, n // 10):
        raise RuntimeError("Most Monte Carlo samples failed — the homography is near degenerate.")
    samples = samples[:succeeded]

    tail = (1.0 - confidence) / 2.0
    low, high = np.percentile(samples, [100 * tail, 100 * (1 - tail)])
    return Measurement(
        value=float(np.mean(samples)),
        std=float(np.std(samples, ddof=1)),
        low=float(low),
        high=float(high),
        confidence=confidence,
        method=f"monte_carlo(n={succeeded})",
        unit=unit,
    )


def _from_params(p: np.ndarray, template: Homography) -> Homography:
    """Throwaway Homography from 8 parameters, only used while differentiating."""
    H = np.append(p, 1.0).reshape(3, 3)
    return Homography(
        H=H, H_inv=np.linalg.inv(H), rms_px=template.rms_px,
        covariance=None, reference_box=template.reference_box,
        converged=template.converged,
    )


def _z_value(confidence: float) -> float:
    """Two-sided normal critical value. The stdlib has this, no need for scipy."""
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"Confidence must be between 0 and 1, {confidence} given.")
    return float(NormalDist().inv_cdf(0.5 + confidence / 2.0))
