"""Correctness of the LM solver and the analytic Jacobian."""

import numpy as np
import pytest

from metrik_goz import lm
from metrik_goz.homography import _jacobian, _residuals
from metrik_goz.synthetic import build_scene


def test_converges_on_a_known_problem():
    """Fitting y = a*exp(b*x): LM must find the right parameters."""
    x = np.linspace(0, 2, 40)
    truth = np.array([2.5, -0.9])
    y = truth[0] * np.exp(truth[1] * x)

    result = lm.solve(lambda p: p[0] * np.exp(p[1] * x) - y, [1.0, -0.1])

    assert result.converged
    np.testing.assert_allclose(result.p, truth, rtol=1e-6)
    assert result.cost < 1e-18


def test_covariance_reasonable_on_noisy_data():
    """As the noise grows, so must the parameter uncertainty."""
    rng = np.random.default_rng(0)
    x = np.linspace(0, 2, 200)
    y0 = 2.5 * np.exp(-0.9 * x)

    stds = []
    for sigma in (0.01, 0.05):
        y = y0 + rng.normal(0, sigma, x.size)
        result = lm.solve(lambda p: p[0] * np.exp(p[1] * x) - y, [1.0, -0.1])
        stds.append(np.sqrt(np.diag(result.covariance))[0])

    assert stds[1] > stds[0] * 2


def test_analytic_jacobian_matches_numerical():
    """The homography Jacobian was derived by hand; it must match the numerical one."""
    scene = build_scene(tilt_deg=30.0)
    world = scene.reference_world
    image = scene.reference_px
    p = (scene.H_true / scene.H_true[2, 2]).ravel()[:8]

    J_analytic = _jacobian(p, world, image)
    J_numerical = lm.numerical_jacobian(lambda q: _residuals(q, world, image), p)

    scale = np.maximum(np.abs(J_numerical).max(axis=0), 1e-9)
    difference = np.abs(J_analytic - J_numerical) / scale
    assert difference.max() < 1e-5, f"largest relative difference {difference.max():.2e}"


@pytest.mark.parametrize("tilt", [0.0, 20.0, 45.0, 60.0])
def test_jacobian_consistent_at_different_angles(tilt):
    scene = build_scene(tilt_deg=tilt)
    world, image = scene.reference_world, scene.reference_px
    p = (scene.H_true / scene.H_true[2, 2]).ravel()[:8]

    J_a = _jacobian(p, world, image)
    J_n = lm.numerical_jacobian(lambda q: _residuals(q, world, image), p)
    scale = np.maximum(np.abs(J_n).max(axis=0), 1e-9)
    assert (np.abs(J_a - J_n) / scale).max() < 1e-5
