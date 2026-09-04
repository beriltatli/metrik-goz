"""
Coverage test — the exam for this package's real claim.

If the system says "95% confidence interval", then across many independent
measurements the true value must really land inside that interval 95% of the
time. If it does not, the error bar is decoration.

This test works on synthetic scenes because a real photo has no right answer. We
place the camera, the reference and the distance to be measured; the system gets
nothing but noisy pixels.
"""

import numpy as np
import pytest

from metrik_goz import Homography, measure, uncertainty
from metrik_goz.synthetic import add_noise, build_scene

SIGMA_PX = 0.5


def _distance_fn(h, n):
    return measure.distance(h, n[0], n[1])


def _trial(rng, *, distance_mm, tilt, ref_mm=100.0, distance_factor=1.0, mc_n=120):
    """One synthetic measurement; returns the true value and the intervals produced."""
    scene = build_scene(reference_size_mm=ref_mm, distance_mm=distance_mm,
                        tilt_deg=tilt, azimuth_deg=float(rng.uniform(0, 360)))

    r = distance_factor * ref_mm
    angle = rng.uniform(0, 2 * np.pi)
    center = r * np.array([np.cos(angle), np.sin(angle)])
    direction = rng.uniform(0, 2 * np.pi)
    span = rng.uniform(0.5, 2.0) * ref_mm
    direction_vec = np.array([np.cos(direction), np.sin(direction)])
    a, b = center - 0.5 * span * direction_vec, center + 0.5 * span * direction_vec

    a_px, b_px = scene.project(a)[0], scene.project(b)[0]
    if not (scene.is_visible([a_px, b_px]) and scene.is_visible(scene.reference_px)):
        return None

    ref_noisy = add_noise(scene.reference_px, SIGMA_PX, rng)
    points_noisy = add_noise(np.array([a_px, b_px]), SIGMA_PX, rng)
    truth = float(np.hypot(*(b - a)))

    mc = uncertainty.monte_carlo(scene.reference_world, ref_noisy, _distance_fn, points_noisy,
                                 sigma_px=SIGMA_PX, n=mc_n,
                                 seed=int(rng.integers(1 << 30)))
    h = Homography.fit(scene.reference_world, ref_noisy)
    an = uncertainty.analytic(h, scene.reference_world, ref_noisy, _distance_fn, points_noisy,
                              sigma_px=SIGMA_PX)
    return truth, mc, an


def _run(seed, n=120, **kw):
    rng = np.random.default_rng(seed)
    trials = [d for d in (_trial(rng, **kw) for _ in range(n)) if d]
    assert len(trials) > n // 2, "most of the scenes did not fit in the frame"
    return trials


# --------------------------------------------------------------- the real tests
def test_monte_carlo_coverage_95_percent():
    """
    The 95% interval must really hold around 95% of the time.

    Across six independent seeds the measured coverage is 94.4%. The one-point
    shortfall from the nominal 95% is real and explainable: parametric bootstrap
    centers the distribution not on the true corners but on the OBSERVED (already
    noisy) ones. The width of the interval is right, its center drifts a little.
    We write that down instead of hiding it; better than saying 95% and
    delivering 80%.
    """
    trials = _run(11, distance_mm=1200, tilt=25.0)
    coverage = np.mean([mc.low <= t <= mc.high for t, mc, _ in trials])
    assert 0.88 <= coverage <= 0.99, f"coverage {coverage:.3f}, should be around 94%"


def test_analytic_coverage_95_percent():
    trials = _run(12, distance_mm=1200, tilt=25.0)
    coverage = np.mean([an.low <= t <= an.high for t, _, an in trials])
    assert 0.88 <= coverage <= 0.99, f"coverage {coverage:.3f}, should be around 94%"


def test_analytic_agrees_with_monte_carlo():
    """
    First-order propagation must give the same magnitude as the assumption-free
    Monte Carlo. If they diverge, the analytic route is invalid in this operating
    envelope.
    """
    trials = _run(13, n=50, distance_mm=1200, tilt=25.0)
    ratios = np.array([an.std / max(mc.std, 1e-9) for _, mc, an in trials])
    assert 0.80 <= np.median(ratios) <= 1.25, f"AN/MC std ratio {np.median(ratios):.2f}"


@pytest.mark.parametrize("tilt", [0.0, 25.0, 50.0])
def test_coverage_independent_of_viewing_angle(tilt):
    """As the view flattens the error grows, but the interval must still hold."""
    trials = _run(20 + int(tilt), n=90, distance_mm=1200, tilt=tilt)
    coverage = np.mean([mc.low <= t <= mc.high for t, mc, _ in trials])
    assert 0.86 <= coverage <= 0.99, f"tilt {tilt}: coverage {coverage:.3f}"


def test_error_under_3_percent_in_operating_envelope():
    """
    The declared operating envelope: 100 mm reference, 0.5 px corner noise, ≤ 2 m
    distance, ≤ 2× the reference size away. Inside it the median relative error
    must stay under 3%.
    """
    trials = _run(31, n=70, distance_mm=1200, tilt=25.0, distance_factor=1.0)
    errors = np.array([abs(mc.value - t) / t for t, mc, _ in trials])
    assert np.median(errors) < 0.03, f"median error {np.median(errors) * 100:.2f}%"


def test_uncertainty_grows_with_distance():
    """
    As you move away from the reference the system must SAY it is less sure.
    Being silently wrong is far worse than a wider error bar.
    """
    near = _run(41, n=40, distance_mm=1200, tilt=25.0, distance_factor=0.5)
    far = _run(42, n=40, distance_mm=1200, tilt=25.0, distance_factor=4.0)

    n = np.median([mc.std / t for t, mc, _ in near])
    f = np.median([mc.std / t for t, mc, _ in far])
    assert f > 1.5 * n, f"relative uncertainty did not grow with distance ({n:.4f} -> {f:.4f})"


# --------------------------------------------- the similarity model's declared limit
def test_similarity_perfect_straight_down():
    """
    A scale built from a single known length must produce no bias in a
    straight-down shot, wherever the reference sits — and the interval must hold
    there too. This is the exam for the panel's sentence "if the photo was taken
    directly above the object the result is correct".
    """
    from metrik_goz.validation import similarity_sweep

    sweep = similarity_sweep(n=15, seed=7)
    assert sweep["top_down_p90_bias"] < 1e-6
    assert sweep["top_down_coverage"] >= 0.88


def test_similarity_biased_under_tilt_and_coverage_collapses():
    """
    Under a tilted shot the model is systematically wrong and that bias is NOT
    INSIDE the error bar — coverage collapses. This is not a defect but a
    declared limit; the test's job is to verify the limit is still there.
    """
    from metrik_goz.validation import similarity_sweep

    sweep = similarity_sweep(n=15, seed=7)
    assert sweep["tilted_median_bias"] > 0.05
    assert sweep["tilted_coverage"] < 0.6


def test_rectangularity_warning_catches_serious_bias():
    """
    The user does not know the camera's tilt; the only observable sign the system
    has is the divergence of opposite edges. If the warning threshold does not
    catch most of the serious bias, the warning is decoration.
    """
    from metrik_goz.validation import similarity_sweep

    sweep = similarity_sweep(n=15, seed=7)
    assert sweep["catch_rate"] > 0.6, f"catch rate {sweep['catch_rate']:.2f}"
    assert sweep["false_alarm"] < 0.35, f"false alarm {sweep['false_alarm']:.2f}"
