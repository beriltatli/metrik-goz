"""Correctness of the homography fit and the measurement layer."""

import numpy as np
import pytest

from metrik_goz import Homography, measure
from metrik_goz.synthetic import add_noise, build_scene


def test_exact_without_noise():
    """With no noise the measurement should give the true value almost exactly."""
    scene = build_scene(tilt_deg=25.0)
    h = Homography.fit(scene.reference_world, scene.reference_px)

    a_world, b_world = np.array([-150.0, 60.0]), np.array([220.0, -40.0])
    truth = float(np.hypot(*(b_world - a_world)))
    measured = measure.distance(h, scene.project(a_world)[0], scene.project(b_world)[0])

    assert abs(measured - truth) / truth < 1e-9


def test_area_correct():
    scene = build_scene(tilt_deg=35.0)
    h = Homography.fit(scene.reference_world, scene.reference_px)

    square = np.array([[-80.0, -80.0], [80.0, -80.0], [80.0, 80.0], [-80.0, 80.0]])
    truth = 160.0 ** 2
    measured = measure.area(h, scene.project(square))

    assert abs(measured - truth) / truth < 1e-9


def test_lm_better_than_dlt():
    """Under noise the LM refinement should lower the reprojection error."""
    rng = np.random.default_rng(3)
    scene = build_scene(tilt_deg=35.0)

    # 5+ points: give LM a degree of freedom to refine
    extra_world = np.array([[-200.0, 150.0], [180.0, -170.0], [40.0, 210.0]])
    world = np.vstack([scene.reference_world, extra_world])
    image = add_noise(scene.project(world), 1.0, rng)

    dlt_only = Homography.fit(world, image, refine=False)
    with_lm = Homography.fit(world, image, refine=True)

    assert with_lm.rms_px <= dlt_only.rms_px + 1e-12
    assert with_lm.converged


def test_fewer_than_four_points_rejected():
    with pytest.raises(ValueError):
        Homography.fit(np.zeros((3, 2)), np.zeros((3, 2)))


def test_extrapolation_warning_grows():
    """The warning value should grow as you move away from the reference."""
    scene = build_scene(reference_size_mm=100.0, tilt_deg=20.0)
    h = Homography.fit(scene.reference_world, scene.reference_px)

    near = h.off_plane_warning(scene.project([[30.0, 0.0]])[0])
    far = h.off_plane_warning(scene.project([[400.0, 0.0]])[0])

    assert near == pytest.approx(0.0, abs=1e-6)
    assert far > 3.0


def test_local_scale_grows_with_distance():
    """
    Under an oblique view a distant pixel covers more millimeters.

    The sampling has to run along the depth direction: at the default azimuth
    `build_scene` puts the camera on the +X side, so depth varies with X, not
    with Y. Sampling along Y leaves both points at the same depth and the scales
    come out equal to within 1e-13 — what the test measures disappears.
    """
    scene = build_scene(tilt_deg=55.0)
    h = Homography.fit(scene.reference_world, scene.reference_px)

    near_px = scene.project([[200.0, 0.0]])[0]     # the side close to the camera
    far_px = scene.project([[-400.0, 0.0]])[0]     # the far side
    assert scene.is_visible([near_px, far_px])

    assert h.scale_mm_px(far_px) > 1.5 * h.scale_mm_px(near_px)


def test_narrowest_passage_finds_known_corridor():
    """
    We build a corridor 300 mm wide on the world plane that narrows to 120 mm in
    the middle, and expect the system to find the narrowing.
    """
    scene = build_scene(tilt_deg=15.0, distance_mm=2500.0, reference_size_mm=200.0)
    height, width = scene.size_px

    # Define the corridor in world coordinates, paint it into a pixel mask
    yy, xx = np.mgrid[0:height, 0:width]
    px = np.column_stack([xx.ravel().astype(float), yy.ravel().astype(float)])
    h_true_inv = np.linalg.inv(scene.H_true)
    hn = np.hstack([px, np.ones((len(px), 1))]) @ h_true_inv.T
    world = hn[:, :2] / hn[:, 2:3]
    X, Y = world[:, 0], world[:, 1]

    half = np.where(np.abs(X) < 250.0, 60.0, 150.0)     # 120 mm in the middle, 300 mm outside
    free = (np.abs(Y) < half) & (np.abs(X) < 900.0)
    mask = free.reshape(height, width)

    h = Homography.fit(scene.reference_world, scene.reference_px)
    passage = measure.narrowest_passage(h, mask, axis=[1.0, 0.0], step_mm=25.0, sample_mm=4.0)

    assert abs(passage.width_mm - 120.0) < 12.0
    assert passage.fits(100.0)
    assert not passage.fits(150.0)


# ----------------------------------------------------------------- similarity model
def test_from_length_exact_looking_straight_down():
    """
    Looking straight down (nadir) the scale is the same everywhere on the plane,
    so a similarity model built from a single known length must be PERFECT there
    — how far from the reference you measure should not matter. That is the
    model's validity claim.
    """
    scene = build_scene(reference_size_mm=26.15, tilt_deg=0.0, distance_mm=520.0)
    ends = scene.project([[-13.075, 0.0], [13.075, 0.0]])
    h = Homography.from_length(ends[0], ends[1], 26.15)
    assert h.model == "similarity"

    for offset in (0.0, 100.0, 250.0):
        a, b = np.array([offset, -60.0]), np.array([offset + 90.0, 40.0])
        truth = float(np.hypot(*(b - a)))
        measured = measure.distance(h, scene.project(a)[0], scene.project(b)[0])
        assert abs(measured - truth) / truth < 1e-9


def test_from_length_wrong_under_tilt():
    """
    The same model must go wrong under a tilted view — and that is not a defect,
    it is a declared limit. Silently being taken for correct is exactly the
    dangerous case.
    """
    scene = build_scene(reference_size_mm=26.15, tilt_deg=35.0, distance_mm=520.0)
    ends = scene.project([[-13.075, 0.0], [13.075, 0.0]])
    h = Homography.from_length(ends[0], ends[1], 26.15)

    a, b = np.array([200.0, 0.0]), np.array([320.0, 0.0])
    truth = float(np.hypot(*(b - a)))
    measured = measure.distance(h, scene.project(a)[0], scene.project(b)[0])
    assert abs(measured - truth) / truth > 0.05


def test_from_length_rejects_degenerate_input():
    with pytest.raises(ValueError):
        Homography.from_length([10.0, 10.0], [10.0, 10.0], 26.15)
    with pytest.raises(ValueError):
        Homography.from_length([0.0, 0.0], [50.0, 0.0], 0.0)


# ----------------------------------------------------------------- box measurement
def test_box_edges_and_area_correct():
    scene = build_scene(tilt_deg=30.0)
    h = Homography.fit(scene.reference_world, scene.reference_px)

    width, height = 146.7, 71.5
    corners = np.array([[-width / 2, -height / 2], [width / 2, -height / 2],
                        [width / 2, height / 2], [-width / 2, height / 2]])
    b = measure.box(h, scene.project(corners))

    assert b.width_mm == pytest.approx(width, rel=1e-9)
    assert b.height_mm == pytest.approx(height, rel=1e-9)
    assert b.area_mm2 == pytest.approx(width * height, rel=1e-9)
    assert b.diagonal_mm == pytest.approx(np.hypot(width, height), rel=1e-9)
    # The projective model corrects perspective, so the deviation must be zero.
    assert b.rectangularity < 1e-9


def test_box_rectangularity_reveals_tilt():
    """
    `rectangularity` is the observable proxy for a tilt the user does not know
    about: under the scale model an oblique view skews the object and the
    opposite edges diverge. The warning layer rests on this number, so it must
    grow with tilt.
    """
    width, height = 146.7, 71.5
    corners = np.array([[-width / 2, -height / 2], [width / 2, -height / 2],
                        [width / 2, height / 2],
                        [-width / 2, height / 2]]) + np.array([150.0, 0.0])

    deviations = []
    for tilt in (0.0, 15.0, 35.0):
        scene = build_scene(reference_size_mm=26.15, tilt_deg=tilt, distance_mm=520.0)
        ends = scene.project([[-13.075, 0.0], [13.075, 0.0]])
        h = Homography.from_length(ends[0], ends[1], 26.15)
        deviations.append(measure.box(h, scene.project(corners)).rectangularity)

    assert deviations[0] < 1e-9
    assert deviations[0] < deviations[1] < deviations[2]


def test_box_requires_four_corners():
    scene = build_scene()
    h = Homography.fit(scene.reference_world, scene.reference_px)
    with pytest.raises(ValueError):
        measure.box(h, scene.project([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]]))
