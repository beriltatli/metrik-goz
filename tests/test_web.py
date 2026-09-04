"""
The web panel's back end.

The real question here is not "did the endpoint return 200" — it is that the
number the panel produces is IDENTICAL to the number the library produces. We
call the same core from two different places; if they ever diverge, the user
cannot know which one to trust.
"""

import io

import numpy as np
import pytest

flask = pytest.importorskip("flask")
cv2 = pytest.importorskip("cv2")

from metrik_goz import Homography, measure, reference, uncertainty
from metrik_goz import sample as sample_module
from metrik_goz.sample import sample_scene
from metrik_goz.web.server import Store, create_app

MANUAL = reference.TYPICAL_SIGMA_PX["manual"]


@pytest.fixture(scope="module")
def client():
    return create_app().test_client()


@pytest.fixture(scope="module")
def table(client):
    """The panel's main scenario: a phone on a table with a 1 TL coin. Straight-down shot."""
    return client.post("/api/sample", json={"name": "flat"}).get_json()


@pytest.fixture(scope="module")
def passage_scene(client):
    """The ArUco scene: for the tests that exercise the projective reference family."""
    scene = client.post("/api/sample", json={"name": "passage"}).get_json()
    aruco = client.post("/api/aruco", json={"image_id": scene["id"],
                                            "edge_mm": 200}).get_json()
    return scene, aruco


def _scale_ref(scene):
    """The similarity reference built from a single known length (the panel's default)."""
    return {"type": "scale", "name": scene["reference"]["name"],
            "points": scene["hint"]["reference"]}


def _aruco_ref(aruco, edge=200):
    return {"type": "aruco", "edge_mm": edge, "corners": aruco["corners"]}


def _fit_fn(length_mm):
    return lambda g: Homography.from_length(g[0], g[1], length_mm)


# ------------------------------------------------------------------ basics
def test_panel_opens(client):
    response = client.get("/")
    assert response.status_code == 200
    body = response.get_data(as_text=True)
    assert 'id="canvas"' in body and "panel.js" in body


def test_panel_only_offers_samples_it_can_drive(client):
    """
    The panel drives a single flow (the size of an object); the passage scene does
    not fit it.

    If we printed its button anyway the user would land on a scene where nothing
    works — so the scene list and the list the panel can drive are kept separate.
    """
    body = client.get("/").get_data(as_text=True)
    for name in sample_module.PANEL_SCENES:
        assert f'data-sample="{name}"' in body
    assert 'data-sample="passage"' not in body
    assert "passage" in sample_module.SCENES          # it stays in the API


def test_upload_and_serve_image(client):
    image = np.full((240, 320, 3), 200, np.uint8)
    data = cv2.imencode(".png", image)[1].tobytes()
    response = client.post("/api/image",
                           data={"file": (io.BytesIO(data), "square.png")})
    assert response.status_code == 200
    summary = response.get_json()
    assert (summary["width"], summary["height"]) == (320, 240)

    served = client.get(summary["url"])
    assert served.status_code == 200
    # The grid the server measures on and the one sent to the browser must be the
    # same size; if they diverge, where the user clicks and where we measure differ.
    back = cv2.imdecode(np.frombuffer(served.data, np.uint8), cv2.IMREAD_COLOR)
    assert back.shape[:2] == (240, 320)


def test_store_drops_the_oldest():
    from metrik_goz.web.server import RequestError, StoredImage

    store = Store(capacity=2)
    for i in range(3):
        store.add(StoredImage(id=str(i), array=np.zeros((4, 4, 3), np.uint8),
                              data=b"", mime="image/png", name=f"{i}.png"))
    assert store.get("2").id == "2"
    with pytest.raises(RequestError):
        store.get("0")


# ------------------------------------------------------------------ measurement
def test_panel_and_library_give_the_same_number(client, table):
    """The value the panel returns is what the library returns for the same input."""
    corners = table["hint"]["box"]
    response = client.post("/api/measure", json={
        "image_id": table["id"], "reference": _scale_ref(table),
        "measurement": {"type": "box", "points": corners},
        "sigma_px": MANUAL, "mc_n": 200, "confidence": 0.95,
    }).get_json()

    length_mm = reference.KNOWN_LENGTHS[table["reference"]["name"]][0]
    image = np.asarray(table["hint"]["reference"], float)
    expected = uncertainty.monte_carlo(
        None, image, lambda h, n: measure.box(h, n).width_mm,
        np.asarray(corners, float), sigma_px=MANUAL, n=200,
        fit_fn=_fit_fn(length_mm))

    assert response["measurement"]["name"] == "width"
    assert response["measurement"]["value"] == pytest.approx(expected.value, rel=1e-12)
    assert response["measurement"]["low"] == pytest.approx(expected.low, rel=1e-12)


def test_measurement_catches_the_true_value(client, table):
    """In a straight-down scene the right answer must lie inside the confidence interval."""
    response = client.post("/api/measure", json={
        "image_id": table["id"], "reference": _scale_ref(table),
        "measurement": {"type": "box", "points": table["hint"]["box"]},
        "mc_n": 400,
    }).get_json()

    measurements = {m["name"]: m for m in response["measurements"]}
    assert set(measurements) == {"width", "height", "area"}
    for name, m in measurements.items():
        truth = table["truth"][name]
        assert m["unit"] == truth["unit"]
        assert m["low"] <= truth["value"] <= m["high"], \
            f"{name}: {truth['value']} ∉ [{m['low']}, {m['high']}]"


def test_perspective_warning_in_a_tilted_shot(client):
    """
    The similarity model's only real weakness is perspective, and the systematic
    bias is NOT INSIDE the error bar. The system must not swallow that silently:
    in a tilted scene the measurement misses the true value, so a high-level
    warning is mandatory.
    """
    scene = client.post("/api/sample", json={"name": "tilted"}).get_json()
    response = client.post("/api/measure", json={
        "image_id": scene["id"], "reference": _scale_ref(scene),
        "measurement": {"type": "box", "points": scene["hint"]["box"]},
        "mc_n": 200,
    }).get_json()

    width = response["measurements"][0]
    truth = scene["truth"]["width"]["value"]
    assert not (width["low"] <= truth <= width["high"]), \
        "the interval must not hold for the tilted scene"

    high = [w["text"] for w in response["warnings"] if w["level"] == "high"]
    assert high and "perspective" in " ".join(high).lower()
    assert response["box"]["rectangularity"] > 0.06


def test_monte_carlo_instead_of_analytic_in_the_similarity_model(client, table):
    """
    Analytic propagation rests on the covariance of the eight projective
    parameters; in the similarity model four of them are not even free. Rather
    than silently producing a wrong number it must fall back and say so.
    """
    response = client.post("/api/measure", json={
        "image_id": table["id"], "reference": _scale_ref(table),
        "measurement": {"type": "box", "points": table["hint"]["box"]},
        "method": "analytic", "mc_n": 100,
    }).get_json()

    assert response["measurement"]["method"].startswith("monte_carlo")
    assert any("Monte Carlo" in w["text"] for w in response["warnings"])


def test_passage_verdict_uses_the_lower_end(client, passage_scene):
    """
    The verdict must look at the LOWER end of the interval, not the point
    estimate: entering a route you cannot pass costs more than missing one you
    could.
    """
    scene, aruco = passage_scene
    common = {"image_id": scene["id"], "reference": _aruco_ref(aruco), "mc_n": 60}

    response = client.post("/api/measure", json={**common, "measurement": {
        "type": "passage", "points": scene["hint"]["passage"], "footprint_mm": 480,
    }}).get_json()
    passage = response["passage"]
    assert passage["verdict"]["fits"] is True
    assert response["measurement"]["low"] <= 520.0 <= response["measurement"]["high"]
    assert len(passage["profile_mm"]) == len(passage["stations_mm"])
    assert passage["edge_margin_mm"] > 0

    # A footprint between the lower end and the point estimate: the verdict must be no.
    between = (response["measurement"]["low"] + response["measurement"]["value"]) / 2.0
    borderline = client.post("/api/measure", json={**common, "measurement": {
        "type": "passage", "points": scene["hint"]["passage"], "footprint_mm": between,
    }}).get_json()["passage"]["verdict"]
    assert borderline["point_estimate_fits"] is True
    assert borderline["fits"] is False


def test_distant_measurement_raises_a_warning(client, passage_scene):
    """
    Moving away from a projective reference, the system must say it is degrading.

    Distance risk is the projective family's weakness: the homography is seated on
    the reference corners, and the further you go the more the parameter error is
    carried and amplified.
    """
    scene, aruco = passage_scene
    response = client.post("/api/measure", json={
        "image_id": scene["id"], "reference": _aruco_ref(aruco),
        "measurement": {"type": "distance", "points": [[20, 20], [1380, 880]]},
        "mc_n": 100,
    }).get_json()
    assert any(w["level"] == "high" for w in response["warnings"])


def test_manual_reference_gives_a_wider_interval(client, passage_scene):
    """A hand-clicked corner is noisier; the interval must be wider than ArUco's."""
    scene, aruco = passage_scene
    common = {"image_id": scene["id"], "mc_n": 200,
              "measurement": {"type": "distance", "points": [[400, 500], [1000, 520]]}}
    automatic = client.post("/api/measure",
                            json={**common, "reference": _aruco_ref(aruco)}).get_json()
    manual = client.post("/api/measure", json={**common, "reference": {
        "type": "square", "edge_mm": 200, "corners": aruco["corners"]}}).get_json()
    assert manual["measurement"]["std"] > 2 * automatic["measurement"]["std"]


# ------------------------------------------------------------------ errors
@pytest.mark.parametrize("body, expected", [
    ({"image_id": "missing"}, 404),
    ({"measurement": {"type": "box", "points": [[1, 1]]}}, 400),
    ({"measurement": {"type": "volume", "points": [[1, 1], [2, 2]]}}, 400),
    ({"sigma_px": -1}, 400),
    ({"method": "divination"}, 400),
    ({"reference": {"type": "object", "object": "missing", "corners": [[0, 0]] * 4}}, 400),
    ({"reference": {"type": "scale", "name": "missing", "points": [[0, 0], [9, 9]]}}, 400),
    ({"reference": {"type": "scale", "length_mm": 26.15, "points": [[0, 0]]}}, 400),
])
def test_bad_requests_give_an_explained_error(client, table, body, expected):
    full = {"image_id": table["id"], "reference": _scale_ref(table),
            "measurement": {"type": "box", "points": table["hint"]["box"]}, **body}
    response = client.post("/api/measure", json=full)
    assert response.status_code == expected
    assert response.get_json()["error"]


def test_unknown_sample_gives_an_explained_error(client):
    response = client.post("/api/sample", json={"name": "counter"})
    assert response.status_code == 400
    assert "flat" in response.get_json()["error"]


def test_corrupt_image_is_rejected(client):
    response = client.post("/api/image",
                           data={"file": (io.BytesIO(b"not a jpeg"), "x.jpg")})
    assert response.status_code == 400
    assert "Could not read" in response.get_json()["error"]


def test_aruco_error_on_an_image_without_a_marker(client):
    plain = cv2.imencode(".png", np.full((200, 200, 3), 128, np.uint8))[1].tobytes()
    summary = client.post("/api/image",
                          data={"file": (io.BytesIO(plain), "plain.png")}).get_json()
    response = client.post("/api/aruco", json={"image_id": summary["id"],
                                               "edge_mm": 50})
    assert response.status_code == 400
    assert "No ArUco marker" in response.get_json()["error"]


# ------------------------------------------------------------------ sample scenes
def test_marker_is_detectable_in_the_passage_scene():
    """
    The marker in the sample scene must really be detectable.

    This is more fragile than it looks: because the camera looks down at the
    plane, the world->image transform flips the orientation, and if the marker is
    pasted mirrored no detector recognizes it.
    """
    scene = sample_scene("passage")
    edge = scene["reference"]["edge_mm"]
    world, image, _ = reference.find_aruco(scene["image"], edge)
    h = Homography.fit(world, image)

    edges = [np.hypot(*(image[i] - image[(i + 1) % 4])) for i in range(4)]
    assert min(edges) > 20, "the marker looks too small to measure"
    assert h.rms_px < 1e-6


@pytest.mark.parametrize("name, max_error", [("flat", 0.03), ("tilted", 1.0)])
def test_sample_hints_hold_the_scene_claim(name, max_error):
    """
    The hint points are noise-free and the right answer is known, so the remaining
    error is the model's own. `flat` must stay under the declared 3% — `tilted`
    must not; showing that difference is the whole point of the scenes.
    """
    scene = sample_scene(name)
    ref = np.asarray(scene["hint"]["reference"], float)
    h = Homography.from_length(ref[0], ref[1], scene["reference"]["length_mm"])
    b = measure.box(h, np.asarray(scene["hint"]["box"], float))

    truth = scene["truth"]["width"]["value"]
    error = abs(b.width_mm - truth) / truth
    assert h.model == "similarity"
    if name == "flat":
        assert error < max_error
        assert b.rectangularity < 0.02
    else:
        assert error > 0.10, "the tilted scene must be clearly wrong"
        assert b.rectangularity > 0.06, "the error must be observable"
