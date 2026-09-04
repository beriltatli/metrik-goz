"""
Flask server: the panel's back end.

Design decision — the server does not measure, it calls the measurement. All the
math lives in `metrik_goz.measure` and `metrik_goz.uncertainty` and is guarded by
tests. The code here is responsible for three things:

  1) Keeping the uploaded image on EXACTLY the same pixel grid as the browser.
     This matters more than it looks: phone photos carry an EXIF rotation flag
     and the browser applies it. If the server does not, the (x, y) the user
     clicks and the (x, y) in our array are different places and the measurement
     comes out silently wrong. The fix: decode the image once and re-encode it
     without EXIF, then send that copy to the browser.
  2) Validating the input — missing points, nonsensical sigma, a lost image.
  3) Returning the result, with its warnings, in a drawable form.
"""

from __future__ import annotations

import io
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field

import numpy as np
from flask import Flask, jsonify, render_template, request, send_file

from .. import __version__, measure, reference, sample, uncertainty
from ..homography import Homography

# Upload limits
MAX_UPLOAD = 32 * 1024 * 1024              # 32 MB
MAX_IMAGES = 8                             # images kept in memory (LRU)
MAX_POINTS = 400                           # per polygon
MC_RANGE = (50, 4000)
PASSAGE_MC_CAP = 120                       # a passage sample is expensive; cap MC here

MEASUREMENT_TYPES = {
    "box": dict(min=4, max=4, unit="mm"),
    "distance": dict(min=2, max=2, unit="mm"),
    "length": dict(min=2, max=MAX_POINTS, unit="mm"),
    "area": dict(min=3, max=MAX_POINTS, unit="cm²"),
    "passage": dict(min=3, max=MAX_POINTS, unit="mm"),
}


class RequestError(Exception):
    """An error caused by user input, to be returned with a 400."""

    def __init__(self, message: str, code: int = 400):
        super().__init__(message)
        self.message = message
        self.code = code


# ------------------------------------------------------------------ image store
@dataclass
class StoredImage:
    id: str
    array: np.ndarray                # BGR, the grid the server measures on
    data: bytes                      # encoded copy sent to the browser (EXIF-free)
    mime: str
    name: str
    added: float = field(default_factory=time.time)

    @property
    def width(self) -> int:
        return int(self.array.shape[1])

    @property
    def height(self) -> int:
        return int(self.array.shape[0])

    def summary(self) -> dict:
        return dict(id=self.id, name=self.name,
                    width=self.width, height=self.height,
                    url=f"/image/{self.id}")


class Store:
    """
    A small store that keeps images in memory and drops the oldest one.

    We do not write to disk: the panel is a single-user tool, and an uploaded
    photo leaving no trace on the server is both cleaner and safer. The limit is
    how many images are kept in memory; past it, the oldest one is dropped.
    """

    def __init__(self, capacity: int = MAX_IMAGES):
        self._capacity = capacity
        self._items: OrderedDict[str, StoredImage] = OrderedDict()
        self._lock = threading.Lock()

    def add(self, image: StoredImage) -> StoredImage:
        with self._lock:
            self._items[image.id] = image
            self._items.move_to_end(image.id)
            while len(self._items) > self._capacity:
                self._items.popitem(last=False)
        return image

    def get(self, image_id: str) -> StoredImage:
        with self._lock:
            image = self._items.get(image_id)
            if image is None:
                raise RequestError(
                    "The image is not on the server — reload the tab and upload it "
                    "again. (The panel keeps only the last few images in memory.)", 404)
            self._items.move_to_end(image_id)
            return image


# ------------------------------------------------------------------ helpers
def _cv2():
    try:
        import cv2
        return cv2
    except ImportError as error:                               # pragma: no cover
        raise RequestError(
            "This operation needs OpenCV: pip install 'metrik-goz[web]'", 500) from error


def _decode_image(data: bytes, name: str) -> StoredImage:
    """Decodes the bytes and prepares the EXIF-free copy for the browser."""
    cv2 = _cv2()
    array = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if array is None:
        raise RequestError("Could not read the image. It must be JPEG, PNG, WEBP or BMP.")
    if min(array.shape[:2]) < 32:
        raise RequestError("The image is too small to measure (shortest edge 32 pixels).")

    # imdecode applied the EXIF orientation; re-encoding carries no EXIF, so the
    # grid the browser sees is exactly the same as ours.
    ok, buffer = cv2.imencode(".jpg", array, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:                                                 # pragma: no cover
        raise RequestError("Could not re-encode the image.")
    return StoredImage(id=uuid.uuid4().hex, array=array, data=buffer.tobytes(),
                       mime="image/jpeg", name=name or "upload")


def _number(body: dict, key: str, default=None, *,
            minimum=None, maximum=None, required=False) -> float | None:
    raw = body.get(key, default)
    if raw is None:
        if required:
            raise RequestError(f"The '{key}' field is required.")
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        raise RequestError(f"'{key}' must be a number, '{raw}' given.") from None
    if not np.isfinite(value):
        raise RequestError(f"'{key}' must be a finite number.")
    if minimum is not None and value < minimum:
        raise RequestError(f"'{key}' must be at least {minimum:g}.")
    if maximum is not None and value > maximum:
        raise RequestError(f"'{key}' can be at most {maximum:g}.")
    return value


def _points(raw, name: str, minimum: int, maximum: int, size: tuple[int, int]) -> np.ndarray:
    if not isinstance(raw, (list, tuple)):
        raise RequestError(f"'{name}' must be a list of points.")
    if len(raw) < minimum:
        raise RequestError(f"{name} needs at least {minimum} points, {len(raw)} given.")
    if len(raw) > maximum:
        raise RequestError(f"{name} takes at most {maximum} points.")
    try:
        p = np.asarray(raw, dtype=float).reshape(len(raw), 2)
    except (ValueError, TypeError):
        raise RequestError(f"'{name}' must be of the form [[x, y], ...].") from None
    if not np.all(np.isfinite(p)):
        raise RequestError(f"'{name}' contains a non-finite coordinate.")

    # A point outside the frame is NOT geometrically invalid — the homography is
    # defined on the whole plane, and marking the edge of a corridor that does not
    # fit in the frame is ordinary. We do not reject it; we report how far outside
    # it is as a warning. The limit is only against overflow: far enough out, the
    # plane becomes numerically meaningless.
    height, width = size
    if np.abs(p[:, 0]).max() > 50 * width or np.abs(p[:, 1]).max() > 50 * height:
        raise RequestError(f"'{name}' is unbelievably far from the image.")
    return p


@dataclass
class ReferenceModel:
    """
    Which model the reference builds.

    There are two families and their difference decides how much of the
    measurement you can trust:

    similarity — a single known LENGTH (the coin's diameter). The scale is known,
                 perspective is NOT corrected. There are two observed points; the
                 remaining two corners are derived from them, which is why
                 `fit_fn` does the fitting: the model knows which numbers are
                 really observations.
    projective — four points (rectangle corners or ArUco). Perspective is
                 corrected.
    """

    type: str
    family: str
    image: np.ndarray                       # the observed points
    sigma: float
    label: str
    world: np.ndarray | None = None
    fit_fn: object | None = None

    def fit(self) -> Homography:
        if self.fit_fn is not None:
            return self.fit_fn(self.image)
        return Homography.fit(self.world, self.image)


def _resolve_reference(body: dict, image: StoredImage) -> ReferenceModel:
    ref = body.get("reference") or {}
    kind = ref.get("type", "scale")
    size = (image.height, image.width)

    if kind == "scale":
        # The simplest route: click the two ends of the coin next to the object
        # and type its diameter.
        length = ref.get("length_mm")
        name = ref.get("name")
        if length is None and name:
            if name not in reference.KNOWN_LENGTHS:
                raise RequestError(f"Unknown reference '{name}'.")
            length = reference.KNOWN_LENGTHS[name][0]
        length = _number({"length_mm": length}, "length_mm", required=True,
                         minimum=0.1, maximum=1_000_000.0)
        points = _points(ref.get("points"), "reference ends", 2, 2, size)
        label = (reference.KNOWN_LENGTHS[name][1] if name in reference.KNOWN_LENGTHS
                 else f"{length:g} mm known length")
        return ReferenceModel(
            type=kind, family="similarity", image=points,
            sigma=reference.TYPICAL_SIGMA_PX["manual"],
            label=f"{label} · {length:g} mm",
            fit_fn=lambda g, u=length: Homography.from_length(g[0], g[1], u),
        )

    if kind == "aruco":
        edge = _number(ref, "edge_mm", required=True, minimum=1.0, maximum=100_000.0)
        dictionary = str(ref.get("dictionary", "DICT_4X4_50"))
        corners = ref.get("corners")
        if corners:
            # The panel finds the marker once and keeps the corners; no need to search again.
            image_px = _points(corners, "reference corners", 4, 4, size)
            label = ref.get("label") or f"ArUco {edge:g} mm"
        else:
            _, image_px, marker_id = _find_aruco(image, edge, dictionary)
            label = f"ArUco #{marker_id} · {edge:g} mm"
        return ReferenceModel(type=kind, family="projective",
                              world=reference.square_world(edge),
                              image=image_px, sigma=reference.TYPICAL_SIGMA_PX["aruco"],
                              label=label)

    if kind == "square":
        edge = _number(ref, "edge_mm", required=True, minimum=1.0, maximum=100_000.0)
        world = reference.square_world(edge)
        label = f"Manual square · {edge:g} mm"
    elif kind == "rectangle":
        width_mm = _number(ref, "width_mm", required=True, minimum=1.0, maximum=100_000.0)
        height_mm = _number(ref, "height_mm", required=True, minimum=1.0, maximum=100_000.0)
        world = reference.rectangle_world(width_mm, height_mm)
        label = f"Manual rectangle · {width_mm:g}×{height_mm:g} mm"
    elif kind == "object":
        name = str(ref.get("object", ""))
        try:
            world = reference.known_object(name)
        except KeyError as error:
            raise RequestError(str(error)) from None
        w, h = reference.KNOWN_OBJECTS[name]
        label = f"{name} · {w:g}×{h:g} mm"
    else:
        raise RequestError(f"Unknown reference type '{kind}'.")

    image_px = _points(ref.get("corners"), "reference corners", 4, 4, size)
    return ReferenceModel(type=kind, family="projective", world=world, image=image_px,
                          sigma=reference.TYPICAL_SIGMA_PX["manual"], label=label)


def _find_aruco(image: StoredImage, edge_mm: float, dictionary: str):
    _cv2()
    try:
        return reference.find_aruco(image.array, edge_mm, dictionary=dictionary)
    except AttributeError:
        raise RequestError(f"Unknown ArUco dictionary '{dictionary}'.") from None
    except ValueError as error:
        raise RequestError(str(error)) from None


def _build_mask(points: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Turns the drawn polygon into a free-space mask."""
    cv2 = _cv2()
    height, width = size
    mask = np.zeros((height, width), np.uint8)
    clipped = np.clip(np.rint(points), -1e6, 1e6).astype(np.int32)
    cv2.fillPoly(mask, [clipped], 255)
    if mask.sum() == 0:
        raise RequestError("The polygon drawn covers an empty area.")
    return mask > 127


def _warnings(h: Homography, points: np.ndarray, ref: ReferenceModel,
              worst_relative: float, size: tuple[int, int],
              sigma_px: float, box_result=None) -> list[dict]:
    """
    The list that says where the measurement can go silently wrong, most dangerous
    first. The user will not read the whole list; they will read the first line.

    The thresholds differ per model, because the two models are weak in different
    places:

      similarity — its only weakness is perspective. The object being FAR from the
                   reference produces no error on its own; the scale is the same
                   everywhere. Distance only magnifies the error when perspective
                   is present, so we combine the two into one warning.
                   `validation.similarity_sweep` measures this: in a straight-down
                   shot the error is zero wherever the coin sits.
      projective — the homography is seated on the reference corners; the further
                   you go from them, the more the parameter error is carried and
                   amplified. There, distance itself is a risk.
    """
    warnings: list[dict] = []
    height, width = size
    farthest = max(float(h.off_plane_warning(p)) for p in points)
    ref_px = float(np.hypot(*(ref.image[1] - ref.image[0]))) if len(ref.image) >= 2 else 0.0

    # 1) Perspective — the similarity model's only real weakness and the costliest error.
    if ref.family == "similarity":
        deviation = box_result.rectangularity if box_result is not None else None
        far = farthest > 1.5
        if deviation is not None and deviation > 0.06:
            warnings.append(dict(level="high", text=(
                f"Opposite edges measure {deviation * 100:.1f}% apart — the photo was "
                f"clearly taken at an angle. A scale built from a single length does NOT "
                f"correct perspective; at this tilt the error can exceed 10% and the "
                f"error bar below does not cover it. Hold the camera straight above the "
                f"object, or use something rectangular (a card, a sheet of A4) as the "
                f"reference and mark its four corners — then I can correct perspective." +
                (" Putting the reference on top of the object also halves the error."
                 if far else ""))))
        elif deviation is not None and deviation > 0.02:
            warnings.append(dict(level="medium", text=(
                f"Opposite edges measure {deviation * 100:.1f}% apart: there is slight "
                f"perspective and the error bar does not cover it." +
                (" The reference is far from the object; putting it on top roughly halves "
                 "the error." if far else " Holding the camera a little more upright fixes it."))))
        else:
            warnings.append(dict(level="info", text=(
                "The scale was built from a single length: perspective is not corrected. "
                "Since the object's opposite edges measure equal, the photo looks upright "
                "enough — in a tilted shot this number grows and a warning appears."
                if box_result is not None else
                "The scale was built from a single length: perspective is not corrected. "
                "If the photo was taken directly above the object the result is correct.")))
    else:
        if farthest > 2.0:
            warnings.append(dict(level="high", text=(
                f"The measurement is {farthest:.1f} box widths outside the reference. The "
                f"error grows quickly in that region — move the reference closer to what "
                f"you are measuring.")))
        elif farthest > 1.0:
            warnings.append(dict(level="medium", text=(
                f"The measurement is {farthest:.1f} box widths outside the reference; "
                f"moving it closer lowers the error.")))

    # 2) The reference's size in the image — most of the uncertainty usually comes from here.
    if 0 < ref_px < 150 and ref.family == "similarity":
        warnings.append(dict(level="medium", text=(
            f"The reference is only {ref_px:.0f} pixels long in the image. The scale "
            f"error is roughly σ√2 / {ref_px:.0f} px, so most of the error bar comes "
            f"from here. A bigger reference (a card, a sheet of A4) or a closer shot "
            f"narrows the interval noticeably.")))

    # 3) Framing and fit
    outside = int(np.sum((points[:, 0] < -2) | (points[:, 0] > width + 2) |
                         (points[:, 1] < -2) | (points[:, 1] > height + 2)))
    if outside:
        warnings.append(dict(level="medium",
                             text=f"{outside} point(s) fall outside the frame."))
    if h.rms_px > 1.5:
        warnings.append(dict(level="high", text=(
            f"Reprojection error {h.rms_px:.2f} px — the reference corners may not be "
            f"seated well.")))
    if not h.converged:
        warnings.append(dict(level="high",
                             text="The homography solver did not converge; do not trust the result."))
    if worst_relative > 0.05:
        warnings.append(dict(level="medium", text=(
            f"Relative uncertainty {worst_relative * 100:.1f}% — you are outside the "
            f"declared operating envelope (3%).")))

    # 4) Limits that always apply
    warnings.append(dict(level="info", text=(
        "The ArUco corners were read at sub-pixel accuracy." if ref.type == "aruco" else
        f"The points were marked by hand; a click noise of {sigma_px:.2f} px was assumed. "
        f"Seating the corner exactly with the magnifier really does lower that number.")))
    warnings.append(dict(level="info", text=(
        "Plane assumption: what you measure must be on the same plane (the same height) "
        "as the reference. You can measure an object on the table with a coin on the "
        "table, but not one on the shelf.")))
    return warnings


def _measurement_dict(m: uncertainty.Measurement, name: str = "") -> dict:
    return dict(name=name, value=m.value, std=m.std, low=m.low, high=m.high,
                confidence=m.confidence, method=m.method, unit=m.unit,
                relative_error=m.relative_error, text=str(m))


# ------------------------------------------------------------------ measurement flow
def _run_measurement(body: dict, store: Store) -> dict:
    image = store.get(str(body.get("image_id", "")))
    size = (image.height, image.width)

    ref = _resolve_reference(body, image)
    sigma_px = _number(body, "sigma_px", ref.sigma, minimum=0.01, maximum=20.0)
    confidence = _number(body, "confidence", 0.95, minimum=0.5, maximum=0.999)
    method = str(body.get("method", "monte_carlo"))
    if method not in ("monte_carlo", "analytic"):
        raise RequestError(f"Unknown method '{method}'.")

    request_measurement = body.get("measurement") or {}
    kind = request_measurement.get("type", "box")
    if kind not in MEASUREMENT_TYPES:
        raise RequestError(f"Unknown measurement type '{kind}'.")
    rule = MEASUREMENT_TYPES[kind]
    points = _points(request_measurement.get("points"), "measurement points",
                     rule["min"], rule["max"], size)

    try:
        h = ref.fit()
    except (ValueError, np.linalg.LinAlgError) as error:
        raise RequestError(f"The reference could not be fitted: {error}") from None

    notes: list[dict] = []
    if method == "analytic" and ref.fit_fn is not None:
        # Analytic propagation rests on the covariance of the eight projective
        # parameters; in the similarity model four of them are not even free.
        method = "monte_carlo"
        notes.append(dict(level="info", text=(
            "Analytic propagation is only defined for a four-point projective "
            "reference; this measurement was done with Monte Carlo.")))

    extra: dict = {}
    box_result = None
    if kind == "passage":
        fns, passage = _prepare_passage(h, points, request_measurement, size, extra)
        mc_points = None
    elif kind == "box":
        box_result = measure.box(h, points)
        fns = [("width", lambda hh, nn: measure.box(hh, nn).width_mm, "mm"),
               ("height", lambda hh, nn: measure.box(hh, nn).height_mm, "mm"),
               ("area", lambda hh, nn: measure.box(hh, nn).area_mm2 / 100.0, "cm²")]
        mc_points, passage = points, None
    else:
        fns = {
            "distance": [("distance", lambda hh, nn: measure.distance(hh, nn[0], nn[1]), "mm")],
            "length": [("length", lambda hh, nn: measure.length(hh, nn), "mm")],
            "area": [("area", lambda hh, nn: measure.area(hh, nn) / 100.0, "cm²")],
        }[kind]
        mc_points, passage = points, None

    mc_cap = PASSAGE_MC_CAP if kind == "passage" else MC_RANGE[1]
    mc_n = int(min(_number(body, "mc_n", 400, minimum=MC_RANGE[0], maximum=MC_RANGE[1]),
                   mc_cap))

    started = time.perf_counter()
    measurements = []
    for name, fn, unit in fns:
        try:
            if method == "analytic":
                result = uncertainty.analytic(h, ref.world, ref.image, fn, mc_points,
                                              sigma_px=sigma_px, confidence=confidence,
                                              unit=unit)
            else:
                # The same seed: all three measures come from the SAME set of
                # perturbations, so "width" and "height" are measured in a world
                # that is consistent with itself.
                result = uncertainty.monte_carlo(ref.world, ref.image, fn, mc_points,
                                                 sigma_px=sigma_px, n=mc_n,
                                                 confidence=confidence, unit=unit,
                                                 fit_fn=ref.fit_fn)
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
            raise RequestError(f"The measurement failed: {error}") from None
        measurements.append(_measurement_dict(result, name))
    duration_ms = (time.perf_counter() - started) * 1000.0

    # An area's relative error is roughly twice a length's; if it set the
    # threshold the warning would fire on every measurement. We look at the
    # length measures.
    lengths = [m["relative_error"] for m in measurements if m["unit"] == "mm"]
    worst = max(lengths or [m["relative_error"] for m in measurements])
    response = dict(
        type=kind,
        measurements=measurements,
        measurement=measurements[0],            # the primary measure
        reference=dict(type=ref.type, family=ref.family, label=ref.label,
                       sigma_px=sigma_px, points=ref.image.tolist(),
                       pixel_length=(float(np.hypot(*(ref.image[1] - ref.image[0])))
                                     if len(ref.image) >= 2 else None)),
        homography=dict(rms_px=h.rms_px, model=h.model, converged=bool(h.converged),
                        scale_mm_px=float(h.scale_mm_px(points.mean(axis=0)))),
        warnings=notes + _warnings(h, points, ref, worst, size, sigma_px, box_result),
        duration_ms=duration_ms,
        **extra,
    )

    if box_result is not None:
        response["box"] = dict(
            edges_mm=box_result.edges_mm.tolist(),
            rectangularity=box_result.rectangularity,
            diagonal_mm=box_result.diagonal_mm,
        )
    if kind in ("distance", "length"):
        d = h.to_world(points)
        response["segments"] = [float(np.hypot(*(d[i + 1] - d[i])))
                                for i in range(len(d) - 1)]
    if passage is not None:
        response["passage"] = _passage_summary(h, passage, request_measurement,
                                               measurements[0])
    return response


def _prepare_passage(h: Homography, points, request_body, size, extra):
    mask = _build_mask(points, size)
    step = _number(request_body, "step_mm", 20.0, minimum=1.0, maximum=1000.0)
    sample_mm = _number(request_body, "sample_mm", 5.0, minimum=0.5, maximum=200.0)
    edge_margin = _number(request_body, "edge_margin_mm", None, minimum=0.0, maximum=100_000.0)
    try:
        passage = measure.narrowest_passage(h, mask, step_mm=step, sample_mm=sample_mm,
                                            edge_margin_mm=edge_margin)
    except ValueError as error:
        raise RequestError(str(error)) from None

    # The axis must stay fixed throughout Monte Carlo: redoing the PCA on every
    # sample would change what is measured, not the uncertainty.
    def fn(hh, _):
        return measure.narrowest_passage(hh, mask, axis=passage.axis, step_mm=step,
                                         sample_mm=sample_mm,
                                         edge_margin_mm=passage.edge_margin_mm).width_mm

    extra["mask_area_px"] = int(mask.sum())
    return [("passage", fn, "mm")], passage


def _passage_summary(h: Homography, passage: measure.Passage, request_body: dict,
                     result: dict) -> dict:
    normal = np.array([-passage.axis[1], passage.axis[0]])
    ends_mm = np.array([passage.position_mm - passage.width_mm / 2.0 * normal,
                        passage.position_mm + passage.width_mm / 2.0 * normal])
    summary = dict(
        width_mm=passage.width_mm,
        edge_margin_mm=passage.edge_margin_mm,
        axis=passage.axis.tolist(),
        stations_mm=passage.stations_mm.tolist(),
        profile_mm=passage.profile_mm.tolist(),
        narrowest_index=int(np.argmin(passage.profile_mm)),
        line_px=h.to_image(ends_mm).tolist(),
    )
    footprint = _number(request_body, "footprint_mm", None, minimum=0.0, maximum=100_000.0)
    if footprint is not None:
        clearance = _number(request_body, "clearance_mm", 0.0, minimum=0.0, maximum=100_000.0)
        required = footprint + clearance
        # The verdict is made on the LOWER end of the interval: entering a route
        # you cannot pass costs more than missing one you could. Asymmetric cost,
        # asymmetric threshold.
        summary["verdict"] = dict(
            footprint_mm=footprint, clearance_mm=clearance, required_mm=required,
            fits=bool(result["low"] >= required),
            point_estimate_fits=bool(result["value"] >= required),
            margin_mm=float(result["low"] - required),
        )
    return summary


# ------------------------------------------------------------------ application
def create_app(*, store: Store | None = None) -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD
    # Flask 3 no longer reads the old JSON_SORT_KEYS setting; sorting the keys
    # alphabetically was silently breaking the meaningful order in the response
    # (for example "distance first, then area").
    app.json.sort_keys = False
    images = store if store is not None else Store()

    @app.errorhandler(RequestError)
    def _request_error(error: RequestError):
        return jsonify(error=error.message), error.code

    @app.errorhandler(413)
    def _too_large(_):
        return jsonify(error=f"The file is too large (limit "
                             f"{MAX_UPLOAD // (1024 * 1024)} MB)."), 413

    @app.get("/")
    def panel():
        return render_template(
            "panel.html",
            version=__version__,
            lengths=[{"name": name, "mm": mm, "description": description}
                     for name, (mm, description) in reference.KNOWN_LENGTHS.items()],
            objects={name: list(size) for name, size in reference.KNOWN_OBJECTS.items()},
            sigmas=reference.TYPICAL_SIGMA_PX,
            # The panel drives a single flow (the size of an object); the passage
            # scene does not fit it, so it is not on the buttons — it stays in the
            # CLI and the API.
            samples=list(sample.PANEL_SCENES),
        )

    @app.get("/image/<image_id>")
    def serve_image(image_id: str):
        img = images.get(image_id)
        return send_file(io.BytesIO(img.data), mimetype=img.mime,
                         download_name=img.name, max_age=3600)

    @app.post("/api/image")
    def upload_image():
        uploaded = request.files.get("file")
        if uploaded is None or not uploaded.filename:
            raise RequestError("The file field is empty.")
        data = uploaded.read()
        if not data:
            raise RequestError("The file is empty.")
        return jsonify(images.add(_decode_image(data, uploaded.filename)).summary())

    @app.post("/api/sample")
    def generate_sample():
        cv2 = _cv2()
        name = str((request.get_json(silent=True) or {}).get("name", "flat"))
        try:
            scene = sample.sample_scene(name)
        except KeyError as error:
            raise RequestError(str(error)) from None
        ok, buffer = cv2.imencode(".png", scene["image"])
        if not ok:                                             # pragma: no cover
            raise RequestError("The sample scene could not be encoded.")
        img = images.add(StoredImage(id=uuid.uuid4().hex, array=scene["image"],
                                     data=buffer.tobytes(), mime="image/png",
                                     name=scene["name"]))
        return jsonify({**img.summary(),
                        "description": scene["description"],
                        "default_measurement": scene["default_measurement"],
                        "reference": scene["reference"],
                        "truth": scene["truth"],
                        "hint": scene["hint"],
                        "footprint_mm": scene.get("footprint_mm")})

    @app.post("/api/aruco")
    def search_aruco():
        body = request.get_json(silent=True) or {}
        img = images.get(str(body.get("image_id", "")))
        edge = _number(body, "edge_mm", required=True, minimum=1.0, maximum=100_000.0)
        dictionary = str(body.get("dictionary", "DICT_4X4_50"))
        _, image_px, marker_id = _find_aruco(img, edge, dictionary)
        return jsonify(corners=image_px.tolist(), marker_id=marker_id,
                       label=f"ArUco #{marker_id} · {edge:g} mm",
                       sigma_px=reference.TYPICAL_SIGMA_PX["aruco"])

    @app.post("/api/measure")
    def measure_endpoint():
        return jsonify(_run_measurement(request.get_json(silent=True) or {}, images))

    @app.get("/api/status")
    def status():
        try:
            import cv2
            cv_version = cv2.__version__
        except ImportError:                                    # pragma: no cover
            cv_version = None
        return jsonify(version=__version__, opencv=cv_version,
                       max_upload_mb=MAX_UPLOAD // (1024 * 1024))

    app.store = images          # so tests and embedded use can reach it
    return app


def run(host: str = "127.0.0.1", port: int = 8000, debug: bool = False) -> None:
    """Starts the development server."""
    app = create_app()
    print(f"metrik-goz panel:  http://{host}:{port}")
    app.run(host=host, port=port, debug=debug, threaded=True)
