"""
Command line interface.

    metrik-goz box      photo.jpg --scale-name 1_tl --end 300,410 --end 372,410 \
                                  --object-corner ... (the object's 4 corners)
    metrik-goz distance photo.jpg --aruco 100 --point 120,340 --point 610,355
    metrik-goz area     photo.jpg --aruco 100 --point ... (at least 3 points)
    metrik-goz passage  photo.jpg --aruco 100 --mask free.png --footprint 480
    metrik-goz validate --out validation/
    metrik-goz panel    --port 8000
    metrik-goz sample   --scene flat --out examples/

The reference can come from one of two families, and the difference decides how
much of the measurement you can trust: `--scale` builds a similarity model from a
single known LENGTH (perspective is not corrected), while `--aruco` / `--object`
/ `--corner` build a projective model from four points (perspective is
corrected). The panel makes the same distinction.
"""

from __future__ import annotations

import argparse
import sys
from typing import Callable, NamedTuple

import numpy as np

from .homography import Homography
from . import measure, reference, sample, uncertainty


def _point(text: str) -> tuple[float, float]:
    try:
        x, y = text.split(",")
        return float(x), float(y)
    except ValueError:
        raise argparse.ArgumentTypeError(f"A point must be given as 'x,y': '{text}'")


class Reference(NamedTuple):
    """
    The resolved reference: which points are observations, how the model is built.

    When `fit_fn` is given, the model builds the homography itself. In the
    similarity model the observation consists of two points and the remaining
    corners are derived from them; perturbing those derived corners independently
    in Monte Carlo would invent noise that does not exist.
    """

    world: np.ndarray | None
    image: np.ndarray
    sigma: float
    name: str
    fit_fn: Callable[[np.ndarray], Homography] | None = None

    def fit(self) -> Homography:
        if self.fit_fn is not None:
            return self.fit_fn(self.image)
        return Homography.fit(self.world, self.image)


def _resolve_reference(args, image) -> Reference:
    if args.aruco is not None:
        world, image_px, marker_id = reference.find_aruco(image, args.aruco)
        return Reference(world, image_px, reference.TYPICAL_SIGMA_PX["aruco"],
                         f"ArUco #{marker_id}")

    scale_mm = args.scale
    if args.scale_name is not None:
        scale_mm, description = reference.KNOWN_LENGTHS[args.scale_name]
    else:
        description = f"{scale_mm:g} mm known length" if scale_mm else ""
    if scale_mm is not None:
        if len(args.end) != 2:
            raise SystemExit("A scale reference needs exactly 2 --end values "
                             "(the two ends of the known length).")
        image_px = np.array(args.end, float)
        return Reference(None, image_px, reference.TYPICAL_SIGMA_PX["manual"], description,
                         fit_fn=lambda g, u=scale_mm: Homography.from_length(g[0], g[1], u))

    if args.object is not None:
        if len(args.corner) != 4:
            raise SystemExit("A known object needs exactly 4 --corner values.")
        world = reference.known_object(args.object)
        return Reference(world, np.array(args.corner, float),
                         reference.TYPICAL_SIGMA_PX["manual"], args.object)

    raise SystemExit("A reference is required: --aruco EDGE_MM, --scale-name NAME --end x,y ×2, "
                     "--scale LENGTH_MM --end x,y ×2 or --object NAME --corner x,y ×4")


def _read_image(path: str):
    import cv2
    img = cv2.imread(path)
    if img is None:
        raise SystemExit(f"Could not read the image: {path}")
    return img


def _measure(ref: Reference, h: Homography, fn, points, args, *, unit: str = "mm"):
    """Propagates the uncertainty using the reference model's own fitting."""
    return uncertainty.monte_carlo(
        ref.world, ref.image, fn, points,
        sigma_px=ref.sigma, n=args.mc, unit=unit, fit_fn=ref.fit_fn,
    )


def _print_warnings(h: Homography, points, *, rectangularity: float | None = None) -> None:
    if h.model == "similarity":
        # The similarity model's only real weakness is perspective, not distance.
        if rectangularity is not None and rectangularity > 0.06:
            print(f"  WARNING: opposite edges measure {rectangularity * 100:.1f}% apart "
                  f"— the photo is tilted. A scale built from a single length does NOT "
                  f"correct perspective and the error bar does not cover it.",
                  file=sys.stderr)
        else:
            print("  Note: the scale was built from a single length, perspective is not "
                  "corrected. If the photo was taken directly above the object the result "
                  "is correct.", file=sys.stderr)
    elif points is not None:
        farthest = max(h.off_plane_warning(p) for p in points)
        if farthest > 2.0:
            print(f"  WARNING: the measurement is {farthest:.1f} box widths outside the "
                  f"reference. Move the reference closer to what you are measuring.",
                  file=sys.stderr)
    if h.rms_px > 1.5:
        print(f"  WARNING: reprojection error {h.rms_px:.2f} px — the reference corners "
              f"may not be seated well.", file=sys.stderr)


def _print_header(ref: Reference, h: Homography) -> None:
    print(f"Reference: {ref.name}   model: {h.model}   "
          f"reprojection RMS: {h.rms_px:.2f} px")


def command_box(args) -> None:
    """Width, height and area of an object marked by its four corners — the panel's flow."""
    image = _read_image(args.image)
    ref = _resolve_reference(args, image)
    if len(args.object_corner) != 4:
        raise SystemExit("box needs exactly 4 --object-corner values.")

    h = ref.fit()
    points = np.array(args.object_corner, float)
    b = measure.box(h, points)

    # All three from the same seed: "width" and "height" are measured in a world
    # that is consistent with itself.
    width = _measure(ref, h, lambda hh, nn: measure.box(hh, nn).width_mm, points, args)
    height = _measure(ref, h, lambda hh, nn: measure.box(hh, nn).height_mm, points, args)
    area_ = _measure(ref, h, lambda hh, nn: measure.box(hh, nn).area_mm2 / 100.0, points,
                     args, unit="cm²")

    _print_header(ref, h)
    print(f"Width:    {width}")
    print(f"Height:   {height}")
    print(f"Area:     {area_}")
    _print_warnings(h, points, rectangularity=b.rectangularity)


def command_distance(args) -> None:
    image = _read_image(args.image)
    ref = _resolve_reference(args, image)
    if len(args.point) != 2:
        raise SystemExit("distance needs exactly 2 --point values.")

    h = ref.fit()
    points = np.array(args.point, float)
    result = _measure(ref, h, lambda hh, nn: measure.distance(hh, nn[0], nn[1]), points, args)

    _print_header(ref, h)
    print(f"Distance: {result}")
    _print_warnings(h, points)


def command_area(args) -> None:
    image = _read_image(args.image)
    ref = _resolve_reference(args, image)
    if len(args.point) < 3:
        raise SystemExit("area needs at least 3 --point values.")

    h = ref.fit()
    points = np.array(args.point, float)
    result = _measure(ref, h, lambda hh, nn: measure.area(hh, nn) / 100.0,   # mm^2 -> cm^2
                      points, args, unit="cm²")

    _print_header(ref, h)
    print(f"Area:     {result}")
    _print_warnings(h, points)


def command_passage(args) -> None:
    import cv2
    image = _read_image(args.image)
    ref = _resolve_reference(args, image)

    mask_img = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE)
    if mask_img is None:
        raise SystemExit(f"Could not read the mask: {args.mask}")
    mask = mask_img > 127

    h = ref.fit()
    passage = measure.narrowest_passage(h, mask, step_mm=args.step, sample_mm=args.sample)

    # The axis stays fixed throughout Monte Carlo: redoing the PCA on every sample
    # would change what is measured, not the uncertainty.
    result = uncertainty.monte_carlo(
        ref.world, ref.image,
        lambda hh, _: measure.narrowest_passage(hh, mask, axis=passage.axis,
                                                step_mm=args.step,
                                                sample_mm=args.sample).width_mm,
        None, sigma_px=ref.sigma, n=min(args.mc, 60), fit_fn=ref.fit_fn,
    )
    _print_header(ref, h)
    print(f"Narrowest passage: {result}")
    if args.footprint is not None:
        verdict = "FITS" if result.low >= args.footprint else "DOES NOT FIT"
        print(f"Verdict for a {args.footprint:.0f} mm footprint: {verdict}")
        print("  (the verdict is made on the LOWER end of the interval — entering a "
              "route you cannot pass costs more than missing one you could)")


def command_validate(args) -> None:
    from .validation import run_validation
    run_validation(args.out)


def command_panel(args) -> None:
    try:
        from .web import create_app
    except ImportError as error:
        raise SystemExit("The panel needs Flask:  pip install 'metrik-goz[web]'") from error
    app = create_app()
    print(f"metrik-goz panel:  http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)


def command_sample(args) -> None:
    """Writes a synthetic sample scene to disk — with the true answer as JSON beside it."""
    import json
    import pathlib

    import cv2

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    names = sorted(sample.SCENES) if args.scene == "all" else [args.scene]

    for name in names:
        scene = sample.sample_scene(name)
        image_path = out / scene["name"]
        cv2.imwrite(str(image_path), scene["image"])

        side = {k: v for k, v in scene.items() if k != "image"}
        side["hint"] = {a: np.asarray(b).round(2).tolist()
                        for a, b in scene["hint"].items()}
        data_path = image_path.with_suffix(".json")
        data_path.write_text(json.dumps(side, ensure_ascii=False, indent=2), encoding="utf-8")

        print(f"Written: {image_path}  and  {data_path}")
        for measure_name, t in scene["truth"].items():
            print(f"  true {measure_name:8s}: {t['value']:.1f} {t['unit']}  "
                  f"({t['description']})")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        prog="metrik-goz",
        description="Real-world measurements from a single photo, with an honest error bar.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p):
        p.add_argument("image")
        p.add_argument("--aruco", type=float, metavar="EDGE_MM",
                       help="Edge length of the ArUco marker (mm)")
        p.add_argument("--scale", type=float, metavar="LENGTH_MM",
                       help="A single known length (mm); its two ends come from --end")
        p.add_argument("--scale-name", choices=sorted(reference.KNOWN_LENGTHS),
                       help="A known length from the table (e.g. 1_tl)")
        p.add_argument("--end", type=_point, action="append", default=[],
                       metavar="X,Y", help="End of the known length (given twice)")
        p.add_argument("--object", choices=sorted(reference.KNOWN_OBJECTS),
                       help="A standard-sized reference object")
        p.add_argument("--corner", type=_point, action="append", default=[],
                       help="Corner of the known object (given 4 times)")
        p.add_argument("--mc", type=int, default=400, help="Monte Carlo sample count")

    p_box = sub.add_parser("box", help="Width/height/area of an object marked by four corners")
    common(p_box)
    p_box.add_argument("--object-corner", type=_point, action="append", default=[],
                       metavar="X,Y", help="Corner of the measured object (given 4 times)")
    p_box.set_defaults(fn=command_box)

    p_distance = sub.add_parser("distance", help="Distance between two points")
    common(p_distance)
    p_distance.add_argument("--point", type=_point, action="append", default=[])
    p_distance.set_defaults(fn=command_distance)

    p_area = sub.add_parser("area", help="Area of a polygon")
    common(p_area)
    p_area.add_argument("--point", type=_point, action="append", default=[])
    p_area.set_defaults(fn=command_area)

    p_passage = sub.add_parser("passage", help="Narrowest passage in the free space")
    common(p_passage)
    p_passage.add_argument("--mask", required=True,
                           help="Free-space mask (white = traversable)")
    p_passage.add_argument("--footprint", type=float, help="Width of the vehicle (mm)")
    p_passage.add_argument("--step", type=float, default=20.0)
    p_passage.add_argument("--sample", type=float, default=5.0)
    p_passage.set_defaults(fn=command_passage)

    p_val = sub.add_parser("validate", help="Run the synthetic validation and produce the plots")
    p_val.add_argument("--out", default="validation")
    p_val.set_defaults(fn=command_validate)

    p_panel = sub.add_parser("panel", help="Start the web panel (drag-and-drop interface)")
    p_panel.add_argument("--host", default="127.0.0.1")
    p_panel.add_argument("--port", type=int, default=8000)
    p_panel.add_argument("--debug", action="store_true",
                         help="Flask debug mode (development only)")
    p_panel.set_defaults(fn=command_panel)

    p_sample = sub.add_parser("sample", help="Generate a synthetic sample scene with a known answer")
    p_sample.add_argument("--scene", default="flat",
                          choices=sorted(sample.SCENES) + ["all"])
    p_sample.add_argument("--out", default="examples")
    p_sample.set_defaults(fn=command_sample)

    args = parser.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
