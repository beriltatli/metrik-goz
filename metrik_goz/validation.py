"""
Synthetic validation: the module that produces and plots the claimed numbers.

Not a single number in the README is typed by hand; all of them are produced here
and written to `validation/results.json`. That way the numbers cannot go stale
when the code changes.

    python -m metrik_goz.cli validate --out validation
"""

from __future__ import annotations

import json
import pathlib

import numpy as np

from .homography import Homography
from . import measure, reference, uncertainty
from .synthetic import add_noise, build_scene

# Validated categorical palette (colour-blind safe, passes on a light ground)
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, INK3 = "#0b0b0b", "#52514e", "#8a8880"
SURFACE, GRID = "#fcfcfb", "#e6e5e1"

SIGMA_PX = 0.5
REF_MM = 100.0

# Sweep for the similarity (single length) model: the panel's default scenario —
# measuring the phone on the table with the 1 TL coin on the table.
SIMILARITY_REF_MM = 26.15                 # 1 TL, diameter
SIMILARITY_OBJECT_MM = (146.7, 71.5)      # phone: width, height
SIMILARITY_TILTS = (0.0, 5.0, 10.0, 20.0, 30.0)
SIMILARITY_DISTANCES = (0.0, 2.0, 4.0, 6.0)   # multiples of the reference diameter
MIN_CELL = 8               # don't average cells that fall out of frame
# The server's "high" warning threshold; the sweep exists exactly to test it.
RECTANGULARITY_THRESHOLD = 0.06
SERIOUS_ERROR = 0.05


# ------------------------------------------------------------------ experiments
def _single_measurement(rng, *, distance_mm, tilt, distance_factor, mc_n=200):
    scene = build_scene(reference_size_mm=REF_MM, distance_mm=distance_mm,
                        tilt_deg=tilt, azimuth_deg=float(rng.uniform(0, 360)))
    r = distance_factor * REF_MM
    angle = rng.uniform(0, 2 * np.pi)
    center = r * np.array([np.cos(angle), np.sin(angle)])
    direction = rng.uniform(0, 2 * np.pi)
    span = rng.uniform(0.5, 2.0) * REF_MM
    v = np.array([np.cos(direction), np.sin(direction)])
    a, b = center - 0.5 * span * v, center + 0.5 * span * v

    a_px, b_px = scene.project(a)[0], scene.project(b)[0]
    if not (scene.is_visible([a_px, b_px]) and scene.is_visible(scene.reference_px)):
        return None

    ref_noisy = add_noise(scene.reference_px, SIGMA_PX, rng)
    points_noisy = add_noise(np.array([a_px, b_px]), SIGMA_PX, rng)
    truth = float(np.hypot(*(b - a)))
    fn = lambda h, n: measure.distance(h, n[0], n[1])

    mc = uncertainty.monte_carlo(scene.reference_world, ref_noisy, fn, points_noisy,
                                 sigma_px=SIGMA_PX, n=mc_n,
                                 seed=int(rng.integers(1 << 30)))
    h = Homography.fit(scene.reference_world, ref_noisy)
    an = uncertainty.analytic(h, scene.reference_world, ref_noisy, fn, points_noisy,
                              sigma_px=SIGMA_PX)
    return dict(truth=truth, mc=mc, an=an,
                inside=bool(mc.low <= truth <= mc.high),
                error=abs(mc.value - truth) / truth)


def _run(seed, n, **kw):
    rng = np.random.default_rng(seed)
    return [d for d in (_single_measurement(rng, **kw) for _ in range(n)) if d]


# ------------------------------------------------------------------ plots
def _style_axes(ax, title, subtitle=None, xlabel=None, ylabel=None):
    ax.set_title(title, fontsize=13, fontweight="600", color=INK, loc="left",
                 pad=30 if subtitle else 12)
    if subtitle:
        ax.text(0, 1.045, subtitle, transform=ax.transAxes, fontsize=10.5,
                color=INK2, va="bottom")
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=10.5, color=INK2)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10.5, color=INK2)
    ax.tick_params(colors=INK2, labelsize=10)
    for k in ("top", "right"):
        ax.spines[k].set_visible(False)
    for k in ("left", "bottom"):
        ax.spines[k].set_color(GRID)


def _plot_coverage(conditions, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [c["name"] for c in conditions]
    values = [c["coverage"] for c in conditions]
    y = np.arange(len(names))

    fig, ax = plt.subplots(figsize=(8.4, 0.52 * len(names) + 2.1))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    ax.axvline(0.95, color=ORANGE, linewidth=1.6, linestyle=(0, (5, 3)), zorder=1)
    ax.text(0.95, len(names) - 0.35, " nominal 95%", color=ORANGE, fontsize=10,
            fontweight="600", va="center")

    ax.hlines(y, 0.85, values, color=GRID, linewidth=1.4, zorder=1)
    ax.scatter(values, y, s=70, color=BLUE, zorder=3,
               edgecolor=SURFACE, linewidth=1.5)
    # Labels in one aligned column: keep them off the nominal line
    for yi, d in zip(y, values):
        ax.text(1.0, yi, f"{d * 100:.1f}%", color=INK, fontsize=10,
                va="center", ha="right", fontweight="600")

    ax.set_yticks(y, names)
    ax.set_xlim(0.85, 1.012)
    ax.set_ylim(-0.6, len(names) - 0.15)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v * 100:.0f}%")
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    _style_axes(ax, "Does the confidence interval actually hold",
                "Independent synthetic measurements per condition · how often the true "
                "value lands in the 95% interval", xlabel="coverage")
    fig.tight_layout()
    fig.savefig(path, dpi=170, facecolor=SURFACE)
    plt.close(fig)


def _plot_error_distance(series, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.array([s["distance"] for s in series])
    median = np.array([s["median"] for s in series]) * 100
    p10 = np.array([s["p10"] for s in series]) * 100
    p90 = np.array([s["p90"] for s in series]) * 100

    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    ax.fill_between(x, p10, p90, color=BLUE, alpha=0.16, linewidth=0, zorder=2)
    ax.plot(x, median, color=BLUE, linewidth=2.0, marker="o", markersize=7,
            markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=3)

    ax.axhline(3.0, color=ORANGE, linewidth=1.6, linestyle=(0, (5, 3)), zorder=1)
    ax.text(x[0], 3.0, " declared limit 3%", color=ORANGE, fontsize=10,
            fontweight="600", va="bottom")
    ax.text(x[-1], median[-1], f"  median {median[-1]:.1f}%", color=INK, fontsize=10,
            fontweight="600", va="center")
    ax.text(x[-1], p90[-1], f"  p90 {p90[-1]:.1f}%", color=INK2, fontsize=9.5, va="center")

    ax.set_xlim(x[0] - 0.15, x[-1] + 0.9)
    ax.set_ylim(0, max(p90.max() * 1.15, 4))
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    _style_axes(ax, "The error grows with distance from the reference",
                "1.2 m distance · 25° view · 100 mm reference · 0.5 px corner noise",
                xlabel="distance of the measured spot from the reference "
                       "(multiples of the reference size)",
                ylabel="relative error (%)")
    fig.tight_layout()
    fig.savefig(path, dpi=170, facecolor=SURFACE)
    plt.close(fig)


def _plot_analytic_mc(trials, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mc = np.array([d["mc"].std for d in trials])
    an = np.array([d["an"].std for d in trials])
    top = max(mc.max(), an.max()) * 1.08

    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    ax.plot([0, top], [0, top], color=INK3, linewidth=1.4, linestyle=(0, (5, 3)), zorder=1)
    ax.text(top * 0.97, top * 0.97, "equal  ", color=INK3, fontsize=10,
            ha="right", va="top", rotation=45, rotation_mode="anchor")
    ax.scatter(mc, an, s=46, color=BLUE, alpha=0.65, zorder=3,
               edgecolor=SURFACE, linewidth=1.0)

    ratio = np.median(an / np.maximum(mc, 1e-9))
    ax.text(0.04, 0.94, f"median ratio {ratio:.2f}", transform=ax.transAxes,
            fontsize=11, color=INK, fontweight="600")

    ax.set_xlim(0, top)
    ax.set_ylim(0, top)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    _style_axes(ax, "The fast route agrees with the slow one",
                "Each point is one measurement · analytic propagation vs Monte Carlo",
                xlabel="Monte Carlo std (mm)", ylabel="analytic std (mm)")
    fig.tight_layout()
    fig.savefig(path, dpi=170, facecolor=SURFACE)
    plt.close(fig)


# --------------------------------------------------- similarity (single length) model
def _widest_diameter(scene, center_mm, radius_mm: float) -> np.ndarray:
    """
    The two ends of the WIDEST part of the circle in the image.

    Under perspective a round reference projects to an ellipse, so which diameter
    you read matters. The widest part (the ellipse's major axis) is the diameter
    that suffered no foreshortening — and a user naturally clicks there. The
    sweep here imitates exactly that well-meaning user; picking a bad diameter
    would make the model look unfairly bad and render the validation useless.
    """
    angle = np.linspace(0, np.pi, 360, endpoint=False)
    direction = np.column_stack([np.cos(angle), np.sin(angle)])
    a = scene.project(np.asarray(center_mm, float) + radius_mm * direction)
    b = scene.project(np.asarray(center_mm, float) - radius_mm * direction)
    k = int(np.argmax(np.hypot(*(a - b).T)))
    return np.array([a[k], b[k]])


def _similarity_single(rng, *, tilt, distance_factor, mc_n=100):
    """
    One trial: a coin on the table, a phone `distance_factor` diameters away.

    Two numbers are measured at once and they must not be confused:

    bias  — the error of the NOISE-FREE measurement, i.e. the model's own error.
            Because a scale built from a single length does not correct
            perspective, this term grows with tilt and is systematic; the error
            bar does NOT cover it.
    error — the error the user actually sees: bias + click noise. A 26 mm coin is
            about 75 px in the image, so even in a straight-down shot a few
            percent of noise remains.

    If we summed the two into one number, the claim "the model is perfect in a
    straight-down shot" would disappear under the noise and the sweep would prove
    nothing.
    """
    scene = build_scene(reference_size_mm=SIMILARITY_REF_MM, focal_px=1500.0,
                        size_px=(900, 1300), distance_mm=520.0, tilt_deg=tilt,
                        azimuth_deg=float(rng.uniform(0, 360)))
    width_mm, height_mm = SIMILARITY_OBJECT_MM

    angle = rng.uniform(0, 2 * np.pi)
    center = distance_factor * SIMILARITY_REF_MM * np.array([np.cos(angle), np.sin(angle)])
    t = rng.uniform(0, 2 * np.pi)
    rotation = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    corner_mm = np.array([[-width_mm / 2, -height_mm / 2], [width_mm / 2, -height_mm / 2],
                          [width_mm / 2, height_mm / 2],
                          [-width_mm / 2, height_mm / 2]]) @ rotation.T + center

    corner_px = scene.project(corner_mm)
    ref_px = _widest_diameter(scene, (0.0, 0.0), SIMILARITY_REF_MM / 2.0)
    if not (scene.is_visible(corner_px) and scene.is_visible(ref_px)):
        return None

    fit_fn = lambda g: Homography.from_length(g[0], g[1], SIMILARITY_REF_MM)
    bias = abs(measure.box(fit_fn(ref_px), corner_px).width_mm - width_mm) / width_mm

    # Both the coin and the phone are clicked by hand; no ArUco sub-pixel accuracy.
    sigma = reference.TYPICAL_SIGMA_PX["manual"]
    ref_noisy = add_noise(ref_px, sigma, rng)
    corner_noisy = add_noise(corner_px, sigma, rng)
    box_result = measure.box(fit_fn(ref_noisy), corner_noisy)
    mc = uncertainty.monte_carlo(
        None, ref_noisy, lambda h, n: measure.box(h, n).width_mm, corner_noisy,
        sigma_px=sigma, n=mc_n, seed=int(rng.integers(1 << 30)), fit_fn=fit_fn)

    return dict(truth=width_mm, mc=mc,
                bias=float(bias),
                error=abs(mc.value - width_mm) / width_mm,
                inside=bool(mc.low <= width_mm <= mc.high),
                # The deviation visible on the user's screen: from noisy corners.
                rectangularity=float(box_result.rectangularity))


def similarity_sweep(n: int = 40, *, seed: int = 900) -> dict:
    """
    Where the SIMILARITY model built from a single known length is valid.

    This model does not correct perspective — the panel's warning texts rest on
    the three things this sweep measures:

    1) What produces the bias is TILT, not distance. In a straight-down shot the
       scale is the same everywhere on the plane, so the bias drops to zero
       wherever the coin sits; as the view flattens it grows, and distance
       multiplies it. In the projective model it is the other way round: there
       distance itself is the risk.
    2) The systematic bias is NOT INSIDE the error bar. That is why coverage
       collapses as tilt grows; the width of the interval is right, its center
       drifts. Instead of hiding it we measure and write it down — the panel's
       "the error bar does not cover this" sentence comes from exactly here.
    3) The user does not know the tilt, but the system does see how differently
       the opposite edges of the measured object come out. `Box.rectangularity`
       is therefore an observable proxy for tilt; the sweep's real exam is how
       much of the serious bias that proxy's threshold catches.
    """
    rng = np.random.default_rng(seed)
    cells, trials = [], []
    for tilt in SIMILARITY_TILTS:
        for distance in SIMILARITY_DISTANCES:
            d = [x for x in (_similarity_single(rng, tilt=tilt, distance_factor=distance)
                             for _ in range(n)) if x]
            # When the object spills out of frame only a few trials remain;
            # a median from three samples looks like a number but is not information.
            if len(d) < MIN_CELL:
                continue
            trials.extend(d)
            errors = np.array([x["error"] for x in d])
            cells.append(dict(
                tilt=tilt, distance=distance, n=len(d),
                median_bias=float(np.median([x["bias"] for x in d])),
                median_error=float(np.median(errors)),
                p90_error=float(np.percentile(errors, 90)),
                median_rectangularity=float(np.median([x["rectangularity"] for x in d])),
                coverage=float(np.mean([x["inside"] for x in d])),
            ))

    bias = np.array([d["bias"] for d in trials])
    deviation = np.array([d["rectangularity"] for d in trials])
    serious = bias > SERIOUS_ERROR
    warned = deviation > RECTANGULARITY_THRESHOLD

    top_down = [c for c in cells if c["tilt"] == 0.0]
    tilted = [c for c in cells if c["tilt"] == SIMILARITY_TILTS[-1]]
    return dict(
        cells=cells,
        n=len(trials),
        threshold=RECTANGULARITY_THRESHOLD,
        serious_error_threshold=SERIOUS_ERROR,
        # How much of the serious bias produces a warning: that is the warning's real job.
        catch_rate=float(np.mean(warned[serious])) if serious.any() else float("nan"),
        false_alarm=float(np.mean(warned[~serious])) if (~serious).any() else float("nan"),
        # In a straight-down shot the bias must drop to zero regardless of distance.
        top_down_p90_bias=float(max(c["median_bias"] for c in top_down)),
        top_down_coverage=float(np.mean([c["coverage"] for c in top_down])),
        tilted_median_bias=float(max(c["median_bias"] for c in tilted)),
        tilted_coverage=float(np.mean([c["coverage"] for c in tilted])),
        worst_median_bias=float(max(c["median_bias"] for c in cells)),
    )


def _plot_similarity(sweep, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12.8, 5.0),
                                 gridspec_kw=dict(width_ratios=[1.2, 1]))
    fig.patch.set_facecolor(SURFACE)
    colors = [BLUE, AQUA, ORANGE, INK3]

    for color, distance in zip(colors, SIMILARITY_DISTANCES):
        series = [c for c in sweep["cells"] if c["distance"] == distance]
        if not series:
            continue
        x = [c["tilt"] for c in series]
        y = [c["median_bias"] * 100 for c in series]
        ax.plot(x, y, color=color, linewidth=2.0, marker="o", markersize=6,
                markeredgecolor=SURFACE, markeredgewidth=1.4, zorder=3)
        ax.text(x[-1], y[-1], f"  {distance:g}× away", color=color, fontsize=9.5,
                va="center", fontweight="600")

    ax.set_facecolor(SURFACE)
    ax.axhline(3.0, color=INK3, linewidth=1.4, linestyle=(0, (5, 3)), zorder=1)
    ax.text(0, 3.0, " declared limit 3%", color=INK3, fontsize=9.5,
            va="bottom", fontweight="600")
    ax.set_xlim(-1.5, SIMILARITY_TILTS[-1] + 9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    _style_axes(ax, "Tilt produces the bias, not distance",
                "Scale from a single length · a phone measured with a 1 TL diameter · "
                "noise-free, i.e. the model's own error",
                xlabel="camera tilt (degrees, 0 = straight down)",
                ylabel="median systematic bias (%)")

    deviation = [c["median_rectangularity"] * 100 for c in sweep["cells"]]
    error = [c["median_bias"] * 100 for c in sweep["cells"]]
    bx.set_facecolor(SURFACE)
    bx.axvline(sweep["threshold"] * 100, color=ORANGE, linewidth=1.6,
               linestyle=(0, (5, 3)), zorder=1)
    bx.text(sweep["threshold"] * 100, 0.0, " warning threshold", color=ORANGE,
            fontsize=9.5, va="bottom", fontweight="600")
    bx.scatter(deviation, error, s=54, color=BLUE, alpha=0.8, zorder=3,
               edgecolor=SURFACE, linewidth=1.2)
    bx.text(0.04, 0.93, f"{sweep['catch_rate'] * 100:.0f}% of serious biases "
                        f"raise a warning", transform=bx.transAxes, fontsize=10.5,
            color=INK, fontweight="600")
    bx.grid(color=GRID, linewidth=0.8)
    bx.set_axisbelow(True)
    _style_axes(bx, "The observable proxy for tilt",
                "The mismatch between opposite edges gives the tilt away without "
                "knowing it",
                xlabel="rectangularity deviation (%)", ylabel="median bias (%)")

    fig.tight_layout(w_pad=3.0)
    fig.savefig(path, dpi=170, facecolor=SURFACE)
    plt.close(fig)


# ------------------------------------------------------------------ main flow
def run_validation(output_dir: str = "validation", n: int = 140) -> dict:
    path = pathlib.Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)

    conditions = []
    definitions = [
        ("0.6 m distance", dict(distance_mm=600, tilt=25.0, distance_factor=1.0)),
        ("1.2 m distance", dict(distance_mm=1200, tilt=25.0, distance_factor=1.0)),
        ("2.0 m distance", dict(distance_mm=2000, tilt=25.0, distance_factor=1.0)),
        ("3.0 m distance", dict(distance_mm=3000, tilt=25.0, distance_factor=1.0)),
        ("0° view (straight down)", dict(distance_mm=1200, tilt=0.0, distance_factor=1.0)),
        ("40° view", dict(distance_mm=1200, tilt=40.0, distance_factor=1.0)),
        ("55° view (very oblique)", dict(distance_mm=1200, tilt=55.0, distance_factor=1.0)),
        ("4× reference away", dict(distance_mm=1200, tilt=25.0, distance_factor=4.0)),
    ]
    for i, (name, kw) in enumerate(definitions):
        d = _run(200 + i, n, **kw)
        errors = np.array([x["error"] for x in d])
        conditions.append(dict(
            name=name, n=len(d),
            coverage=float(np.mean([x["inside"] for x in d])),
            median_error=float(np.median(errors)),
            p90_error=float(np.percentile(errors, 90)),
        ))

    distance_series = []
    for i, k in enumerate((0.5, 1.0, 2.0, 3.0, 4.0)):
        d = _run(300 + i, n, distance_mm=1200, tilt=25.0, distance_factor=k)
        e = np.array([x["error"] for x in d])
        distance_series.append(dict(distance=k, n=len(d),
                                    median=float(np.median(e)),
                                    p10=float(np.percentile(e, 10)),
                                    p90=float(np.percentile(e, 90))))

    comparison = _run(400, 90, distance_mm=1200, tilt=25.0, distance_factor=1.0)
    similarity = similarity_sweep(max(20, n // 3))

    _plot_coverage(conditions, path / "coverage.png")
    _plot_error_distance(distance_series, path / "error_distance.png")
    _plot_analytic_mc(comparison, path / "analytic_vs_mc.png")
    _plot_similarity(similarity, path / "similarity.png")

    ratios = np.array([d["an"].std / max(d["mc"].std, 1e-9) for d in comparison])
    summary = dict(
        sigma_px=SIGMA_PX,
        reference_mm=REF_MM,
        conditions=conditions,
        distance_series=distance_series,
        analytic_mc_ratio_median=float(np.median(ratios)),
        mean_coverage=float(np.mean([c["coverage"] for c in conditions])),
        similarity=similarity,
    )
    (path / "results.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                       encoding="utf-8")

    print(f"Written: {path}/coverage.png, error_distance.png, analytic_vs_mc.png, "
          f"similarity.png, results.json")
    print(f"  mean coverage        : {summary['mean_coverage'] * 100:.1f}%")
    print(f"  analytic/MC std ratio: {summary['analytic_mc_ratio_median']:.2f}")
    for c in conditions:
        print(f"  {c['name']:24s} coverage {c['coverage']*100:5.1f}%  "
              f"median error {c['median_error']*100:5.2f}%")
    print("  similarity model (single length, perspective not corrected):")
    print(f"    bias straight down    : {similarity['top_down_p90_bias'] * 100:.3f}% "
          f"(independent of distance), coverage {similarity['top_down_coverage'] * 100:.1f}%")
    print(f"    bias at {SIMILARITY_TILTS[-1]:.0f}° tilt    : "
          f"{similarity['tilted_median_bias'] * 100:.1f}%, "
          f"coverage {similarity['tilted_coverage'] * 100:.1f}% — a systematic bias "
          f"is not inside the error bar")
    print(f"    rectangularity warning: catches "
          f"{similarity['catch_rate'] * 100:.0f}% of serious biases, "
          f"{similarity['false_alarm'] * 100:.0f}% false alarms")
    return summary
