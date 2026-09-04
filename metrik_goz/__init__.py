"""
metrik-goz — real-world measurements from a single photo, with an honest error bar.

Basic usage:

    from metrik_goz import Homography, measure, uncertainty

    h = Homography.fit(reference_world_mm, reference_image_px)
    mm = measure.distance(h, (120, 340), (610, 355))

    result = uncertainty.monte_carlo(
        reference_world_mm, reference_image_px,
        lambda hh, nn: measure.distance(hh, nn[0], nn[1]),
        points_px=[(120, 340), (610, 355)],
        sigma_px=0.5,
    )
    print(result)     # 412.3 ± 8.7 mm (95%: 395.1–429.4)

Limit: everything you measure must lie on the same plane as the reference.
"""

from .homography import Homography, dlt
from .measure import Box, Passage, area, box, distance, length, narrowest_passage
from .uncertainty import Measurement, analytic, monte_carlo, parameter_covariance
from . import lm, measure, uncertainty, synthetic, reference

__all__ = [
    "Homography", "dlt",
    "Passage", "Box", "distance", "length", "area", "box", "narrowest_passage",
    "Measurement", "monte_carlo", "analytic", "parameter_covariance",
    "lm", "measure", "uncertainty", "synthetic", "reference",
]

__version__ = "0.1.0"
