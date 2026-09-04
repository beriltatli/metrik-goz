"""
Synthetic scene generator — the backbone of validation.

A real photo has no "right answer"; if it did, there would be nothing to
measure. So the only way to prove the uncertainty claim is to build scenes whose
answer we put there ourselves: we place the camera, we place the reference, we
know the distance to be measured. Then we hand the system nothing but pixels and
look at how close it gets and whether the confidence interval really holds.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Scene:
    """A synthetic measurement scene with a known answer."""

    H_true: np.ndarray            # world (mm) -> image (px), noise-free
    reference_world: np.ndarray   # reference corners, mm
    reference_px: np.ndarray      # reference corners, px (noise-free)
    size_px: tuple[int, int]      # (height, width)

    def project(self, world_points) -> np.ndarray:
        d = np.atleast_2d(np.asarray(world_points, dtype=float))
        h = np.hstack([d, np.ones((len(d), 1))]) @ self.H_true.T
        return h[:, :2] / h[:, 2:3]

    def is_visible(self, px) -> bool:
        height, width = self.size_px
        px = np.atleast_2d(px)
        return bool(np.all((px[:, 0] >= 0) & (px[:, 0] < width) &
                           (px[:, 1] >= 0) & (px[:, 1] < height)))


def camera_homography(
    *,
    focal_px: float,
    center_px: tuple[float, float],
    distance_mm: float,
    tilt_deg: float,
    azimuth_deg: float = 0.0,
) -> np.ndarray:
    """
    The homography between the z=0 plane and the image.

    `tilt_deg` = 0 is a straight-down (nadir) view; as it grows the view flattens
    out and foreshortening increases. This is the parameter that grows the
    measurement error the most, which is why validation sweeps it.
    """
    tilt = np.radians(tilt_deg)
    azimuth = np.radians(azimuth_deg)

    # Camera position: above the plane, at the given tilt and azimuth
    C = distance_mm * np.array([
        np.sin(tilt) * np.cos(azimuth),
        np.sin(tilt) * np.sin(azimuth),
        np.cos(tilt),
    ])

    zc = -C / np.linalg.norm(C)                      # camera forward axis: look at the origin
    world_up = np.array([0.0, 0.0, 1.0])
    if abs(zc @ world_up) > 0.999:                   # exact nadir: pick another up
        world_up = np.array([0.0, 1.0, 0.0])
    xc = np.cross(world_up, zc)
    xc /= np.linalg.norm(xc)
    yc = np.cross(zc, xc)

    R = np.vstack([xc, yc, zc])                      # world -> camera
    t = -R @ C

    K = np.array([
        [focal_px, 0.0, center_px[0]],
        [0.0, focal_px, center_px[1]],
        [0.0, 0.0, 1.0],
    ])
    H = K @ np.column_stack([R[:, 0], R[:, 1], t])
    return H / H[2, 2]


def build_scene(
    *,
    reference_size_mm: float = 100.0,
    reference_center_mm: tuple[float, float] = (0.0, 0.0),
    focal_px: float = 1400.0,
    size_px: tuple[int, int] = (1080, 1920),
    distance_mm: float = 1200.0,
    tilt_deg: float = 25.0,
    azimuth_deg: float = 0.0,
) -> Scene:
    """
    A typical scene: a square reference placed on a table, with the things to be
    measured a little way off. The defaults correspond to a kitchen-counter
    scenario (1.2 m distance, slightly oblique view, 100 mm reference).
    """
    height, width = size_px
    H = camera_homography(
        focal_px=focal_px,
        center_px=(width / 2.0, height / 2.0),
        distance_mm=distance_mm,
        tilt_deg=tilt_deg,
        azimuth_deg=azimuth_deg,
    )
    half = reference_size_mm / 2.0
    cx, cy = reference_center_mm
    ref_world = np.array([
        [cx - half, cy - half],
        [cx + half, cy - half],
        [cx + half, cy + half],
        [cx - half, cy + half],
    ])
    h = np.hstack([ref_world, np.ones((4, 1))]) @ H.T
    ref_px = h[:, :2] / h[:, 2:3]

    return Scene(H_true=H, reference_world=ref_world, reference_px=ref_px, size_px=size_px)


def add_noise(points, sigma_px: float, rng: np.random.Generator) -> np.ndarray:
    """Adds independent Gaussian noise to pixel coordinates."""
    points = np.asarray(points, dtype=float)
    return points + rng.normal(0.0, sigma_px, size=points.shape)
