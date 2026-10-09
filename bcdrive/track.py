"""Closed race tracks and the geometry the robot and the expert need.

A track is a wobbly loop: its centreline is a circle whose radius is bent by
two sine waves, so every random seed gives a different set of curves. The
walls sit `half_width` either side of the centreline.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def wrap_angle(a):
    """Map any angle (or array of angles) to [-pi, pi)."""
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


@dataclass
class Track:
    centre: np.ndarray        # (N, 2) points, counter-clockwise, not repeated at the end
    half_width: float

    porosity: float = 0.0     # share of wall that is missing (gaps the lidar sees through)
    seed: int = 0

    def __post_init__(self):
        c = self.centre
        nxt = np.roll(c, -1, axis=0)
        seg = nxt - c
        self.seg_len = np.linalg.norm(seg, axis=1)
        self.tangent = seg / self.seg_len[:, None]
        self.normal = np.stack([-self.tangent[:, 1], self.tangent[:, 0]], axis=1)  # points left (inwards)
        self.heading = np.arctan2(self.tangent[:, 1], self.tangent[:, 0])
        self.cum_len = np.concatenate([[0.0], np.cumsum(self.seg_len)[:-1]])
        self.length = float(self.seg_len.sum())
        # Wall polylines: offset the centreline along the averaged vertex normal.
        vn = self.normal + np.roll(self.normal, 1, axis=0)
        vn /= np.linalg.norm(vn, axis=1, keepdims=True)
        self.left_wall = c + vn * self.half_width
        self.right_wall = c - vn * self.half_width
        walls = []
        for w in (self.left_wall, self.right_wall):
            walls.append(np.concatenate([w, np.roll(w, -1, axis=0)], axis=1))
        segments = np.concatenate(walls, axis=0)   # (2N, 4): x1, y1, x2, y2
        if self.porosity > 0:
            # Knock out runs of 3-8 wall segments until the target share is missing.
            rng = np.random.default_rng(self.seed + 7919)
            keep = np.ones(len(segments), dtype=bool)
            while (~keep).mean() < self.porosity:
                start, run = rng.integers(len(segments)), rng.integers(3, 9)
                keep[np.arange(start, start + run) % len(segments)] = False
            segments = segments[keep]
        self.wall_segments = segments
        self.wall_mid = (segments[:, :2] + segments[:, 2:]) / 2
        self.wall_reach = float(np.max(np.hypot(*(segments[:, 2:] - segments[:, :2]).T)) / 2)

    # ---------- construction ----------

    @classmethod
    def random(cls, seed: int, half_width: float = 0.5, wobble: float = 1.0, porosity: float = 0.0,
               n_points: int = 360) -> "Track":
        """wobble scales how curvy the loop is; porosity opens gaps in the walls."""
        rng = np.random.default_rng(seed)
        radius = rng.uniform(4.0, 6.0)
        k1, k2 = rng.choice([2, 3, 4, 5], size=2, replace=False)
        a1, a2 = wobble * rng.uniform(0.06, 0.16), wobble * rng.uniform(0.03, 0.10)
        p1, p2 = rng.uniform(0, 2 * np.pi, size=2)
        theta = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
        r = radius * (1 + a1 * np.sin(k1 * theta + p1) + a2 * np.sin(k2 * theta + p2))
        centre = np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1)
        return cls(centre, half_width, porosity, seed)

    @classmethod
    def oval(cls, half_width: float = 0.5, n_points: int = 360) -> "Track":
        theta = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
        centre = np.stack([6.0 * np.cos(theta), 3.5 * np.sin(theta)], axis=1)
        return cls(centre, half_width)

    # ---------- queries ----------

    def project(self, xy):
        """Nearest point on the centreline.

        Returns (segment index, distance along the track, signed lateral
        offset: positive = left of the centreline).
        """
        p = np.asarray(xy, dtype=float)
        rel = p - self.centre
        t = np.clip(np.einsum("ij,ij->i", rel, self.tangent), 0.0, self.seg_len)
        closest = self.centre + self.tangent * t[:, None]
        d2 = np.sum((p - closest) ** 2, axis=1)
        i = int(np.argmin(d2))
        lateral = float(np.dot(p - closest[i], self.normal[i]))
        return i, float(self.cum_len[i] + t[i]), lateral

    def errors(self, x, y, theta):
        """(heading error, cross-track error, progress along the track)."""
        i, s, lateral = self.project((x, y))
        heading_error = float(wrap_angle(self.heading[i] - theta))
        return heading_error, lateral, s

    def point_at(self, s):
        """Centreline position and heading at distance s along the track."""
        s = s % self.length
        i = int(np.searchsorted(self.cum_len, s, side="right") - 1)
        t = s - self.cum_len[i]
        xy = self.centre[i] + self.tangent[i] * t
        return xy, float(self.heading[i])

    def curvature(self):
        """Absolute turning per metre at each centreline point."""
        dh = wrap_angle(np.roll(self.heading, -1) - self.heading)
        return np.abs(dh) / self.seg_len
