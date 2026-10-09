"""A differential-drive robot with a 2D lidar, driving on a Track."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .track import Track, wrap_angle

N_BEAMS = 20
LIDAR_RANGE = 2.0
LIDAR_FOV = np.pi            # 180 degrees, centred on the heading
MAX_V = 1.5                  # m/s
MAX_W = 4.0                  # rad/s
DT = 0.1                     # s per step
RADIUS = 0.15                # m


def lidar_scan(track: Track, x, y, theta, n_beams=N_BEAMS, max_range=LIDAR_RANGE, fov=LIDAR_FOV):
    """Distance to the nearest wall along each beam (max_range if nothing is hit).

    Vectorised ray-segment intersection against every wall segment.
    """
    angles = theta + np.linspace(-fov / 2, fov / 2, n_beams)
    d = np.stack([np.cos(angles), np.sin(angles)], axis=1)            # (B, 2)
    # Only walls within lidar range can be hit; skip the rest of the track.
    near = np.hypot(track.wall_mid[:, 0] - x, track.wall_mid[:, 1] - y) < max_range + track.wall_reach
    seg = track.wall_segments[near]
    p1, p2 = seg[:, :2], seg[:, 2:]
    e = p2 - p1                                                       # (S, 2)
    o = np.array([x, y]) - p1                                         # (S, 2)
    # Solve origin + t*d = p1 + u*e  for t (along the beam) and u (along the wall).
    denom = d[:, None, 0] * e[None, :, 1] - d[:, None, 1] * e[None, :, 0]   # (B, S)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (e[None, :, 0] * o[None, :, 1] - e[None, :, 1] * o[None, :, 0]) / denom
        u = (d[:, None, 0] * o[None, :, 1] - d[:, None, 1] * o[None, :, 0]) / denom
    hit = (np.abs(denom) > 1e-12) & (t >= 0) & (u >= 0) & (u <= 1)
    t = np.where(hit, t, np.inf)
    return np.minimum(t.min(axis=1), max_range)


@dataclass
class Robot:
    x: float
    y: float
    theta: float
    trail: list = field(default_factory=list)

    def step(self, v, w, dt=DT):
        v = float(np.clip(v, 0.0, MAX_V))
        w = float(np.clip(w, -MAX_W, MAX_W))
        self.x += v * np.cos(self.theta) * dt
        self.y += v * np.sin(self.theta) * dt
        self.theta = float(wrap_angle(self.theta + w * dt))
        self.trail.append((self.x, self.y))
        return v, w

    def crashed(self, track: Track):
        _, lateral, _ = track.errors(self.x, self.y, self.theta)
        return abs(lateral) > track.half_width - RADIUS


def observe(track: Track, robot: Robot, lidar_noise: float = 0.0, rng=None):
    """What the policy sees: 20 lidar distances plus the heading error.

    The heading error (angle between the robot and the track direction) is the
    one non-lidar input the course assignment used: without it the network
    can't tell which way round the loop it is meant to drive.
    """
    scan = lidar_scan(track, robot.x, robot.y, robot.theta)
    if lidar_noise:
        rng = rng or np.random.default_rng()
        scan = np.clip(scan + rng.normal(0, lidar_noise, scan.shape), 0.0, LIDAR_RANGE)
    heading_error, _, _ = track.errors(robot.x, robot.y, robot.theta)
    return np.concatenate([scan, [heading_error]]).astype(np.float32)
