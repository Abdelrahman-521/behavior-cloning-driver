"""The expert the robot learns to imitate.

It cheats: it reads the exact heading and cross-track error from the track
geometry instead of using the lidar. That's the point of behaviour cloning:
the network has to recover the same behaviour from raw lidar.
"""

from __future__ import annotations

import numpy as np

from .robot import MAX_V, MAX_W
from .track import Track


class Expert:
    def __init__(self, heading_gain=2.0, cross_gain=3.0, min_speed_fraction=0.15):
        self.heading_gain = heading_gain
        self.cross_gain = cross_gain
        self.min_speed_fraction = min_speed_fraction

    def act(self, track: Track, x, y, theta):
        heading_error, lateral, _ = track.errors(x, y, theta)
        # Turn towards the track direction, and back towards the centreline.
        w = self.heading_gain * heading_error - self.cross_gain * lateral
        w = float(np.clip(w, -MAX_W, MAX_W))
        # Slow down when pointing the wrong way or drifting towards a wall.
        alignment = min(max(0.0, np.cos(heading_error)), max(0.0, 1.0 - abs(lateral) / track.half_width))
        v = MAX_V * (self.min_speed_fraction + (1 - self.min_speed_fraction) * alignment)
        return float(v), w
