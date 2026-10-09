"""Collecting demonstrations, DAgger, and measuring how well a policy drives."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .expert import Expert
from .robot import MAX_V, MAX_W, Robot, observe
from .track import Track, wrap_angle


def spawn(track: Track, rng, heading_noise=0.0, lateral_noise=0.0):
    """Put the robot on the track at a random spot, optionally off-centre and
    pointing the wrong way a little."""
    s = rng.uniform(0, track.length)
    xy, heading = track.point_at(s)
    i, _, _ = track.project(xy)
    xy = xy + track.normal[i] * rng.uniform(-lateral_noise, lateral_noise)
    return Robot(float(xy[0]), float(xy[1]), float(wrap_angle(heading + rng.uniform(-heading_noise, heading_noise))))


def collect_expert(tracks, n_samples, rng, teleport_prob=0.1, action_noise=0.0, expert=None):
    """Drive the expert and record (observation, expert action) pairs.

    teleport_prob: chance per step of jumping to a random spot with a random
      heading (the course's way of getting recovery examples).
    action_noise: std of noise added to the *executed* action while the clean
      expert action is what gets recorded (DART-style noise injection), so the
      data shows how to recover from small mistakes.
    """
    expert = expert or Expert()
    X, Y = [], []
    k = 0
    while len(X) < n_samples:
        track = tracks[k % len(tracks)]
        k += 1
        robot = spawn(track, rng, heading_noise=0.3, lateral_noise=0.1)
        for _ in range(400):
            obs = observe(track, robot)
            v, w = expert.act(track, robot.x, robot.y, robot.theta)
            X.append(obs)
            Y.append((v, w))
            if action_noise:
                v += rng.normal(0, action_noise * MAX_V * 0.3)
                w += rng.normal(0, action_noise * MAX_W)
            robot.step(v, w)
            if rng.random() < teleport_prob:
                robot = spawn(track, rng, heading_noise=np.pi / 2, lateral_noise=0.15)
            if robot.crashed(track) or len(X) >= n_samples:
                break
    return np.array(X[:n_samples], dtype=np.float32), np.array(Y[:n_samples], dtype=np.float32)


def collect_on_policy(policy, tracks, n_samples, rng, beta=0.0, expert=None):
    """DAgger data: let the *learned* policy drive (mixed with the expert with
    probability beta), and label every state it visits with what the expert
    would have done there."""
    expert = expert or Expert()
    X, Y = [], []
    k = 0
    while len(X) < n_samples:
        track = tracks[k % len(tracks)]
        k += 1
        robot = spawn(track, rng, heading_noise=0.3, lateral_noise=0.1)
        for _ in range(400):
            obs = observe(track, robot)
            label = expert.act(track, robot.x, robot.y, robot.theta)
            X.append(obs)
            Y.append(label)
            action = label if rng.random() < beta else policy.act(obs)
            robot.step(*action)
            if robot.crashed(track) or len(X) >= n_samples:
                break
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)


@dataclass
class Episode:
    completed: bool
    crashed: bool
    progress: float          # fraction of a lap
    mean_abs_lateral: float  # m
    mean_speed: float        # m/s
    trail: np.ndarray


def drive(policy_or_expert, track: Track, robot: Robot, max_steps=1200, lidar_noise=0.0, rng=None):
    """Drive until one full lap, a crash, or max_steps."""
    is_expert = isinstance(policy_or_expert, Expert)
    _, _, s_prev = track.errors(robot.x, robot.y, robot.theta)
    progress, laterals, speeds = 0.0, [], []
    crashed = False
    for _ in range(max_steps):
        if is_expert:
            v, w = policy_or_expert.act(track, robot.x, robot.y, robot.theta)
        else:
            v, w = policy_or_expert.act(observe(track, robot, lidar_noise, rng))
        v, _ = robot.step(v, w)
        _, lateral, s = track.errors(robot.x, robot.y, robot.theta)
        ds = s - s_prev
        if ds < -track.length / 2:
            ds += track.length
        elif ds > track.length / 2:
            ds -= track.length
        progress += ds
        s_prev = s
        laterals.append(abs(lateral))
        speeds.append(v)
        if robot.crashed(track):
            crashed = True
            break
        if progress >= track.length:
            break
    return Episode(
        completed=progress >= track.length and not crashed,
        crashed=crashed,
        progress=max(0.0, progress) / track.length,
        mean_abs_lateral=float(np.mean(laterals)),
        mean_speed=float(np.mean(speeds)),
        trail=np.array(robot.trail),
    )


def evaluate(policy, tracks, starts_per_track=3, seed=1234, lidar_noise=0.0):
    """Lap-completion rate and driving quality over many tracks and starts."""
    rng = np.random.default_rng(seed)
    episodes = []
    for track in tracks:
        for _ in range(starts_per_track):
            robot = spawn(track, rng, heading_noise=0.2, lateral_noise=0.05)
            episodes.append(drive(policy, track, robot, lidar_noise=lidar_noise, rng=rng))
    return {
        "lap_completion": float(np.mean([e.completed for e in episodes])),
        "crash_rate": float(np.mean([e.crashed for e in episodes])),
        "mean_progress": float(np.mean([min(e.progress, 1.0) for e in episodes])),
        "mean_abs_lateral": float(np.mean([e.mean_abs_lateral for e in episodes])),
        "mean_speed": float(np.mean([e.mean_speed for e in episodes])),
        "episodes": len(episodes),
    }
