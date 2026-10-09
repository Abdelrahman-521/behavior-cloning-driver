"""Figures for the README."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import animation  # noqa: E402

from .robot import LIDAR_FOV, N_BEAMS, Robot, lidar_scan  # noqa: E402
from .sim import drive  # noqa: E402
from .track import Track  # noqa: E402

COLOURS = ["#c0392b", "#e67e22", "#2980b9", "#27ae60"]


def draw_track(ax, track: Track):
    seg = track.wall_segments
    ax.plot(seg[:, [0, 2]].T, seg[:, [1, 3]].T, color="#333", lw=1.2)
    ax.plot(*np.vstack([track.centre, track.centre[:1]]).T, color="#bbb", lw=0.6, ls="--")
    ax.set_aspect("equal")
    ax.axis("off")


def training_curves(curves, path):
    fig, ax = plt.subplots(figsize=(7, 4))
    for (name, hist), colour in zip(curves.items(), COLOURS):
        ax.plot(hist["train"], color=colour, label=f"{name} (train)")
        if "val" in hist:
            ax.plot(hist["val"], color=colour, ls="--", alpha=0.7)
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE loss (dashed: validation)")
    ax.set_title("Training loss")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def results_chart(results, test_names, path):
    drivers = [d for d in results if not d.startswith("Expert")]
    x = np.arange(len(test_names))
    width = 0.8 / len(drivers)
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, (name, colour) in enumerate(zip(drivers, COLOURS)):
        vals = [results[name][t]["lap_completion"] * 100 for t in test_names]
        bars = ax.bar(x + (i - (len(drivers) - 1) / 2) * width, vals, width, label=name, color=colour)
        ax.bar_label(bars, fmt="%.0f", fontsize=7)
    ax.set_xticks(x, test_names, fontsize=8)
    ax.set_ylabel("laps completed (%)")
    ax.set_ylim(0, 110)
    ax.set_title("Can the cloned driver finish a lap on a track it has never seen?", pad=34)
    ax.legend(fontsize=7, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=len(drivers), frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _start(track):
    xy, heading = track.point_at(0.0)
    return Robot(float(xy[0]), float(xy[1]), heading)


def completes(driver, track):
    return drive(driver, track, _start(track)).completed


def compare_runs(track, drivers, path):
    fig, axes = plt.subplots(1, len(drivers), figsize=(6 * len(drivers), 6))
    for ax, (name, driver), colour in zip(np.atleast_1d(axes), drivers.items(), [COLOURS[0], COLOURS[3]]):
        draw_track(ax, track)
        ep = drive(driver, track, _start(track))
        ax.plot(ep.trail[:, 0], ep.trail[:, 1], color=colour, lw=2)
        ax.plot(*ep.trail[0], "o", color="black")
        if ep.crashed:
            ax.plot(*ep.trail[-1], "X", color=colour, ms=14)
        outcome = "full lap" if ep.completed else f"crashed after {ep.progress:.0%} of a lap"
        ax.set_title(f"{name}\n{outcome}", fontsize=11)
    fig.suptitle("Same unseen track with 30% of the wall missing", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def animate(track, driver, path, every=3, max_frames=160, window=3.0):
    """Follow-camera GIF: the robot, its lidar beams and its trail, with a
    small map of the whole track in the corner."""
    ep = drive(driver, track, _start(track))
    trail = ep.trail
    frames = list(range(0, len(trail), every))[:max_frames]
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_axes([0, 0, 1, 1])
    draw_track(ax, track)
    inset = fig.add_axes([0.70, 0.70, 0.28, 0.28])
    draw_track(inset, track)
    inset.set_axis_on()
    inset.set_xticks([])
    inset.set_yticks([])
    inset.set_facecolor("white")
    for spine in inset.spines.values():
        spine.set_color("#ccc")
    path_line, = ax.plot([], [], color=COLOURS[3], lw=2.5)
    body, = ax.plot([], [], "o", color="black", ms=9)
    dot, = inset.plot([], [], "o", color=COLOURS[3], ms=4)
    beams = [ax.plot([], [], color="#e74c3c", lw=0.8, alpha=0.7)[0] for _ in range(N_BEAMS)]
    label = ax.text(0.03, 0.04, "", transform=ax.transAxes, fontsize=9, color="#333")

    def update(k):
        x, y = trail[k]
        nxt = trail[min(k + 1, len(trail) - 1)]
        prv = trail[max(k - 1, 0)]
        theta = np.arctan2(nxt[1] - prv[1], nxt[0] - prv[0])
        path_line.set_data(trail[: k + 1, 0], trail[: k + 1, 1])
        body.set_data([x], [y])
        dot.set_data([x], [y])
        ranges = lidar_scan(track, x, y, theta)
        for beam, angle, r in zip(beams, theta + np.linspace(-LIDAR_FOV / 2, LIDAR_FOV / 2, N_BEAMS), ranges):
            beam.set_data([x, x + r * np.cos(angle)], [y, y + r * np.sin(angle)])
        ax.set_xlim(x - window, x + window)
        ax.set_ylim(y - window, y + window)
        label.set_text(f"20-beam lidar, unseen track with gaps in the walls   {k / len(trail):.0%} of lap")
        return [path_line, body, dot, label, *beams]

    ani = animation.FuncAnimation(fig, update, frames=frames, blit=False)
    ani.save(path, writer=animation.PillowWriter(fps=15), dpi=80)
    plt.close(fig)
