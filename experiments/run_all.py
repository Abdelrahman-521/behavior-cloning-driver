#!/usr/bin/env python3
"""Reproduce every number and figure in the README.

  python experiments/run_all.py            # full run, about 5 minutes on a laptop
  python experiments/run_all.py --quick    # small smoke test for CI

Four drivers, each one adding a fix to the one before:
  1. course version      500 samples from one track, the assignment's training loop
  2. + more tracks       12 training tracks, 6,000 samples, noise injection, better training
  3. + wall-gap training some training walls have gaps (domain randomisation)
  4. + DAgger            4 rounds of on-policy data labelled by the expert
Each is tested on tracks it has never seen: solid walls, walls with 30% gaps,
and walls with gaps plus lidar noise.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bcdrive.expert import Expert  # noqa: E402
from bcdrive.policy import train, train_course_style  # noqa: E402
from bcdrive.sim import collect_expert, collect_on_policy, evaluate  # noqa: E402
from bcdrive.track import Track  # noqa: E402
from bcdrive import viz  # noqa: E402


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true", help="tiny version for CI")
    parser.add_argument("--out", default=str(ROOT / "results"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    torch.set_num_threads(2)

    n_test = 4 if args.quick else 20
    scale = 0.25 if args.quick else 1.0
    dagger_rounds = 1 if args.quick else 4

    course_track = Track.random(0)
    train_solid = [Track.random(s) for s in range(1, 13)]
    train_gappy = [Track.random(s, porosity=float(rng.uniform(0.0, 0.4))) for s in range(1, 13)]
    tests = {
        "unseen tracks, solid walls": dict(tracks=[Track.random(s) for s in range(1000, 1000 + n_test)]),
        "unseen tracks, 30% wall gaps": dict(tracks=[Track.random(s, porosity=0.3) for s in range(2000, 2000 + n_test)]),
        "gaps + lidar noise": dict(tracks=[Track.random(s, porosity=0.3) for s in range(3000, 3000 + n_test)],
                                   lidar_noise=0.03),
    }

    drivers, curves = {}, {}

    log("1/4 course version: 500 samples from one track, the assignment's training loop")
    X, Y = collect_expert([course_track], 500, rng, teleport_prob=0.1)
    drivers["1. Course version"], curves["1. Course version"] = train_course_style(X, Y)

    log("2/4 more tracks + noise injection + better training")
    X2, Y2 = collect_expert(train_solid, int(6000 * scale), rng, teleport_prob=0.1, action_noise=0.2)
    drivers["2. + more tracks, noise injection"], curves["2. + more tracks, noise injection"] = train(X2, Y2)

    log("3/4 + wall-gap training (domain randomisation)")
    X3, Y3 = collect_expert(train_gappy, int(6000 * scale), rng, teleport_prob=0.1, action_noise=0.2)
    policy, history = train(X3, Y3)
    drivers["3. + wall-gap training"], curves["3. + wall-gap training"] = policy, history

    log(f"4/4 + DAgger ({dagger_rounds} rounds)")
    Xa, Ya = X3, Y3
    dagger_progress = []
    for r in range(dagger_rounds):
        beta = 0.5 * (1 - r / dagger_rounds)
        Xn, Yn = collect_on_policy(policy, train_gappy, int(1500 * scale), rng, beta=beta)
        Xa, Ya = np.concatenate([Xa, Xn]), np.concatenate([Ya, Yn])
        policy, history = train(Xa, Ya, seed=r + 1)
        score = evaluate(policy, tests["unseen tracks, 30% wall gaps"]["tracks"], seed=99)["lap_completion"]
        dagger_progress.append({"round": r + 1, "samples": int(len(Xa)), "gappy_lap_completion": score})
        log(f"    round {r + 1}: {len(Xa)} samples, gappy-track laps {score:.0%}")
    drivers["4. + DAgger"], curves["4. + DAgger"] = policy, history

    log("evaluating")
    results = {"Expert (cheats: reads the true track geometry)": {}}
    for test_name, spec in tests.items():
        results["Expert (cheats: reads the true track geometry)"][test_name] = evaluate(Expert(), **spec)
    for name, driver in drivers.items():
        results[name] = {test_name: evaluate(driver, **spec) for test_name, spec in tests.items()}
        log(f"    {name}: " + ", ".join(f"{t}: {r['lap_completion']:.0%}" for t, r in results[name].items()))

    torch.save(drivers["1. Course version"].state_dict(), out / "course_policy.pt")
    torch.save(drivers["3. + wall-gap training"].state_dict(), out / "wall_gap_policy.pt")
    torch.save(drivers["4. + DAgger"].state_dict(), out / "dagger_policy.pt")
    (out / "metrics.json").write_text(json.dumps({"results": results, "dagger_rounds": dagger_progress}, indent=2))

    # Results table for the README.
    names = list(tests)
    lines = ["| Driver | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for driver_name, row in results.items():
        cells = [f"{row[t]['lap_completion']:.0%} laps, {row[t]['crash_rate']:.0%} crashes" for t in names]
        lines.append(f"| {driver_name} | " + " | ".join(cells) + " |")
    episodes = results["1. Course version"][names[0]]["episodes"]
    lines.append("")
    lines.append(f"Each cell: {episodes} runs ({n_test} tracks x 3 random starts). A run counts if the robot "
                 "finishes one full lap without touching a wall.")
    (out / "results.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    if not args.quick:
        log("figures")
        # Showcase: the first unseen gappy track where the improved driver finishes from the
        # standard start (it does on ~95% of runs), next to the course driver on the same track.
        best = drivers["3. + wall-gap training"]
        gappy = tests["unseen tracks, 30% wall gaps"]["tracks"]
        showcase = next((t for t in gappy if viz.completes(best, t)), gappy[0])
        viz.training_curves(curves, out / "training_curves.png")
        viz.results_chart(results, names, out / "lap_completion.png")
        viz.compare_runs(showcase, {"Course version": drivers["1. Course version"],
                                    "With wall-gap training": best},
                         out / "course_vs_final.png")
        viz.animate(showcase, best, out / "drive.gif")
    log("done")


if __name__ == "__main__":
    main()
