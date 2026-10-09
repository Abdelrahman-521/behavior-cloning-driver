| Driver | unseen tracks, solid walls | unseen tracks, 30% wall gaps | gaps + lidar noise |
|---|---|---|---|
| Expert (cheats: reads the true track geometry) | 95% laps, 5% crashes | 100% laps, 0% crashes | 100% laps, 0% crashes |
| 1. Course version | 100% laps, 0% crashes | 0% laps, 100% crashes | 0% laps, 100% crashes |
| 2. + more tracks, noise injection | 100% laps, 0% crashes | 0% laps, 98% crashes | 2% laps, 98% crashes |
| 3. + wall-gap training | 100% laps, 0% crashes | 95% laps, 5% crashes | 100% laps, 0% crashes |
| 4. + DAgger | 100% laps, 0% crashes | 92% laps, 8% crashes | 100% laps, 0% crashes |

Each cell: 60 runs (20 tracks x 3 random starts). A run counts if the robot finishes one full lap without touching a wall.
