import numpy as np
import pytest
import torch

from bcdrive.expert import Expert
from bcdrive.policy import N_INPUTS, Policy, train, train_course_style
from bcdrive.robot import LIDAR_RANGE, N_BEAMS, Robot, lidar_scan, observe
from bcdrive.sim import collect_expert, collect_on_policy, drive, evaluate
from bcdrive.track import Track, wrap_angle


def test_wrap_angle():
    assert wrap_angle(3 * np.pi / 2) == pytest.approx(-np.pi / 2)
    assert wrap_angle(-3 * np.pi / 2) == pytest.approx(np.pi / 2)


def test_circle_track_geometry():
    theta = np.linspace(0, 2 * np.pi, 720, endpoint=False)
    track = Track(np.stack([5 * np.cos(theta), 5 * np.sin(theta)], axis=1), half_width=0.5)
    assert track.length == pytest.approx(2 * np.pi * 5, rel=1e-3)
    # A point inside the circle is to the left of a counter-clockwise centreline.
    _, _, lateral = track.project((4.8, 0.0))
    assert lateral == pytest.approx(0.2, abs=1e-3)
    heading_error, lateral, _ = track.errors(5.0, 0.0, np.pi / 2)
    assert heading_error == pytest.approx(0.0, abs=0.01)


def test_lidar_sees_walls_at_the_right_distance():
    theta = np.linspace(0, 2 * np.pi, 720, endpoint=False)
    track = Track(np.stack([5 * np.cos(theta), 5 * np.sin(theta)], axis=1), half_width=0.5)
    # Robot on the centreline at (5, 0) facing +y (along the track).
    scan = lidar_scan(track, 5.0, 0.0, np.pi / 2)
    assert scan.shape == (N_BEAMS,)
    # First beam points right (outwards): outer wall 0.5 m away.
    assert scan[0] == pytest.approx(0.5, abs=0.01)
    # Last beam points left (inwards): inner wall 0.5 m away.
    assert scan[-1] == pytest.approx(0.5, abs=0.01)
    assert np.all(scan <= LIDAR_RANGE)


def test_wall_gaps_let_beams_through():
    solid = Track.random(5)
    gappy = Track.random(5, porosity=0.5)
    assert len(gappy.wall_segments) < len(solid.wall_segments)
    rng = np.random.default_rng(0)
    longer = 0
    for _ in range(50):
        xy, heading = solid.point_at(rng.uniform(0, solid.length))
        longer += lidar_scan(gappy, *xy, heading).sum() > lidar_scan(solid, *xy, heading).sum() + 1e-6
    assert longer > 10


def test_observation_shape():
    track = Track.random(1)
    xy, heading = track.point_at(0)
    obs = observe(track, Robot(*xy, heading))
    assert obs.shape == (N_INPUTS,)


def test_expert_drives_a_lap_on_random_tracks():
    for seed in range(3):
        track = Track.random(seed)
        xy, heading = track.point_at(0)
        ep = drive(Expert(), track, Robot(*xy, heading))
        assert ep.completed and not ep.crashed


def test_collection_shapes():
    rng = np.random.default_rng(0)
    X, Y = collect_expert([Track.random(0)], 200, rng, teleport_prob=0.1, action_noise=0.2)
    assert X.shape == (200, N_INPUTS) and Y.shape == (200, 2)
    policy, _ = train_course_style(X, Y, epochs=5)
    Xd, Yd = collect_on_policy(policy, [Track.random(1)], 100, rng, beta=0.5)
    assert Xd.shape[1] == N_INPUTS and len(Xd) == len(Yd) >= 100


def test_course_training_loop_reduces_loss():
    rng = np.random.default_rng(0)
    X, Y = collect_expert([Track.random(0)], 300, rng)
    _, hist = train_course_style(X, Y)
    assert hist["train"][-1] < hist["train"][0] / 5


def test_improved_training_and_save_load_round_trip(tmp_path):
    rng = np.random.default_rng(1)
    X, Y = collect_expert([Track.random(0), Track.random(1)], 600, rng)
    policy, hist = train(X, Y, epochs=20, hidden=(16,))
    assert len(hist["val"]) <= 20
    torch.save(policy.state_dict(), tmp_path / "p.pt")
    loaded = Policy.from_state_dict(torch.load(tmp_path / "p.pt", weights_only=True))
    assert loaded.act(X[0]) == pytest.approx(policy.act(X[0]))


def test_evaluate_reports_rates():
    result = evaluate(Expert(), [Track.random(7)], starts_per_track=2)
    assert result["episodes"] == 2
    assert 0.0 <= result["lap_completion"] <= 1.0
