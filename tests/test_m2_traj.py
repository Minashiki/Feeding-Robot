"""T05 circle and T14 replay. One nominal pose; the runner repeats the other poses."""

import numpy as np

from feedingrobot.controllers.so3 import orientation_error
from feedingrobot.sim.scene import FeedingScene
from tests.m2_support import load_m2_config, place_arm, run_commands, start_controller


def test_t05_circle_stays_within_bounds():
    scene = FeedingScene("configs/m1_scene.json")
    cfg = load_m2_config()
    state = place_arm(scene, scene.config["q_torque_poses"][1])
    ctl, guard = start_controller(scene, cfg, state)
    radius = 0.01
    period = 4.0
    omega = 2.0 * np.pi / period

    def twist(t, c):
        if c.power_on:
            return np.zeros(6)
        elapsed = max(t - c.power_time, 0.0)
        if elapsed > 8.0:
            return np.zeros(6)
        out = np.zeros(6)
        out[0] = -radius * omega * np.sin(omega * elapsed)
        out[1] = radius * omega * np.cos(omega * elapsed)
        return out

    samples = run_commands(ctl, guard, 9.0, twist)
    assert guard.failure is None, guard.failure
    err = np.array([np.linalg.norm(s["tcp_pos"] - info["p_ref"]) for s, info in samples])
    # Score the moving window after power-on and before the stop transient ends.
    window = err[1500:8000]
    assert np.sqrt(np.mean(window**2)) <= 0.002
    assert np.max(window) <= 0.005


def test_t14_replay_matches():
    cfg = load_m2_config()

    def once():
        scene = FeedingScene("configs/m1_scene.json")
        state = place_arm(scene, scene.config["q_torque_poses"][0], seed=3)
        ctl, guard = start_controller(scene, cfg, state)
        samples = run_commands(ctl, guard, 0.5, lambda t, c: np.zeros(6))
        return np.array([s["q"] for s, _info in samples]), np.array([s["tcp_pos"] for s, _info in samples])

    q1, p1 = once()
    q2, p2 = once()
    assert np.max(np.abs(q1 - q2)) <= 1e-9
    assert np.max(np.abs(p1 - p2)) <= 1e-9
