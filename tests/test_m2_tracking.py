"""T04 and T05: hold, shaped step, circle, and orientation sine."""

import numpy as np

from feedingrobot.controllers.so3 import orientation_error
from feedingrobot.sim.scene import FeedingScene
from tests.m2_support import load_m2_config, place_arm, run_commands, start_controller


def _rms(values):
    arr = np.asarray(values, dtype=float)
    return float(np.sqrt(np.mean(arr**2)))


def test_t04_hold_three_poses():
    scene = FeedingScene("configs/m1_scene.json")
    cfg = load_m2_config()
    for q in scene.config["q_torque_poses"]:
        state = place_arm(scene, q)
        ctl, guard = start_controller(scene, cfg, state)
        samples = run_commands(ctl, guard, 6.0, lambda t, c: np.zeros(6))
        assert guard.failure is None, guard.failure
        tail = [np.linalg.norm(s["tcp_pos"] - info["p_ref"]) for s, info in samples[-1000:]]
        ori = [np.linalg.norm(orientation_error(info["r_ref"], s["tcp_mat"])) for s, info in samples[-1000:]]
        assert _rms(tail) <= 0.001
        assert _rms(ori) <= np.deg2rad(0.5)


def test_t04_position_step_x():
    scene = FeedingScene("configs/m1_scene.json")
    cfg = load_m2_config()
    state = place_arm(scene, scene.config["q_torque_poses"][0])
    ctl, guard = start_controller(scene, cfg, state)
    origin = state["tcp_pos"].copy()

    def twist(t, c):
        if c.power_on:
            return np.zeros(6)
        elapsed = t - c.power_time
        out = np.zeros(6)
        if 0.0 <= elapsed < 0.2:
            out[0] = 0.05
        return out

    samples = run_commands(ctl, guard, 4.0, twist)
    assert guard.failure is None, guard.failure
    moved = np.linalg.norm(ctl.reference.p_ref - origin)
    assert 0.008 <= moved <= 0.014
    err = [np.linalg.norm(s["tcp_pos"] - info["p_ref"]) for s, info in samples[-500:]]
    assert _rms(err) <= 0.002
