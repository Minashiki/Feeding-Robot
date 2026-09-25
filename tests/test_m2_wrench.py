"""T06 and T07: unloaded compensation, known load, and food weight left in the signal."""

import numpy as np

from feedingrobot.sim.model import load_model
from feedingrobot.sim.scene import FeedingScene
from feedingrobot.controllers.wrench import tool_body_ids
from tests.m2_support import load_m2_config, place_arm, run_commands, start_controller


def _tail_wrench(samples, key="ft_compensated_wrench_tcp"):
    rows = [info[key] for _state, info in samples[-1000:]]
    return np.asarray(rows, dtype=float)


def test_tool_mass_is_80_grams():
    model = load_model("assets/scenes/feeding_p0.xml")
    from feedingrobot.sim.model import ModelIndex

    index = ModelIndex(model)
    ids = tool_body_ids(model, index.tool_body_id)
    mass = float(sum(model.body_mass[i] for i in ids))
    assert abs(mass - 0.08) < 1e-9


def test_t06_unloaded_and_known_force():
    scene = FeedingScene("configs/m1_scene.json")
    cfg = load_m2_config()
    state = place_arm(scene, scene.config["q_torque_poses"][0])
    ctl, guard = start_controller(scene, cfg, state)
    samples = run_commands(ctl, guard, 2.0, lambda t, c: np.zeros(6))
    assert guard.failure is None, guard.failure
    wrench = _tail_wrench(samples)
    assert np.linalg.norm(np.mean(wrench[:, :3], axis=0)) <= 0.02
    assert np.linalg.norm(np.mean(wrench[:, 3:], axis=0)) <= 0.002
    point = samples[-1][0]["tcp_pos"]
    scene.set_external_wrench([1.0, 0.0, 0.0], [0.0, 0.0, 0.0], point)
    loaded = run_commands(ctl, guard, 1.5, lambda t, c: np.zeros(6))
    assert guard.failure is None, guard.failure
    mean = np.mean(_tail_wrench(loaded)[:, :3], axis=0)
    assert abs(mean[0] - 1.0) <= max(0.02, 0.02 * 1.0)
    assert abs(mean[1]) <= 0.02 and abs(mean[2]) <= 0.02


def test_t07_food_is_not_zeroed():
    scene = FeedingScene("configs/m1_scene.json")
    cfg = load_m2_config()
    state = scene.reset(seed=0, preset="food_on_spoon", settle_steps=30)
    ctl, guard = start_controller(scene, cfg, state, phase="TRANSPORT")
    samples = run_commands(ctl, guard, 1.5, lambda t, c: np.zeros(6))
    assert guard.failure is None, guard.failure
    mean = np.mean(_tail_wrench(samples)[:, :3], axis=0)
    weight = 0.003 * 9.81
    assert abs(mean[2] + weight) < 0.02 or abs(np.linalg.norm(mean) - weight) < 0.02
    assert np.linalg.norm(mean) > 0.01
