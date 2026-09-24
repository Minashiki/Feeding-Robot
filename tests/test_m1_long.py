"""T09 and T12–T14: head motion, nominal run, timestep and solver checks."""

import numpy as np

from feedingrobot.sim.contacts import pair_set
from feedingrobot.sim.model import repo_root
from feedingrobot.sim.scene import FeedingScene


def _scene():
    return FeedingScene(str(repo_root() / "configs" / "m1_scene.json"))


def test_t09_head_tracks_for_10s():
    scene = _scene()
    scene.reset(seed=3, preset="food_on_plate")
    prev = None
    saturated = 0
    for _ in range(10000):
        state = scene.step_physics(scene.bias_tau(), hold_driver=False)
        assert state["finite"]
        assert sum(state["warnings"]) == 0
        q = np.concatenate([state["head_q"], [state["jaw_q"]]])
        if prev is not None:
            assert np.max(np.abs(q - prev)) < 0.02
        prev = q
        limits = scene.model.actuator_ctrlrange[scene.index.scene_actuator_ids]
        cmd = state["head_tau"]
        saturated += int(np.any(np.abs(cmd - limits[:, 0]) < 1e-6) or np.any(np.abs(cmd - limits[:, 1]) < 1e-6))
    assert saturated < 200
    t = 10.0
    w = 2 * np.pi * scene.config["head"]["freq_hz"]
    amp = scene.config["head"]["amp_m"]
    # position has moved; final sine is not required to be exact, motion must be visible
    assert np.max(np.abs(state["head_q"][:2])) > 0.5 * amp
    assert abs(state["head_q"][0] - amp * np.sin(w * t)) < 0.003
    ranges = scene.model.jnt_range[scene.index.head_joint_ids]
    assert np.all(state["head_q"] >= ranges[:, 0]) and np.all(state["head_q"] <= ranges[:, 1])


def test_t12_nominal_10s():
    scene = _scene()
    scene.reset(seed=4, preset="food_on_plate")
    for _ in range(10000):
        state = scene.step_physics(scene.bias_tau(), hold_driver=False)
        assert state["finite"] and sum(state["warnings"]) == 0
        assert min((c["dist"] for c in state["contacts"]), default=0.0) > -1e-3
    ranges = scene.model.jnt_range[scene.index.arm_joint_ids]
    assert np.all(state["q"] > ranges[:, 0] - 1e-4) and np.all(state["q"] < ranges[:, 1] + 1e-4)


def _carry_force(scene, seconds):
    steps = int(round(seconds / scene.model.opt.timestep))
    forces = []
    supported = False
    for _ in range(steps):
        state = scene.step_physics(scene.bias_tau(), hold_driver=True)
        fz = 0.0
        for c in state["contacts"]:
            if {c["group1"], c["group2"]} == {"food", "plate"}:
                fz += c["force_on_geom2_world"][2] if c["group2"] == "food" else c["force_on_geom1_world"][2]
        forces.append(fz)
        supported = ("food", "plate") in pair_set(state["contacts"]) or ("plate", "food") in pair_set(state["contacts"])
    return float(np.mean(forces[-200:])), float(np.max(np.abs(forces[-200:]))), supported and state["finite"]


def test_t13_timestep_and_t14_solver():
    scene = _scene()
    scene.reset(seed=5, preset="food_on_plate")
    mean_a, peak_a, ok_a = _carry_force(scene, 1.0)
    scene.reset(seed=5, preset="food_on_plate")
    scene.model.opt.timestep = 0.0005
    scene.dt = 0.0005
    mean_b, peak_b, ok_b = _carry_force(scene, 1.0)
    assert ok_a and ok_b
    assert abs(mean_a - mean_b) <= 0.1 * max(abs(mean_a), 1e-6) + 1e-4
    assert abs(peak_a - peak_b) <= 0.1 * max(abs(peak_a), abs(peak_b), 1e-6) + 1e-4

    scene.restore_model_options()
    scene.dt = float(scene.model.opt.timestep)
    scene.reset(seed=5, preset="food_on_plate")
    mean_c, _, ok_c = _carry_force(scene, 0.5)
    scene.reset(seed=5, preset="food_on_plate")
    scene.model.opt.iterations = 100
    scene.model.opt.tolerance = scene._opt_baseline["tolerance"] * 0.1
    mean_d, _, ok_d = _carry_force(scene, 0.5)
    assert ok_c and ok_d
    assert abs(mean_c - mean_d) <= 0.1 * max(abs(mean_c), 1e-6) + 1e-4
    scene.restore_model_options()
