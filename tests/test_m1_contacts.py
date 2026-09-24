"""T07, T08, T10: support, drop, mouth passage."""

import mujoco
import numpy as np

from feedingrobot.sim.contacts import pair_set
from feedingrobot.sim.model import repo_root
from feedingrobot.sim.scene import FeedingScene


def _scene():
    return FeedingScene(str(repo_root() / "configs" / "m1_scene.json"))


def _supported(scene, preset, seeds):
    for seed in seeds:
        state = scene.reset(seed=seed, preset=preset)
        speed = []
        for _ in range(2000):
            state = scene.step_physics(scene.bias_tau(), hold_driver=True)
            speed.append(np.linalg.norm(state["food_vel"]))
        assert state["finite"]
        assert min(c["dist"] for c in state["contacts"]) > -1e-3
        assert speed[-1] < 0.01
        if preset == "food_on_plate":
            assert ("food", "plate") in pair_set(state["contacts"]) or ("plate", "food") in pair_set(state["contacts"])
            assert np.linalg.norm(state["food_pos"][:2] - np.array([0.45, -0.18])) < 0.09
        else:
            tcp = state["tcp_pos"]
            assert np.linalg.norm(state["food_pos"] - tcp) < 0.03
            assert ("food", "spoon") in pair_set(state["contacts"]) or ("spoon", "food") in pair_set(state["contacts"])


def test_t07_plate_and_spoon_support():
    scene = _scene()
    _supported(scene, "food_on_plate", range(10))
    _supported(scene, "food_on_spoon", range(10))


def test_t08_tilt_and_accel_drop():
    scene = _scene()
    q_tilt = np.array(scene.config["q_tilt"], dtype=float)
    for seed in range(5):
        scene.reset(seed=seed, preset="food_on_spoon")
        left = False
        for _ in range(2500):
            state = scene.step_physics(scene.diagnostic_pd_tau(q_tilt, kp=25, kd=6), hold_driver=True)
            away = np.linalg.norm(state["food_pos"] - state["tcp_pos"])
            touching = ("food", "spoon") in pair_set(state["contacts"]) or ("spoon", "food") in pair_set(state["contacts"])
            if away > 0.05 and not touching and state["finite"]:
                left = True
                break
        assert left, seed
        assert state["finite"]
        assert np.max(np.abs(state["dq"])) < 5.0

    for seed in range(5):
        scene.reset(seed=10 + seed, preset="food_on_spoon")
        left = False
        for k in range(700):
            tau = scene.bias_tau()
            if 30 <= k < 180:
                tau = tau + np.array([0.0, 0.0, 0.0, 0.0, 0.0, 12.0, 0.0])
            state = scene.step_physics(tau, hold_driver=True)
            away = np.linalg.norm(state["food_pos"] - state["tcp_pos"])
            touching = ("food", "spoon") in pair_set(state["contacts"]) or ("spoon", "food") in pair_set(state["contacts"])
            table_hit = any(
                {c["group1"], c["group2"]} & {"arm", "spoon"} and {c["group1"], c["group2"]} & {"table", "plate"}
                for c in state["contacts"]
            )
            if away > 0.05 and not touching and state["finite"] and sum(state["warnings"]) == 0 and not table_hit:
                left = True
                break
        assert left, seed


def test_t10_mouth_passage_and_receiver():
    scene = _scene()
    state = scene.reset(seed=0, preset="near_mouth")
    entry = state["tcp_pos"] * 0 + scene.data.site_xpos[scene.index.site_ids["mouth_entry"]]
    direction = np.array([1.0, 0.0, 0.0])
    geomid = np.array([-1], dtype=np.int32)
    dist = mujoco.mj_ray(scene.model, scene.data, entry + np.array([0.001, 0, 0]), direction, None, 1, -1, geomid)
    assert dist > 0.02
    # lip ray should hit immediately
    lip = entry + np.array([0.0, 0.0, 0.017])
    dist_lip = mujoco.mj_ray(scene.model, scene.data, lip + np.array([-0.02, 0, 0]), direction, None, 1, -1, geomid)
    assert 0.0 <= dist_lip < 0.03

    q_in = np.array(scene.config["q_mouth_in"], dtype=float)
    q_out = np.array(scene.config["q_mouth_out"], dtype=float)
    entered = False
    for _ in range(1500):
        state = scene.step_physics(scene.diagnostic_pd_tau(q_in, kp=30, kd=8), hold_driver=True)
        if state["tcp_pos"][0] > 0.54:
            entered = True
            break
    assert entered and state["finite"]
    exited = False
    for _ in range(1500):
        state = scene.step_physics(scene.diagnostic_pd_tau(q_out, kp=30, kd=8), hold_driver=True)
        if state["tcp_pos"][0] < 0.50:
            exited = True
            break
    assert exited

    scene.data.qpos[scene.index.arm_qpos_adr] = q_out
    adr = scene.index.food_qpos_adr
    scene.data.qpos[adr : adr + 7] = [0.57, 0.12, 0.36, 1, 0, 0, 0]
    scene.data.qvel[:] = 0
    for _ in range(800):
        state = scene.step_physics(scene.bias_tau(), hold_driver=True)
    assert ("food", "mouth") in pair_set(state["contacts"]) or ("mouth", "food") in pair_set(state["contacts"])
    assert np.linalg.norm(state["food_vel"]) < 0.02

    scene.data.qpos[scene.index.arm_qpos_adr] = np.array(scene.config["q_mouth_lip"])
    scene.data.qvel[scene.index.arm_dof_adr] = 0
    adr = scene.index.food_qpos_adr
    scene.data.qpos[adr : adr + 3] = [2, 2, 0.5]
    mujoco.mj_forward(scene.model, scene.data)
    names = {(c["geom1"], c["geom2"]) for c in scene.snapshot()["contacts"]}
    assert any("mouth" in a or "mouth" in b for a, b in names)
