"""T01 and T05: model structure, frames, tool mass."""

import numpy as np

from feedingrobot.sim.model import ModelIndex, load_model, repo_root
from feedingrobot.sim.scene import FeedingScene
from feedingrobot.sim.sensors import shift_torque


def test_t01_names_and_motors():
    scene = FeedingScene(str(repo_root() / "configs" / "m1_scene.json"))
    model = scene.model
    assert model.nu == 12
    for name in ["ft_site", "tcp", "plate_frame", "mouth_entry", "mouth_receiver", "ft_force", "ft_torque"]:
        assert name
    scene.index.assert_arm_motors()
    # free joint is not packed into the arm slice
    assert int(scene.index.food_qpos_adr) == 7
    assert int(scene.index.arm_qpos_adr[0]) == 0
    assert model.body_mass[scene.index.tool_body_id] == np.float64(scene.config["tool_mass_kg"]) or abs(
        model.body_mass[scene.index.tool_body_id] - scene.config["tool_mass_kg"]
    ) < 1e-9


def test_t05_frame_roundtrip_and_mass():
    scene = FeedingScene(str(repo_root() / "configs" / "m1_scene.json"))
    scene.reset(seed=1, preset="food_on_plate")
    state = scene.snapshot()
    rot = state["ft_mat"]
    vector = np.array([0.2, -0.4, 0.7])
    back = rot.T @ (rot @ vector)
    assert np.linalg.norm(back - vector) <= 1e-10
    # 90 deg about Z
    c, s = 0.0, 1.0
    rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    assert np.linalg.norm(rz.T @ (rz @ vector) - vector) <= 1e-10
    point = np.array([0.1, 0.2, 0.3])
    force = np.array([0.0, 0.0, 2.0])
    torque = shift_torque(np.zeros(3), force, point, point + np.array([0.0, 0.05, 0.0]))
    expect = np.cross(np.array([0.0, -0.05, 0.0]), force)
    assert np.linalg.norm(torque - expect) <= 1e-12
    assert abs(scene.model.body_mass[scene.index.tool_body_id] - 0.08) < 1e-9
    visual = 0
    for gid in range(scene.model.ngeom):
        if scene.model.geom_contype[gid] == 0 and scene.model.geom_bodyid[gid] == scene.index.tool_body_id:
            visual += 1
    assert visual == 0


def test_sphere_food_settles_on_plate():
    import json

    cfg = json.loads((repo_root() / "configs" / "m1_scene.json").read_text())
    cfg["model"] = "assets/scenes/feeding_p0_sphere.xml"
    scene = FeedingScene(cfg)
    state = scene.reset(seed=1, preset="food_on_plate")
    for _ in range(500):
        state = scene.step_physics(scene.bias_tau(), hold_driver=True)
    assert state["finite"]
    assert abs(state["food_pos"][2] - 0.016) < 0.004
    assert np.linalg.norm(state["food_vel"]) < 0.01


def test_upstream_joint_ranges_kept():
    model = load_model("assets/scenes/feeding_p0.xml")
    index = ModelIndex(model)
    expected = {
        "joint1": (-2.8973, 2.8973),
        "joint2": (-1.7628, 1.7628),
        "joint4": (-3.0718, -0.0698),
        "joint6": (-0.0175, 3.7525),
    }
    for name, (lo, hi) in expected.items():
        jid = int(index.arm_joint_ids[int(name[-1]) - 1])
        assert abs(model.jnt_range[jid, 0] - lo) < 1e-6
        assert abs(model.jnt_range[jid, 1] - hi) < 1e-6
