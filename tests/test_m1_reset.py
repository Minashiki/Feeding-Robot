"""T11: contaminated reset is deterministic for a fixed seed."""

import numpy as np

from feedingrobot.sim.model import repo_root
from feedingrobot.sim.scene import FeedingScene


def _contaminate(scene):
    scene.set_external_wrench([1, 0, 0], [0, 0, 0], scene.snapshot()["tcp_pos"])
    scene.data.ctrl[:] = 3
    scene.data.qpos[scene.index.jaw_qpos_adr] = 0.3
    scene.contact_events.append({"tick": -1})
    scene.peak_contact_force = 10
    scene.step_physics(np.ones(7))


def _signature(scene):
    return np.concatenate(
        [
            scene.data.qpos,
            scene.data.qvel,
            scene.data.ctrl,
            scene.data.xfrc_applied.ravel(),
            scene.data.qfrc_applied,
        ]
    )


def test_t11_reset_replay():
    cfg = str(repo_root() / "configs" / "m1_scene.json")
    seeds = list(range(50))
    tau = np.array([0.2, -0.1, 0.0, 0.15, 0.05, -0.05, 0.0])
    for seed in seeds:
        a = FeedingScene(cfg)
        b = FeedingScene(cfg)
        _contaminate(a)
        state_a = a.reset(seed=seed, preset="food_on_plate")
        state_b = b.reset(seed=seed, preset="food_on_plate")
        assert a.applied_wrench is None
        assert a.tick == 0
        assert a.contact_events == []
        assert a.peak_contact_force == 0
        assert np.max(np.abs(a.data.xfrc_applied)) == 0
        assert np.max(np.abs(a.data.ctrl)) == 0
        assert np.allclose(_signature(a), _signature(b), atol=1e-9)
        for _ in range(30):
            sa = a.step_physics(tau, hold_driver=True)
            sb = b.step_physics(tau, hold_driver=True)
        assert np.allclose(sa["q"], sb["q"], atol=1e-9)
        assert np.allclose(sa["dq"], sb["dq"], atol=1e-9)
        assert state_a["finite"] and state_b["finite"]

    span = []
    scene = FeedingScene(cfg)
    for seed in scene.config["fixed_seeds"]:
        _contaminate(scene)
        state = scene.reset(seed=seed, preset="food_on_plate")
        span.append(state["food_pos"][:2].copy())
        ranges = scene.model.jnt_range[scene.index.arm_joint_ids]
        q = state["q"]
        assert np.all(q >= ranges[:, 0] - 1e-8) and np.all(q <= ranges[:, 1] + 1e-8)
    span = np.array(span)
    assert np.ptp(span[:, 0]) > 0.01 and np.ptp(span[:, 1]) > 0.01
