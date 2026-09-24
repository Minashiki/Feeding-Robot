"""T11: contaminated reset is deterministic for a fixed seed."""

import numpy as np

from feedingrobot.sim.cases import save_case
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


def test_t11_reset_replay(m1_config):
    scene = FeedingScene(m1_config)
    tau = np.array([0.2, -0.1, 0.0, 0.15, 0.05, -0.05, 0.0])
    reference = None
    replay_reference = None
    for _ in range(50):
        _contaminate(scene)
        scene.reset(seed=7, preset="food_on_plate")
        assert scene._hold_head_target is None and scene.velocity_fault is False
        assert np.max(np.abs(scene.data.ctrl)) == 0
        assert np.max(np.abs(scene.data.xfrc_applied)) == 0
        start = np.concatenate([scene.data.qpos, scene.data.qvel])
        if reference is None:
            reference = start.copy()
        assert np.allclose(start, reference, rtol=0, atol=1e-9)
        for _step in range(30):
            scene.step_physics(tau, hold_driver=True)
        end = np.concatenate([scene.data.qpos, scene.data.qvel])
        if replay_reference is None:
            replay_reference = end.copy()
        assert np.allclose(end, replay_reference, rtol=0, atol=1e-9)
    span = []
    for seed in range(50):
        _contaminate(scene)
        state = scene.reset(seed=seed, preset="food_on_plate")
        span.append(state["food_pos"][:2].copy())
        assert scene._hold_latched is False
        ranges = scene.model.jnt_range[scene.index.arm_joint_ids]
        assert np.all(state["q"] >= ranges[:, 0] - 1e-8) and np.all(state["q"] <= ranges[:, 1] + 1e-8)
    span = np.asarray(span)
    assert np.ptp(span[:, 0]) > 0.01 and np.ptp(span[:, 1]) > 0.01
    save_case("T11", {"replay_atol": 1e-9, "food_span_m": float(np.ptp(span[:, 0]))})
