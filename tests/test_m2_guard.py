"""T12 and T13: contact whitelist, first fault, and command expiry."""

import numpy as np

from feedingrobot.controllers.cartesian_impedance import control_step
from feedingrobot.controllers.guard import contact_allowed
from feedingrobot.sim.scene import FeedingScene
from tests.m2_support import load_m2_config, place_arm, run_commands, start_controller


def test_t12_contact_whitelist_is_ordered():
    bowl = "bowl_bottom"
    plate = "plate_bottom"
    handle = "tool_handle"
    lip = "jaw_lip"
    assert contact_allowed("ACQUIRE", bowl, plate) is True
    assert contact_allowed("ACQUIRE", plate, bowl) is True
    assert contact_allowed("TRANSPORT", bowl, plate) is False
    assert contact_allowed("ACQUIRE", handle, plate) is False
    assert contact_allowed("APPROACH", bowl, lip) is True
    assert contact_allowed("TRANSPORT", bowl, lip) is False
    assert contact_allowed("SELECT", "bowl_bottom", "table") is False
    assert contact_allowed("ACQUIRE", bowl, "spring_pad", {"press_test": False}) is False
    assert contact_allowed("ACQUIRE", bowl, "spring_pad", {"press_test": True}) is True
    assert contact_allowed("TRANSPORT", "food_box", "bowl_bottom") is True


def test_t13_expired_command_latches_and_stops():
    scene = FeedingScene("configs/m1_scene.json")
    cfg = load_m2_config()
    state = place_arm(scene, scene.config["q_torque_poses"][0])
    ctl, guard = start_controller(scene, cfg, state)
    run_commands(ctl, guard, 0.3, lambda t, c: np.zeros(6))
    assert guard.failure is None
    ctl.set_command(np.zeros(6), ctl.sim_time, ctl.sim_time + 0.02, "TRANSPORT")
    for _ in range(200):
        control_step(ctl, guard)
    assert guard.failure is not None
    assert guard.failure["reason"] == "command_expired"
    first = guard.failure["tick"]
    for _ in range(100):
        control_step(ctl, guard)
    assert guard.failure["tick"] == first
    assert guard.status in {"STOPPING", "STOPPED"}


def test_t13_reset_clears_fault():
    scene = FeedingScene("configs/m1_scene.json")
    cfg = load_m2_config()
    state = place_arm(scene, scene.config["q_torque_poses"][2])
    ctl, guard = start_controller(scene, cfg, state)
    ctl.wrench.filtered[:] = 3.0
    ctl.tau_prev[:] = 4.0
    guard.latch({"tick": 1, "time": 0.0, "reason": "wrench", "phase": "TRANSPORT", "pair": None, "geoms": None, "measured": 9, "limit": 8})
    fresh = place_arm(scene, scene.config["q_torque_poses"][2])
    ctl.reset(fresh, np.zeros(7), "TRANSPORT")
    assert guard.failure is None
    assert np.allclose(ctl.wrench.filtered, 0)
    assert np.allclose(ctl.tau_prev, 0)
