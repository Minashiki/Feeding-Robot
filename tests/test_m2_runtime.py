"""Public stepping, controlled stops, and corrupt F/T regressions."""
import copy

import numpy as np
import pytest

from feedingrobot.controllers.cartesian_impedance import control_step
from feedingrobot.sim.scene import FeedingScene
from tests.m2_support import load_m2_config, place_arm, start_controller


def rig(pose=0, cfg=None):
    scene = FeedingScene("configs/m1_scene.json")
    state = place_arm(scene, scene.config["q_torque_poses"][pose])
    ctl, guard = start_controller(scene, cfg or load_m2_config(), state)
    drive(ctl, guard, 0.8, np.zeros(6))
    assert guard.failure is None
    return scene, ctl, guard


def drive(ctl, guard, duration, twist):
    samples = []
    for k in range(round(duration / ctl.scene.dt)):
        if k % round(0.05 / ctl.scene.dt) == 0:
            ctl.set_command(twist, ctl.now(), ctl.now() + 0.1, "TRANSPORT")
        samples.append(control_step(ctl, guard))
    return samples


@pytest.mark.parametrize("phase", ["STOP", "RECOVER"])
def test_explicit_stop_brakes_and_requires_reset(phase):
    scene, ctl, guard = rig()
    drive(ctl, guard, 0.4, np.array([0., .04, 0, 0, 0, 0]))
    assert ctl.set_command(np.zeros(6), ctl.now(), ctl.now() + .1, phase)["accepted"]
    _, info = control_step(ctl, guard)
    assert info["execution"] == "stop" and guard.failure is None
    np.testing.assert_allclose(info["tau_raw"], info["tau_bias"] - ctl.stop_damping * info["dq"])
    assert not np.any(info["tau_null"]) and not np.any(info["tau_task"])
    clock = guard._stop_since
    assert ctl.set_command(np.zeros(6), ctl.now(), ctl.now() + .1, phase)["idempotent"]
    assert guard._stop_since == clock
    assert not ctl.set_command(np.ones(6), ctl.now(), ctl.now() + .1, "TRANSPORT")["accepted"]
    for _ in range(800):
        control_step(ctl, guard)
    assert guard.status == "STOPPED" and guard.stopped_ok
    assert guard.events[0]["kind"] == "stop_request"


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("delay", [0., .005, .01])
def test_bad_wrench_aborts_without_polluting_filter(monkeypatch, bad, delay):
    cfg = load_m2_config()
    cfg["delay_s"] = delay
    scene, ctl, guard = rig(cfg=cfg)
    original = scene.snapshot
    target = scene.tick + 1
    filtered, buffer = ctl.wrench.filtered.copy(), copy.deepcopy(ctl.wrench._buffer)
    def snapshot():
        state = original()
        if scene.tick == target:
            state["raw_wrench_sensor"][0] = bad
        return state
    monkeypatch.setattr(scene, "snapshot", snapshot)
    _, info = control_step(ctl, guard)
    assert guard.status == "ABORTED" and not info["ft_valid"]
    np.testing.assert_array_equal(ctl.wrench.filtered, filtered)
    assert len(ctl.wrench._buffer) == len(buffer)
    tick, time = scene.tick, scene.data.time
    monkeypatch.setattr(scene, "snapshot", original)
    for _ in range(3):
        _, info = control_step(ctl, guard)
        assert not info["apply"] and scene.tick == tick and scene.data.time == time
    state = place_arm(scene, scene.config["q_torque_poses"][0])
    ctl.reset(state, np.zeros(7))
    assert guard.failure is None and guard.status == "RUNNING"
    assert not ctl.wrench._cache["valid"]
    drive(ctl, guard, .1, np.zeros(6))
    assert ctl.wrench._cache["valid"] and np.all(np.isfinite(ctl.wrench._cache["delivered_wrench_tcp"]))


@pytest.mark.parametrize("pose", range(3))
@pytest.mark.parametrize("axis", range(6))
@pytest.mark.parametrize("sign", [-1, 1])
def test_moving_expiry_stops(pose, axis, sign):
    scene, ctl, guard = rig(pose)
    twist = np.zeros(6)
    twist[axis] = sign * (.04 if axis < 3 else .2)
    drive(ctl, guard, .6, twist)
    assert guard.failure is None
    ctl.set_command(twist, ctl.now(), ctl.now() + .02, "TRANSPORT")
    states = [control_step(ctl, guard)[0] for _ in range(850)]
    assert guard.failure["reason"] == "command_expired"
    times = np.array([s["episode_time"] for s in states]) - guard.failure["time"]
    speed = np.array([np.max(np.abs(s["dq"])) for s in states])
    candidates = np.flatnonzero((times >= 0) & (times <= .5 + 1e-9))
    count = round(.2 / scene.dt) + 1
    assert any(len(speed[k:k+count]) == count and np.all(speed[k:k+count] <= .02 + 1e-12) for k in candidates)
    assert guard.stopped_ok


@pytest.mark.parametrize("pose", range(3))
def test_public_step_updates_torque_and_measurements(pose):
    scene, ctl, guard = rig(pose)
    for state, info in drive(ctl, guard, .1, np.zeros(6)):
        assert info["ft_sample_tick"] == state["tick"]
        assert info["ft_sample_time"] == state["episode_time"]
        assert info["ft_valid"] and info["ft_estimated_valid"]
    np.testing.assert_array_equal(ctl.tau_prev, scene.last_tau)
    assert guard.failure is None


@pytest.mark.parametrize('phase', ['STOP', 'RECOVER'])
def test_phase_argument_uses_same_stop_entry(phase):
    _, ctl, guard = rig()
    _, info = control_step(ctl, guard, phase)
    assert info['execution'] == 'stop'
    assert guard.failure is None
    first_time = guard._stop_since
    for _ in range(4):
        control_step(ctl, guard, phase)
    assert guard._stop_since == first_time
    assert len(guard.events) == 1


def test_estimator_uses_causal_velocity_and_rotational_inertia():
    _, ctl, _ = rig()
    pipeline = ctl.wrench
    pipeline.reset()
    bodies = len(pipeline.body_ids)
    sample = {
        'tool_velocity': np.zeros((bodies,6)),
        'tool_com': np.zeros((bodies,3)),
        'tool_inertia': np.tile(np.diag([.01,.02,.03]),(bodies,1,1)),
        'tcp_position': np.zeros(3),
        'tcp_wrench_world': np.zeros(6),
        'bias_wrench_tcp': np.zeros(6),
        # A bogus exact result must have no influence on the estimator.
        'exact_model': np.full(6,12345.),
    }
    _, valid = pipeline._estimated(sample,0.)
    assert not valid
    sample['tool_velocity'][:,:3] = [.1,.2,.3]
    sample['tool_velocity'][:,3:] = [.01,.02,.03]
    value, valid = pipeline._estimated(sample,.1)
    assert valid
    mass = sum(pipeline.model.body_mass[i] for i in pipeline.body_ids)
    expected_force = -mass*(pipeline.model.opt.gravity-np.array([.1,.2,.3]))
    inertia = np.diag([.01,.02,.03])
    expected_moment = bodies*(inertia@np.array([1.,2.,3.])+np.cross([.1,.2,.3],inertia@np.array([.1,.2,.3])))
    np.testing.assert_allclose(value,np.r_[expected_force,expected_moment])
    pipeline.reset()
    assert pipeline._kinematics_prev is None and pipeline._kinematics_time is None


@pytest.mark.parametrize('bad', [np.nan,np.inf,-np.inf])
def test_nonfinite_compensation_aborts_before_filter(monkeypatch,bad):
    scene,ctl,guard = rig()
    original = ctl.wrench.measure
    before = ctl.wrench.filtered.copy()
    updates = ctl.wrench.filter_updates
    def measure(*args,**kwargs):
        result = original(*args,**kwargs)
        result['compensated_wrench_tcp'][3] = bad
        return result
    monkeypatch.setattr(ctl.wrench,'measure',measure)
    _,info = control_step(ctl,guard)
    assert guard.status == 'ABORTED' and not info['ft_valid']
    np.testing.assert_array_equal(ctl.wrench.filtered,before)
    assert ctl.wrench.filter_updates == updates
    tick = scene.tick
    assert not control_step(ctl,guard)[1]['apply'] and scene.tick == tick


@pytest.mark.parametrize('pose',range(3))
def test_executable_public_example(pose):
    from feedingrobot.scripts.m2_control_example import run
    assert run(pose)['passed']
