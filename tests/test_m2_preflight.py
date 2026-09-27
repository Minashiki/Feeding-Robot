"""Public entry points must reject corrupt pre-step samples without advancing."""
import copy

import numpy as np
import pytest

from feedingrobot.controllers.cartesian_impedance import control_step
from feedingrobot.sim.scene import FeedingScene
from feedingrobot.validation.m2.spec import VARIANTS
from tests.m2_support import load_m2_config, place_arm, start_controller


@pytest.mark.parametrize('pose', [0, 1, 2])
@pytest.mark.parametrize('variant', ['A', 'B', 'C'])
@pytest.mark.parametrize('status', ['power_on', 'RUNNING', 'STOPPING', 'STOPPED'])
@pytest.mark.parametrize('fault', ['qacc', 'outside_qvel', 'warning'])
@pytest.mark.parametrize('entry', ['compute', 'control_step'])
def test_preflight_no_advance(pose, variant, status, fault, entry):
    scene = FeedingScene('configs/m1_scene.json')
    dt, iterations, tolerance = VARIANTS[variant]
    scene.apply_experiment(timestep=dt, iterations=iterations, tolerance=tolerance)
    state = place_arm(scene, scene.config['q_torque_poses'][pose])
    ctl, guard = start_controller(scene, load_m2_config(), state)
    if status != 'power_on':
        ctl.power_on = False
        guard.status = status
    if status in {'STOPPING', 'STOPPED'}:
        guard.failure = {'reason': 'prior-fault', 'tick': -1}
    first = copy.deepcopy(guard.failure)
    if fault == 'qacc':
        scene.data.qacc[-1] = np.nan
    elif fault == 'outside_qvel':
        scene.data.qvel[-1] = np.inf
    else:
        scene.data.warning[0].number = 1
    tick, time = scene.tick, scene.data.time
    for _ in range(2):
        if entry == 'compute':
            _, info = ctl.compute(scene.snapshot(), dt)
        else:
            _, info = control_step(ctl, guard)
        assert not info['apply'] and guard.status == 'ABORTED'
        assert (scene.tick, scene.data.time) == (tick, time)
        assert ctl._command is None
    if first is not None:
        assert guard.failure == first
    else:
        assert guard.failure['reason'] == ('warning' if fault == 'warning' else 'non-finite')
        assert guard.failure['tick'] == tick


@pytest.mark.parametrize('power_on', [False, True])
def test_unknown_phase_has_no_side_effect(power_on):
    scene = FeedingScene('configs/m1_scene.json')
    state = place_arm(scene, scene.config['q_torque_poses'][0])
    ctl, guard = start_controller(scene, load_m2_config(), state)
    ctl.power_on = power_on
    ctl.set_command(np.zeros(6), ctl.now(), ctl.now() + .1, 'TRANSPORT')
    before = copy.deepcopy((ctl._command, ctl.phase, ctl._k, guard.failure))
    tick, time = scene.tick, scene.data.time
    with pytest.raises(ValueError, match='unknown phase'):
        control_step(ctl, guard, 'TYPO')
    after = (ctl._command, ctl.phase, ctl._k, guard.failure)
    for old, new in zip(before, after):
        if isinstance(old, np.ndarray):
            np.testing.assert_array_equal(old, new)
        elif isinstance(old, dict):
            for key in old:
                np.testing.assert_equal(old[key], new[key])
        else:
            assert old == new
    assert (scene.tick, scene.data.time) == (tick, time)


def test_fatal_state_requires_reset_without_guard():
    import mujoco
    scene = FeedingScene('configs/m1_scene.json')
    state = place_arm(scene, scene.config['q_torque_poses'][0])
    ctl, _ = start_controller(scene, load_m2_config(), state)
    ctl.guard = None
    scene.data.qacc[-1] = np.nan
    assert not ctl.compute(scene.snapshot(), scene.dt)[1]['apply']
    mujoco.mj_forward(scene.model, scene.data)
    assert scene.snapshot()['finite']
    assert not ctl.compute(scene.snapshot(), scene.dt)[1]['apply']
    ctl.reset(scene.snapshot(), np.zeros(7))
    assert ctl.compute(scene.snapshot(), scene.dt)[1]['apply']


@pytest.mark.parametrize('entry', ['compute', 'control_step'])
def test_existing_abort_cancels_cached_command(entry):
    scene = FeedingScene('configs/m1_scene.json')
    state = place_arm(scene, scene.config['q_torque_poses'][0])
    ctl, guard = start_controller(scene, load_m2_config(), state)
    ctl.set_command(np.ones(6) * .001, ctl.now(), ctl.now() + .1, 'TRANSPORT')
    assert ctl._command is not None
    guard.latch({'reason': 'non-finite', 'tick': scene.tick})
    if entry == 'compute':
        _, info = ctl.compute(scene.snapshot(), scene.dt)
    else:
        _, info = control_step(ctl, guard)
    assert not info['apply'] and ctl._command is None
    assert scene.tick == 0
