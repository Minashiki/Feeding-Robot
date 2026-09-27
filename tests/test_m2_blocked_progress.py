"""Progress is measured from real positions; compliance alone is not blockage."""
import numpy as np
import pytest

from feedingrobot.controllers.reference import ReferenceShaper
from feedingrobot.validation.m2.reference_checks import check_analytic_motion
from feedingrobot.validation.m2.spec import Case
from tests.m2_support import load_m2_config


LIMITS = dict(v=.05, w=.3, a=.25, alpha=1.5, pos_dev=.02, rot_dev=.14)


def shaper():
    result = ReferenceShaper(load_m2_config())
    result.anchor_pose(np.zeros(3), np.eye(3), np.zeros(7))
    return result


def sample(ref, t, dt, actual, command, force=(-1., 0., 0.)):
    pos = np.asarray(actual) * t
    ref.p_ref = pos + np.array([.004, 0., 0.])
    return ref.shape(np.r_[command, np.zeros(3)], pos, np.eye(3), np.array(force),
                     dt, LIMITS, 1., 1., True, False)


@pytest.mark.parametrize('dt', [.001, .0005])
@pytest.mark.parametrize('speed,command_speed,blocked', [(.011, .015, False), (.0001, .0001, False),
                                                 (0., .002, True), (.00001, .002, True),
                                                 (-.001, .002, True)])
def test_progress_distinguishes_motion_from_stall(dt, speed, command_speed, blocked):
    ref = shaper()
    outputs = [sample(ref, k*dt, dt, [speed, 0., 0.], [command_speed, 0., 0.]) for k in range(round(.36/dt))]
    assert bool(any(row['blocked_now'] for row in outputs)) is blocked
    if blocked:
        first = next(k for k, row in enumerate(outputs) if row['blocked_now'])
        assert abs(first*dt - (.27-dt)) < 1e-10
    else:
        assert outputs[-1]['twist_limited'][0] > 0.


@pytest.mark.parametrize('dt', [.001, .0005])
def test_tangent_does_not_hide_normal_blockage(dt):
    ref = shaper()
    outputs = [sample(ref, k*dt, dt, [0., .002, 0.], [.002, .002, 0.]) for k in range(round(.36/dt))]
    final = next(row for row in outputs if row['blocked_now'])
    assert abs(final['twist_limited'][0]) < 1e-12
    assert final['twist_limited'][1] > .0019
    assert final['block_correction_pos'][0] == pytest.approx(-.002)
    np.testing.assert_array_equal(final['block_correction_pos'][1:], 0.)


@pytest.mark.parametrize('dt', [.001, .0005])
def test_zero_net_oscillation_cannot_clear_blockage(dt):
    ref = shaper()
    outputs = []
    for k in range(round(.64/dt)):
        pos = np.array([.00001*np.sin(2*np.pi*k*dt/.01), 0., 0.])
        ref.p_ref = pos + [.004, 0., 0.]
        outputs.append(ref.shape(np.array([.002, 0., 0., 0., 0., 0.]), pos, np.eye(3),
                                 np.array([-1., 0., 0.]), dt, LIMITS, 1., 1., True, False))
    assert sum(row['blocked_now'] for row in outputs) >= 2


@pytest.mark.parametrize('action', ['unload', 'reverse', 'zero', 'stop', 'saturation', 'reanchor', 'reset'])
def test_release_and_cancellation_clear_progress(action):
    ref = shaper()
    for k in range(400):
        row = sample(ref, k*.001, .001, [0., 0., 0.], [.002, 0., 0.])
        if row['blocked_now']:
            break
    assert row['blocked_now']
    if action == 'reanchor':
        ref.reanchor(np.zeros(3), np.eye(3))
    elif action == 'reset':
        ref.reset_motion()
        ref.anchor_pose(np.zeros(3), np.eye(3), np.zeros(7))
    command = [-.002, 0., 0.] if action == 'reverse' else [0., 0., 0.] if action in {'zero', 'stop'} else [.002, 0., 0.]
    force = [0., 0., 0.] if action == 'unload' else [-1., 0., 0.]
    row = ref.shape(np.r_[command, np.zeros(3)], np.zeros(3), np.eye(3), np.array(force),
                    .001, LIMITS, 1., 1., True, action == 'saturation',
                    execution='stop' if action == 'stop' else 'run')
    assert not row['blocked_now']
    assert row['block_progress'][-1] == 0.
    if action in {'reverse', 'zero', 'stop', 'saturation', 'reanchor', 'reset'}:
        assert row['block_progress'][3] == 0.
    if action == 'unload':
        assert row['twist_limited'][0] > 0.


def test_short_compliance_transient_is_not_confirmed():
    ref = shaper()
    pos = np.zeros(3)
    for k in range(400):
        if not 30 <= k < 230:
            pos[0] += .002*.001
        ref.p_ref = pos + [.004, 0., 0.]
        row = ref.shape(np.array([.002, 0., 0., 0., 0., 0.]), pos, np.eye(3), np.array([-1., 0., 0.]),
                        .001, LIMITS, 1., 1., True, False)
        assert not row['blocked_now']


def test_right_angle_command_clears_confirmation_window():
    ref = shaper()
    for k in range(80):
        row = sample(ref, k*.001, .001, [0., 0., 0.], [.002, 0., 0.])
    assert row['block_progress'][4] > .05
    row = sample(ref, .08, .001, [0., 0., 0.], [.002*np.cos(np.pi/2), .002, 0.])
    assert row['block_progress'][3] == 0. and row['block_progress'][4] == 0.


def test_progress_does_not_bypass_deviation_cap():
    ref = shaper()
    for k in range(150):
        pos = np.array([k*.001*.01, 0., 0.])
        ref.p_ref = pos + [.02, 0., 0.]
        row = ref.shape(np.array([.01, 0., 0., 0., 0., 0.]), pos, np.eye(3), np.zeros(3),
                        .001, LIMITS, 1., 1., True, False)
        assert np.linalg.norm(row['p_ref']-pos) <= .02+1e-12


@pytest.mark.parametrize('mutation', ['none', 'frozen', 'partial', 'small', 'shifted', 'orientation'])
def test_loaded_motion_uses_fixed_physical_compliance_center(mutation):
    t = np.linspace(0., 10., 1001)
    angle = .5*np.pi*np.clip(t-1., 0., 8.)
    target = np.column_stack([.01*(np.cos(angle)-1.), .01*np.sin(angle), np.zeros(len(t))])
    actual = target.copy()
    actual[t>1., 0] += 1./300.
    rotation = np.tile(np.eye(3), (len(t), 1, 1))
    if mutation == 'frozen': actual[:] = 0.
    if mutation == 'partial': actual[500:] = actual[499]
    if mutation == 'small': actual[:,:2] *= .3
    if mutation == 'shifted': actual[:,0] += .01
    if mutation == 'orientation': rotation[:] = [[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]]
    data = dict(t=t, origin=np.zeros(3), rotation=np.eye(3), p_ref=target[1:], tcp_pos=actual, tcp_mat=rotation)
    errors = check_analytic_motion(Case('loaded_circle', load=1.), data)
    assert bool(errors) is (mutation != 'none')
    assert all(row['code'].startswith('motion:loaded_circle') for row in errors)


@pytest.mark.parametrize('family', ['release', 'release_fast'])
def test_current_legacy_release_rebuilds_progress(tmp_path, monkeypatch, family):
    from pathlib import Path
    from feedingrobot.controllers.v3spec import _check_physical
    from tests.test_m2_v2 import test_physical as run_physical
    monkeypatch.setenv('M2_EVIDENCE_DIR', str(tmp_path/'cases'))
    (tmp_path/'controller_config.json').write_bytes(Path('configs/m2_controller.json').read_bytes())
    case = family + '-s0-A'
    run_physical(case)
    errors = []
    _check_physical(case, tmp_path, errors)
    assert errors == [], errors
    path = tmp_path/'cases'/(case+'.npz')
    with np.load(path) as archive:
        original = dict(archive)
    for field, index in [('block_progress', (100, 0)), ('block_progress', (100, 3)), ('block_progress', (100, 5)),
                         ('block_correction_pos', (100, 0))]:
        altered = {**original, field: original[field].copy()}
        altered[field][index] += .01
        np.savez(path, **altered)
        errors = []
        _check_physical(case, tmp_path, errors)
        assert any(row['code'] == 'event_mismatch' and row['field'] == field for row in errors)


def test_legacy_contract_cannot_be_downgraded_by_snapshot(tmp_path):
    import hashlib
    import json
    from pathlib import Path
    from feedingrobot.controllers.v3spec import evaluate_evidence
    config_bytes = Path('configs/m2_controller.json').read_bytes()
    recorded = {'files': [{'path': 'configs/m2_controller.json',
                           'sha256': hashlib.sha256(config_bytes).hexdigest()}]}
    for name in ['input_hash_before.json', 'input_hash_after.json']:
        (tmp_path/name).write_text(json.dumps(recorded))
    config = json.loads(config_bytes)
    config['version'] = 'm2-full-v2'
    (tmp_path/'controller_config.json').write_text(json.dumps(config))
    errors = evaluate_evidence(tmp_path)['reasons']
    assert any(row['code'] == 'input_hash' and row.get('field') == 'controller_config.json' for row in errors)


@pytest.mark.parametrize('dt', [.001, .0005])
def test_large_progress_deficit_confirms_before_slow_timeout(dt):
    ref = shaper()
    outputs = [sample(ref, k*dt, dt, [0., 0., 0.], [.05, 0., 0.]) for k in range(round(.1/dt))]
    first = next(k for k, row in enumerate(outputs) if row['blocked_now'])
    assert first*dt == pytest.approx(.05-dt, abs=1e-10)
    assert outputs[first]['block_correction_pos'][0] == pytest.approx(-.002)


def test_progress_deficit_does_not_accumulate_across_force_gaps():
    ref = shaper()
    for k in range(600):
        force = (-1., 0., 0.) if k % 30 < 25 else (0., 0., 0.)
        row = sample(ref, k*.001, .001, [0., 0., 0.], [.05, 0., 0.], force)
        assert not row['blocked_now']
        if force[0] == 0.:
            assert row['block_progress'][4] == row['block_progress'][5] == 0.
