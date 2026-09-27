"""Public verification and the S5 event/steady-load acceptance contracts."""
import copy
import json
import sys

import numpy as np
import pytest

from feedingrobot.scripts import verify_m2_report as cli
from feedingrobot.validation.m2 import package
from feedingrobot.validation.m2.spec import SCHEMA


@pytest.mark.parametrize('stage', ['calibration', 'full'])
@pytest.mark.parametrize('scope', ['calibration', 'full', 'fixes-v6'])
def test_current_schema_cli_dispatch(tmp_path, monkeypatch, capsys, stage, scope):
    (tmp_path/'run.json').write_text(json.dumps({'schema_version': SCHEMA, 'stage': stage}))
    calls = []
    def evaluate(run, **kwargs):
        calls.append((run, kwargs))
        return package.result([], {}, [], stage == 'calibration')
    def forbid(*args, **kwargs):
        raise AssertionError('full evidence routed to legacy')
    monkeypatch.setattr(package, 'evaluate', evaluate)
    monkeypatch.setattr(cli, 'verify_finished_report', forbid)
    monkeypatch.setattr(sys, 'argv', ['verify_m2_report', '--run', str(tmp_path), '--scope', scope,
                                    '--check-current-workspace'])
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    if stage == scope:
        cli.main()
    else:
        with pytest.raises(SystemExit) as error:
            cli.main()
        assert error.value.code == 1
    verdict = json.loads(capsys.readouterr().out)
    assert verdict['scope_passed'] == (stage == scope)
    assert verdict['m3_ready'] == (stage == scope == 'full')
    assert calls == [(tmp_path, {'check_workspace': True, 'calibration': stage == 'calibration'})]
    assert before == {p.name: p.read_bytes() for p in tmp_path.iterdir()}


@pytest.mark.parametrize('schema', ['m2-full-v1', 'm2-full-v3', 'm2-full-v4', 'unknown'])
def test_unsupported_schema_is_rejected(tmp_path, schema):
    (tmp_path/'run.json').write_text(json.dumps({'schema_version': schema, 'stage': 'calibration'}))
    verdict = cli.assess_m2(tmp_path, 'calibration', False)
    assert not verdict['scope_passed'] and not verdict['m3_ready']
    assert 'unsupported_schema' in {r['code'] for r in verdict['reasons']}


def pair_metrics(family='spring'):
    left = dict.fromkeys(['position_rms', 'rotation_rms', 'tracking_position_rms',
                         'tracking_rotation_rms', 'force_peak', 'force_mean', 'moment_peak',
                         'tau_peak', 'impulse', 'contact_force_peak', 'contact_moment_peak',
                         'contact_impulse'], 0.)
    left.update(family=family, contact_force_mean=1., initial_qpos=[0.], initial_qvel=[0.],
                initial_source_identity='shared', execution_id='A', variant='A', events={})
    return left, {**left, 'execution_id': 'B', 'variant': 'B'}


@pytest.mark.parametrize('family', ['spring', 'press'])
@pytest.mark.parametrize('change', ['double', 'missing_one', 'missing_both', 'nan', 'within'])
def test_steady_contact_pair_required(family, change):
    left, right = pair_metrics(family)
    if change == 'double': right['contact_force_mean'] = 2.
    elif change == 'missing_one': right.pop('contact_force_mean')
    elif change == 'missing_both':
        left.pop('contact_force_mean'); right.pop('contact_force_mean')
    elif change == 'nan': right['contact_force_mean'] = float('nan')
    else: right['contact_force_mean'] = 1.1
    errors = package.compare(left, right)
    assert bool(errors) == (change != 'within'), errors
    if errors:
        assert any('contact_force_mean' in r['code'] for r in errors)


@pytest.mark.parametrize('change', ['missing', 'late', 'boundary', 'nan'])
def test_block_confirmation_pair(change):
    left, right = pair_metrics('wall')
    left['events'] = {'blocked_confirm': 1.6}
    right['events'] = {'blocked_confirm': 1.62 if change == 'boundary' else 1.621}
    if change == 'missing': right['events'] = {}
    if change == 'nan': right['events']['blocked_confirm'] = float('nan')
    assert bool(package.compare(left, right)) == (change != 'boundary')


@pytest.mark.parametrize('dt', [.001, .0005])
@pytest.mark.parametrize('force', [-7., 0.])
def test_confirmation_is_independent_of_backcalculation(dt, force):
    from feedingrobot.controllers.reference import ReferenceShaper
    from feedingrobot.validation.m2.reference_checks import check_reference_path
    from tests.m2_support import load_m2_config
    cfg = load_m2_config()
    ref = ReferenceShaper(cfg)
    origin = np.array([0., .006, 0.])
    ref.anchor_pose(origin, np.eye(3), np.zeros(7))
    limits = {**cfg['gears']['FREE'], 'pos_dev': .02, 'rot_dev': cfg['gears']['FREE']['rot_dev_rad']}
    n = round(.31/dt)
    twist = np.array([.001, .02, 0., 0., 0., 0.])
    rows = []
    for k in range(n):
        before = {'p_ref_before': ref.p_ref.copy(), 'r_ref_before': ref.r_ref.copy()}
        row = ref.shape(twist, np.zeros(3), np.eye(3), np.array([force, 0., 0.]), dt,
                        limits, 1., 1., True, False)
        rows.append({**row, **before})
    data = {key: np.array([r[key] for r in rows]) for key in
            ['p_ref_before', 'r_ref_before', 'p_ref', 'r_ref', 'v_ref', 'v_hist', 'scale', 'beta',
             'twist_limited', 'reference_correction_pos', 'reference_correction_rot', 'blocked',
             'blocked_now', 'block_progress', 'block_correction_pos']}
    data.update(t=np.arange(n+1)*dt, tau_cmd=np.zeros((n, 7)), tau_raw=np.zeros((n, 7)),
                phase_k=np.full(n, 'TRANSPORT'), execution=np.full(n, 'run'), origin=origin,
                rotation=np.eye(3), tcp_pos=np.zeros((n+1, 3)), tcp_mat=np.tile(np.eye(3), (n+1, 1, 1)),
                rho=np.full(n, .05), command_times=np.c_[np.arange(n)*dt, np.arange(n)*dt+.1],
                supplied=np.tile(twist, (n, 1)), twist_command=np.tile(twist, (n, 1)),
                candidate_twist=np.array([r['candidate'] for r in rows]),
                initial_compensated=np.array([force, 0., 0., 0., 0., 0.]),
                ft_compensated_wrench_tcp=np.tile([force, 0., 0., 0., 0., 0.], (n, 1)))
    errors, events = check_reference_path(data, cfg, dt)
    assert not errors, errors
    assert not np.any(data['block_correction_pos'])
    assert events == ({'blocked_confirm': pytest.approx(.02+.25-dt)} if force else {})
    altered = copy.deepcopy(data)
    altered['blocked_now'][:] = not bool(force)
    failures, rebuilt = check_reference_path(altered, cfg, dt)
    assert rebuilt == events
    assert 'reference:blocked_now' in {r['code'] for r in failures}


def test_steady_force_uses_physical_hold_window():
    from feedingrobot.validation.m2.runner import prepare, run_variant
    from feedingrobot.validation.m2.checks import score
    from feedingrobot.validation.m2.spec import Case
    from tests.m2_support import load_m2_config
    case = Case('spring')
    scene, cfg, initial = prepare(case, 0, load_m2_config())
    data, meta = run_variant(case, 0, 'A', scene, cfg, initial)
    errors, values = score(case, 0, 'A', data, meta)
    assert not errors, errors
    hold = (data['t'][1:] >= 5.25) & (data['t'][1:] <= 6.25)
    expected = np.mean(np.linalg.norm(data['ft_compensated_wrench_tcp'][hold, :3], axis=1))
    assert expected > .01
    assert values['force_mean'] == pytest.approx(expected, abs=1e-12)
    assert np.mean(np.linalg.norm(data['ft_compensated_wrench_tcp'][data['t'][1:] >= 9., :3], axis=1)) < .01
    right = {**values, 'execution_id': 'independent-B', 'variant': 'B',
             'force_mean': values['force_mean']*2, 'contact_force_mean': values['contact_force_mean']*2}
    assert {'convergence:force_mean', 'convergence:contact_force_mean'} <= {r['code'] for r in package.compare(values, right)}
    errors, _ = score(case, 0, 'A', {**data, 't': data['t']+20.}, meta)
    assert 'steady_force_window' in {r['code'] for r in errors}


@pytest.mark.parametrize('change', ['inf', 'nan'])
def test_nonfinite_steady_pair_rejected(change):
    # The score-level physical test above covers real sampling. This small test
    # proves nonfinite pair metrics cannot silently compare as "within tolerance".
    left, right = pair_metrics()
    right['force_mean'] = float('nan') if change == 'nan' else float('inf')
    assert 'convergence:force_mean' in {r['code'] for r in package.compare(left, right)}
