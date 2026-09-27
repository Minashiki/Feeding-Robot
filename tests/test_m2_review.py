"""Regressions for the six M2 review findings."""
import copy

import numpy as np
import pytest

from feedingrobot.controllers.cartesian_impedance import control_step
from feedingrobot.controllers.guard import contact_allowed
from feedingrobot.sim.scene import FeedingScene
from tests.m2_support import load_m2_config, place_arm, start_controller


@pytest.mark.parametrize('phase', ['STOP', 'RECOVER'])
@pytest.mark.parametrize('variant', ['A', 'B', 'C'])
@pytest.mark.parametrize('elapsed', [0., .02])
@pytest.mark.parametrize('pose', [0, 1, 2])
def test_stop_during_takeover(phase, variant, elapsed, pose):
    from feedingrobot.validation.m2.spec import VARIANTS
    scene = FeedingScene('configs/m1_scene.json')
    state = place_arm(scene, scene.config['q_torque_poses'][pose])
    dt, iterations, tolerance = VARIANTS[variant]
    scene.apply_experiment(timestep=dt, iterations=iterations, tolerance=tolerance)
    ctl, guard = start_controller(scene, load_m2_config(), state)
    for _ in range(round(elapsed / dt)):
        control_step(ctl, guard)
    assert ctl.power_on
    tick = scene.tick
    _, info = control_step(ctl, guard, phase)
    assert info['execution'] == 'stop'
    assert guard.events == [{'kind': 'stop_request', 'tick': tick, 'time': pytest.approx(elapsed), 'phase': phase}]
    np.testing.assert_allclose(info['tau_raw'], info['tau_bias'] - ctl.stop_damping * info['dq'])
    assert not np.any(info['tau_task']) and not np.any(info['tau_null'])
    assert not ctl.set_command(np.ones(6), ctl.now(), ctl.now()+.1, 'TRANSPORT')['accepted']
    clock = guard._stop_since
    for _ in range(round(.75 / dt)):
        state, info = control_step(ctl, guard, phase)
        assert info['execution'] == 'stop'
        assert np.max(np.abs(state['dq'])) <= .5+1e-9
        assert state['finite'] and not any(state['warnings'])
        assert np.all(np.abs(info['tau_cmd']) <= ctl.tau_max+1e-9)
        assert np.max(np.abs(info['tau_cmd']-info['tau_before'])) <= ctl.rate*dt+1e-9
    assert guard.stopped_ok and guard.status == 'STOPPED', (guard.failure, guard._slow_since, guard._stop_since, scene.snapshot()['dq'])
    assert guard._stop_since == clock
    assert len([e for e in guard.events if e['kind'] == 'stop_request']) == 1


@pytest.mark.parametrize('reverse', [False, True])
def test_unknown_tool_contact_is_rejected(reverse):
    names = ['bowl_front', 'unclassified_obstacle']
    if reverse:
        names.reverse()
    assert not contact_allowed('TRANSFER', *names)
    scene = FeedingScene('configs/m1_scene.json')
    state = place_arm(scene, scene.config['q_torque_poses'][0])
    _, guard = start_controller(scene, load_m2_config(), state)
    state['contacts'] = [{'geom1': names[0], 'geom2': names[1], 'dist': -.0001}]
    guard.observe(state, 'TRANSFER')
    assert guard.failure['reason'] == 'forbidden_contact'
    assert guard.status == 'STOPPING'


@pytest.fixture(scope='module')
def identity_trace(tmp_path_factory):
    from feedingrobot.validation.m2.spec import Case
    from feedingrobot.validation.m2.runner import run_group
    case = Case('stop', axis=1, sign=-1, event='expired', context='spring')
    out = tmp_path_factory.mktemp('identity')
    traces = run_group(case, 0, load_m2_config(), str(out))
    return case, traces, out


@pytest.mark.parametrize('change', ['seed', 'variant', 'config', 'solver', 'initial', 'geom', 'execution'])
def test_identity_tamper_rejected(identity_trace, change):
    from feedingrobot.validation.m2.checks import score
    case, traces, _ = identity_trace
    data, meta = traces[0]
    assert score(case, 0, 'A', data, meta)[0] == []
    meta = copy.deepcopy(meta)
    seed, variant = 0, 'A'
    if change == 'seed':
        seed = 3
        meta.update(case.manifest(seed, variant))
    elif change == 'variant':
        variant = 'C'
        meta.update(case.manifest(seed, variant))
    elif change == 'config':
        meta['effective_controller']['gears']['FREE']['kp'] = 99999.
    elif change == 'solver':
        meta['execution']['solver_after']['iterations'] = 49
    elif change == 'initial':
        meta['initial_source']['settled']['qpos'][0] += .01
    elif change == 'geom':
        meta['geometry_registry'][0]['group'] = 'food'
    else:
        meta['execution']['id'] = 'another-execution'
    assert score(case, seed, variant, data, meta)[0]


def test_independent_pair_and_receipts(identity_trace):
    import json
    from feedingrobot.validation.m2.package import compare
    from feedingrobot.validation.m2.checks import score
    from feedingrobot.validation.m2.provenance import check_receipt
    from feedingrobot.validation.m2.package import digest
    case, traces, out = identity_trace
    ids = []
    values = []
    for variant, (data, meta) in zip('ABC', traces):
        path = out/case.manifest(0, variant)['case_id']
        receipt = json.loads(path.with_suffix('.execution.json').read_text())
        assert check_receipt(case, 0, variant, data, meta, receipt, digest(str(path)+'.npz'), digest(str(path)+'.json')) == []
        receipt['execution']['id'] = 'reused'
        assert check_receipt(case, 0, variant, data, meta, receipt, digest(str(path)+'.npz'), digest(str(path)+'.json'))
        errors, metrics = score(case, 0, variant, data, meta)
        assert not errors
        values.append(metrics)
        ids.append(meta['execution']['id'])
    assert len(set(ids)) == 3
    assert compare(values[0], values[1]) == compare(values[0], values[2]) == []
    # Equal numerical outcomes do not imply a reused execution.
    equal = {**values[0], 'execution_id': 'independent-C', 'variant': 'C'}
    assert compare(values[0], equal) == []
    assert compare(values[0], {**equal, 'execution_id': values[0]['execution_id']})


@pytest.mark.parametrize('change', ['missing', 'duplicate', 'unknown', 'failed', 'skipped', 'xfail', 'teardown', 'mixed', 'run'])
def test_execution_tamper_rejected(change):
    from feedingrobot.validation.m2.provenance import check_execution
    nodes = ['test_a', 'test_b']
    execution = {'run_id': 'run', 'exitstatus': 0, 'collected': nodes.copy(),
                 'reports': [{'nodeid': n, 'when': w, 'outcome': 'passed'} for n in nodes for w in ('setup', 'call', 'teardown')]}
    assert check_execution(execution, nodes, 'run') == []
    if change == 'missing': execution['collected'].pop()
    elif change == 'duplicate': execution['collected'] = ['test_a']*363
    elif change == 'unknown': execution['collected'][0] = 'test_other'
    elif change in {'failed', 'skipped'}: execution['reports'][1]['outcome'] = change
    elif change == 'xfail': execution['reports'][1]['wasxfail'] = 'expected'
    elif change == 'teardown': execution['reports'].pop()
    elif change == 'mixed': execution['reports'].append({**execution['reports'][1], 'outcome': 'failed'})
    else: execution['run_id'] = 'other'
    assert check_execution(execution, nodes, 'run')


def test_model_contact_ids_and_names():
    from feedingrobot.controllers.guard import geometry_registry, recorded_contact_allowed
    scene = FeedingScene('configs/m1_scene.json')
    registry = geometry_registry(scene.model)
    ids = {r['name']: r['id'] for r in registry if r['name']}
    row = {'geom1': 'bowl_bottom', 'geom2': 'plate_bottom',
           'geom1_id': ids['bowl_bottom'], 'geom2_id': ids['plate_bottom']}
    assert recorded_contact_allowed('ACQUIRE', row, {}, registry)
    assert not recorded_contact_allowed('TRANSFER', row, {}, registry)
    row['geom2_id'] = -1
    assert not recorded_contact_allowed('ACQUIRE', row, {}, registry)
    row['geom2_id'] = ids['jaw_lip']
    assert not recorded_contact_allowed('ACQUIRE', row, {}, registry)
    arm = next(r for r in registry if r['group'] == 'arm' and r['name'] is None)
    row.update(geom2=None, geom2_id=arm['id'])
    assert not recorded_contact_allowed('TRANSFER', row, {}, registry)


def test_calibration_package_fail_closed(identity_trace, tmp_path, monkeypatch):
    import json
    import shutil
    from feedingrobot.validation.m2 import package as a
    from feedingrobot.validation.m2.provenance import CONFIG_PATHS
    from feedingrobot.validation.m2.spec import SCHEMA
    from feedingrobot.sim.model import repo_root
    case, _, folder = identity_trace
    shutil.copytree(folder, tmp_path/'cases')
    monkeypatch.setattr(a, 'cases', lambda: [case])
    monkeypatch.setattr(a, 'CALIBRATION_SEEDS', (0,))
    manifest = [case.manifest(0, v) for v in 'ABC']
    monkeypatch.setattr(a, 'manifest', lambda seeds: manifest)
    def write(name, value):
        (tmp_path/name).write_text(json.dumps(value))
    header = {'schema_version': SCHEMA, 'stage': 'calibration', 'run_id': 'unit', 'incomplete': True, 'm3_ready': False}
    write('run.json', header)
    write('input_hash_before.json', a.inputs())
    write('input_hash_after.json', a.inputs())
    write('case_manifest.json', manifest)
    write('evidence_manifest.json', {p.name: a.digest(p) for p in (tmp_path/'cases').iterdir()})
    for path in CONFIG_PATHS:
        shutil.copyfile(repo_root()/path, tmp_path/path.rsplit('/', 1)[-1])
    assert not a.evaluate(tmp_path, calibration=True)['scope_passed']
    core = a.evaluate(tmp_path, calibration=True, _require_negatives=False)
    assert core['scope_passed'] and not core['m3_ready'], core['reasons']
    assert a.finalize(tmp_path, calibration=True) == core
    import mujoco
    from feedingrobot.validation.m2 import negatives, runner
    def forbid(*args, **kwargs):
        raise AssertionError('verification must be read-only')
    monkeypatch.setattr(mujoco, 'mj_step', forbid)
    monkeypatch.setattr(negatives, 'run_negative', forbid)
    monkeypatch.setattr(runner, 'run_group', forbid)
    before_verify = {str(p.relative_to(tmp_path)): a.digest(p) for p in tmp_path.rglob('*') if p.is_file()}
    assert a.evaluate(tmp_path, calibration=True, check_workspace=True) == core
    from feedingrobot.scripts.verify_m2_report import assess_m2, main
    import sys
    assert assess_m2(tmp_path, 'calibration', True) == core
    monkeypatch.setattr(sys, 'argv', ['verify_m2_report', '--run', str(tmp_path),
                                    '--scope', 'calibration', '--check-current-workspace'])
    main()
    assert before_verify == {str(p.relative_to(tmp_path)): a.digest(p) for p in tmp_path.rglob('*') if p.is_file()}
    write('report.json', {**core, 'm3_ready': True})
    assert 'report_mismatch' in {r['code'] for r in a.evaluate(tmp_path, calibration=True)['reasons']}
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 1
    write('report.json', core)
    cfg=json.loads((tmp_path/'m2_controller.json').read_text())
    cfg['gears']['FREE']['kp']=99999.
    write('m2_controller.json', cfg)
    assert 'config_snapshot' in {r['code'] for r in a.evaluate(tmp_path, calibration=True)['reasons']}
