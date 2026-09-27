"""Stop and expiry must interrupt loaded motion, blockage and wall release."""
import json
import numpy as np
import pytest

from feedingrobot.controllers.cartesian_impedance import CartesianImpedance
from feedingrobot.validation.m2 import runner
from feedingrobot.validation.m2.spec import Case, VARIANTS, TAU_MAX
from tests.m2_support import load_m2_config


@pytest.mark.parametrize('variant', ['A', 'B', 'C'])
@pytest.mark.parametrize('event', ['STOP', 'expired'])
@pytest.mark.parametrize('context,at', [('loaded', 1.5), ('blocked', 2.), ('release', 3.2)])
def test_stop_in_motion_context(tmp_path, monkeypatch, variant, event, context, at):
    case = Case('loaded_circle', load=1.) if context == 'loaded' else Case('wall')
    scene, config, initial = runner.prepare(case, 0, load_m2_config())
    step = runner.control_step
    setter = CartesianImpedance.set_command
    if event == 'STOP':
        def stop(ctl, guard, *args, **kwargs):
            if ctl.scene.snapshot()['episode_time'] >= at-1e-9:
                kwargs['phase'] = 'STOP'
            return step(ctl, guard, *args, **kwargs)
        monkeypatch.setattr(runner, 'control_step', stop)
    else:
        def expire(ctl, twist, timestamp, valid_until, phase=None):
            if timestamp >= at-.05-1e-9:
                return {'accepted': False, 'reason': 'test publisher stopped'}
            return setter(ctl, twist, timestamp, valid_until, phase)
        monkeypatch.setattr(CartesianImpedance, 'set_command', expire)
    data, meta = runner.run_variant(case, 0, variant, scene, config, initial)
    np.savez_compressed(tmp_path/'trace.npz', **data)
    (tmp_path/'metadata.json').write_text(json.dumps({'context':context,'event':event,'at':at,**meta}, indent=2))
    t = data['t'];dt = VARIANTS[variant][0]
    stopping = np.flatnonzero(data['execution'] == 'stop')
    assert stopping.size and abs(t[stopping[0]]-at) <= dt+1e-9
    k = int(stopping[0])
    assert np.all(data['execution'][k:] == 'stop')
    assert np.max(np.abs(data['twist_command'][k:])) == 0.
    assert np.max(np.abs(data['tau_task'][k:])) == 0.
    assert meta['stopped_ok'] and data['status'][-1] == 'STOPPED'
    window = (t >= at+.5-1e-9) & (t <= at+.7+1e-9)
    assert np.sum(window) >= round(.2/dt)
    assert np.max(np.abs(data['physical_dq'][window])) <= .02+1e-9
    assert np.max(np.abs(data['physical_dq'])) <= .5+1e-9
    assert np.all(np.abs(data['tau_cmd']) <= TAU_MAX+1e-9)
    assert np.max(np.abs(data['tau_cmd']-data['tau_before'])) <= config['tau_rate_nm_s']*dt+1e-6
    assert np.max(np.linalg.norm(data['ft_compensated_wrench_tcp'][:,:3],axis=1)) <= 8.+1e-9
    assert np.max(np.linalg.norm(data['ft_compensated_wrench_tcp'][:,3:],axis=1)) <= .8+1e-9
    assert not np.any(data['warnings']) and np.all(data['finite'])
    assert np.all(data['contact_dist'] >= -.001-1e-9)
    if context == 'loaded':
        assert np.any(np.linalg.norm(data['external'][:k,:3],axis=1) > .9)
        assert not np.any(data['blocked_now'][:k])
    else:
        assert np.any(np.linalg.norm(data['block_correction_pos'][:k],axis=1) > 1e-12)
    if context == 'release':
        assert np.max(data['fixture'][:k,3]) > 0.
    if event == 'expired':
        assert meta['fault']['reason'] == 'command_expired'
        assert abs(meta['fault']['time']-at) <= dt+1e-9
    else:
        assert meta['fault'] is None


@pytest.mark.parametrize('variant', ['A', 'B', 'C'])
def test_valid_withdrawal_after_wall_release(tmp_path, monkeypatch, variant):
    case = Case('wall')
    scene, config, initial = runner.prepare(case, 0, load_m2_config())
    setter = CartesianImpedance.set_command
    normal = scene.fixture_normal.copy()
    def withdraw(ctl, twist, timestamp, valid_until, phase, frame='world'):
        if timestamp >= 3.6-1e-9:
            twist = np.r_[-.002*normal, np.zeros(3)]
        return setter(ctl, twist, timestamp, valid_until, phase, frame)
    monkeypatch.setattr(CartesianImpedance, 'set_command', withdraw)
    data, meta = runner.run_variant(case, 0, variant, scene, config, initial)
    np.savez_compressed(tmp_path/'trace.npz', **data)
    (tmp_path/'metadata.json').write_text(json.dumps(meta, indent=2))
    dt = VARIANTS[variant][0]
    start, end = round(3.6/dt), round(5.6/dt)
    assert np.any(data['blocked_now'][:start])
    assert not np.any(data['blocked_now'][start:])
    assert meta['fault'] is None
    assert np.allclose(data['twist_command'][start:,:3], -.002*normal, atol=1e-9)
    assert -(data['tcp_pos'][end]-data['tcp_pos'][start]) @ normal >= .002
    assert np.max(np.abs(data['physical_dq'])) <= .5+1e-9
    assert np.max(np.linalg.norm(data['tcp_pos'][:-1]-data['p_ref'],axis=1)) <= .02+1e-9
