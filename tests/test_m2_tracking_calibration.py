"""Tracking failure diagnostics and version-bound damping reconstruction."""
import numpy as np
import pytest
from feedingrobot.controllers.v4fields import rebuild_gain_path, check_logged_gains
from feedingrobot.controllers.v3spec import _check_unit
from feedingrobot.validation.m2.reference_checks import check_analytic_motion
from feedingrobot.validation.m2.spec import Case


@pytest.mark.parametrize('metric', ['rms', 'maximum', 'both', 'none'])
def test_loaded_circle_failure_reports_triggering_metric(metric):
    t = np.arange(10001) * .001
    a = .5*np.pi*np.clip(t-1., 0., 8.)
    target = np.column_stack([.01*(np.cos(a)-1), .01*np.sin(a), np.zeros(len(t))])
    actual = target.copy()
    actual[t>1., 1] += 1/330.
    if metric in {'rms', 'both'}:
        actual[(t>=1.) & (t<=9.), 2] += .00203
    if metric in {'maximum', 'both'}:
        actual[2000, 2] += .006
    data = dict(t=t, origin=np.zeros(3), rotation=np.eye(3), p_ref=target[1:],
                tcp_pos=actual, tcp_mat=np.tile(np.eye(3), (len(t), 1, 1)))
    failures = check_analytic_motion(Case('loaded_circle',axis=1,load=1.), data)
    if metric == 'none':
        assert failures == []
        return
    assert len(failures) == 1
    row = failures[0]
    assert row['code'] == 'motion:loaded_circle_target'
    assert row['metric'] == ('rms' if metric in {'rms','both'} else 'maximum')
    assert row['observed'] == row[row['metric']]
    assert row['observed'] > row['limit']
    assert row['rms_limit'] == .002 and row['max_limit'] == .005
    if metric == 'rms':
        assert row['maximum'] < row['max_limit']
    if metric == 'maximum':
        assert row['rms'] < row['rms_limit']


@pytest.mark.parametrize('dt', [.001, .0005])
@pytest.mark.parametrize('version,kp,dp', [('m2-full-v5',300.,49.), ('m2-full-v5-tracking5',330.,40.)])
def test_legacy_gain_profile_and_blend_are_version_bound(dt, version, kp, dp):
    phase = np.array(['TRANSPORT']*round(.05/dt)+['ACQUIRE']*round(.25/dt))
    built = rebuild_gain_path(phase, dt, controller_version=version)
    assert built['K'][0,0] == kp
    assert built['D'][0,0] == dp
    switch = round(.05/dt)
    assert built['D'][switch,0] == pytest.approx(dp+(35.-dp)*dt/.2)
    assert built['D'][-1,0] == 35.
    errors = []
    args = (built['K'], built['D'], built['trans'], built['active'], built['target'], phase, dt, 'gear', errors)
    check_logged_gains(*args, require_acquire=True, controller_version=version)
    assert errors == []
    other = 'm2-full-v5' if version != 'm2-full-v5' else 'm2-full-v5-tracking5'
    check_logged_gains(*args, require_acquire=True, controller_version=other)
    assert any(row['code']=='gain_path_mismatch' for row in errors)
    assert rebuild_gain_path(phase,dt)['D'][0,0] == 49.  # Historical default was not mutated.


@pytest.mark.parametrize('version,kp,dp', [('m2-full-v5',300.,49.), ('m2-full-v5-tracking5',330.,40.)])
def test_legacy_unit_gain_target_is_version_bound(version, kp, dp):
    name = 'F5-FREE-ACQUIRE-50-0.001'
    start = np.array([150.,5.,35.,.63]); end = np.array([kp,8.,dp,.8])
    row = dict(case_id=name, gains=np.linspace(start,end,201),k_start=start,k_target=end,dt=.001)
    errors = [];_check_unit(name,row,errors,version)
    assert errors == []
    other = 'm2-full-v5' if version != 'm2-full-v5' else 'm2-full-v5-tracking5'
    _check_unit(name,row,errors,other)
    assert any(r['code']=='threshold_mismatch' and r['field']=='k_target' for r in errors)
