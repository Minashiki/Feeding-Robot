"""The full-matrix checker must validate schedules rather than result flags."""
import numpy as np
import pytest

from feedingrobot.controllers.full_checks import expected_stimulus
from feedingrobot.controllers.full_spec import Case, cases, manifest
from feedingrobot.controllers.full_runner import target


def test_manifest_contains_formal_seeds_and_all_pairs():
    rows = manifest()
    assert len({r['case_id'] for r in rows}) == len(rows)
    assert {r['seed'] for r in rows} == set(range(3,9))
    for case in cases():
        selected = [r for r in rows if r['case_id'].startswith(case.key+'-s')]
        assert len(selected) == 18


@pytest.mark.parametrize('family', ['step','circle','sine','spring','press','carry','plate','mouth'])
def test_independent_target_schedule_matches_generator(family):
    case = Case(family)
    times = np.arange(0,case.duration,.05)
    command, _, _ = expected_stimulus(case,times)
    expected = np.array([(target(case,t+.05)-target(case,t))/.05 for t in times])
    np.testing.assert_allclose(command,expected,atol=1e-12,rtol=0)


def test_contact_stop_expiry_keeps_fixture_direction(tmp_path):
    from feedingrobot.controllers.full_runner import run_group
    from feedingrobot.controllers.full_checks import score, compare
    from feedingrobot.sim.model import load_config
    case=Case('stop',axis=1,sign=-1,event='expired',context='spring')
    traces=run_group(case,0,load_config('configs/m2_controller.json'),str(tmp_path))
    metrics=[]
    for variant,(data,meta) in zip('ABC',traces):
        reasons,values=score(case,0,variant,data,meta)
        assert reasons==[], reasons
        metrics.append(values)
        altered={**data,'supplied':data['supplied'].copy()}
        tick=round(1.4/meta['dt'])
        altered['supplied'][tick,:3]=[0.,-.04,0.]
        reasons,_=score(case,0,variant,altered,meta)
        assert 'command_stimulus' in {r['code'] for r in reasons}
    assert compare(metrics[0],metrics[1])==[]
    assert compare(metrics[0],metrics[2])==[]
    assert not list(tmp_path.glob('*.partial'))
