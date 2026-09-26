"""Full-package counterexamples run only in the explicit second stage."""
import json
import os
from pathlib import Path

import pytest

from feedingrobot.controllers.full_negatives import NEGATIVES, check_negatives, run_negative


def test_boolean_negative_summary_is_rejected(tmp_path):
    (tmp_path/'negative_results.json').write_text(json.dumps({name: True for name in NEGATIVES}))
    assert check_negatives(tmp_path) == [{'code': 'negative_results'}]


def test_full_counterexample_pipeline(tmp_path, monkeypatch):
    """Exercise every mutation on real traces; only the unrelated M1/legacy prerequisite is stubbed."""
    import shutil
    from feedingrobot.controllers import full_acceptance as a, acceptance
    from feedingrobot.controllers.full_contracts import CONFIG_PATHS
    from feedingrobot.controllers.full_runner import run_group
    from feedingrobot.controllers.full_spec import Case, SCHEMA
    from feedingrobot.controllers.full_negatives import negative_nodes
    from feedingrobot.sim.model import repo_root
    cases = [Case('hold'), Case('stop', event='nan'), Case('stop', axis=1, sign=-1, event='expired', context='spring')]
    monkeypatch.setattr(a, 'cases', lambda: cases)
    monkeypatch.setattr(a, 'CALIBRATION_SEEDS', (0,))
    monkeypatch.setattr(a, 'FORMAL_SEEDS', (3, 4))
    monkeypatch.setattr(a, 'manifest', lambda seeds: [c.manifest(s, v) for c in cases for s in seeds for v in 'ABC'])
    monkeypatch.setattr(a, 'required_nodes', lambda: ['unit'])
    monkeypatch.setattr(acceptance, 'verify_finished_report', lambda *args: {'scope_passed': True})
    cfg=json.loads((repo_root()/'configs/m2_controller.json').read_text())
    def write(path,value):path.write_text(json.dumps(value,indent=2))
    def execution(nodes, run_id):
        return {'run_id':run_id,'exitstatus':0,'collected':nodes,
                'reports':[{'nodeid':n,'when':w,'outcome':'passed'} for n in nodes for w in ('setup','call','teardown')]}
    def build(path,seeds,stage):
        path.mkdir();(path/'cases').mkdir()
        write(path/'run.json',{'schema_version':SCHEMA,'stage':stage,'run_id':path.name,'incomplete':True,'m3_ready':False})
        for file in CONFIG_PATHS:shutil.copyfile(repo_root()/file,path/Path(file).name)
        write(path/'input_hash_before.json',a.inputs());write(path/'input_hash_after.json',a.inputs())
        write(path/'case_manifest.json',a.manifest(seeds))
        for case in cases:
            for seed in seeds:run_group(case,seed,cfg,str(path/'cases'))
        write(path/'evidence_manifest.json',{p.name:a.digest(p) for p in (path/'cases').iterdir()})
    cal=tmp_path/'cal';build(cal,(0,),'calibration')
    verdict=a.evaluate(cal,calibration=True,_require_negatives=False)
    assert verdict['scope_passed'],verdict['reasons'][:10]
    write(cal/'report.json',verdict)
    header=a.read(cal/'run.json');header['incomplete']=False;write(cal/'run.json',header)
    source=tmp_path/'full';build(source,(3,4),'full')
    legacy=tmp_path/'legacy';legacy.mkdir()
    write(legacy/'report.json',{'unit_test_prerequisite_stub':True})
    write(legacy/'input_hash_after.json',{'files':a.inputs()})
    write(source/'calibration_binding.json',{'path':str(cal),'sha256':a.digest(cal/'report.json')})
    write(source/'legacy_binding.json',{'path':str(legacy),'sha256':a.digest(legacy/'report.json')})
    write(source/'execution.json',execution(['unit'],source.name))
    clean=a.evaluate(source,_require_negatives=False)
    assert clean['scope_passed'] and not clean['m3_ready'],clean['reasons'][:10]
    rows=[]
    write(source/'negative_results.json',{'source':a.package_identity(source),'clean_passed':True,'results':rows})
    for name in NEGATIVES:
        rows.append(run_negative(source,name))
        write(source/'negative_results.json',{'source':a.package_identity(source),'clean_passed':True,'results':rows})
    for row in rows:
        assert row['rejected'] and row['source_unchanged'] and row['expected'] in row['actual'],row
    payload={'source':a.package_identity(source),'clean_passed':True,'results':rows}
    write(source/'negative_results.json',payload)
    write(source/'negative_execution.json',execution(negative_nodes(),source.name+'-negatives'))
    assert check_negatives(source)==[]
    assert not a.evaluate(source)['m3_ready']  # staged packages never release
    payload['results'][0]['actual']=[]
    write(source/'negative_results.json',payload)
    assert check_negatives(source)


if os.environ.get('M2_FULL_PACKAGE'):
    @pytest.mark.parametrize('name', list(NEGATIVES))
    def test_full_negative(name):
        from feedingrobot.controllers.full_acceptance import package_identity
        package = Path(os.environ['M2_FULL_PACKAGE'])
        row = run_negative(package, name)
        assert row['rejected'] and row['source_unchanged']
        assert NEGATIVES[name] in row['actual'], row
        path = Path(os.environ['M2_FULL_RESULTS'])
        payload = json.loads(path.read_text())
        assert payload['source'] == package_identity(package)
        payload['results'].append(row)
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(payload, indent=2))
        temporary.replace(path)
