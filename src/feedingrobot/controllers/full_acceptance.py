"""Full-matrix package verification, isolated from historical fixes schemas."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np

from feedingrobot.controllers.full_spec import SCHEMA, cases, manifest, VARIANTS, FORMAL_SEEDS, CALIBRATION_SEEDS
from feedingrobot.controllers.full_checks import score, compare
from feedingrobot.controllers.acceptance import collect_inputs
from feedingrobot.sim.model import repo_root


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def inputs():
    rows=collect_inputs()['files']
    root=repo_root()
    extras=['assets/tests/m2_full_surface.xml','assets/tests/m2_full_wall.xml','M2Plan.md','SimModelPlan.md','docs/m2_interface.md']
    return sorted(rows+[{'path':p,'sha256':digest(root/p)} for p in extras],key=lambda r:r['path'])


def read(path):
    return json.loads(Path(path).read_text(),parse_constant=lambda x: (_ for _ in ()).throw(ValueError('nonfinite JSON')))


def evaluate(run, *, check_workspace=False, calibration=False, _require_negatives=True):
    run=Path(run)
    reasons=[]; metrics={}; comparisons=[]
    seeds=CALIBRATION_SEEDS if calibration else FORMAL_SEEDS
    expected=manifest(seeds)
    def fail(code,**kwargs):reasons.append({'code':code,**kwargs})
    try:
        header=read(run/'run.json')
        if header.get('schema_version')!=SCHEMA or header.get('stage')!=('calibration' if calibration else 'full') or header.get('incomplete') is not False:fail('run_identity')
        before=read(run/'input_hash_before.json');after=read(run/'input_hash_after.json')
        if before!=after:fail('inputs_changed')
        if check_workspace and after!=inputs():fail('workspace_changed')
        if read(run/'case_manifest.json')!=expected:fail('case_manifest')
        listed=read(run/'evidence_manifest.json')
        wanted={row['case_id']+suffix for row in expected for suffix in ('.npz','.json')}
        actual={p.name for p in (run/'cases').iterdir() if p.is_file()}
        if actual!=wanted:fail('case_files',missing=sorted(wanted-actual),extra=sorted(actual-wanted))
        if set(listed)!=wanted:fail('evidence_manifest')
        for name,sha in listed.items():
            if name not in wanted or not (run/'cases'/name).is_file() or digest(run/'cases'/name)!=sha:fail('evidence_hash',case_id=name)
        cfg=read(run/'controller_config.json')
        if cfg['stop']['damping_nm_s_per_rad'] not in (8.,12.,16.,24.,32.) or cfg['hybrid_normal_force']['enabled'] is not False:fail('controller_profile')
        if calibration:
            if cfg['stop']['damping_nm_s_per_rad']!=8.:fail('unproved_smaller_stop_candidates')
        else:
            binding=read(run/'calibration_binding.json')
            cal=Path(binding['path'])
            if digest(cal/'report.json')!=binding['sha256']:fail('calibration_binding')
            verdict=evaluate(cal,calibration=True)
            if not verdict['scope_passed'] or read(cal/'input_hash_after.json')!=after or read(cal/'controller_config.json')!=cfg:fail('calibration_failed')
            legacy=read(run/'legacy_binding.json')
            from feedingrobot.controllers.acceptance import verify_finished_report
            path=Path(legacy['path'])
            if digest(path/'report.json')!=legacy['sha256'] or not verify_finished_report(path,'fixes-v6',check_workspace).get('scope_passed'):fail('legacy_regression_failed')
            legacy_inputs={r['path']:r['sha256'] for r in read(path/'input_hash_after.json')['files']}
            if any(legacy_inputs.get(r['path'],r['sha256'])!=r['sha256'] for r in after):fail('legacy_inputs')
            execution=read(run/'execution.json')
            if execution.get('exitstatus')!=0 or len(execution.get('collected',[]))<363:fail('regression_execution')
            for node in execution.get('collected',[]):
                for when in ('setup','call','teardown'):
                    if not any(r.get('nodeid')==node and r.get('when')==when and r.get('outcome')=='passed' for r in execution.get('reports',[])):fail('regression_node',nodeid=node,when=when)
    except (OSError,ValueError,KeyError,TypeError) as error:
        fail('package_structure',detail=str(error))
    if reasons:
        return result(reasons,metrics,comparisons,calibration)
    by_key={c.key:c for c in cases()}
    for row in expected:
        name=row['case_id'];case=by_key[name.rsplit('-s',1)[0]]
        try:
            with np.load(run/'cases'/f'{name}.npz',allow_pickle=False) as z:data=dict(z)
            meta=read(run/'cases'/f'{name}.json')
            errors,values=score(case,row['seed'],row['variant'],data,meta,cfg['stop']['damping_nm_s_per_rad'])
            reasons.extend({'case_id':name,**r} for r in errors)
            if values:metrics[name]=values
        except (OSError,ValueError,KeyError,TypeError,IndexError) as error:fail('case_structure',case_id=name,detail=str(error))
    for case in cases():
        for seed in seeds:
            names=[case.manifest(seed,v)['case_id'] for v in VARIANTS]
            if not all(name in metrics for name in names):fail('missing_pair',case_id=case.key,seed=seed);continue
            for name in names[1:]:
                errors=compare(metrics[names[0]],metrics[name])
                comparisons.append({'left':names[0],'right':name,'reasons':errors})
                reasons.extend({'case_id':name,**r} for r in errors)
            for variant in VARIANTS:
                current=metrics[case.manifest(seed,variant)['case_id']]
                if case.family in {'stiffness','press'} and case.stiffness==150.:
                    high=metrics.get(replace(case,stiffness=450.).manifest(seed,variant)['case_id'])
                    if high:
                        ok=current['stiffness_displacement']>=1.5*high['stiffness_displacement'] if case.family=='stiffness' else high['contact_force_mean']>current['contact_force_mean']
                        if not ok:fail('stiffness_trend',case_id=case.key)
                if case.event.startswith('sensor'):
                    base=metrics.get(replace(case,noise=False,delay=0.).manifest(seed,variant)['case_id'])
                    if base:
                        for key,floor in (('tracking_position_rms',.0002),('tracking_rotation_rms',np.deg2rad(.1))):
                            if current[key]>1.25*base[key]+floor:fail('noise_tracking',case_id=case.key,metric=key)
    if not calibration and _require_negatives:
        try:
            negative=read(run/'negative_results.json')
            required={'late_stop','ordinary_stop','valid_nan','abort_step','missing_case','missing_pair','torque_limit','gain','stimulus','event_shift'}
            if set(negative)!=required or any(v is not True for v in negative.values()):fail('negative_tests')
        except (OSError,ValueError):fail('negative_tests')
    verdict=result(reasons,metrics,comparisons,calibration)
    if not _require_negatives:verdict['m3_ready']=False
    return verdict


def result(reasons,metrics,comparisons,calibration):
    passed=not reasons
    return {'schema_version':SCHEMA,'evidence_valid':passed,'scope_passed':passed,'fixes_passed':passed,
            'm3_ready':passed and not calibration,'hybrid_force_status':'disabled','reasons':reasons,
            'remaining_m2_requirements':[] if passed and not calibration else sorted({r['code'] for r in reasons}) or ['formal_matrix'],
            'metrics':metrics,'comparisons':comparisons}
