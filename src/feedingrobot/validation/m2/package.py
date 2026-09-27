"""Full-matrix package verification, isolated from historical fixes schemas."""
from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from feedingrobot.validation.m2.spec import SCHEMA, cases, manifest, VARIANTS, FORMAL_SEEDS, CALIBRATION_SEEDS
from feedingrobot.validation.m2.checks import score
from feedingrobot.validation.m2.provenance import digest, inputs, read
from feedingrobot.validation.m2.provenance import check_config_snapshots, check_execution, required_nodes
from feedingrobot.validation.m2.provenance import check_receipt, identity


def package_identity(run):
    names = ['input_hash_before.json', 'input_hash_after.json', 'case_manifest.json', 'evidence_manifest.json',
             'm1_scene.json', 'm2_controller.json', 'm2_acceptance.json', 'execution.json',
             'calibration_binding.json', 'legacy_binding.json']
    files={name: digest(Path(run)/name) for name in names if (Path(run)/name).is_file()}
    files.update({'cases/'+p.name: digest(p) for p in sorted((Path(run)/'cases').iterdir()) if p.is_file()})
    return identity(files)


def evaluate(run, *, check_workspace=False, calibration=False, _require_negatives=True, _case_cache=None, _finalizing=False):
    run=Path(run)
    reasons=[]; metrics={}; comparisons=[]
    seeds=CALIBRATION_SEEDS if calibration else FORMAL_SEEDS
    expected=manifest(seeds)
    def fail(code,**kwargs):reasons.append({'code':code,**kwargs})
    try:
        header=read(run/'run.json')
        if header.get('schema_version')!=SCHEMA or header.get('stage')!=('calibration' if calibration else 'full') or (_require_negatives and not _finalizing and header.get('incomplete') is not False):fail('run_identity')
        before=read(run/'input_hash_before.json');after=read(run/'input_hash_after.json')
        if before!=after:fail('inputs_changed')
        paths = [row['path'] for row in after]
        if len(paths) != len(set(paths)) or set(paths) != {row['path'] for row in inputs()}:
            fail('input_coverage')
        if (check_workspace or not calibration) and after!=inputs():fail('workspace_changed')
        if read(run/'case_manifest.json')!=expected:fail('case_manifest')
        listed=read(run/'evidence_manifest.json')
        wanted={row['case_id']+suffix for row in expected for suffix in ('.npz','.json','.execution.json')}
        actual={p.name for p in (run/'cases').iterdir() if p.is_file()}
        if actual!=wanted:fail('case_files',missing=sorted(wanted-actual),extra=sorted(actual-wanted))
        if set(listed)!=wanted:fail('evidence_manifest')
        for name,sha in listed.items():
            if name not in wanted or not (run/'cases'/name).is_file() or digest(run/'cases'/name)!=sha:fail('evidence_hash',case_id=name)
        reasons.extend(check_config_snapshots(run, after))
        cfg=read(run/'m2_controller.json')
        if not calibration:
            binding=read(run/'calibration_binding.json')
            cal=Path(binding['path'])
            if digest(cal/'report.json')!=binding['sha256']:fail('calibration_binding')
            verdict=evaluate(cal,calibration=True,_case_cache=_case_cache)
            if not verdict['scope_passed'] or read(cal/'input_hash_after.json')!=after or read(cal/'m2_controller.json')!=cfg:fail('calibration_failed')
            legacy=read(run/'legacy_binding.json')
            from feedingrobot.validation.m2.legacy import verify_finished_report
            path=Path(legacy['path'])
            if digest(path/'report.json')!=legacy['sha256'] or not verify_finished_report(path,'fixes-v6',check_workspace).get('scope_passed'):fail('legacy_regression_failed')
            legacy_inputs={r['path']:r['sha256'] for r in read(path/'input_hash_after.json')['files']}
            if any(legacy_inputs.get(r['path'],r['sha256'])!=r['sha256'] for r in after):fail('legacy_inputs')
            execution=read(run/'execution.json')
            reasons.extend(check_execution(execution, required_nodes(), header['run_id']))
    except (OSError,ValueError,KeyError,TypeError) as error:
        fail('package_structure',detail=str(error))
    if reasons:
        return result(reasons,metrics,comparisons,calibration)
    by_key={c.key:c for c in cases()}
    executions=set()
    scoring_identity=identity({'inputs':after,'config':cfg})
    for row in expected:
        name=row['case_id'];case=by_key[name.rsplit('-s',1)[0]]
        try:
            meta=read(run/'cases'/f'{name}.json')
            eid=meta['execution']['id']
            if eid in executions:fail('duplicate_execution',case_id=name)
            executions.add(eid)
            key=(scoring_identity,name,listed[name+'.npz'],listed[name+'.json'],listed[name+'.execution.json'])
            if _case_cache is not None and key in _case_cache:
                errors,values=_case_cache[key]
            else:
                with np.load(run/'cases'/f'{name}.npz',allow_pickle=False) as z:data=dict(z)
                receipt=read(run/'cases'/f'{name}.execution.json')
                errors,values=score(case,row['seed'],row['variant'],data,meta,cfg['stop']['damping_nm_s_per_rad'],cfg)
                errors=check_receipt(case,row['seed'],row['variant'],data,meta,receipt,listed[name+'.npz'],listed[name+'.json'])+errors
                if _case_cache is not None:
                    _case_cache[key]=(errors,values)
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
        from feedingrobot.validation.m2.negatives import check_negatives
        reasons.extend(check_negatives(run))
    verdict=result(reasons,metrics,comparisons,calibration)
    if not _require_negatives:
        verdict['m3_ready']=False
    elif not _finalizing:
        try:
            saved=read(run/'report.json')
            if saved != verdict or header.get('m3_ready') != verdict['m3_ready']:
                fail('report_mismatch')
        except (OSError,ValueError):fail('report_missing')
        verdict=result(reasons,metrics,comparisons,calibration)
    return verdict


def result(reasons,metrics,comparisons,calibration):
    passed=not reasons
    return {'schema_version':SCHEMA,'evidence_valid':passed,'scope_passed':passed,'fixes_passed':passed,
            'm3_ready':passed and not calibration,'hybrid_force_status':'disabled','reasons':reasons,
            'remaining_m2_requirements':[] if passed and not calibration else sorted({r['code'] for r in reasons}) or ['formal_matrix'],
            'metrics':metrics,'comparisons':comparisons}


def finalize(run, *, calibration=False):
    """Validate all evidence, then publish the completion record last."""
    run = Path(run)
    cache = {}
    verdict = evaluate(run, check_workspace=True, calibration=calibration,
                       _case_cache=cache, _finalizing=True)
    if not verdict['scope_passed']:
        return verdict
    header = read(run/'run.json')
    for filename, payload in (
        ('report.json', verdict),
        ('run.json', {**header, 'incomplete': False, 'm3_ready': verdict['m3_ready']}),
    ):
        temporary = run/(filename + '.tmp')
        temporary.write_text(json.dumps(payload, indent=2, allow_nan=False))
        temporary.replace(run/filename)
    verified = evaluate(run, check_workspace=True, calibration=calibration, _case_cache=cache)
    if not verified['scope_passed']:
        temporary = run/'run.json.tmp'
        temporary.write_text(json.dumps({**header, 'incomplete': True, 'm3_ready': False}, indent=2))
        temporary.replace(run/'run.json')
    return verified


def compare(left, right):
    failures = []
    if left.get('execution_id') == right.get('execution_id'):
        failures.append({'code': 'duplicate_execution'})
    if left.get('initial_source_identity') != right.get('initial_source_identity'):
        failures.append({'code': 'initial_pair:source'})
    if left.get('variant') != 'A' or right.get('variant') not in {'B', 'C'}:
        failures.append({'code': 'variant_pair'})
    for key, floor in (("position_rms",.0001),("rotation_rms",np.deg2rad(.05)),("tracking_position_rms",.0001),("tracking_rotation_rms",np.deg2rad(.05)),("force_peak",.02),("force_mean",.02),("moment_peak",.002),("tau_peak",.05),("impulse",1e-4),("contact_force_peak",.02),("contact_moment_peak",.002),("contact_impulse",1e-4)):
        a,b = np.asarray(left[key]),np.asarray(right[key])
        if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))) or np.any(np.abs(a-b)>np.maximum(.1*np.maximum(np.abs(a),np.abs(b)),floor)+1e-12):
            failures.append({"code":"convergence:"+key,"left":a.tolist(),"right":b.tolist()})
    if left.get('family') in {'spring', 'press'} or right.get('family') in {'spring', 'press'}:
        a, b = left.get('contact_force_mean'), right.get('contact_force_mean')
        if a is None or b is None:
            failures.append({'code': 'missing_metric:contact_force_mean'})
        elif not (np.isfinite(a) and np.isfinite(b)) or abs(a-b) > max(.1*max(abs(a),abs(b)), .02)+1e-12:
            failures.append({'code': 'convergence:contact_force_mean', 'left': a, 'right': b})
    for key in ("initial_qpos","initial_qvel"):
        if not np.array_equal(left[key],right[key]): failures.append({"code":"initial_pair:"+key})
    if set(left["events"]) != set(right["events"]):
        failures.append({"code":"event_pair"})
    else:
        for event in left["events"]:
            if not (np.isfinite(left['events'][event]) and np.isfinite(right['events'][event])) or abs(left["events"][event]-right["events"][event]) > .020+1e-12:
                failures.append({"code":"event_time:"+event})
    return failures
