"""Serial full-schema counterexamples, bound to a clean evidence package."""
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np

from feedingrobot.controllers.full_spec import Case
from feedingrobot.controllers.full_identity import identity

NEGATIVES = {
    'late_stop': 'late_stop', 'ordinary_stop': 'unexpected_stop', 'valid_nan': 'valid_nan',
    'abort_step': 'abort_advanced', 'missing_case': 'case_files', 'missing_pair': 'case_files',
    'torque_limit': 'torque_limit', 'gain': 'gain_k', 'stimulus': 'command_stimulus',
    'event_shift': 'first_fault', 'seed_swap': 'execution_identity', 'solver_swap': 'execution_identity',
    'duplicate_execution': 'duplicate_execution', 'effective_config': 'effective_config',
    'config_snapshot': 'config_snapshot', 'missing_test': 'regression_nodes',
    'duplicate_test': 'regression_nodes', 'mixed_test': 'regression_outcome',
    'negative_flags': 'negative_results', 'negative_missing': 'negative_results',
}

# Only the serial counterexample process reuses immutable case scores. File hashes,
# execution identities, package structure and pair checks are still checked each time.
# Public independent verification does not supply this cache.
_CASE_CACHE = {}
_CLEAN_SOURCE = None


def negative_nodes():
    return [f'tests/test_m2_full_negatives.py::test_full_negative[{name}]' for name in NEGATIVES]


def check_results(run, names=None):
    from feedingrobot.controllers.full_acceptance import read, package_identity
    try:
        payload = read(Path(run)/'negative_results.json')
        rows = payload['results']
        if payload['source'] != package_identity(run) or payload['clean_passed'] is not True:
            return [{'code': 'negative_results'}]
        if [r['name'] for r in rows] != (list(NEGATIVES) if names is None else names):
            return [{'code': 'negative_results'}]
        for row in rows:
            if (row['expected'] != NEGATIVES[row['name']] or row['expected'] not in row['actual']
                    or row['rejected'] is not True or row['source_unchanged'] is not True
                    or not row['mutation_sha256'] or row['mutation_sha256'] == payload['source']):
                return [{'code': 'negative_results'}]
        return []
    except (OSError, ValueError, KeyError, TypeError):
        return [{'code': 'negative_results'}]


def check_negatives(run):
    from feedingrobot.controllers.full_acceptance import read
    from feedingrobot.controllers.full_contracts import check_execution
    errors = check_results(run)
    if errors:
        return errors
    try:
        execution = read(Path(run)/'negative_execution.json')
        header = read(Path(run)/'run.json')
        return check_execution(execution, negative_nodes(), header['run_id']+'-negatives')
    except (OSError, ValueError, KeyError, TypeError):
        return [{'code': 'negative_results'}]


def mutate(run, name):
    from feedingrobot.controllers.full_acceptance import read, digest
    def write(path, value):
        Path(path).write_text(json.dumps(value, indent=2, allow_nan=False))
    if name in {'negative_flags', 'negative_missing'}:
        path = run/'negative_results.json'
        if name == 'negative_flags': write(path, {key: True for key in NEGATIVES})
        else: path.unlink(missing_ok=True)
        return
    if name in {'missing_test', 'duplicate_test', 'mixed_test'}:
        path=run/'execution.json'; ex=read(path)
        if name == 'missing_test': ex['collected'].pop()
        elif name == 'duplicate_test': ex['collected']=[ex['collected'][0]]*363
        else: ex['reports'].append({**ex['reports'][0], 'outcome': 'failed'})
        write(path, ex); return
    if name == 'config_snapshot':
        path=run/'m2_controller.json'; cfg=read(path);cfg['gears']['FREE']['kp']=99999.
        write(path,cfg);return
    case = Case('stop', event='nan') if name == 'abort_step' else Case('hold') if name == 'ordinary_stop' else Case('stop', axis=1, sign=-1, event='expired', context='spring')
    source_id = case.manifest(3, 'A')['case_id']
    target_id = case.manifest(4, 'A')['case_id'] if name == 'seed_swap' else case.manifest(3, 'C')['case_id'] if name in {'solver_swap', 'duplicate_execution', 'missing_pair'} else source_id
    folder=run/'cases'
    manifest=read(run/'evidence_manifest.json')
    if name in {'missing_case', 'missing_pair'}:
        suffixes=('.npz',) if name=='missing_case' else ('.npz','.json','.execution.json')
        for suffix in suffixes:
            (folder/(target_id+suffix)).unlink()
            manifest.pop(target_id+suffix)
        write(run/'evidence_manifest.json',manifest);return
    with np.load(folder/(source_id+'.npz'), allow_pickle=False) as z:
        data = {k: z[k].copy() for k in z.files}
    meta=read(folder/(source_id+'.json'))
    receipt=read(folder/(source_id+'.execution.json'))
    k=round(1.42/.001)
    if name=='late_stop':data['execution'][k]='run'
    elif name=='ordinary_stop':data['status'][100]='STOPPED'
    elif name=='valid_nan':data['ft_delivered_wrench_tcp'][1000,0]=np.nan
    elif name=='abort_step':meta['abort_probe']['tick_after']+=1
    elif name=='torque_limit':data['tau_cmd'][100,0]=100.
    elif name=='gain':data['k'][100,0]+=1.
    elif name=='stimulus':data['supplied'][100,0]+=.1
    elif name=='event_shift':meta['fault']['tick']+=1
    elif name=='effective_config':meta['effective_controller']['gears']['FREE']['kp']=99999.
    elif name=='seed_swap':meta.update(case.manifest(4,'A'))
    elif name in {'solver_swap','duplicate_execution'}:meta.update(case.manifest(3,'C'))
    path=folder/target_id
    np.savez_compressed(str(path)+'.npz', **data)
    write(str(path)+'.json',meta)
    # Rehash both the manifest and receipt: identity/physics must reject, not just stale file hashes.
    receipt.update(case=case.manifest(4 if name=='seed_swap' else 3,'C' if name in {'solver_swap','duplicate_execution'} else 'A'),
                   json_sha256=digest(str(path)+'.json'), npz_sha256=digest(str(path)+'.npz'))
    write(str(path)+'.execution.json',receipt)
    for suffix in ('.npz','.json','.execution.json'):
        manifest[target_id+suffix]=digest(str(path)+suffix)
    write(run/'evidence_manifest.json',manifest)


def run_negative(source, name):
    from feedingrobot.controllers.full_acceptance import evaluate, package_identity, digest
    source=Path(source)
    before=package_identity(source)
    global _CLEAN_SOURCE
    if _CLEAN_SOURCE != before:
        _CASE_CACHE.clear()
        clean=evaluate(source, _require_negatives=False, _case_cache=_CASE_CACHE)
        if not clean['scope_passed']:
            raise RuntimeError('counterexample source is not clean: '+str(clean['reasons'][:3]))
        _CLEAN_SOURCE=before
    with tempfile.TemporaryDirectory(prefix='m2-full-negative-') as directory:
        target=Path(directory)/'run'
        shutil.copytree(source, target, copy_function=shutil.copy2)
        if package_identity(target)!=before:
            raise RuntimeError('copy identity mismatch')
        if name.startswith('negative_'):
            # The preceding counterexamples are real completed runs. Prove their
            # partial payload valid before mutation; pytest execution is still open.
            from feedingrobot.controllers.full_acceptance import read
            completed = [r['name'] for r in read(target/'negative_results.json')['results']]
            if completed != list(NEGATIVES)[:list(NEGATIVES).index(name)] or check_results(target, completed):
                raise RuntimeError('negative payload baseline is not clean')
        mutate(target,name)
        if name.startswith('negative_'):
            reasons=check_results(target, completed)
        else:
            reasons=evaluate(target, _require_negatives=False, _case_cache=_CASE_CACHE)['reasons']
        after=package_identity(source)
        mutation=package_identity(target)
        if name.startswith('negative_'):
            mutation=(digest(target/'negative_results.json') if (target/'negative_results.json').is_file()
                      else identity({'missing': 'negative_results.json'}))
        return {'name':name, 'expected':NEGATIVES[name], 'actual':sorted({r['code'] for r in reasons}),
                'rejected':bool(reasons), 'source_unchanged':before==after, 'mutation_sha256':mutation}
