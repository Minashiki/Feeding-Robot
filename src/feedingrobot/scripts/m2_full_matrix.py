"""Generate calibration or full M2 evidence; incomplete runs cannot release M3."""
import json
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import subprocess
import sys

from feedingrobot.controllers.full_spec import SCHEMA, cases, manifest, VARIANTS, FORMAL_SEEDS, CALIBRATION_SEEDS
from feedingrobot.controllers.full_runner import run_group
from feedingrobot.controllers.full_checks import score, compare
from feedingrobot.controllers.full_acceptance import inputs, digest, evaluate
from feedingrobot.sim.model import repo_root


def write(path,value):
    path=Path(path)
    temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False))
    temporary.replace(path)


def bounded_jobs(jobs,config,out,workers):
    if workers==1:
        for case,seed in jobs:
            yield group_job(case,seed,config,out)
        return
    iterator=iter(jobs)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        pending=set()
        for _ in range(workers):
            item=next(iterator,None)
            if item is not None:pending.add(pool.submit(group_job,*item,config,out))
        while pending:
            completed,pending=wait(pending,return_when=FIRST_COMPLETED)
            for future in completed:
                yield future.result()
                item=next(iterator,None)
                if item is not None:pending.add(pool.submit(group_job,*item,config,out))


def group_job(case,seed,config,out):
    try:
        traces=run_group(case,seed,config,str(Path(out)/'cases'))
        scored=[score(case,seed,v,*trace,config['stop']['damping_nm_s_per_rad']) for v,trace in zip(VARIANTS,traces)]
        return {'case':case.key,'seed':seed,'failures':[r[0] for r in scored],
                'comparisons':[compare(scored[0][1],r[1]) for r in scored[1:]] if all(r[1] for r in scored) else [{'code':'missing_pair'}]}
    except Exception as error:
        return {'case':case.key,'seed':seed,'error':repr(error)}


def run(args):
    from feedingrobot.scripts.m2_limited import current_limits,verify_limits
    verify_limits(current_limits())
    if args.workers not in (1,2):raise SystemExit('local workers must be 1 or 2')
    root=repo_root();out=Path(args.output).resolve()
    out.mkdir(parents=True,exist_ok=False)
    (out/'cases').mkdir()
    calibration=getattr(args,'calibrate',False)
    stage='calibration' if calibration else 'full'
    write(out/'run.json',{'schema_version':SCHEMA,'stage':stage,'incomplete':True,'m3_ready':False})
    before=inputs();write(out/'input_hash_before.json',before)
    config=json.loads((root/args.controller_config).read_text())
    write(out/'controller_config.json',config)
    seeds=CALIBRATION_SEEDS if calibration else FORMAL_SEEDS
    write(out/'case_manifest.json',manifest(seeds))
    write(out/'environment.json',{'python':sys.version,'workers':args.workers})
    if not calibration:
        if not args.calibration:raise SystemExit('full requires --calibration pointing to a passed, frozen calibration package')
        cal=Path(args.calibration).resolve()
        verdict=evaluate(cal,check_workspace=True,calibration=True)
        if not verdict['scope_passed']:raise SystemExit('calibration failed: '+str(verdict['reasons'][:5]))
        write(out/'calibration_binding.json',{'path':str(cal),'sha256':digest(cal/'report.json')})
        commands=[
            [sys.executable,'-m','feedingrobot.scripts.validate_m1','--output',str(out/'m1_regression')],
            [sys.executable,'-m','feedingrobot.scripts.validate_m2','--scope','fixes-v6','--m1-baseline',args.m1_baseline,'--m1-regression',str(out/'m1_regression'),'--output',str(out/'legacy')],
        ]
        for index,command in enumerate(commands):
            with (out/f'prerequisite-{index}.txt').open('w') as log:
                proc=subprocess.run(command,cwd=root,stdout=log,stderr=subprocess.STDOUT)
            if proc.returncode:raise SystemExit(f'prerequisite {index} failed; see {out}')
        write(out/'legacy_binding.json',{'path':str(out/'legacy'),'sha256':digest(out/'legacy'/'report.json')})
        env=os.environ.copy();env['M1_EXECUTION_PATH']=str(out/'execution.json');env['M1_RUN_ID']=out.name
        tests=sorted(str(p.relative_to(root)) for p in (root/'tests').glob('test_m2_*.py'))
        with (out/'pytest.txt').open('w') as log:
            proc=subprocess.run([sys.executable,'-m','pytest',*tests,'-q'],cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
        if proc.returncode:raise SystemExit('M2 regression failed')
    jobs=[(case,seed) for case in cases() for seed in seeds]
    progress=[]
    for row in bounded_jobs(jobs,config,str(out),args.workers):
        progress.append(row)
        write(out/'progress.json',progress)
        failed='error' in row or any(row.get('failures',[])) or any(row.get('comparisons',[]))
        print(f'{len(progress)}/{len(jobs)} {row["case"]} seed={row["seed"]}: {"FAIL" if failed else "pass"}',flush=True)
    write(out/'input_hash_after.json',inputs())
    write(out/'evidence_manifest.json',{p.name:digest(p) for p in sorted((out/'cases').iterdir()) if p.is_file()})
    write(out/'run.json',{'schema_version':SCHEMA,'stage':stage,'incomplete':False,'m3_ready':False})
    verdict=evaluate(out,check_workspace=True,calibration=calibration)
    write(out/'report.json',verdict)
    print(json.dumps({key:value for key,value in verdict.items() if key not in {'metrics','comparisons'}},indent=2))
    return 0 if verdict['scope_passed'] else 1
