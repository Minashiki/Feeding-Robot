"""Small, resource-limited calibration diagnostics. Never an M3 certification."""
import argparse
import json
from pathlib import Path

from feedingrobot.controllers.full_spec import cases
from feedingrobot.controllers.full_acceptance import inputs,digest
from feedingrobot.scripts.m2_full_matrix import group_job,write
from feedingrobot.scripts.m2_limited import current_limits,verify_limits
from feedingrobot.sim.model import load_config


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',action='append',required=True)
    parser.add_argument('--seed',type=int,choices=(0,1,2),default=0)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    verify_limits(current_limits())
    available={c.key:c for c in cases()}
    if any(key not in available for key in args.case):parser.error('unknown case key')
    out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=False);(out/'cases').mkdir()
    before=inputs()
    write(out/'run.json',{'schema_version':'m2-diagnostic-v1','incomplete':True,'m3_ready':False})
    write(out/'input_hash_before.json',before)
    config=load_config('configs/m2_controller.json')
    write(out/'controller_config.json',config)
    rows=[]
    for key in args.case:
        row=group_job(available[key],args.seed,config,str(out))
        rows.append(row);write(out/'progress.json',rows)
        print(json.dumps(row),flush=True)
    after=inputs();write(out/'input_hash_after.json',after)
    write(out/'evidence_manifest.json',{p.name:digest(p) for p in sorted((out/'cases').iterdir())})
    passed=before==after and all(not r.get('error') and not any(r.get('failures',[])) and not any(r.get('comparisons',[])) for r in rows)
    write(out/'run.json',{'schema_version':'m2-diagnostic-v1','incomplete':False,'diagnostics_passed':passed,'m3_ready':False})
    return 0 if passed else 1


if __name__=='__main__':raise SystemExit(main())
