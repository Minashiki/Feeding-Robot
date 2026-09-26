"""Run one M2 job under verified cgroup limits; never fall back to an unlimited job."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

THREAD_VARS = ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS')
MEMORY_MAX = 6 * 1024**3


def current_limits():
    group = next(line.split('::',1)[1] for line in Path('/proc/self/cgroup').read_text().splitlines() if line.startswith('0::'))
    path = Path('/sys/fs/cgroup') / group.lstrip('/')
    names = ('cpu.max','memory.high','memory.max','memory.swap.max','memory.peak','cpu.stat','memory.events')
    limits = {name:(path/name).read_text().strip() for name in names if (path/name).exists()}
    limits['cgroup'] = group
    return limits


def verify_limits(limits):
    quota,period = limits.get('cpu.max','max 100000').split()
    if quota == 'max' or int(quota) > int(period):
        raise RuntimeError('CPU quota is missing or exceeds one logical CPU')
    if limits.get('memory.max','max') == 'max' or int(limits['memory.max']) > MEMORY_MAX:
        raise RuntimeError('memory cap is missing or exceeds 6 GiB')
    if limits.get('memory.swap.max') != '0':
        raise RuntimeError('swap must be disabled for this job')
    if any(os.environ.get(name) != '1' for name in THREAD_VARS):
        raise RuntimeError('numerical library thread limit is missing')


def inside(output,command):
    limits=current_limits()
    verify_limits(limits)
    limits.update(nice=os.nice(0),threads={name:os.environ[name] for name in THREAD_VARS})
    (output/'limits.json').write_text(json.dumps(limits,indent=2))
    start=time.monotonic()
    # The wrapper stays alive to preserve statistics after all descendants exit.
    result=subprocess.run(command)
    finish={'returncode':result.returncode,'elapsed_s':time.monotonic()-start,'limits':current_limits()}
    (output/'result.json').write_text(json.dumps(finish,indent=2))
    return result.returncode


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    parser.add_argument('--inside',action='store_true',help=argparse.SUPPRESS)
    parser.add_argument('command',nargs=argparse.REMAINDER)
    args=parser.parse_args()
    command=args.command[1:] if args.command[:1]==['--'] else args.command
    if not command:parser.error('supply a command after --')
    output=Path(args.output).resolve()
    if args.inside:
        return inside(output,command)
    root=Path(__file__).resolve().parents[3]
    lock_path=root/'outputs/m2/.resource.lock'
    lock_path.parent.mkdir(parents=True,exist_ok=True)
    with lock_path.open('a+') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise SystemExit('another M2 job holds the project resource lock')
        active=subprocess.run(['systemctl','--user','list-units','m2-limited-*.scope','--state=active','--no-legend','--plain'],capture_output=True,text=True)
        if active.returncode:
            raise SystemExit('cannot inspect existing resource scopes; no job started')
        if active.stdout.strip():
            raise SystemExit('an M2 resource scope is still active; inspect and stop it before starting another job')
        output.mkdir(parents=True,exist_ok=False)
        unit='m2-limited-'+uuid.uuid4().hex[:12]+'.scope'
        env=os.environ.copy()
        env.update({name:'1' for name in THREAD_VARS})
        env['M2_RESOURCE_GUARDED']='1'
        launch=['systemd-run','--user','--scope','--unit='+unit,
                '-p','CPUQuota=100%','-p','MemoryHigh=4G','-p','MemoryMax=6G','-p','MemorySwapMax=0',
                'nice','-n','15','ionice','-c','3',sys.executable,'-m','feedingrobot.scripts.m2_limited',
                '--inside','--output',str(output),'--',*command]
        (output/'launch.json').write_text(json.dumps({'unit':unit,'command':command,'cwd':str(Path.cwd()),'m3_ready':False},indent=2))
        def interrupted(signum,frame):
            raise KeyboardInterrupt
        old={s:signal.signal(s,interrupted) for s in (signal.SIGINT,signal.SIGTERM)}
        try:
            code=subprocess.call(launch,env=env)
            (output/'launcher_result.json').write_text(json.dumps({'returncode':code,'incomplete':not (output/'result.json').is_file(),'m3_ready':False}))
            return code
        except KeyboardInterrupt:
            (output/'interrupted.json').write_text(json.dumps({'incomplete':True,'m3_ready':False}))
            return 130
        finally:
            subprocess.run(['systemctl','--user','stop',unit],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            for s,handler in old.items():signal.signal(s,handler)


if __name__=='__main__':
    raise SystemExit(main())
