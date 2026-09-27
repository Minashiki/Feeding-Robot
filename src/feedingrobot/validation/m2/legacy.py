"""Read-only fixes-v6 boundary and binding of generator-produced counterexamples."""
from pathlib import Path
import json

from feedingrobot.validation.m2.provenance import digest, identity, read


def record_identity(run):
    run = Path(run)
    return identity({str(p.relative_to(run)): digest(p)
                     for p in sorted(run.rglob('*')) if p.is_file()
                     and p.name not in {'adversarial_binding.json', 'adversarial_binding.json.tmp'}})


def seal_execution(run):
    """Called by the generator only after real counterexample execution completes."""
    run = Path(run)
    execution = read(run/'adversarial_execution.json')
    from feedingrobot.controllers.v3spec import check_second_stage
    errors = []
    check_second_stage(execution, errors)
    if errors or execution.get('run_id') != run.name + '-adversarial':
        raise ValueError('legacy counterexample execution is incomplete')
    payload = {'source': record_identity(run),
               'results': digest(run/'adversarial_results.json'),
               'execution': digest(run/'adversarial_execution.json')}
    temporary = run/'adversarial_binding.json.tmp'
    temporary.write_text(json.dumps(payload, indent=2))
    temporary.replace(run/'adversarial_binding.json')


def verify_finished_report(run, scope, check_workspace):
    from feedingrobot.controllers.v3spec import verify_finished_report as verify
    result = verify(run, scope, check_workspace, rerun_adversarial=False)
    if scope != 'fixes-v6':
        return result
    run = Path(run)
    try:
        binding = read(run/'adversarial_binding.json')
        expected = {'source': record_identity(run),
                    'results': digest(run/'adversarial_results.json'),
                    'execution': digest(run/'adversarial_execution.json')}
        execution = read(run/'adversarial_execution.json')
        valid = binding == expected and execution.get('run_id') == run.name + '-adversarial'
    except (OSError, ValueError, TypeError, KeyError):
        valid = False
    if not valid:
        result['reasons'].append({'code': 'legacy_execution_binding'})
        result.update(scope_passed=False, fixes_passed=False, evidence_valid=False, m3_ready=False)
    return result
