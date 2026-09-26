import json
import os
from pathlib import Path

import pytest

from feedingrobot.sim.model import load_config

_EXECUTION = []
_COLLECTED = []


def pytest_addoption(parser):
    parser.addoption("--m1-config", default="configs/m1_scene.json")


def pytest_collection_finish(session):
    _COLLECTED.clear()
    _COLLECTED.extend(item.nodeid for item in session.items)


def pytest_runtest_logreport(report):
    path = os.environ.get("M1_EXECUTION_PATH")
    if not path:
        return
    _EXECUTION.append({
        "nodeid": report.nodeid,
        "when": report.when,
        "outcome": report.outcome,
        "duration_s": float(getattr(report, "duration", 0.0)),
        "wasxfail": getattr(report, "wasxfail", None),
    })


def pytest_sessionfinish(session, exitstatus):
    path = os.environ.get("M1_EXECUTION_PATH")
    if not path:
        return
    Path(path).write_text(json.dumps({
        "run_id": os.environ.get("M1_RUN_ID"),
        "exitstatus": int(exitstatus),
        "collected": list(_COLLECTED),
        "reports": _EXECUTION,
    }, indent=2))


@pytest.fixture
def m1_config(request):
    return load_config(request.config.getoption("--m1-config"))
