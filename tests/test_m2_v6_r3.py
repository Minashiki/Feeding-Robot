"""Regression contracts for V6 review gaps; all mutations use private copies."""

import copy
import json

import numpy as np
import pytest

from feedingrobot.controllers.acceptance import V6_SECOND_STAGE_NODEIDS
from feedingrobot.controllers.v3spec import _abc, check_release_stimulus, check_second_stage
from feedingrobot.controllers.v6checks import CheckContext, check_trace, check_zero_grid
from tests.test_m2_v6 import _trace, test_zero_projection_grid as make_grid


def codes(reasons):
    return {row["code"] for row in reasons}


@pytest.fixture(scope="module")
def traces(tmp_path_factory):
    root = tmp_path_factory.mktemp("v6-r3-traces")
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("M2_EVIDENCE_DIR", str(root))
        for name in ("upper", "lower", "rest", "cross"):
            for variant in ("A", "B", "C"):
                case = f"V6-zero-phys-{name}-{variant}" if name in {"upper", "lower"} else f"V6-stop-{name}-{variant}"
                _trace(case)
        make_grid("V6-zero-j0-upper-linear-A")
    return root


def read(root, case):
    with np.load(root / f"{case}.npz") as data:
        arrays = {key: data[key] for key in data.files}
    return arrays, json.loads((root / f"{case}.json").read_text())


def test_v6_common_contracts(traces):
    case = "V6-zero-phys-upper-A"
    original, meta = read(traces, case)
    context = CheckContext()
    assert check_trace(case, original, meta, context) == []
    def tau_limit(data):
        data["tau_applied"][:, 6] = np.minimum(np.arange(len(data["command"])) + 1, 50)
    edits = [
        (tau_limit, "torque_rate"),
        (lambda d: d["K"].fill(1e9), "gain_path_mismatch"),
        (lambda d: d["tau_before"].__setitem__(0, np.nan), "nonfinite"),
        (lambda d: d.__setitem__("transition_active", np.full(len(d["command"]), np.nan)), "bad_dtype"),
        (lambda d: d.__setitem__("dq", d["dq"][:-1]), "bad_shape"),
        (lambda d: d.__setitem__("tick", d["tick"].astype(float)), "bad_dtype"),
        (lambda d: d.__setitem__("fault_tick", np.int64(100)), "unexpected_fault"),
        (lambda d: d["history_twist0"].fill(0), "stimulus"),
        (lambda d: d["candidate"].fill(0), "stimulus"),
        (lambda d: d.__setitem__("execution", np.full(len(d["command"]), "run")), "unexpected_fault"),
        (lambda d: d["r_ref"].__setitem__((0, 0, 0), 2.), "invalid_rotation"),
        (lambda d: d["wrench_compensated"].__setitem__((50, 0), 10.), "wrench_limit"),
    ]
    for edit, expected in edits:
        data = copy.deepcopy(original)
        edit(data)
        assert expected in codes(check_trace(case, data, meta, context)), expected


def test_v6_fixed_grid_contract(traces):
    case = "V6-zero-j0-upper-linear-A"
    row = json.loads((traces / f"{case}.json").read_text())
    context = CheckContext()
    assert check_zero_grid(case, row, context) == []
    for field, value in (("jbar", np.zeros((7, 6)).tolist()), ("candidate", [0.] * 6), ("q", [0.] * 7), ("dt", 0.0005)):
        altered = copy.deepcopy(row)
        altered[field] = value
        assert "stimulus" in codes(check_zero_grid(case, altered, context))
    for field, value, expected in (("dt", float("nan"), "nonfinite"), ("q", [0.] * 6, "bad_shape"), ("dt", "0.001", "bad_dtype")):
        altered = copy.deepcopy(row)
        altered[field] = value
        assert expected in codes(check_zero_grid(case, altered, context))


def test_v6_fixed_release_clock():
    for dt in (0.001, 0.0005):
        unload = round(0.15 / dt)
        command = np.zeros((unload + 10, 6))
        external = np.zeros_like(command)
        command[round(0.05 / dt):unload, 0] = 0.05
        external[round(0.04 / dt) + 1:unload, 0] = -3.
        reasons = []
        check_release_stimulus(reasons, "release_fast-s0-A", command, external, dt, unload)
        assert reasons == []
        for index, value in ((round(0.04 / dt), -3.), (round(0.04 / dt) + 1, 0.), (round(0.05 / dt), -0.001)):
            altered = external.copy()
            altered[index, 0] = value
            reasons = []
            check_release_stimulus(reasons, "release_fast-s0-A", command, altered, dt, unload)
            assert "stimulus" in codes(reasons)


def test_v6_second_stage_nodes():
    clean = {"exitstatus": 0, "collected": list(V6_SECOND_STAGE_NODEIDS), "reports": [{"nodeid": node, "when": when, "outcome": "passed"} for node in V6_SECOND_STAGE_NODEIDS for when in ("setup", "call", "teardown")]}
    reasons = []
    check_second_stage(clean, reasons)
    assert not reasons
    for node in V6_SECOND_STAGE_NODEIDS:
        for change in ("drop", "skip", "fail", "duplicate", "duplicate_collect"):
            row = copy.deepcopy(clean)
            report = next(item for item in row["reports"] if item["nodeid"] == node and item["when"] == "call")
            if change == "drop":
                row["reports"].remove(report)
            elif change == "duplicate":
                row["reports"].append(report.copy())
            elif change == "duplicate_collect":
                row["collected"].append(node)
            else:
                report["outcome"] = "skipped" if change == "skip" else "failed"
            reasons = []
            check_second_stage(row, reasons)
            assert "nodeid" in codes(reasons)


def test_v6_pair_contracts(traces):
    context = CheckContext()
    metrics = {}
    for path in sorted(traces.glob("*.npz")):
        data, meta = read(traces, path.stem)
        derived = {}
        assert check_trace(path.stem, data, meta, context, derived) == []
        metrics[path.stem] = derived
    data, meta = read(traces, "V6-stop-cross-A")
    fault = int(data["fault_tick"])
    data["mode"][fault + 1:] = "STOPPED"
    data["guard_status"][fault + 1:] = "STOPPED"
    assert "singularity_stop" in codes(check_trace("V6-stop-cross-A", data, meta, context))
    reasons = []
    comparisons = _abc(metrics, reasons)
    assert not [row for row in reasons if str(row.get("case_id", "")).startswith("V6")]
    assert len(comparisons) == 144
    for change in ("late_fault", "missing_event", "missing_c", "initial"):
        altered = copy.deepcopy(metrics)
        if change == "late_fault":
            altered["V6-stop-cross-B"]["events"]["fault"] += 0.1
        elif change == "missing_event":
            altered["V6-stop-cross-B"]["events"].pop("stop_confirm")
        elif change == "missing_c":
            altered.pop("V6-stop-cross-C")
        else:
            altered["V6-stop-cross-B"]["qpos0"][0] += 0.01
        reasons = []
        _abc(altered, reasons)
        expected = {"late_fault": "convergence", "missing_event": "event_missing", "missing_c": "missing_comparison", "initial": "initial_state"}[change]
        assert expected in codes(reasons)


def test_v6_corrupted_copy_rejected(tmp_path, monkeypatch):
    """A damaged clean copy must never reach the adversarial mutation/evaluation."""
    import shutil
    from feedingrobot.controllers import v3spec

    source = tmp_path / "source"
    source.mkdir()
    (source / "trace.bin").write_bytes(b"original")
    original_copytree = shutil.copytree
    evaluations = []

    def clean(run, check_workspace):
        evaluations.append(Path(run))
        return {"reasons": [], "fixes_passed": True, "evidence_valid": True}

    def damaged_copy(*args, **kwargs):
        destination = Path(original_copytree(*args, **kwargs))
        (destination / "trace.bin").write_bytes(b"corrupted")
        return destination

    def must_not_mutate(*args):
        raise AssertionError("damaged clean copy reached adversarial mutation")

    from pathlib import Path
    monkeypatch.setattr(v3spec, "ADVERSARIAL", ("auto_resume",))
    monkeypatch.setattr(v3spec, "evaluate_evidence", clean)
    monkeypatch.setattr(v3spec.shutil, "copytree", damaged_copy)
    monkeypatch.setattr(v3spec, "apply_tamper", must_not_mutate)
    result = v3spec.run_adversarial(source)
    assert result == [{"name": "auto_resume", "fixes_passed": False, "evidence_valid": False, "codes": ["clean_copy", "io_error"]}]
    assert evaluations == [source]
    assert (source / "trace.bin").read_bytes() == b"original"
