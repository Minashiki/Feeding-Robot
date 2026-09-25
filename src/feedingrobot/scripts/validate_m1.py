"""Rebuild the M1 acceptance report. A pytest pass alone does not set m2_ready."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import struct
import subprocess
import sys
import zlib
from datetime import datetime, timezone
from pathlib import Path

import mujoco
import numpy as np

from feedingrobot.sim.model import load_config, repo_root
from feedingrobot.sim.report import assess_run


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tracked_files(root: Path, config_path: Path) -> list[Path]:
    import re

    required = [config_path.resolve()]
    for rel in ("pyproject.toml", "requirements.txt", "third_party_manifest.json"):
        required.append((root / rel).resolve())
    for folder in (root / "src" / "feedingrobot" / "sim", root / "src" / "feedingrobot" / "scripts", root / "tests"):
        required.extend(path.resolve() for path in folder.rglob("*") if path.is_file() and path.suffix in {".py", ".xml", ".json"})
    cfg = json.loads(config_path.read_text())
    for key in ("model", "fixture", "sphere_model"):
        required.append((root / cfg[key]).resolve())
    pending = [path for path in required if path.suffix == ".xml"]
    seen = set()
    while pending:
        xml = pending.pop()
        if xml in seen:
            continue
        seen.add(xml)
        text = xml.read_text()
        mesh_match = re.search(r"""meshdir=["']([^"']+)["']""", text)
        meshdir = (xml.parent / mesh_match.group(1)).resolve() if mesh_match else xml.parent
        for match in re.findall(r"""file=["']([^"']+)["']""", text):
            child = (xml.parent / match).resolve()
            if not child.is_file():
                child = (meshdir / match).resolve()
            required.append(child)
            if child.suffix == ".xml":
                pending.append(child)
    manifest = json.loads((root / "third_party_manifest.json").read_text())
    for entry in manifest["entries"]:
        for item in entry["files"]:
            required.append((root / item["local_path"]).resolve())
    missing = sorted({str(path.relative_to(root)) for path in required if not path.is_file()})
    if missing:
        raise SystemExit("missing required input: " + missing[0])
    return sorted(set(required))


def _bundle_hash(root: Path, files: list[Path]) -> dict:
    rows = []
    for path in files:
        rel = str(path.relative_to(root))
        rows.append({"path": rel, "sha256": _sha256(path)})
    digest = hashlib.sha256()
    for row in rows:
        digest.update(row["path"].encode())
        digest.update(b"\0")
        digest.update(row["sha256"].encode())
        digest.update(b"\n")
    return {"files": rows, "aggregate_sha256": digest.hexdigest()}


def _png(path: Path, image: np.ndarray) -> None:
    image = np.asarray(image, dtype=np.uint8)
    height, width = image.shape[:2]
    raw = b"".join(b"\x00" + image[y].tobytes() for y in range(height))

    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, 9))
        + chunk(b"IEND", b"")
    )


def _render(cfg, out: Path) -> tuple[str, list[dict]]:
    os.environ.setdefault("MUJOCO_GL", "egl")
    try:
        from feedingrobot.sim.scene import FeedingScene

        scene = FeedingScene(cfg)
        renderer = mujoco.Renderer(scene.model, 480, 640)
    except Exception as exc:
        (out / "render_error.txt").write_text(str(exc))
        return "unavailable", []
    index = []
    option = mujoco.MjvOption()
    fixed = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_CAMERA, "overview")
    shots = [
        ("overview_visual", "food_on_plate", 2, "fixed"),
        ("overview_collision", "food_on_plate", 3, "fixed"),
        ("overview_scene", "food_on_plate", None, "fixed"),
        ("spoon_close", "food_on_spoon", None, "tcp"),
        ("plate_close", "food_on_plate", None, "plate"),
        ("mouth_close", "near_mouth", None, "mouth"),
    ]
    for name, preset, group, look in shots:
        scene.reset(seed=0, preset=preset)
        option.geomgroup[:] = 1 if group is None else 0
        if group is not None:
            option.geomgroup[group] = 1
        option.frame = mujoco.mjtFrame.mjFRAME_NONE
        cam = mujoco.MjvCamera()
        if look == "fixed":
            cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            cam.fixedcamid = fixed
        else:
            cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            cam.lookat[:] = {
                "tcp": scene.data.site_xpos[scene.index.site_ids["tcp"]],
                "plate": [0.45, -0.18, 0.05],
                "mouth": [0.55, 0.12, 0.35],
            }[look]
            cam.distance = 0.45
            cam.azimuth = 140
            cam.elevation = -35
        renderer.update_scene(scene.data, camera=cam, scene_option=option)
        path = out / f"{name}.png"
        _png(path, renderer.render())
        index.append({"file": path.name, "preset": preset, "group": group, "seed": 0})
    renderer.close()
    return "passed", index


def main():
    parser = argparse.ArgumentParser(description="Run M1 tests and write a traceable acceptance report.")
    parser.add_argument("--config", default="configs/m1_scene.json")
    parser.add_argument("--output", default="outputs/m1/acceptance/m1-fix-validation")
    args = parser.parse_args()
    root = repo_root()
    out = Path(args.output)
    if not out.is_absolute():
        out = root / out
    if (out / "run.json").exists() or (out / "report.json").exists() or (out / "cases").exists():
        raise SystemExit(f"refusing to reuse {out}; choose a new output directory")
    out.mkdir(parents=True)
    case_dir = out / "cases"
    case_dir.mkdir()
    run_id = out.name
    (out / "run.json").write_text(json.dumps({"schema_version": 2, "run_id": run_id, "status": "running", "incomplete": True}))
    before = _bundle_hash(root, _tracked_files(root, (root / args.config).resolve()))
    (out / "input_hash_before.json").write_text(json.dumps(before, indent=2))
    env = os.environ.copy()
    env["M1_CASE_DIR"] = str(case_dir)
    env["M1_RUN_DIR"] = str(out)
    env["M1_RUN_ID"] = run_id
    env["M1_EXECUTION_PATH"] = str(out / "execution.json")
    env["M1_CONFIG_PATH"] = args.config
    env.setdefault("MUJOCO_GL", "egl")
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_m1_model.py", "tests/test_m1_torque.py", "tests/test_m1_wrench.py", "tests/test_m1_contacts.py", "tests/test_m1_reset.py", "tests/test_m1_long.py", "tests/test_m1_report.py", "-q", "--m1-config", args.config],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
    )
    (out / "pytest.txt").write_text(proc.stdout + "\n" + proc.stderr)
    tracked = _tracked_files(root, (root / args.config).resolve())
    after = _bundle_hash(root, tracked)
    hashes_match = before["aggregate_sha256"] == after["aggregate_sha256"]
    cfg = load_config(args.config)
    (out / "config_snapshot.json").write_text(json.dumps(cfg, indent=2))
    visual_status, frames = _render(cfg, out)
    (out / "frames.json").write_text(json.dumps(frames, indent=2))
    final_hash = _bundle_hash(root, tracked)
    hashes_match = before["aggregate_sha256"] == after["aggregate_sha256"] == final_hash["aggregate_sha256"]
    (out / "input_hash_after.json").write_text(json.dumps(final_hash, indent=2))
    status = "completed" if proc.returncode == 0 and hashes_match else "failed"
    (out / "run.json").write_text(json.dumps({
        "schema_version": 2,
        "run_id": run_id,
        "status": status,
        "incomplete": status != "completed",
        "finished": datetime.now(timezone.utc).isoformat(),
        "pytest_returncode": proc.returncode,
        "visual_status": visual_status,
    }))
    verdict = assess_run(out)
    (out / "report.json").write_text(json.dumps({"run_id": run_id, **verdict}, indent=2))
    draft = {"run_id": run_id, **verdict}
    print(proc.stdout)
    print(json.dumps({key: draft[key] for key in ("physics_passed", "provenance_passed", "visual_status", "m2_ready", "overall_status", "reasons")}, indent=2))
    if not draft["m2_ready"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
