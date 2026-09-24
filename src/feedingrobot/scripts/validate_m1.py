"""Run the M1 pytest suite and write a JSON report plus optional frames."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import mujoco
import numpy as np

from feedingrobot.sim.model import load_config, repo_root


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _try_frames(cfg, out: Path) -> list[str]:
    saved = []
    try:
        from feedingrobot.sim.scene import FeedingScene

        scene = FeedingScene(cfg)
        renderer = mujoco.Renderer(scene.model, 480, 640)
    except Exception as exc:
        (out / "render_error.txt").write_text(str(exc))
        return saved
    option = mujoco.MjvOption()
    option.frame = mujoco.mjtFrame.mjFRAME_SITE
    cases = {
        "carry": ("food_on_spoon", 0),
        "plate": ("food_on_plate", 0),
        "mouth": ("near_mouth", 0),
    }
    for name, (preset, seed) in cases.items():
        scene.reset(seed=seed, preset=preset)
        renderer.update_scene(scene.data, camera="overview", scene_option=option)
        image = renderer.render()
        path = out / f"{name}.png"
        try:
            from PIL import Image
        except ImportError:
            np.save(out / f"{name}.npy", image)
            saved.append(str(out / f"{name}.npy"))
            continue
        Image.fromarray(image).save(path)
        saved.append(str(path))
    renderer.close()
    return saved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/m1_scene.json")
    parser.add_argument("--output", default="outputs/m1/acceptance")
    args = parser.parse_args()
    root = repo_root()
    out = Path(args.output)
    if not out.is_absolute():
        out = root / out
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_config(args.config)
    (out / "m1_scene.json").write_text(json.dumps(cfg, indent=2))
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_m1_model.py", "tests/test_m1_torque.py", "tests/test_m1_wrench.py", "tests/test_m1_contacts.py", "tests/test_m1_reset.py", "tests/test_m1_long.py", "-q"],
        cwd=root,
        capture_output=True,
        text=True,
    )
    (out / "pytest.txt").write_text(proc.stdout + "\n" + proc.stderr)
    model_path = root / cfg["model"]
    report = {
        "passed": proc.returncode == 0,
        "pytest_returncode": proc.returncode,
        "python": sys.version,
        "platform": platform.platform(),
        "mujoco": mujoco.__version__,
        "numpy": np.__version__,
        "model_sha256": _sha256(model_path),
        "robot_sha256": _sha256(root / "assets/robots/panda_torque.xml"),
        "config_sha256": _sha256(root / cfg["_config_path"]),
        "frames": _try_frames(cfg, out),
        "interface": {
            "step": "FeedingScene.step_physics(tau_arm) -> one physics step",
            "tau_arm": "7-vector of joint torque, N*m, software-clipped to motor ranges",
            "wrench_sign": cfg["wrench_sign"],
            "compensated_wrench": None,
        },
        "known_limits": [
            "compensated_wrench is absent until M2",
            "head and jaw torques are a bounded tracking fixture inside the scene driver",
            "diagnostic_pd_tau is a test fixture and is not the feeding controller",
            "same-seed replay is bit-close on this machine only",
        ],
    }
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print(proc.stdout)
    if proc.returncode != 0:
        print(proc.stderr)
        raise SystemExit(proc.returncode)
    print(f"wrote {out / 'report.json'}")


if __name__ == "__main__":
    main()
