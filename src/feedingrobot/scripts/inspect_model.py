"""Print the M1 model structure, actuator audit, and tool mass."""

from __future__ import annotations

import argparse
import json

import mujoco
import numpy as np

from feedingrobot.sim.model import ARM_ACTUATORS, load_config, repo_root
from feedingrobot.sim.scene import FeedingScene


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/m1_scene.json")
    args = parser.parse_args()
    cfg = load_config(args.config)
    scene = FeedingScene(cfg)
    model = scene.model
    rows = []
    for name in ARM_ACTUATORS:
        act = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        rows.append(
            {
                "name": name,
                "dyntype": int(model.actuator_dyntype[act]),
                "biastype": int(model.actuator_biastype[act]),
                "gain": float(model.actuator_gainprm[act, 0]),
                "gear": float(model.actuator_gear[act, 0]),
                "ctrlrange": model.actuator_ctrlrange[act].tolist(),
                "forcerange": model.actuator_forcerange[act].tolist(),
            }
        )
    summary = {
        "nq": int(model.nq),
        "nv": int(model.nv),
        "nu": int(model.nu),
        "timestep": float(model.opt.timestep),
        "gravity": model.opt.gravity.tolist(),
        "tool_mass": float(model.body_mass[scene.index.tool_body_id]),
        "tool_com": model.body_ipos[scene.index.tool_body_id].tolist(),
        "food_mass": float(model.body_mass[scene.index.food_body_id]),
        "arm_actuators": rows,
        "sites": list(scene.index.site_ids),
        "repo": str(repo_root()),
    }
    print(json.dumps(summary, indent=2))
    assert abs(summary["tool_mass"] - cfg["tool_mass_kg"]) < 1e-9
    assert np.allclose(summary["gravity"], [0, 0, -9.81])


if __name__ == "__main__":
    main()
