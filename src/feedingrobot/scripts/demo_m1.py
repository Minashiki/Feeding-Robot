"""Diagnostic joint-torque demo. This is not the feeding controller."""

from __future__ import annotations

import argparse

import numpy as np

from feedingrobot.sim.scene import FeedingScene


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/m1_scene.json")
    parser.add_argument("--case", default="carry_drop", choices=["carry_drop", "head"])
    parser.add_argument("--viewer", action="store_true")
    args = parser.parse_args()
    scene = FeedingScene(args.config)
    viewer = None
    if args.viewer:
        import mujoco.viewer

        scene.reset(seed=0, preset="food_on_spoon" if args.case == "carry_drop" else "food_on_plate")
        viewer = mujoco.viewer.launch_passive(scene.model, scene.data)
    else:
        scene.reset(seed=0, preset="food_on_spoon" if args.case == "carry_drop" else "food_on_plate")
    q_tilt = np.array(scene.config["q_tilt"], dtype=float)
    steps = 2500 if args.case == "carry_drop" else 2000
    for k in range(steps):
        if args.case == "carry_drop":
            tau = scene.diagnostic_pd_tau(q_tilt, kp=25, kd=6)
            scene.step_physics(tau, hold_driver=True)
        else:
            scene.step_physics(scene.bias_tau(), hold_driver=False)
        if viewer is not None:
            viewer.sync()
        if k % 500 == 0:
            state = scene.snapshot()
            print(f"t={state['sim_time']:.3f} food={np.round(state['food_pos'], 3)} ncon={len(state['contacts'])}")
    if viewer is not None:
        viewer.close()


if __name__ == "__main__":
    main()
