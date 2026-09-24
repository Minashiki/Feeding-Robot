"""Print contact pairs. --viewer opens the passive viewer when a display exists."""

from __future__ import annotations

import argparse

from feedingrobot.sim.contacts import pair_set
from feedingrobot.sim.scene import FeedingScene


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/m1_scene.json")
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--preset", default="food_on_plate")
    args = parser.parse_args()
    scene = FeedingScene(args.config)
    state = scene.reset(seed=0, preset=args.preset)
    print(f"ncon={len(state['contacts'])} pairs={sorted(pair_set(state['contacts']))}")
    for row in state["contacts"]:
        print(f"  {row['geom1']} -- {row['geom2']} dist={row['dist']:.5f} f={row['force_on_geom2_world']}")
    if not args.viewer:
        return
    try:
        import mujoco.viewer as viewer
    except Exception as exc:
        print(f"viewer unavailable: {exc}")
        return
    viewer.launch_passive(scene.model, scene.data)


if __name__ == "__main__":
    main()
