"""Contact records. World force is contact-frame force times the contact axes.

frame[0:3], frame[3:6], frame[6:9] are the normal and two tangents in world.
mj_contactForce[:3] is (normal, tangent, tangent). force_on_geom2 = +that map;
force_on_geom1 is the opposite. A plate/food support then sums to +Z on the food.
"""

from __future__ import annotations

import mujoco
import numpy as np

_GROUPS = (
    ("spoon", ("tool_", "bowl_")),
    ("food", ("food_",)),
    ("plate", ("plate_",)),
    ("table", ("table",)),
    ("floor", ("floor",)),
    ("mouth", ("mouth_", "jaw_")),
)


def geom_group(name: str | None) -> str:
    if not name:
        return "arm"
    for group, prefixes in _GROUPS:
        if name.startswith(prefixes):
            return group
    return "arm"


def contact_world_force(force_contact: np.ndarray, frame: np.ndarray) -> np.ndarray:
    axes = np.asarray(frame, dtype=float).reshape(9)
    return (
        force_contact[0] * axes[0:3]
        + force_contact[1] * axes[3:6]
        + force_contact[2] * axes[6:9]
    )


def read_contacts(model, data) -> list[dict]:
    rows = []
    for i in range(data.ncon):
        con = data.contact[i]
        force = np.zeros(6)
        mujoco.mj_contactForce(model, data, i, force)
        n1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(con.geom1))
        n2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(con.geom2))
        fw = contact_world_force(force[:3], con.frame)
        rows.append(
            {
                "geom1": n1,
                "geom2": n2,
                "geom1_id": int(con.geom1),
                "geom2_id": int(con.geom2),
                "group1": geom_group(n1),
                "group2": geom_group(n2),
                "dist": float(con.dist),
                "pos": np.array(con.pos, dtype=float).copy(),
                "frame": np.array(con.frame, dtype=float).copy(),
                "force_contact": np.array(force[:3], dtype=float).copy(),
                "force_on_geom2_world": fw,
                "force_on_geom1_world": -fw,
            }
        )
    return rows


def pair_set(contacts: list[dict]) -> set[tuple[str, str]]:
    pairs = set()
    for row in contacts:
        pairs.add(tuple(sorted((row["group1"], row["group2"]))))
    return pairs


def min_distance(contacts: list[dict]) -> float:
    if not contacts:
        return 0.0
    return float(min(row["dist"] for row in contacts))
