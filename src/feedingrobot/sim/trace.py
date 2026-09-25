"""Per-step records for one physics run. Saved as uncompressed numeric arrays."""

from __future__ import annotations

import numpy as np

from feedingrobot.sim.trial import FORBIDDEN, is_forbidden_contact


class Trace:
    def __init__(self):
        self.rows = []
        self.contacts = []
        self.geom_names = {}

    def add(self, state: dict, tau_requested, tau_command, group_force, single_peak, outside: bool, departure) -> None:
        qacc = np.asarray(state.get("qacc", state.get("qacc_arm")), dtype=np.float64)
        self.rows.append(
            {
                "time_s": float(state["episode_time"]),
                "input_interval": np.asarray(state["input_interval"], dtype=np.float64),
                "q": np.asarray(state["q"], dtype=np.float64),
                "dq": np.asarray(state["dq"], dtype=np.float64),
                "qacc": qacc,
                "tau_requested": np.asarray(tau_requested, dtype=np.float64),
                "tau_command": np.asarray(tau_command, dtype=np.float64),
                "tau_actual": np.asarray(state["actuator_force"], dtype=np.float64),
                "tcp_pos": np.asarray(state["tcp_pos"], dtype=np.float64),
                "tcp_rot": np.asarray(state["tcp_mat"], dtype=np.float64).reshape(9),
                "mouth_pos": np.asarray(state["mouth_pos"], dtype=np.float64),
                "mouth_rot": np.asarray(state["mouth_mat"], dtype=np.float64).reshape(9),
                "food_pos": np.asarray(state["food_pos"], dtype=np.float64),
                "food_quat": np.asarray(state["food_quat"], dtype=np.float64),
                "food_vel": np.asarray(state["food_vel"], dtype=np.float64),
                "raw_wrench": np.asarray(state["raw_wrench_sensor"], dtype=np.float64),
                "contact_force_group": np.asarray(group_force, dtype=np.float64),
                "single_contact_peak": float(single_peak),
                "min_contact_dist": float(min((c["dist"] for c in state["contacts"]), default=0.0)),
                "warning_count": int(sum(state["warnings"])),
                "velocity_fault": bool(state["velocity_fault"]),
                "forbidden_contact": any(
                    is_forbidden_contact(row["group1"], row["group2"], FORBIDDEN) for row in state["contacts"]
                ),
                "event_candidate": bool(departure.candidate or departure.confirmed_s is not None),
                "event_confirmed": bool(departure.confirmed_s is not None and state["episode_time"] + 1e-12 >= departure.confirmed_s),
                "mouth_contact": any("mouth_upper" in (c["geom1"], c["geom2"]) for c in state["contacts"]),
                "head_target": np.asarray(state["head_target"], dtype=np.float64),
                "head_actual": np.concatenate([state["head_q"], [state["jaw_q"]]]).astype(np.float64),
                "head_command": np.asarray(state["head_tau"], dtype=np.float64),
            }
        )
        tick = len(self.rows) - 1
        for row in state["contacts"]:
            self.contacts.append(
                (
                    tick,
                    int(row.get("geom1_id", -1)),
                    int(row.get("geom2_id", -1)),
                    float(row["pos"][0]),
                    float(row["pos"][1]),
                    float(row["pos"][2]),
                    float(row["dist"]),
                    float(row["force_on_geom2_world"][0]),
                    float(row["force_on_geom2_world"][1]),
                    float(row["force_on_geom2_world"][2]),
                )
            )
            self.geom_names[int(row.get("geom1_id", -1))] = row["geom1"] or ""
            self.geom_names[int(row.get("geom2_id", -1))] = row["geom2"] or ""

    def arrays(self) -> dict:
        if not self.rows:
            raise RuntimeError("empty trace")
        flags = ("velocity_fault", "forbidden_contact", "event_candidate", "event_confirmed", "mouth_contact")
        keys = [k for k in self.rows[0] if k not in flags]
        out = {key: np.stack([row[key] for row in self.rows]) for key in keys}
        out["input_interval_s"] = out["input_interval"]
        for key in flags:
            out[key] = np.asarray([row[key] for row in self.rows], dtype=np.int8)
        out["warning_count"] = out["warning_count"].astype(np.int32)
        flat = np.asarray(self.contacts, dtype=np.float64) if self.contacts else np.zeros((0, 10))
        out["contact_flat"] = flat
        ids = sorted(self.geom_names)
        out["geom_ids"] = np.asarray(ids, dtype=np.int32)
        out["geom_name_bytes"] = np.asarray([self.geom_names[i].encode() for i in ids], dtype="S64")
        return out
