"""Load the M1 scene and resolve named indexes. Do not assume qpos[:7] is the arm."""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]

ARM_JOINTS = [f"joint{i}" for i in range(1, 8)]
ARM_ACTUATORS = [f"arm_tau_{i}" for i in range(1, 8)]
HEAD_JOINTS = ["head_x", "head_y", "head_z", "head_yaw"]
HEAD_ACTUATORS = ["head_fx", "head_fy", "head_fz", "head_tau_yaw", "jaw_tau"]
SITES = ["ft_site", "tcp", "plate_frame", "mouth_entry", "mouth_receiver"]
SENSORS = ["ft_force", "ft_torque"]


def repo_root() -> Path:
    return REPO_ROOT


def load_config(path: str | Path) -> dict:
    path = Path(path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    with path.open() as f:
        cfg = json.load(f)
    cfg["_config_path"] = str(path)
    return cfg


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def _id(model, obj, name: str) -> int:
    index = mujoco.mj_name2id(model, obj, name)
    if index < 0:
        raise KeyError(f"missing {name}")
    return int(index)


def load_model(xml_path: str | Path) -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_path(str(resolve_path(xml_path)))


class ModelIndex:
    def __init__(self, model: mujoco.MjModel):
        self.model = model
        self.arm_joint_ids = np.array([_id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in ARM_JOINTS])
        self.arm_qpos_adr = np.array([model.jnt_qposadr[i] for i in self.arm_joint_ids])
        self.arm_dof_adr = np.array([model.jnt_dofadr[i] for i in self.arm_joint_ids])
        self.arm_actuator_ids = np.array([_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in ARM_ACTUATORS])
        self.food_joint_id = _id(model, mujoco.mjtObj.mjOBJ_JOINT, "food_joint")
        self.food_qpos_adr = int(model.jnt_qposadr[self.food_joint_id])
        self.food_dof_adr = int(model.jnt_dofadr[self.food_joint_id])
        self.head_joint_ids = np.array([_id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in HEAD_JOINTS])
        self.jaw_joint_id = _id(model, mujoco.mjtObj.mjOBJ_JOINT, "jaw")
        self.head_qpos_adr = np.array([model.jnt_qposadr[i] for i in self.head_joint_ids])
        self.head_dof_adr = np.array([model.jnt_dofadr[i] for i in self.head_joint_ids])
        self.jaw_qpos_adr = int(model.jnt_qposadr[self.jaw_joint_id])
        self.jaw_dof_adr = int(model.jnt_dofadr[self.jaw_joint_id])
        self.scene_actuator_ids = np.array(
            [_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in HEAD_ACTUATORS]
        )
        self.site_ids = {n: _id(model, mujoco.mjtObj.mjOBJ_SITE, n) for n in SITES}
        self.force_sensor_id = _id(model, mujoco.mjtObj.mjOBJ_SENSOR, "ft_force")
        self.torque_sensor_id = _id(model, mujoco.mjtObj.mjOBJ_SENSOR, "ft_torque")
        self.force_adr = int(model.sensor_adr[self.force_sensor_id])
        self.torque_adr = int(model.sensor_adr[self.torque_sensor_id])
        self.force_dim = int(model.sensor_dim[self.force_sensor_id])
        self.torque_dim = int(model.sensor_dim[self.torque_sensor_id])
        self.tool_body_id = _id(model, mujoco.mjtObj.mjOBJ_BODY, "tool_mount")
        self.food_body_id = _id(model, mujoco.mjtObj.mjOBJ_BODY, "food")

    def assert_arm_motors(self) -> None:
        model = self.model
        if len(self.arm_actuator_ids) != 7:
            raise AssertionError("expected 7 arm actuators")
        for act_id, joint_id in zip(self.arm_actuator_ids, self.arm_joint_ids):
            if int(model.actuator_trntype[act_id]) != int(mujoco.mjtTrn.mjTRN_JOINT):
                raise AssertionError("arm actuator is not joint-transmitted")
            if int(model.actuator_trnid[act_id, 0]) != int(joint_id):
                raise AssertionError("arm actuator drives the wrong joint")
            if int(model.actuator_biastype[act_id]) != 0:
                raise AssertionError("arm actuator has a bias")
            if int(model.actuator_dyntype[act_id]) != int(mujoco.mjtDyn.mjDYN_NONE):
                raise AssertionError("arm actuator dynamics are active")
            if abs(float(model.actuator_gainprm[act_id, 0]) - 1.0) > 1e-12:
                raise AssertionError("arm actuator gain is not 1")
            if abs(float(model.actuator_gear[act_id, 0]) - 1.0) > 1e-12:
                raise AssertionError("arm actuator gear is not 1")
        if self.force_dim != 3 or self.torque_dim != 3:
            raise AssertionError("F/T sensors must be 3+3")
