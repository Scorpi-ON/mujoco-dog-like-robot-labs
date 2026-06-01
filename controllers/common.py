import math
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import mujoco
import numpy as np

HOME_KEY = "home"
ROUTE_SITE_NAMES = ("wp_a", "wp_b", "wp_c", "wp_d")
LINE_ROUTE_SITE_NAMES = ("wp_a", "wp_b")
ROOT = Path(__file__).resolve().parent.parent
DEFAULT_POLICY = ROOT / "vendor" / "unitree-go2-velocity-flat" / "policy.onnx"
DEFAULT_POLICY_CONFIG = ROOT / "vendor" / "unitree-go2-velocity-flat" / "params" / "deploy.yaml"


@dataclass(frozen=True)
class RobotSpec:
    base_body: str
    frequency_hz: float
    stride: float
    turn_stride: float
    knee_lift: float
    knee_push: float
    waypoint_radius: float


class Controller(Protocol):
    def reset(self, data: mujoco.MjData) -> None: ...

    def step(self, data: mujoco.MjData) -> ControllerStatus: ...


@dataclass(frozen=True)
class ControllerStatus:
    mode: str
    target_name: str
    distance: float
    yaw_error: float
    base_position: tuple[float, float, float]

    @property
    def base_xy(self) -> tuple[float, float]:
        return (self.base_position[0], self.base_position[1])


@dataclass(frozen=True)
class ObstacleScan:
    front_distance: float
    left_clearance: float
    right_clearance: float
    front_geom_id: int = -1


def wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def yaw_from_quat(quat: np.ndarray) -> float:
    w, x, y, z = quat
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def keyframe_id(model: mujoco.MjModel, name: str) -> int:
    return object_id(model, mujoco.mjtObj.mjOBJ_KEY, name)


def infer_robot_spec(model: mujoco.MjModel) -> RobotSpec:
    if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link") >= 0:
        return RobotSpec(
            base_body="base_link",
            frequency_hz=1.7,
            stride=0.18,
            turn_stride=0.18,
            knee_lift=0.25,
            knee_push=0.06,
            waypoint_radius=0.24,
        )

    if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "body") >= 0:
        return RobotSpec(
            base_body="body",
            frequency_hz=1.8,
            stride=0.24,
            turn_stride=0.16,
            knee_lift=0.32,
            knee_push=0.08,
            waypoint_radius=0.32,
        )

    if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base") >= 0:
        return RobotSpec(
            base_body="base",
            frequency_hz=1.7,
            stride=0.18,
            turn_stride=0.18,
            knee_lift=0.25,
            knee_push=0.06,
            waypoint_radius=0.24,
        )

    raise ValueError("Could not infer robot model: expected body 'body' or 'base'.")


def object_id(model: mujoco.MjModel, object_type: mujoco.mjtObj, name: str) -> int:
    object_id_value = mujoco.mj_name2id(model, object_type, name)
    if object_id_value < 0:
        raise ValueError(f"Required MuJoCo object '{name}' was not found.")
    return object_id_value


def joint_addresses_for_actuators(model: mujoco.MjModel, joint_map: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    qpos_adr = []
    qvel_adr = []
    for actuator_i in joint_map:
        joint_id = model.actuator_trnid[int(actuator_i), 0]
        qpos_adr.append(model.jnt_qposadr[joint_id])
        qvel_adr.append(model.jnt_dofadr[joint_id])
    return np.array(qpos_adr, dtype=np.int32), np.array(qvel_adr, dtype=np.int32)


def quat_conjugate_rotate(quat: np.ndarray, vector: np.ndarray) -> np.ndarray:
    w, x, y, z = quat
    quat_vector = np.array([-x, -y, -z], dtype=float)
    t = 2.0 * np.cross(quat_vector, vector)
    return vector + w * t + np.cross(quat_vector, t)
