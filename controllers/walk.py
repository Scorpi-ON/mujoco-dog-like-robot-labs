import math
from typing import TYPE_CHECKING, cast

import mujoco
import numpy as np
import onnxruntime as ort
import yaml

from .common import (
    DEFAULT_POLICY,
    DEFAULT_POLICY_CONFIG,
    HOME_KEY,
    ROUTE_SITE_NAMES,
    ControllerStatus,
    joint_addresses_for_actuators,
    keyframe_id,
    object_id,
    quat_conjugate_rotate,
    wrap_angle,
    yaw_from_quat,
)

if TYPE_CHECKING:
    from pathlib import Path


class RouteWalkingController:
    """Stable single-target walking demo based on the pretrained velocity policy."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        *,
        policy_path: Path = DEFAULT_POLICY,
        config_path: Path = DEFAULT_POLICY_CONFIG,
        target_name: str = ROUTE_SITE_NAMES[1],
        waypoint_radius: float = 0.35,
    ) -> None:
        self._model = model
        self._home_key_id = keyframe_id(model, HOME_KEY)
        self._base_id = object_id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
        self._target_name = target_name
        self._target_site_id = object_id(model, mujoco.mjtObj.mjOBJ_SITE, target_name)
        self._waypoint_radius = waypoint_radius

        with config_path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        self._policy_dt = float(cfg["step_dt"])
        self._joint_map = np.array(cfg["joint_ids_map"], dtype=np.int32)
        self._default_joint_pos = np.array(cfg["default_joint_pos"], dtype=np.float32)
        self._stiffness = np.array(cfg["stiffness"], dtype=np.float32)
        self._damping = np.array(cfg["damping"], dtype=np.float32)
        action_cfg = cfg["actions"]["JointPositionAction"]
        self._action_scale = np.array(action_cfg["scale"], dtype=np.float32)
        self._action_offset = np.array(action_cfg["offset"], dtype=np.float32)

        self._qpos_adr, self._qvel_adr = joint_addresses_for_actuators(model, self._joint_map)
        self._ctrl_low = model.actuator_ctrlrange[:, 0]
        self._ctrl_high = model.actuator_ctrlrange[:, 1]
        self._gyro_sensor_adr = self._sensor_adr(model, ("imu_gyro", "imu_ang_vel"))

        self._session = ort.InferenceSession(str(policy_path), providers=["CPUExecutionProvider"])
        self._input_name = self._session.get_inputs()[0].name
        self._output_name = self._session.get_outputs()[0].name

        self._command = np.zeros(3, dtype=np.float32)
        self._command_target = np.zeros(3, dtype=np.float32)
        self._gravity_world = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        self._obs = np.zeros((1, 45), dtype=np.float32)
        self._last_action = np.zeros(12, dtype=np.float32)
        self._target_joint_pos = self._default_joint_pos.copy()
        self._target_xy = np.zeros(2, dtype=float)
        self._next_policy_time = 0.0

        mujoco.mj_forward(model, data)
        self._target_xy[:] = data.site_xpos[self._target_site_id, :2]

    def reset(self, data: mujoco.MjData) -> None:
        mujoco.mj_resetDataKeyframe(self._model, data, self._home_key_id)
        for policy_i, actuator_i in enumerate(self._joint_map):
            joint_id = self._model.actuator_trnid[actuator_i, 0]
            data.qpos[self._model.jnt_qposadr[joint_id]] = self._default_joint_pos[policy_i]
        data.qvel[:] = 0.0
        data.ctrl[:] = 0.0
        self._command[:] = 0.0
        self._command_target[:] = 0.0
        self._last_action[:] = 0.0
        self._target_joint_pos[:] = self._default_joint_pos
        self._next_policy_time = 0.0
        mujoco.mj_forward(self._model, data)

    def step(self, data: mujoco.MjData) -> ControllerStatus:
        command, status = self._walk_command(data)
        if data.time >= self._next_policy_time - 1e-9:
            obs = self._observation(data, command)
            action_batch = cast("np.ndarray", self._session.run([self._output_name], {self._input_name: obs})[0])
            action = action_batch[0]
            self._last_action[:] = action.astype(np.float32)
            self._target_joint_pos[:] = self._last_action * self._action_scale + self._action_offset
            self._next_policy_time += self._policy_dt

        self._apply_pd_torques(data)
        return status

    def _walk_command(self, data: mujoco.MjData) -> tuple[np.ndarray, ControllerStatus]:
        base_position = data.xpos[self._base_id]
        error_xy = self._target_xy - base_position[:2]
        distance = float(np.linalg.norm(error_xy))
        desired_yaw = math.atan2(float(error_xy[1]), float(error_xy[0]))
        current_yaw = yaw_from_quat(data.qpos[3:7])
        yaw_error = wrap_angle(desired_yaw - current_yaw)

        if distance <= self._waypoint_radius:
            self._command_target[:] = 0.0
            mode = "stand_at_target"
        else:
            speed = min(0.42, 0.12 + 0.16 * distance)
            vx = speed * math.cos(yaw_error)
            vy = float(np.clip(speed * math.sin(yaw_error), -0.30, 0.30))
            yaw_rate = float(np.clip(0.9 * yaw_error, -0.55, 0.55))
            if abs(yaw_error) > 1.25:
                vx = 0.0
                vy = 0.0
            self._command_target[:] = (vx, vy, yaw_rate)
            mode = "single_target_walk"

        self._command[:] = 0.75 * self._command + 0.25 * self._command_target
        return self._command, ControllerStatus(
            mode=mode,
            target_name=self._target_name,
            distance=distance,
            yaw_error=yaw_error,
            base_position=(
                float(base_position[0]),
                float(base_position[1]),
                float(base_position[2]),
            ),
        )

    def _observation(self, data: mujoco.MjData, command: np.ndarray) -> np.ndarray:
        obs = self._obs[0]
        obs[0:3] = data.sensordata[self._gyro_sensor_adr : self._gyro_sensor_adr + 3]
        obs[3:6] = quat_conjugate_rotate(data.qpos[3:7], self._gravity_world)
        obs[6:9] = command
        obs[9:21] = data.qpos[self._qpos_adr] - self._default_joint_pos
        obs[21:33] = data.qvel[self._qvel_adr]
        obs[33:45] = self._last_action
        return self._obs

    def _apply_pd_torques(self, data: mujoco.MjData) -> None:
        joint_pos = data.qpos[self._qpos_adr]
        joint_vel = data.qvel[self._qvel_adr]
        torques = self._stiffness * (self._target_joint_pos - joint_pos) - self._damping * joint_vel

        data.ctrl[:] = 0.0
        for policy_i, actuator_i in enumerate(self._joint_map):
            data.ctrl[actuator_i] = float(
                np.clip(torques[policy_i], self._ctrl_low[actuator_i], self._ctrl_high[actuator_i])
            )

    @staticmethod
    def _sensor_adr(model: mujoco.MjModel, names: tuple[str, ...]) -> int:
        for name in names:
            sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
            if sensor_id >= 0:
                return int(model.sensor_adr[sensor_id])
        raise ValueError(f"Required gyro sensor not found. Tried: {names}")
