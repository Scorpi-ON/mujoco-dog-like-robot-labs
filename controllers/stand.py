import mujoco
import numpy as np
import yaml

from .common import (
    DEFAULT_POLICY_CONFIG,
    HOME_KEY,
    ControllerStatus,
    infer_robot_spec,
    joint_addresses_for_actuators,
    keyframe_id,
    object_id,
)


class StandController:
    def __init__(self, model: mujoco.MjModel, *, level_tilt: bool = False) -> None:
        self._model = model
        self._home_key_id = keyframe_id(model, HOME_KEY)
        self._home_ctrl = model.key_ctrl[self._home_key_id].copy()
        self._spec = infer_robot_spec(model)
        self._base_id = object_id(model, mujoco.mjtObj.mjOBJ_BODY, self._spec.base_body)
        self._torque_pd_stand = self._spec.base_body == "base_link"
        self._level_tilt = level_tilt

        if self._torque_pd_stand:
            with DEFAULT_POLICY_CONFIG.open("r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            self._joint_map = np.array(cfg["joint_ids_map"], dtype=np.int32)
            self._default_joint_pos = np.array(cfg["default_joint_pos"], dtype=np.float32)
            self._stiffness = np.array(cfg["stiffness"], dtype=np.float32)
            self._damping = np.array(cfg["damping"], dtype=np.float32)
            self._qpos_adr, self._qvel_adr = joint_addresses_for_actuators(model, self._joint_map)
            self._ctrl_low = model.actuator_ctrlrange[:, 0]
            self._ctrl_high = model.actuator_ctrlrange[:, 1]
            self._target_joint_pos = self._default_joint_pos.copy()

    def reset(self, data: mujoco.MjData) -> None:
        mujoco.mj_resetDataKeyframe(self._model, data, self._home_key_id)
        if self._torque_pd_stand:
            for policy_i, actuator_i in enumerate(self._joint_map):
                joint_id = self._model.actuator_trnid[actuator_i, 0]
                data.qpos[self._model.jnt_qposadr[joint_id]] = self._default_joint_pos[policy_i]
            data.ctrl[:] = 0.0
        else:
            data.ctrl[:] = self._home_ctrl
        mujoco.mj_forward(self._model, data)

    def step(self, data: mujoco.MjData) -> ControllerStatus:
        if self._torque_pd_stand:
            self._target_joint_pos[:] = self._default_joint_pos
            if self._level_tilt:
                pitch = self._pitch(data.qpos[3:7])
                thigh_correction = float(np.clip(6.0 * pitch, -0.25, 0.25))
                calf_correction = float(np.clip(3.6 * pitch, -0.15, 0.15))
                for joint_offset in (0, 3):
                    self._target_joint_pos[joint_offset + 1] -= thigh_correction
                    self._target_joint_pos[joint_offset + 2] += calf_correction
                for joint_offset in (6, 9):
                    self._target_joint_pos[joint_offset + 1] += thigh_correction
                    self._target_joint_pos[joint_offset + 2] -= calf_correction

            joint_pos = data.qpos[self._qpos_adr].astype(np.float32)
            joint_vel = data.qvel[self._qvel_adr].astype(np.float32)
            torques = self._stiffness * (self._target_joint_pos - joint_pos) - self._damping * joint_vel
            data.ctrl[:] = 0.0
            for policy_i, actuator_i in enumerate(self._joint_map):
                data.ctrl[actuator_i] = float(
                    np.clip(torques[policy_i], self._ctrl_low[actuator_i], self._ctrl_high[actuator_i])
                )
        else:
            data.ctrl[:] = self._home_ctrl

        base_position = data.xpos[self._base_id]
        return ControllerStatus(
            mode="stand",
            target_name=HOME_KEY,
            distance=0.0,
            yaw_error=0.0,
            base_position=(
                float(base_position[0]),
                float(base_position[1]),
                float(base_position[2]),
            ),
        )

    @staticmethod
    def _pitch(quat: np.ndarray) -> float:
        w, x, y, z = quat
        sin_pitch = 2.0 * (w * y - z * x)
        return float(np.arcsin(np.clip(sin_pitch, -1.0, 1.0)))
