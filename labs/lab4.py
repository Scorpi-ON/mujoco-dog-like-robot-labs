import math

import mujoco
import numpy as np
import yaml
from lab_common import artifact_dir, csv_float, load_bare_model, open_lab_viewer, roll_pitch_yaw, write_csv

from controllers.common import (
    DEFAULT_POLICY_CONFIG,
    HOME_KEY,
    ControllerStatus,
    joint_addresses_for_actuators,
    keyframe_id,
    object_id,
)


class SmoothJointMotionController:
    """Плавно меняет целевые углы суставов и применяет их через ПД-регулятор."""

    def __init__(self, model: mujoco.MjModel) -> None:
        self._model = model
        self._home_key_id = keyframe_id(model, HOME_KEY)
        self._base_id = object_id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
        with DEFAULT_POLICY_CONFIG.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        self._joint_map = np.array(cfg["joint_ids_map"], dtype=np.int32)
        self._default_joint_pos = np.array(cfg["default_joint_pos"], dtype=np.float32)
        self._stiffness = 0.75 * np.array(cfg["stiffness"], dtype=np.float32)
        self._damping = 1.10 * np.array(cfg["damping"], dtype=np.float32)
        self._qpos_adr, self._qvel_adr = joint_addresses_for_actuators(model, self._joint_map)
        self._ctrl_low = model.actuator_ctrlrange[:, 0]
        self._ctrl_high = model.actuator_ctrlrange[:, 1]
        self._target_joint_pos = self._default_joint_pos.copy()

    def reset(self, data: mujoco.MjData) -> None:
        mujoco.mj_resetDataKeyframe(self._model, data, self._home_key_id)
        for policy_i, actuator_i in enumerate(self._joint_map):
            joint_id = self._model.actuator_trnid[actuator_i, 0]
            data.qpos[self._model.jnt_qposadr[joint_id]] = self._default_joint_pos[policy_i]
        data.qpos[0] = 0.0
        data.qpos[1] = 0.0
        data.qpos[2] = 0.34
        data.qvel[:] = 0.0
        data.ctrl[:] = 0.0
        mujoco.mj_forward(self._model, data)

    def step(self, data: mujoco.MjData) -> ControllerStatus:
        phase = math.sin(2.0 * math.pi * 0.18 * data.time)
        self._target_joint_pos[:] = self._default_joint_pos

        for joint_offset in (0, 3):
            self._target_joint_pos[joint_offset + 1] += 0.10 * phase
            self._target_joint_pos[joint_offset + 2] -= 0.22 * phase
        for joint_offset in (6, 9):
            self._target_joint_pos[joint_offset + 1] -= 0.10 * phase
            self._target_joint_pos[joint_offset + 2] += 0.22 * phase

        joint_pos = data.qpos[self._qpos_adr].astype(np.float32)
        joint_vel = data.qvel[self._qvel_adr].astype(np.float32)
        torques = self._stiffness * (self._target_joint_pos - joint_pos) - self._damping * joint_vel

        data.ctrl[:] = 0.0
        for policy_i, actuator_i in enumerate(self._joint_map):
            data.ctrl[actuator_i] = float(
                np.clip(torques[policy_i], self._ctrl_low[actuator_i], self._ctrl_high[actuator_i])
            )

        base_position = data.xpos[self._base_id]
        mode = "front_bend_rear_extend" if phase >= 0.0 else "front_extend_rear_bend"
        return ControllerStatus(
            mode=mode,
            target_name="floor",
            distance=abs(float(phase)),
            yaw_error=0.0,
            base_position=(
                float(base_position[0]),
                float(base_position[1]),
                float(base_position[2]),
            ),
        )


def run_motion(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    controller: SmoothJointMotionController,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for step in range(2600):
        status = controller.step(data)
        mujoco.mj_step(model, data)
        if step % 50 == 0 or step == 2599:
            roll, pitch, yaw = roll_pitch_yaw(data.qpos[3:7])
            rows.append(
                {
                    "step": step,
                    "time": round(float(data.time), 4),
                    "mode": status.mode,
                    "motion_phase": round(float(status.distance), 4),
                    "base_z": round(float(status.base_position[2]), 4),
                    "roll": round(roll, 4),
                    "pitch": round(pitch, 4),
                    "yaw": round(yaw, 4),
                    "mean_abs_torque": round(float(np.mean(np.abs(data.ctrl))), 4),
                }
            )
    return rows


def main() -> None:
    out = artifact_dir("lab04")
    model, data = load_bare_model()
    controller = SmoothJointMotionController(model)
    controller.reset(data)

    rows = run_motion(model, data, controller)
    write_csv(out / "joint_motion.csv", rows)

    max_pitch = max(abs(csv_float(row["pitch"])) for row in rows)
    max_torque = max(csv_float(row["mean_abs_torque"]) for row in rows)
    print(f"max_abs_pitch={max_pitch:.3f}, max_mean_abs_torque={max_torque:.3f}")
    print(f"Saved: {out / 'joint_motion.csv'}")
    print("Вывод: ПД-регулятор плавно ведет суставы к изменяющимся целевым углам на ровном полу.")

    model, data = load_bare_model()
    controller = SmoothJointMotionController(model)
    controller.reset(data)
    open_lab_viewer(model, data, controller, "Лабораторная 4: плавная работа суставов", duration_s=10.0)


if __name__ == "__main__":
    main()
