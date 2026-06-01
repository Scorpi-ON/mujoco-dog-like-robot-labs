import math
from itertools import pairwise

import mujoco
import numpy as np
from lab_common import (
    artifact_dir,
    csv_float,
    csv_int,
    load_full_route_model,
    open_lab_viewer,
    roll_pitch_yaw,
    write_csv,
)

from controllers import ONNXGo2RoutePolicyController
from controllers.common import ControllerStatus, object_id, wrap_angle, yaw_from_quat

SPEED_MULTIPLIERS = (0.5, 2.0, 2.0, 1.0)
COLOR_WALL_SIDES = {"blue_wall_bc": 1.0, "red_wall_cd": -1.0}


class FullMissionController(ONNXGo2RoutePolicyController):
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        super().__init__(model, data, speed_multipliers=SPEED_MULTIPLIERS)
        self._blue_wall_id = object_id(model, mujoco.mjtObj.mjOBJ_GEOM, "blue_wall_bc")
        self._blue_wall_y = float(model.geom_pos[self._blue_wall_id, 1])
        self._red_wall_id = object_id(model, mujoco.mjtObj.mjOBJ_GEOM, "red_wall_cd")
        self._red_wall_x = float(model.geom_pos[self._red_wall_id, 0])
        self._wall_state = "route"
        self._active_wall = ""

    def reset(self, data: mujoco.MjData) -> None:
        super().reset(data)
        self._wall_state = "route"
        self._active_wall = ""
        self._last_wall_name = ""

    def _route_command(self, data: mujoco.MjData) -> tuple[np.ndarray, ControllerStatus]:  # noqa: C901
        base_position = data.xpos[self._base_id]
        current_yaw = yaw_from_quat(data.qpos[3:7])
        scan = self._scan_obstacles(data, current_yaw)
        if (
            self._route_names[self._target_index] == "wp_c"
            and self._wall_state == "route"
            and scan.front_distance < 1.05
            and scan.front_geom_id == self._blue_wall_id
        ):
            self._wall_state = "avoid"
            self._active_wall = "blue"
            self._last_wall_name = "blue_wall_bc"
        elif (
            self._route_names[self._target_index] == "wp_d"
            and self._wall_state == "route"
            and scan.front_distance < 1.05
            and scan.front_geom_id == self._red_wall_id
        ):
            self._wall_state = "avoid"
            self._active_wall = "red"
            self._last_wall_name = "red_wall_cd"

        if self._wall_state not in {"avoid", "rejoin"}:
            return super()._route_command(data)

        target_xy = self._route_xy[self._target_index]
        error_xy = target_xy - base_position[:2]
        distance = float(np.linalg.norm(error_xy))
        desired_yaw = math.atan2(float(error_xy[1]), float(error_xy[0]))
        yaw_error = wrap_angle(desired_yaw - current_yaw)
        cos_yaw = math.cos(current_yaw)
        sin_yaw = math.sin(current_yaw)

        def local_velocity(world_x: float, world_y: float) -> tuple[float, float]:
            return (
                cos_yaw * world_x + sin_yaw * world_y,
                -sin_yaw * world_x + cos_yaw * world_y,
            )

        if self._wall_state == "avoid" and self._active_wall == "blue":
            vx, vy = local_velocity(-0.58, 0.16)
            yaw_rate = float(np.clip(0.45 * yaw_error, -0.40, 0.40))
            mode = "avoid_blue_left"
            if base_position[0] < 4.35:
                self._wall_state = "rejoin"
        elif self._wall_state == "avoid" and self._active_wall == "red":
            vx, vy = local_velocity(-0.30, 0.42)
            yaw_rate = float(np.clip(0.45 * yaw_error, -0.40, 0.40))
            mode = "avoid_red_right"
            if base_position[0] < self._red_wall_x - 0.55:
                self._wall_state = "rejoin"
        elif self._active_wall == "blue":
            rejoin_xy = np.array(
                [5.5, min(max(float(base_position[1] + 1.2), self._blue_wall_y + 1.05), float(target_xy[1]))],
                dtype=float,
            )
            rejoin_error = rejoin_xy - base_position[:2]
            rejoin_distance = float(np.linalg.norm(rejoin_error))
            rejoin_yaw = math.atan2(float(rejoin_error[1]), float(rejoin_error[0]))
            rejoin_yaw_error = wrap_angle(rejoin_yaw - current_yaw)
            speed = min(0.38, 0.12 + 0.18 * rejoin_distance)
            vx = speed * math.cos(rejoin_yaw_error)
            vy = float(np.clip(speed * math.sin(rejoin_yaw_error), -0.35, 0.35))
            yaw_rate = float(np.clip(0.9 * rejoin_yaw_error, -0.55, 0.55))
            mode = "rejoin_route"
            if abs(float(base_position[0] - 5.5)) < 0.18 and base_position[1] > self._blue_wall_y + 0.85:
                self._wall_state = "route"
                self._active_wall = ""
        else:
            rejoin_xy = np.array([max(float(base_position[0] - 1.0), float(target_xy[0])), 3.2], dtype=float)
            rejoin_error = rejoin_xy - base_position[:2]
            rejoin_distance = float(np.linalg.norm(rejoin_error))
            rejoin_yaw = math.atan2(float(rejoin_error[1]), float(rejoin_error[0]))
            rejoin_yaw_error = wrap_angle(rejoin_yaw - current_yaw)
            speed = min(0.38, 0.12 + 0.18 * rejoin_distance)
            vx = speed * math.cos(rejoin_yaw_error)
            vy = float(np.clip(speed * math.sin(rejoin_yaw_error), -0.35, 0.35))
            yaw_rate = float(np.clip(0.9 * rejoin_yaw_error, -0.55, 0.55))
            mode = "rejoin_route"
            if abs(float(base_position[1] - 3.2)) < 0.18 and base_position[0] < self._red_wall_x - 0.85:
                self._wall_state = "route"
                self._active_wall = ""

        self._command_target[:] = (float(vx), float(vy), float(yaw_rate))
        self._command[:] = 0.62 * self._command + 0.38 * self._command_target
        return self._command, ControllerStatus(
            mode=mode,
            target_name=self._route_names[self._target_index],
            distance=distance,
            yaw_error=yaw_error,
            base_position=(
                float(base_position[0]),
                float(base_position[1]),
                float(base_position[2]),
            ),
        )


def make_controller(model: mujoco.MjModel, data: mujoco.MjData) -> ONNXGo2RoutePolicyController:
    return FullMissionController(model, data)


def ramp_contact_count(model: mujoco.MjModel, data: mujoco.MjData) -> int:
    ramp_ids = {
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ramp_up_ab"),
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ramp_down_ab"),
    }
    count = 0
    for i in range(data.ncon):
        contact = data.contact[i]
        if int(contact.geom1) in ramp_ids or int(contact.geom2) in ramp_ids:
            count += 1
    return count


def run_mission(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    controller: ONNXGo2RoutePolicyController,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for step in range(26000):
        status = controller.step(data)
        mujoco.mj_step(model, data)
        if step % 100 == 0 or step == 25999 or status.mode == "finale":
            roll, pitch, yaw = roll_pitch_yaw(data.qpos[3:7])
            rows.append(
                {
                    "step": step,
                    "time": round(float(data.time), 4),
                    "mode": status.mode,
                    "target": status.target_name,
                    "speed_multiplier": round(controller.current_speed_multiplier, 2),
                    "wall": controller.last_wall_name,
                    "base_x": round(float(status.base_position[0]), 4),
                    "base_y": round(float(status.base_position[1]), 4),
                    "base_z": round(float(status.base_position[2]), 4),
                    "roll": round(roll, 4),
                    "pitch": round(pitch, 4),
                    "yaw": round(yaw, 4),
                    "ramp_contacts": ramp_contact_count(model, data),
                }
            )
        if status.mode == "finale" and step > 100:
            break
    return rows


def path_length(rows: list[dict[str, object]]) -> float:
    total = 0.0
    for previous, current in pairwise(rows):
        dx = csv_float(current["base_x"]) - csv_float(previous["base_x"])
        dy = csv_float(current["base_y"]) - csv_float(previous["base_y"])
        total += math.hypot(dx, dy)
    return total


def main() -> None:
    out = artifact_dir("lab11")
    model, data = load_full_route_model()
    controller = make_controller(model, data)
    controller.reset(data)

    rows = run_mission(model, data, controller)
    write_csv(out / "mission_metrics.csv", rows)

    avoid_samples = sum(1 for row in rows if str(row["mode"]).startswith("avoid"))
    complete = rows[-1]["target"] == "complete"
    metrics = {
        "complete": complete,
        "finish_time": rows[-1]["time"],
        "path_length": round(path_length(rows), 3),
        "avoid_samples": avoid_samples,
        "blue_wall_seen": any(row["wall"] == "blue_wall_bc" for row in rows),
        "red_wall_seen": any(row["wall"] == "red_wall_cd" for row in rows),
        "ramp_contact_samples": sum(1 for row in rows if csv_int(row["ramp_contacts"]) > 0),
        "min_base_z": round(min(csv_float(row["base_z"]) for row in rows), 3),
        "max_abs_roll": round(max(abs(csv_float(row["roll"])) for row in rows), 3),
        "max_abs_pitch": round(max(abs(csv_float(row["pitch"])) for row in rows), 3),
    }

    report_path = out / "mission_summary.txt"
    report_path.write_text("\n".join(f"{k}: {v}" for k, v in metrics.items()), encoding="utf-8")
    for key, value in metrics.items():
        print(f"{key}: {value}")
    print(f"Saved: {out / 'mission_metrics.csv'}")
    print(f"Saved: {report_path}")
    print("Вывод: полный прогон объединяет горку, изменение скорости, обход синей стены влево и красной стены вправо.")

    model, data = load_full_route_model()
    controller = make_controller(model, data)
    controller.reset(data)
    open_lab_viewer(model, data, controller, "Лабораторная 11: горка, скорости и цветные стены", duration_s=48.0)


if __name__ == "__main__":
    main()
