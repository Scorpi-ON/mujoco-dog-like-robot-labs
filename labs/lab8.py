import math
from collections import Counter

import mujoco
import numpy as np
from lab_common import COLOR_WALLS_SCENE, artifact_dir, load_model_from_path, open_lab_viewer, roll_pitch_yaw, write_csv

from controllers import ONNXGo2RoutePolicyController
from controllers.common import LINE_ROUTE_SITE_NAMES, ControllerStatus, wrap_angle, yaw_from_quat

WALL_SEQUENCE = (
    ("red_wall_1", -1.0),
    ("red_wall_2", -1.0),
    ("blue_wall_3", 1.0),
    ("red_wall_4", -1.0),
    ("blue_wall_5", 1.0),
    ("blue_wall_6", 1.0),
)

COLOR_WALL_LABELS = {
    "red_wall_1": "красная стена: обход вправо",
    "red_wall_2": "красная стена: обход вправо",
    "blue_wall_3": "синяя стена: обход влево",
    "red_wall_4": "красная стена: обход вправо",
    "blue_wall_5": "синяя стена: обход влево",
    "blue_wall_6": "синяя стена: обход влево",
}


class ColorWallLineController(ONNXGo2RoutePolicyController):
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        super().__init__(
            model,
            data,
            route_site_names=LINE_ROUTE_SITE_NAMES,
            finish_at_last=True,
            color_wall_sides=dict(WALL_SEQUENCE),
        )
        self._avoid_wall_id = -1
        self._avoid_wall_x = 0.0
        self._line_state = "route"

    def reset(self, data: mujoco.MjData) -> None:
        super().reset(data)
        self._avoid_wall_id = -1
        self._avoid_wall_x = 0.0
        self._line_state = "route"

    def _route_command(self, data: mujoco.MjData) -> tuple[np.ndarray, ControllerStatus]:
        base_position = data.xpos[self._base_id]
        base_xy = base_position[:2].copy()
        target_xy = self._route_xy[self._target_index]
        error_xy = target_xy - base_xy
        distance = float(np.linalg.norm(error_xy))

        if distance < self._waypoint_radius:
            self._start_finale(data)
            return self._command, self._finale_status(data)

        current_yaw = yaw_from_quat(data.qpos[3:7])
        scan = self._scan_obstacles(data, current_yaw)
        if (
            scan.front_distance < 0.95
            and scan.front_geom_id in self._color_wall_sides
            and scan.front_geom_id not in self._handled_walls
        ):
            self._avoid_wall_id = scan.front_geom_id
            self._avoid_wall_x = float(self._model.geom_pos[scan.front_geom_id, 0])
            self._avoid_side = self._color_wall_sides[scan.front_geom_id]
            self._last_wall_name = self._color_wall_names[scan.front_geom_id]
            self._handled_walls.add(scan.front_geom_id)
            self._line_state = "avoid"

        desired_yaw = math.atan2(float(error_xy[1]), float(error_xy[0]))
        yaw_error = wrap_angle(desired_yaw - current_yaw)
        cos_yaw = math.cos(current_yaw)
        sin_yaw = math.sin(current_yaw)

        def local_velocity(world_x: float, world_y: float) -> tuple[float, float]:
            return (
                cos_yaw * world_x + sin_yaw * world_y,
                -sin_yaw * world_x + cos_yaw * world_y,
            )

        if self._line_state == "avoid":
            vx, vy = local_velocity(0.30, self._avoid_side * 0.42)
            yaw_rate = float(np.clip(0.35 * yaw_error, -0.35, 0.35))
            mode = self._avoid_mode()
            if base_position[0] > self._avoid_wall_x + 0.45:
                self._line_state = "rejoin"
        elif self._line_state == "rejoin":
            rejoin_xy = np.array(
                [min(float(self._route_xy[-1, 0]), max(float(base_position[0] + 1.2), self._avoid_wall_x + 1.4)), 0.0],
                dtype=float,
            )
            rejoin_error = rejoin_xy - base_xy
            rejoin_distance = float(np.linalg.norm(rejoin_error))
            rejoin_yaw = math.atan2(float(rejoin_error[1]), float(rejoin_error[0]))
            rejoin_yaw_error = wrap_angle(rejoin_yaw - current_yaw)
            speed = min(0.34, 0.12 + 0.18 * rejoin_distance)
            vx = speed * math.cos(rejoin_yaw_error)
            vy = float(np.clip(speed * math.sin(rejoin_yaw_error), -0.35, 0.35))
            yaw_rate = float(np.clip(0.9 * rejoin_yaw_error, -0.55, 0.55))
            mode = "rejoin_line"
            if abs(float(base_position[1])) < 0.14 and base_position[0] > self._avoid_wall_x + 0.90:
                self._line_state = "route"
        else:
            speed = min(0.50, 0.18 + 0.20 * distance)
            vx = speed * math.cos(yaw_error)
            vy = float(np.clip(speed * math.sin(yaw_error), -0.40, 0.40))
            yaw_rate = float(np.clip(1.0 * yaw_error, -0.60, 0.60))
            mode = "route"

        self._command_target[:] = (float(vx), float(vy), float(yaw_rate))
        self._command[:] = 0.60 * self._command + 0.40 * self._command_target
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


def load_lab8_model() -> tuple[mujoco.MjModel, mujoco.MjData]:
    return load_model_from_path(COLOR_WALLS_SCENE)


def make_controller(model: mujoco.MjModel, data: mujoco.MjData) -> ColorWallLineController:
    return ColorWallLineController(model, data)


def sample_row(
    step: int,
    data: mujoco.MjData,
    controller: ONNXGo2RoutePolicyController,
    status: ControllerStatus,
) -> dict[str, object]:
    roll, pitch, yaw = roll_pitch_yaw(data.qpos[3:7])
    return {
        "step": step,
        "time": round(float(data.time), 4),
        "mode": status.mode,
        "target": status.target_name,
        "wall": controller.last_wall_name,
        "rule": COLOR_WALL_LABELS.get(controller.last_wall_name, ""),
        "base_x": round(float(status.base_position[0]), 4),
        "base_y": round(float(status.base_position[1]), 4),
        "base_z": round(float(status.base_position[2]), 4),
        "route_line_error": round(float(status.base_position[1]), 4),
        "roll": round(roll, 4),
        "pitch": round(pitch, 4),
        "yaw": round(yaw, 4),
    }


def run_trial(steps: int = 60000) -> list[dict[str, object]]:
    model, data = load_lab8_model()
    controller = make_controller(model, data)
    controller.reset(data)

    rows: list[dict[str, object]] = []
    for step in range(steps):
        status = controller.step(data)
        mujoco.mj_step(model, data)
        if step % 100 == 0 or step == steps - 1 or status.mode == "finale":
            rows.append(sample_row(step, data, controller, status))
        if status.mode == "finale" and step > 100:
            break
    return rows


def main() -> None:
    out = artifact_dir("lab08")
    rows = run_trial()
    write_csv(out / "color_wall_avoidance.csv", rows)

    mode_counts = Counter(row["mode"] for row in rows)
    wall_order = []
    for row in rows:
        wall = row["wall"]
        if wall and (not wall_order or wall_order[-1] != wall):
            wall_order.append(wall)

    print(f"Modes: {dict(mode_counts)}")
    print(f"Wall sequence seen: {wall_order}")
    print(
        f"Final: target={rows[-1]['target']}, mode={rows[-1]['mode']}, x={rows[-1]['base_x']}, y={rows[-1]['base_y']}"
    )
    print(f"Saved: {out / 'color_wall_avoidance.csv'}")
    print("Вывод: красные стены вызывают обход вправо, синие - влево, после чего робот возвращается к прямой линии.")

    model, data = load_lab8_model()
    controller = make_controller(model, data)
    controller.reset(data)
    open_lab_viewer(model, data, controller, "Лабораторная 8: шесть цветных стен на прямой линии", duration_s=60.0)


if __name__ == "__main__":
    main()
