import math
from typing import TYPE_CHECKING

import mujoco
from lab_common import artifact_dir, load_wall_model, object_id, open_lab_viewer, ray_distance, write_csv

from controllers import RouteWalkingController
from controllers.common import ControllerStatus, yaw_from_quat

if TYPE_CHECKING:
    import numpy as np

SAMPLE_EVERY = 50


class RayApproachController(RouteWalkingController):
    """Walks toward a wall and stops when the forward ray becomes too short."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, *, stop_distance: float = 0.42) -> None:
        super().__init__(model, data)
        self._stop_distance = stop_distance
        self.last_ray_distance = math.inf
        self.last_ray_geom = ""

    def _walk_command(self, data: mujoco.MjData) -> tuple[np.ndarray, ControllerStatus]:
        command, status = super()._walk_command(data)
        self.measure_front_ray(data)
        if self.last_ray_distance <= self._stop_distance:
            self._command_target[:] = 0.0
            self._command[:] = 0.55 * self._command
            command = self._command
            status = ControllerStatus(
                mode="stop_before_wall",
                target_name=status.target_name,
                distance=status.distance,
                yaw_error=status.yaw_error,
                base_position=status.base_position,
            )
        return command, status

    def measure_front_ray(self, data: mujoco.MjData) -> tuple[float, str]:
        yaw = yaw_from_quat(data.qpos[3:7])
        origin = (
            float(data.xpos[self._base_id, 0] + 0.35 * math.cos(yaw)),
            float(data.xpos[self._base_id, 1] + 0.35 * math.sin(yaw)),
            0.32,
        )
        distance, geom = ray_distance(
            self._model,
            data,
            origin=origin,
            yaw=yaw,
            local_angle=0.0,
            exclude_body=self._base_id,
        )
        self.last_ray_distance = distance
        self.last_ray_geom = geom
        return distance, geom


def run_approach(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    controller: RayApproachController,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    stop_samples = 0
    wrote_stop_sample = False
    for step in range(5000):
        status = controller.step(data)
        mujoco.mj_step(model, data)
        distance, geom = controller.measure_front_ray(data)
        should_sample = step % SAMPLE_EVERY == 0 or (status.mode == "stop_before_wall" and not wrote_stop_sample)
        if should_sample:
            rows.append(
                {
                    "step": step,
                    "time": round(float(data.time), 4),
                    "mode": status.mode,
                    "base_x": round(float(status.base_position[0]), 4),
                    "base_y": round(float(status.base_position[1]), 4),
                    "base_z": round(float(status.base_position[2]), 4),
                    "target_distance": round(float(status.distance), 4),
                    "front_ray_distance": "inf" if math.isinf(distance) else round(distance, 4),
                    "ray_geom": geom,
                }
            )
            wrote_stop_sample = wrote_stop_sample or status.mode == "stop_before_wall"
        stop_samples = stop_samples + 1 if status.mode == "stop_before_wall" else 0
        if stop_samples > 200:
            break
    if rows and rows[-1]["step"] != step:
        rows.append(
            {
                "step": step,
                "time": round(float(data.time), 4),
                "mode": status.mode,
                "base_x": round(float(status.base_position[0]), 4),
                "base_y": round(float(status.base_position[1]), 4),
                "base_z": round(float(status.base_position[2]), 4),
                "target_distance": round(float(status.distance), 4),
                "front_ray_distance": "inf" if math.isinf(distance) else round(distance, 4),
                "ray_geom": geom,
            }
        )
    return rows


def main() -> None:
    out = artifact_dir("lab07")
    model, data = load_wall_model()
    object_id(model, mujoco.mjtObj.mjOBJ_GEOM, "blocking_wall")
    controller = RayApproachController(model, data)
    controller.reset(data)

    rows = run_approach(model, data, controller)
    write_csv(out / "ray_walk.csv", rows)

    first_hit = next((row for row in rows if row["ray_geom"] == "blocking_wall"), None)
    final = rows[-1]
    if first_hit:
        print(
            "First wall ray hit: "
            f"t={first_hit['time']}, x={first_hit['base_x']}, distance={first_hit['front_ray_distance']}"
        )
    print(
        "Final ray sample: "
        f"mode={final['mode']}, x={final['base_x']}, distance={final['front_ray_distance']}, geom={final['ray_geom']}"
    )
    print(f"Saved: {out / 'ray_walk.csv'}")
    print("Вывод: последовательные измерения луча показывают, как расстояние до стены уменьшается при движении вперед.")

    model, data = load_wall_model()
    controller = RayApproachController(model, data)
    controller.reset(data)
    open_lab_viewer(model, data, controller, "Лабораторная 7: движение к стене и лучевое расстояние", duration_s=10.0)


if __name__ == "__main__":
    main()
