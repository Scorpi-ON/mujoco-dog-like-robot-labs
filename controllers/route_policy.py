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
    ObstacleScan,
    joint_addresses_for_actuators,
    keyframe_id,
    object_id,
    quat_conjugate_rotate,
    wrap_angle,
    yaw_from_quat,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path


class ONNXGo2RoutePolicyController:
    """Unitree Go2 velocity policy wrapped as a waypoint follower."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        *,
        policy_path: Path = DEFAULT_POLICY,
        config_path: Path = DEFAULT_POLICY_CONFIG,
        waypoint_radius: float = 0.35,
        route_site_names: Sequence[str] = ROUTE_SITE_NAMES,
        finish_at_last: bool = False,
        speed_multipliers: Sequence[float] | None = None,
        color_wall_sides: Mapping[str, float] | None = None,
    ) -> None:
        self._model = model
        self._home_key_id = keyframe_id(model, HOME_KEY)
        self._base_id = object_id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
        self._waypoint_radius = waypoint_radius
        self._route_names = tuple(route_site_names)
        if len(self._route_names) < 2:
            raise ValueError("Route must contain at least two sites.")
        self._target_index = 1
        self._finish_at_last = finish_at_last
        self._route_xy = self._load_route_sites(model, data, self._route_names)
        if speed_multipliers is None:
            self._speed_multipliers = np.ones(len(self._route_names), dtype=np.float32)
        else:
            if len(speed_multipliers) != len(self._route_names):
                raise ValueError("speed_multipliers length must match route_site_names length.")
            self._speed_multipliers = np.array(speed_multipliers, dtype=np.float32)

        self._stage = "route"
        self._finale_start_time = 0.0
        self._finale_settle_s = 1.5

        self._ray_geomid = np.array([-1], dtype=np.int32)
        self._ray_origin = np.zeros(3, dtype=np.float64)
        self._ray_direction = np.zeros(3, dtype=np.float64)
        self._front_ray_angles = tuple(math.radians(angle) for angle in (-25.0, -12.0, 0.0, 12.0, 25.0))
        self._obstacle_distance = 0.85
        self._avoid_until = 0.0
        self._avoid_side = 1.0
        self._handled_walls: set[int] = set()
        self._last_wall_name = ""
        self._color_wall_sides: dict[int, float] = {}
        self._color_wall_names: dict[int, str] = {}
        if color_wall_sides:
            for geom_name, side in color_wall_sides.items():
                geom_id = object_id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
                self._color_wall_sides[geom_id] = float(side)
                self._color_wall_names[geom_id] = geom_name

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

        self._finale_start_joint_pos = self._default_joint_pos.copy()
        self._finale_target_joint_pos = self._default_joint_pos.copy()

        self._qpos_adr, self._qvel_adr = joint_addresses_for_actuators(model, self._joint_map)
        self._ctrl_low = model.actuator_ctrlrange[:, 0]
        self._ctrl_high = model.actuator_ctrlrange[:, 1]

        self._gyro_sensor_adr = self._sensor_adr(model, ("imu_gyro", "imu_ang_vel"))
        self._session = ort.InferenceSession(str(policy_path), providers=["CPUExecutionProvider"])
        self._input_name = self._session.get_inputs()[0].name
        self._output_name = self._session.get_outputs()[0].name

        self._command = np.zeros(3, dtype=np.float32)
        self._command_target = np.zeros(3, dtype=np.float32)
        self._obs = np.zeros((1, 45), dtype=np.float32)
        self._gravity_world = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        self._last_action = np.zeros(12, dtype=np.float32)
        self._target_joint_pos = self._default_joint_pos.copy()
        self._next_policy_time = 0.0
        self._last_speed_multiplier = float(self._speed_multipliers[self._target_index])
        self._last_status = ControllerStatus(
            mode="rl",
            target_name=self._route_names[self._target_index],
            distance=0.0,
            yaw_error=0.0,
            base_position=(0.0, 0.0, 0.0),
        )

    def reset(self, data: mujoco.MjData) -> None:
        self._target_index = 1
        self._stage = "route"
        self._last_action[:] = 0.0
        self._target_joint_pos[:] = self._default_joint_pos
        self._next_policy_time = 0.0
        self._last_speed_multiplier = float(self._speed_multipliers[self._target_index])
        self._avoid_until = 0.0
        self._avoid_side = 1.0
        self._handled_walls.clear()
        self._last_wall_name = ""

        mujoco.mj_resetDataKeyframe(self._model, data, self._home_key_id)
        for policy_i, actuator_i in enumerate(self._joint_map):
            joint_id = self._model.actuator_trnid[actuator_i, 0]
            data.qpos[self._model.jnt_qposadr[joint_id]] = self._default_joint_pos[policy_i]
        mujoco.mj_forward(self._model, data)

    def step(self, data: mujoco.MjData) -> ControllerStatus:
        if self._stage == "finale":
            status = self._step_finale(data)
            self._last_status = status
            return status

        command, status = self._route_command(data)
        if self._stage == "finale":
            status = self._step_finale(data)
            self._last_status = status
            return status

        if data.time >= self._next_policy_time - 1e-9:
            obs = self._observation(data, command)
            action_batch = cast("np.ndarray", self._session.run([self._output_name], {self._input_name: obs})[0])
            action = action_batch[0]
            self._last_action[:] = action.astype(np.float32)
            self._target_joint_pos[:] = self._last_action * self._action_scale + self._action_offset
            self._next_policy_time += self._policy_dt

        self._apply_pd_torques(data)
        self._last_status = status
        return status

    def _route_command(self, data: mujoco.MjData) -> tuple[np.ndarray, ControllerStatus]:
        base_position = data.xpos[self._base_id]
        base_xy = base_position[:2].copy()
        target_xy = self._route_xy[self._target_index]
        error_xy = target_xy - base_xy
        distance = float(np.linalg.norm(error_xy))

        if distance < self._waypoint_radius:
            reached_index = self._target_index
            if self._finish_at_last and reached_index == len(self._route_names) - 1:
                self._start_finale(data)
                return self._command, self._finale_status(data)

            self._target_index = (self._target_index + 1) % len(self._route_xy)
            if not self._finish_at_last and reached_index == 0:
                self._start_finale(data)
                return self._command, self._finale_status(data)

            target_xy = self._route_xy[self._target_index]
            error_xy = target_xy - base_xy
            distance = float(np.linalg.norm(error_xy))

        desired_yaw = math.atan2(float(error_xy[1]), float(error_xy[0]))
        current_yaw = yaw_from_quat(data.qpos[3:7])
        yaw_error = wrap_angle(desired_yaw - current_yaw)

        speed_multiplier = float(self._speed_multipliers[self._target_index])
        self._last_speed_multiplier = speed_multiplier
        speed = speed_multiplier * min(0.58, 0.22 + 0.28 * distance)
        vx = speed * math.cos(yaw_error)
        vy = float(np.clip(speed * math.sin(yaw_error), -0.45, 0.45))
        yaw_rate = float(np.clip(1.2 * yaw_error, -0.7, 0.7))
        mode = "rl" if np.allclose(self._speed_multipliers, 1.0) else f"speed_x{speed_multiplier:g}"

        if abs(yaw_error) > 1.25:
            vx = 0.0
            vy = 0.0

        scan = self._scan_obstacles(data, current_yaw)
        front_blocked = scan.front_distance < self._obstacle_distance
        if front_blocked:
            if scan.front_geom_id in self._color_wall_sides and scan.front_geom_id not in self._handled_walls:
                self._avoid_side = self._color_wall_sides[scan.front_geom_id]
                self._last_wall_name = self._color_wall_names[scan.front_geom_id]
                self._handled_walls.add(scan.front_geom_id)
            elif not self._color_wall_sides:
                self._avoid_side = 1.0 if scan.left_clearance >= scan.right_clearance else -1.0
            self._avoid_until = max(self._avoid_until, data.time + 0.9)

        if front_blocked or data.time < self._avoid_until:
            near_scale = float(np.clip(scan.front_distance / self._obstacle_distance, 0.0, 1.0))
            vx = 0.10 + 0.18 * near_scale
            vy = self._avoid_side * 0.55
            yaw_rate = float(np.clip(0.55 * yaw_error, -0.45, 0.45))
            mode = self._avoid_mode()

        self._command_target[0] = float(vx)
        self._command_target[1] = float(vy)
        self._command_target[2] = float(yaw_rate)
        self._command[:] = 0.65 * self._command + 0.35 * self._command_target
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

    def _start_finale(self, data: mujoco.MjData) -> None:
        self._stage = "finale"
        self._finale_start_time = float(data.time)
        data.qvel[:] = 0.0
        self._finale_start_joint_pos[:] = data.qpos[self._qpos_adr]
        self._finale_target_joint_pos[:] = self._finale_start_joint_pos
        self._command[:] = 0.0

    def _step_finale(self, data: mujoco.MjData) -> ControllerStatus:
        elapsed = max(0.0, float(data.time - self._finale_start_time))
        if elapsed < self._finale_settle_s:
            blend = self._smoothstep01(np.clip(elapsed / self._finale_settle_s, 0.0, 1.0))
            self._finale_target_joint_pos[:] = (
                1.0 - blend
            ) * self._finale_start_joint_pos + blend * self._default_joint_pos
        else:
            self._finale_target_joint_pos[:] = self._default_joint_pos

        self._target_joint_pos[:] = self._finale_target_joint_pos
        self._apply_pd_torques(data)
        return self._finale_status(data)

    def _finale_status(self, data: mujoco.MjData) -> ControllerStatus:
        base_position = data.xpos[self._base_id]
        return ControllerStatus(
            mode="finale",
            target_name="complete",
            distance=0.0,
            yaw_error=0.0,
            base_position=(
                float(base_position[0]),
                float(base_position[1]),
                float(base_position[2]),
            ),
        )

    @property
    def last_wall_name(self) -> str:
        return self._last_wall_name

    @property
    def current_speed_multiplier(self) -> float:
        return self._last_speed_multiplier

    def policy_observation(self, data: mujoco.MjData, command: np.ndarray) -> np.ndarray:
        return self._observation(data, command)

    def _avoid_mode(self) -> str:
        if self._color_wall_sides:
            if self._avoid_side < 0.0:
                return "avoid_red_right"
            return "avoid_blue_left"
        return "avoid"

    def _scan_obstacles(self, data: mujoco.MjData, current_yaw: float) -> ObstacleScan:
        front_distance = math.inf
        front_geom_id = -1
        for angle in self._front_ray_angles:
            distance, geom_id = self._ray_distance_with_geom(data, current_yaw, angle)
            if distance < front_distance:
                front_distance = distance
                front_geom_id = geom_id
        left_clearance = self._ray_distance(data, current_yaw, math.radians(90.0))
        right_clearance = self._ray_distance(data, current_yaw, math.radians(-90.0))
        return ObstacleScan(
            front_distance=front_distance,
            left_clearance=left_clearance,
            right_clearance=right_clearance,
            front_geom_id=front_geom_id,
        )

    def _ray_distance(self, data: mujoco.MjData, current_yaw: float, local_angle: float) -> float:
        distance, _ = self._ray_distance_with_geom(data, current_yaw, local_angle)
        return distance

    def _ray_distance_with_geom(self, data: mujoco.MjData, current_yaw: float, local_angle: float) -> tuple[float, int]:
        self._ray_origin[:] = data.xpos[self._base_id]
        self._ray_origin[0] += 0.35 * math.cos(current_yaw)
        self._ray_origin[1] += 0.35 * math.sin(current_yaw)
        self._ray_origin[2] = 0.32

        world_angle = current_yaw + local_angle
        self._ray_direction[0] = math.cos(world_angle)
        self._ray_direction[1] = math.sin(world_angle)
        self._ray_direction[2] = 0.0
        self._ray_geomid[0] = -1
        distance = mujoco.mj_ray(
            self._model,
            data,
            self._ray_origin,
            self._ray_direction,
            None,
            1,
            self._base_id,
            self._ray_geomid,
        )
        if distance < 0.0:
            return math.inf, -1
        return float(distance), int(self._ray_geomid[0])

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

    @staticmethod
    def _load_route_sites(model: mujoco.MjModel, data: mujoco.MjData, route_names: tuple[str, ...]) -> np.ndarray:
        route = []
        for site_name in route_names:
            site_id = object_id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
            route.append(data.site_xpos[site_id, :2].copy())
        return np.array(route, dtype=float)

    @staticmethod
    def _smoothstep01(value: float) -> float:
        return value * value * (3.0 - 2.0 * value)
