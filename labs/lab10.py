from typing import cast

import mujoco
import numpy as np
import onnxruntime as ort
import yaml
from lab_common import artifact_dir, load_route_model, open_lab_viewer, roll_pitch_yaw, write_csv

from controllers import ONNXGo2RoutePolicyController
from controllers.common import DEFAULT_POLICY, DEFAULT_POLICY_CONFIG

SPEED_MULTIPLIERS = (0.5, 2.0, 2.0, 1.0)


def make_speed_controller(model: mujoco.MjModel, data: mujoco.MjData) -> ONNXGo2RoutePolicyController:
    return ONNXGo2RoutePolicyController(model, data, speed_multipliers=SPEED_MULTIPLIERS)


def run_speed_schedule(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    controller: ONNXGo2RoutePolicyController,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for step in range(22000):
        status = controller.step(data)
        mujoco.mj_step(model, data)
        if step % 100 == 0 or step == 21999 or status.mode == "finale":
            roll, pitch, yaw = roll_pitch_yaw(data.qpos[3:7])
            rows.append(
                {
                    "step": step,
                    "time": round(float(data.time), 4),
                    "mode": status.mode,
                    "target": status.target_name,
                    "speed_multiplier": round(controller.current_speed_multiplier, 2),
                    "base_x": round(float(status.base_position[0]), 4),
                    "base_y": round(float(status.base_position[1]), 4),
                    "base_z": round(float(status.base_position[2]), 4),
                    "roll": round(roll, 4),
                    "pitch": round(pitch, 4),
                    "yaw": round(yaw, 4),
                }
            )
        if status.mode == "finale" and step > 100:
            break
    return rows


def main() -> None:
    out = artifact_dir("lab10")
    model, data = load_route_model()

    with DEFAULT_POLICY_CONFIG.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    session = ort.InferenceSession(str(DEFAULT_POLICY), providers=["CPUExecutionProvider"])
    input_info = session.get_inputs()[0]
    output_info = session.get_outputs()[0]

    controller = make_speed_controller(model, data)
    controller.reset(data)
    command = np.array([0.35, 0.0, 0.0], dtype=np.float32)
    obs = controller.policy_observation(data, command)
    action_batch = cast("np.ndarray", session.run([output_info.name], {input_info.name: obs})[0])
    action = action_batch[0]

    report = [
        "Отчет по управляющей ONNX-политике",
        f"вход: имя={input_info.name}, форма={input_info.shape}, тип={input_info.type}",
        f"выход: имя={output_info.name}, форма={output_info.shape}, тип={output_info.type}",
        f"период расчета политики: {cfg['step_dt']} с",
        f"форма вектора наблюдений: {obs.shape}",
        f"форма вектора действий: {action.shape}",
        f"минимум/максимум действия: {float(np.min(action)):.4f}/{float(np.max(action)):.4f}",
        f"среднее/стандартное отклонение действия: {float(np.mean(action)):.4f}/{float(np.std(action)):.4f}",
    ]
    path = out / "onnx_policy_report.txt"
    path.write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))
    print(f"Saved: {path}")
    print("Вывод: политика получает 45 чисел состояния и возвращает 12 команд для суставов робота.")

    rows = run_speed_schedule(model, data, controller)
    write_csv(out / "speed_schedule.csv", rows)
    transitions = []
    previous_target = None
    for row in rows:
        if row["target"] != previous_target:
            transitions.append((row["time"], row["target"], row["speed_multiplier"], row["mode"]))
            previous_target = row["target"]
    print("Speed schedule transitions:")
    for time_value, target, multiplier, mode in transitions:
        print(f"  t={time_value}: target={target}, speed=x{multiplier}, mode={mode}")
    print(f"Saved: {out / 'speed_schedule.csv'}")

    model, data = load_route_model()
    controller = make_speed_controller(model, data)
    controller.reset(data)
    open_lab_viewer(model, data, controller, "Лабораторная 10: ONNX-политика и разные скорости", duration_s=28.0)


if __name__ == "__main__":
    main()
