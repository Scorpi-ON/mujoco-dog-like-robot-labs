import mujoco
from lab_common import (
    artifact_dir,
    body_position,
    load_tilted_model,
    open_lab_viewer,
    roll_pitch_yaw,
    step_controller,
    write_csv,
)

from controllers import StandController


def place_on_tilted_platform(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    data.qpos[0] = 0.0
    data.qpos[1] = 0.0
    data.qpos[2] = 0.45
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)


def main() -> None:
    out = artifact_dir("lab03")
    model, data = load_tilted_model()
    controller = StandController(model, level_tilt=True)
    controller.reset(data)
    place_on_tilted_platform(model, data)

    rows = step_controller(model, data, controller, steps=1200, sample_every=20)
    write_csv(out / "state_log.csv", rows)

    base = body_position(model, data)
    roll, pitch, yaw = roll_pitch_yaw(data.qpos[3:7])
    print(f"qpos size: {model.nq}, qvel size: {model.nv}, ctrl size: {model.nu}")
    print(f"base position: x={base[0]:.3f}, y={base[1]:.3f}, z={base[2]:.3f}")
    print(f"base orientation: roll={roll:.3f}, pitch={pitch:.3f}, yaw={yaw:.3f}")
    print(f"simulation time: {data.time:.3f}")
    print(f"Saved: {out / 'state_log.csv'}")
    print("Вывод: CSV показывает, как стойка с компенсацией тангажа выравнивает корпус на наклонной платформе.")

    model, data = load_tilted_model()
    controller = StandController(model, level_tilt=True)
    controller.reset(data)
    place_on_tilted_platform(model, data)
    open_lab_viewer(model, data, controller, "Лабораторная 3: стойка на наклонной платформе", duration_s=8.0)


if __name__ == "__main__":
    main()
