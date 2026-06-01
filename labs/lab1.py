import mujoco
from lab_common import artifact_dir, load_bare_model, name_or_empty, open_lab_viewer

from controllers import StandController


def main() -> None:
    out = artifact_dir("lab01")
    model, data = load_bare_model()

    lines = [
        "Сводка MuJoCo-модели",
        "MuJoCo - физический симулятор: он хранит модель мира, состояние робота и считает контакты.",
        f"тел: {model.nbody}",
        f"суставов: {model.njnt}",
        f"степеней свободы: {model.nv}",
        f"исполнительных приводов: {model.nu}",
        f"сенсоров: {model.nsensor}",
        f"геометрических объектов: {model.ngeom}",
        "",
        "В этой первой сцене нет маршрута, стен и горки: только робот, пол и базовая физика.",
    ]

    lines.append("")
    lines.append("Исполнительные приводы:")
    for i in range(model.nu):
        actuator_name = name_or_empty(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        joint_id = int(model.actuator_trnid[i, 0])
        joint_name = name_or_empty(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        low, high = model.actuator_ctrlrange[i]
        lines.append(f"  {i:02d}: {actuator_name} -> {joint_name}, управляющий сигнал=[{low:.1f}, {high:.1f}]")

    lines.append("")
    lines.append("Сенсоры:")
    for i in range(model.nsensor):
        name = name_or_empty(model, mujoco.mjtObj.mjOBJ_SENSOR, i)
        adr = int(model.sensor_adr[i])
        dim = int(model.sensor_dim[i])
        lines.append(f"  {i:02d}: {name}, adr={adr}, dim={dim}")

    path = out / "model_summary.txt"
    path.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nSaved: {path}")
    print("Вывод: по этой таблице видно, сколько у робота суставов, приводов и сенсоров.")

    controller = StandController(model)
    controller.reset(data)
    open_lab_viewer(model, data, controller, "Лабораторная 1: модель Go2 в MuJoCo", duration_s=8.0)


if __name__ == "__main__":
    main()
