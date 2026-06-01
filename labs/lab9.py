import mujoco
from lab_common import artifact_dir, csv_float, load_ramp_model, open_lab_viewer, roll_pitch_yaw, write_csv

from controllers import ONNXGo2RoutePolicyController


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


def main() -> None:
    out = artifact_dir("lab09")
    model, data = load_ramp_model()

    for geom_name in ("ramp_up_ab", "ramp_down_ab"):
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        friction = model.geom_friction[geom_id]
        print(
            f"{geom_name}: friction={friction}, "
            f"size={model.geom_size[geom_id]}, euler-derived quat={model.geom_quat[geom_id]}"
        )

    controller = ONNXGo2RoutePolicyController(model, data)
    controller.reset(data)

    rows: list[dict[str, object]] = []
    for step in range(5500):
        status = controller.step(data)
        mujoco.mj_step(model, data)
        if step % 50 == 0 or step == 5499:
            roll, pitch, yaw = roll_pitch_yaw(data.qpos[3:7])
            rows.append(
                {
                    "step": step,
                    "time": round(float(data.time), 4),
                    "mode": status.mode,
                    "target": status.target_name,
                    "base_x": round(float(status.base_position[0]), 4),
                    "base_y": round(float(status.base_position[1]), 4),
                    "base_z": round(float(status.base_position[2]), 4),
                    "roll": round(roll, 4),
                    "pitch": round(pitch, 4),
                    "yaw": round(yaw, 4),
                    "ramp_contacts": ramp_contact_count(model, data),
                }
            )

    write_csv(out / "ramp_run.csv", rows)
    max_z = max(csv_float(row["base_z"]) for row in rows)
    max_pitch = max(abs(csv_float(row["pitch"])) for row in rows)
    print(f"max_base_z={max_z:.3f}, max_abs_pitch={max_pitch:.3f}")
    print(f"Saved: {out / 'ramp_run.csv'}")
    print("Вывод: рост base_z показывает подъем на горку, а pitch показывает наклон корпуса на подъеме и спуске.")

    model, data = load_ramp_model()
    controller = ONNXGo2RoutePolicyController(model, data)
    controller.reset(data)
    open_lab_viewer(model, data, controller, "Лабораторная 9: горка и контакт с поверхностью", duration_s=12.0)


if __name__ == "__main__":
    main()
