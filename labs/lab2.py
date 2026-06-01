import mujoco
from lab_common import artifact_dir, load_lab2_model, open_lab_viewer

from controllers import StandController


def geom_color(model: mujoco.MjModel, geom_id: int) -> str:
    mat_id = int(model.geom_matid[geom_id])
    rgba = model.mat_rgba[mat_id] if mat_id >= 0 else model.geom_rgba[geom_id]
    return str(rgba)


def main() -> None:
    out = artifact_dir("lab02")
    model, data = load_lab2_model()

    geom_names = ("demo_marker", "demo_route_strip", "demo_wall", "demo_ramp")
    geom_ids = {name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name) for name in geom_names}

    lines = [
        "Редактирование мира для робота",
        "Файл этой сцены: vendor/unitree_rl_mjlab/src/assets/robots/unitree_go2/xmls/lab2_editing_scene.xml",
        "",
        "Что обычно меняют в MJCF:",
        "- geom: физические или визуальные объекты мира.",
        "- material: цвет объекта.",
        "- pos и size: положение центра и половинные размеры объекта.",
        "- euler: поворот объекта; в этой сцене углы задаются в радианах.",
        "- friction и condim: параметры контакта и сцепления с поверхностью.",
        "",
        "В сцену добавлены разные типы учебных объектов:",
        "- demo_marker: плоский цилиндр-маркер, который не мешает движению робота.",
        "- demo_route_strip: визуальная полоса, показывающая, как можно размечать направление.",
        "- demo_wall: простая стена-препятствие с физическим контактом.",
        "- demo_ramp: наклонная плоскость с повышенным трением.",
        "",
        "Параметры объектов из загруженной модели:",
        *[
            (
                f"- {name}: pos={model.geom_pos[geom_id]}, "
                f"size={model.geom_size[geom_id]}, color={geom_color(model, geom_id)}"
            )
            for name, geom_id in geom_ids.items()
        ],
        "",
        "Вывод: мир редактируется декларативно в XML, а Python-код затем загружает уже измененную сцену.",
    ]

    path = out / "world_editing_notes.txt"
    path.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"Saved: {path}")

    controller = StandController(model)
    controller.reset(data)
    open_lab_viewer(model, data, controller, "Лабораторная 2: редактирование мира", duration_s=10.0)


if __name__ == "__main__":
    main()
