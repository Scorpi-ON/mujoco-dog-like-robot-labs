from lab_common import artifact_dir, load_line_route_model, open_lab_viewer, step_controller, summarize_rows, write_csv

from controllers import RouteWalkingController


def main() -> None:
    out = artifact_dir("lab05")
    model, data = load_line_route_model()
    controller = RouteWalkingController(model, data)
    controller.reset(data)

    rows = step_controller(model, data, controller, steps=7000, sample_every=100)
    write_csv(out / "single_target_walk.csv", rows)

    summary = summarize_rows(rows)
    print("Single-target walking summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    print(f"Saved: {out / 'single_target_walk.csv'}")
    print("Вывод: предобученная политика устойчиво превращает простую команду скорости в движения суставов.")

    model, data = load_line_route_model()
    controller = RouteWalkingController(model, data)
    controller.reset(data)
    open_lab_viewer(model, data, controller, "Лабораторная 5: ходьба к одной цели", duration_s=10.0)


if __name__ == "__main__":
    main()
