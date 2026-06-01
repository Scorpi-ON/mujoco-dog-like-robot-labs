from lab_common import artifact_dir, load_route_model, open_lab_viewer, route_sites, step_controller, write_csv

from controllers import ONNXGo2RoutePolicyController


def main() -> None:
    out = artifact_dir("lab06")
    model, data = load_route_model()
    controller = ONNXGo2RoutePolicyController(model, data)
    controller.reset(data)

    rows = step_controller(model, data, controller, steps=18000, sample_every=100)
    write_csv(out / "waypoint_fsm.csv", rows)

    transitions = []
    previous = None
    for row in rows:
        current = row["target"]
        if current != previous:
            transitions.append((row["time"], current, row["mode"]))
            previous = current

    print("Route sites:")
    for name, x, y in route_sites(model, data):
        print(f"  {name}: ({x:.2f}, {y:.2f})")
    print("Target transitions:")
    for time_value, target, mode in transitions:
        print(f"  t={time_value}: target={target}, mode={mode}")
    print(f"Saved: {out / 'waypoint_fsm.csv'}")
    print("Вывод: переходы target показывают, когда конечный автомат считает очередную точку достигнутой.")

    model, data = load_route_model()
    controller = ONNXGo2RoutePolicyController(model, data)
    controller.reset(data)
    open_lab_viewer(model, data, controller, "Лабораторная 6: маршрут по точкам", duration_s=24.0)


if __name__ == "__main__":
    main()
