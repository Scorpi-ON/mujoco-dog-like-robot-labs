from typing import Any, ClassVar

import numpy as np

class mjtObj:
    mjOBJ_ACTUATOR: ClassVar[mjtObj]
    mjOBJ_BODY: ClassVar[mjtObj]
    mjOBJ_CAMERA: ClassVar[mjtObj]
    mjOBJ_GEOM: ClassVar[mjtObj]
    mjOBJ_JOINT: ClassVar[mjtObj]
    mjOBJ_KEY: ClassVar[mjtObj]
    mjOBJ_SENSOR: ClassVar[mjtObj]
    mjOBJ_SITE: ClassVar[mjtObj]


class mjtCamera:
    mjCAMERA_FIXED: ClassVar[mjtCamera]
    mjCAMERA_FREE: ClassVar[mjtCamera]


class mjtFontScale:
    mjFONTSCALE_150: ClassVar[mjtFontScale]


class mjtGridPos:
    mjGRID_TOPLEFT: ClassVar[mjtGridPos]


class MjContact:
    geom1: int
    geom2: int


class MjModel:
    nq: int
    nv: int
    nu: int
    nbody: int
    njnt: int
    nsensor: int
    ngeom: int
    actuator_trnid: np.ndarray[Any, Any]
    actuator_ctrlrange: np.ndarray[Any, Any]
    jnt_qposadr: np.ndarray[Any, Any]
    jnt_dofadr: np.ndarray[Any, Any]
    key_ctrl: np.ndarray[Any, Any]
    sensor_adr: np.ndarray[Any, Any]
    sensor_dim: np.ndarray[Any, Any]
    geom_matid: np.ndarray[Any, Any]
    geom_rgba: np.ndarray[Any, Any]
    geom_pos: np.ndarray[Any, Any]
    geom_size: np.ndarray[Any, Any]
    geom_quat: np.ndarray[Any, Any]
    geom_friction: np.ndarray[Any, Any]
    mat_rgba: np.ndarray[Any, Any]

    @classmethod
    def from_xml_path(cls, path: str) -> MjModel: ...


class MjData:
    time: float
    ncon: int
    qpos: np.ndarray[Any, Any]
    qvel: np.ndarray[Any, Any]
    ctrl: np.ndarray[Any, Any]
    xpos: np.ndarray[Any, Any]
    site_xpos: np.ndarray[Any, Any]
    sensordata: np.ndarray[Any, Any]
    contact: list[MjContact]

    def __init__(self, model: MjModel) -> None: ...


def mj_forward(model: MjModel, data: MjData) -> None: ...
def mj_id2name(model: MjModel, obj_type: mjtObj, obj_id: int) -> str | None: ...
def mj_name2id(model: MjModel, obj_type: mjtObj, name: str) -> int: ...
def mj_resetDataKeyframe(model: MjModel, data: MjData, key: int) -> None: ...
def mj_step(model: MjModel, data: MjData) -> None: ...
def mj_ray(
    model: MjModel,
    data: MjData,
    pnt: np.ndarray[Any, Any],
    vec: np.ndarray[Any, Any],
    geomgroup: Any,
    flg_static: int,
    bodyexclude: int,
    geomid: np.ndarray[Any, Any],
) -> float: ...
