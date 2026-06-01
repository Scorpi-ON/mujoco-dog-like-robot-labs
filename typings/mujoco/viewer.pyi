from contextlib import AbstractContextManager
from typing import Any

from . import MjData, MjModel

class _Camera:
    type: Any
    fixedcamid: int
    lookat: Any
    distance: float
    azimuth: float
    elevation: float


class Handle:
    cam: _Camera

    def is_running(self) -> bool: ...
    def set_texts(self, texts: tuple[Any, ...]) -> None: ...
    def sync(self) -> None: ...


def launch_passive(model: MjModel, data: MjData) -> AbstractContextManager[Handle]: ...
