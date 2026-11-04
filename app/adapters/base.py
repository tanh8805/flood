"""Interface duy nhất giữa app và model: predict(frame_bgr) -> {"p_ngap": float, "mask": bool ndarray HxW | None}."""
from typing import Protocol

import numpy as np


class ModelError(Exception):
    """Lỗi nạp/dùng model, thông báo hiển thị thẳng cho người dùng."""


class FloodModel(Protocol):
    kind: str
    description: str

    def predict(self, frame_bgr: np.ndarray) -> dict: ...


def result(p: float, mask=None) -> dict:
    return {"p_ngap": float(min(1.0, max(0.0, p))), "mask": mask}
