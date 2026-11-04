"""Model/embedder giả cho test."""
import numpy as np


class T:
    """Giả tensor torch: .cpu().numpy() và .data."""
    def __init__(self, a):
        self.a = np.asarray(a)
        self.data = self

    def cpu(self):
        return self

    def numpy(self):
        return self.a

    def __array__(self, dtype=None, copy=None):
        return self.a if dtype is None else self.a.astype(dtype)

    def __len__(self):
        return len(self.a)


class FakeResult:
    """masks: ndarray (N, mh, mw) giống masks.data của Ultralytics (độ phân giải suy luận, có thể có viền letterbox)."""
    def __init__(self, probs=None, masks=None, cls=None):
        self.probs = T(probs) if probs is not None else None
        self.masks = type("M", (), {"data": T(masks)})() if masks is not None else None
        n = len(masks) if masks is not None else 0
        self.boxes = type("B", (), {"cls": T(cls), "__len__": lambda s: n})() if masks is not None else None


class FakeYolo:
    def __init__(self, task, names, result):
        self.task, self.names, self._r = task, names, result

    def __call__(self, frame, verbose=False, **kw):
        self.calls = getattr(self, "calls", []) + [(frame.shape[:2], kw)]
        return [self._r(frame) if callable(self._r) else self._r]
