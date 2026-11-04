"""Model VẼ vùng nước: models/seg.pt (Ultralytics segment). Chỉ tô lên video và ảnh minh chứng, không ảnh hưởng %."""
import re
from pathlib import Path

import cv2
import numpy as np

from .base import ModelError, result

MIN_BLOB_FRAC = 0.01        # bỏ mảng nước nhỏ hơn 1% diện tích khung


def _match(names: dict, pattern: str) -> list[int]:
    rx = re.compile(pattern, re.I)
    return [i for i, n in names.items() if rx.search(str(n))]


class SegAdapter:
    """Model segment (models/seg.pt) CHỈ để tô vùng nước, không quyết định ngập: predict trả {"mask", "p_ngap": 0.0}."""
    kind = "segment"

    def __init__(self, cfg, model=None, path=None):
        """`model` cho phép tiêm model giả (test); mặc định nạp file .pt từ `path`."""
        if model is None:
            try:
                from ultralytics import YOLO
                model = YOLO(str(path))
            except Exception as e:
                raise ModelError(f"Không nạp được {Path(path).name}: {e}") from e
        self.model = model
        self.cfg = cfg
        self.task = getattr(model, "task", None)
        names = getattr(model, "names", None) or {}
        self.names = dict(names) if isinstance(names, dict) else dict(enumerate(names))
        if self.task != "segment":
            raise ModelError(f"seg.pt phải là model segment (vùng nước), file này là '{self.task}'")
        self.ids = _match(self.names, cfg.water_classes)
        if not self.ids:
            raise ModelError(f"Không có class nào khớp '{cfg.water_classes}'. model.names = {self.names}")
        self.name = f"{Path(path).name if path else 'model'} (segment)"
        self.classes = [str(self.names[i]) for i in self.ids]
        self.description = f"Ultralytics segment, class nước: {self.classes}"

    def predict(self, frame_bgr):
        # Tô vùng nước: thu khung về rộng draw_width, conf = water_conf, retina_masks (mask cùng cỡ khung đã thu)
        h, w = frame_bgr.shape[:2]
        dw = int(self.cfg.draw_width)
        small = frame_bgr
        if dw and w > dw:
            small = cv2.resize(frame_bgr, (dw, max(2, round(h * dw / w))), interpolation=cv2.INTER_AREA)
        r = self.model(small, conf=self.cfg.water_conf, retina_masks=True, verbose=False)[0]
        return self._segment(r, small.shape[:2], (h, w))

    def _segment(self, r, shape, out_shape=None):
        """shape = cỡ khung đưa vào model; out_shape = cỡ khung gốc (mask phóng lên cỡ này)."""
        h, w = shape
        oh, ow = out_shape or shape
        if r.masks is None or r.boxes is None or len(r.boxes) == 0:
            return result(0.0, None)
        cls = _np(r.boxes.cls).astype(int)
        keep = [i for i, c in enumerate(cls) if c in self.ids]
        if not keep:
            return result(0.0, None)
        # Dựng từ bitmap masks.data, KHÔNG dùng masks.xy: xy ghép mọi mảng rời của một mask thành một polyline nên có đường nối thẳng giữa các mảng.
        union = _np(r.masks.data)[keep].max(axis=0)
        mask = upscale_mask(clean_mask(letterbox_crop(union, (h, w)) > 0.5), (oh, ow))
        return result(0.0, mask if mask.any() else None)


def _np(x):
    return np.asarray(x.cpu().numpy() if hasattr(x, "cpu") else x)


def letterbox_crop(m, shape):
    """Mask ở độ phân giải suy luận (có viền letterbox) -> phần tương ứng với khung gốc `shape` (h, w), chưa đổi kích thước."""
    mh, mw = m.shape
    h, w = shape
    if (mh, mw) == (h, w):                      # retina_masks hoặc cùng kích thước: không có letterbox
        return m
    gain = min(mh / h, mw / w)
    pw, ph = (mw - w * gain) / 2, (mh - h * gain) / 2
    top, left = int(round(ph - 0.1)), int(round(pw - 0.1))
    bottom, right = mh - int(round(ph + 0.1)), mw - int(round(pw + 0.1))
    return m[top:bottom, left:right]


def clean_mask(m, min_frac=MIN_BLOB_FRAC):
    """Làm sạch nhẹ: open (bỏ nhiễu/cầu mảnh) rồi close (vá lỗ nhỏ), sau đó bỏ mảng nhỏ hơn min_frac diện tích khung."""
    h, w = m.shape
    k = max(3, int(round(min(h, w) * 0.008)) | 1)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    u8 = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, ker)
    u8 = cv2.morphologyEx(u8, cv2.MORPH_CLOSE, ker)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(u8, connectivity=8)
    big = np.zeros(n, bool)
    big[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_frac * h * w
    return big[lab]


def upscale_mask(m, shape):
    """Phóng mask bool lên (h, w) bằng nội suy tuyến tính rồi cắt ngưỡng: viền mượt, không răng cưa."""
    h, w = shape
    if m.shape == (h, w):
        return m
    return cv2.resize(m.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR) > 0.5
