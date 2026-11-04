"""Nạp model theo models/ + models/model.yaml + biến môi trường. Phần còn lại của app chỉ dùng ModelHolder.

Hai vai trò tách bạch:
  QUYẾT ĐỊNH (tính %): models/best.pt, TorchScript, p = m(khung BGR).
  TÔ vùng nước: models/seg.pt (segment), chỉ để tô lên video và ảnh minh chứng khi đang NGẬP, không ảnh hưởng %.
"""
import threading
from dataclasses import asdict
from pathlib import Path

from .. import config
from .base import FloodModel, ModelError  # noqa: F401
from .best import BestAdapter
from .seg import SegAdapter


def build_decider(cfg, models_dir: Path):
    """Model quyết định: models/best.pt (TorchScript)."""
    path = Path(models_dir) / cfg.path
    if not path.exists():
        raise FileNotFoundError(config.NO_MODEL_MSG)
    return BestAdapter(path, device=cfg.device)


def build_drawer(cfg, models_dir: Path):
    """Model tô vùng nước: models/seg.pt (segment). Không có file thì None (video giữ khung gốc)."""
    path = Path(models_dir) / cfg.seg_path
    if not path.exists():
        return None
    return SegAdapter(cfg, path=path)


class Composite:
    """Gộp hai vai trò thành interface duy nhất predict(frame) -> {p_ngap, mask}: % do model quyết định, mask do model vẽ."""

    def __init__(self, decider, drawer=None):
        self.decider, self.drawer = decider, drawer
        self.kind = decider.kind
        self.name = decider.name + (f" + vẽ vùng nước: {drawer.name}" if drawer else "")
        self.description = decider.description

    gated_mask = True       # predict nhận draw_if: chỉ chạy model vẽ khi hàm này trả True cho p vừa tính

    def predict(self, frame, draw_if=None):
        """draw_if(p) -> bool: chỉ chạy model vẽ khi True (analyzer truyền "đang NGẬP"); None = luôn vẽ."""
        p = self.decider.predict(frame)["p_ngap"]
        mask = None
        if self.drawer is not None and (draw_if is None or draw_if(p)):
            try:
                mask = self.drawer.predict(frame)["mask"]
            except Exception:          # lỗi vẽ không được làm hỏng việc quyết định
                mask = None
        return {"p_ngap": p, "mask": mask}


def _mtime(p):
    try:
        return p.stat().st_mtime_ns
    except OSError:
        return None


class ModelHolder:
    """Giữ hai model và tự nạp lại từng cái khi file của nó đổi (mtime): gọi refresh() trước mỗi lần dùng.
    Lỗi nạp được ghi lại, không làm app sập; file hỏng không bị nạp lại cho tới khi file đổi tiếp."""

    def __init__(self, models_dir=None, decider_builder=build_decider, drawer_builder=build_drawer):
        self.models_dir = Path(models_dir or config.MODELS_DIR)
        self.decider_builder, self.drawer_builder = decider_builder, drawer_builder
        self.lock = threading.RLock()
        self.model = self.decider = self.drawer = self.cfg = None      # model = Composite(decider, drawer) hoặc None
        self.error, self.drawer_error = config.NO_MODEL_MSG, None
        self._sig = {"decide": object(), "draw": object()}              # chưa nạp lần nào

    def _signatures(self):
        yml = _mtime(self.models_dir / "model.yaml")
        try:
            cfg = config.load_model_config(self.models_dir)
        except Exception as e:                      # model.yaml hỏng: chờ tới khi file đổi
            return {"decide": ("cfg-error", str(e), yml), "draw": ("cfg-error", str(e), yml)}
        return {"decide": (_mtime(self.models_dir / cfg.path), cfg.device, yml),
                "draw": (_mtime(self.models_dir / cfg.seg_path), yml)}

    def refresh(self):
        """Nạp lại model nào có file thay đổi so với lần nạp trước. Rẻ khi không đổi (vài lệnh stat)."""
        with self.lock:
            sig = self._signatures()
            if sig["decide"] != self._sig["decide"]:
                self._load("decide", sig["decide"])
            if sig["draw"] != self._sig["draw"]:
                self._load("draw", sig["draw"])

    def reload(self):
        with self.lock:
            sig = self._signatures()
            self._load("decide", sig["decide"])
            self._load("draw", sig["draw"])
            return self.status()

    def _load(self, role, sig):
        try:                                        # chụp chữ ký trước khi nạp: file đổi giữa chừng thì lần refresh sau nạp lại
            cfg = config.load_model_config(self.models_dir)
            self.cfg = cfg
            if role == "decide":
                self.decider, self.error = self.decider_builder(cfg, self.models_dir), None
            else:
                self.drawer, self.drawer_error = self.drawer_builder(cfg, self.models_dir), None
        except Exception as e:                      # FileNotFoundError, ModelError, YAML sai, torch hỏng...
            if role == "decide":
                self.decider, self.error = None, str(e)
            else:
                self.drawer, self.drawer_error = None, str(e)
        self._sig[role] = sig
        self.model = Composite(self.decider, self.drawer) if self.decider is not None else None

    def snapshot(self):
        with self.lock:
            return self.model, self.cfg, self.error

    def status(self):
        d, w = self.decider, self.drawer
        with self.lock:
            return {
                "loaded": self.model is not None, "error": self.error, "config": asdict(self.cfg) if self.cfg else None,
                "decide": {"loaded": d is not None, "name": getattr(d, "name", None), "kind": getattr(d, "kind", None), "error": self.error},
                "draw": {"loaded": w is not None, "name": getattr(w, "name", None), "classes": getattr(w, "classes", None),
                         "error": self.drawer_error},
            }
