"""Cấu hình: mặc định < models/model.yaml < biến môi trường."""
import os
from dataclasses import dataclass, fields
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = Path(os.getenv("MODELS_DIR", ROOT / "models"))
STORAGE_DIR = Path(os.getenv("STORAGE_DIR", ROOT / "storage"))
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{STORAGE_DIR / 'app.db'}")
KEEP_VIDEO = os.getenv("KEEP_VIDEO") == "1"
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "500"))
RECENT_HOURS = float(os.getenv("RECENT_HOURS", "6"))      # khu vực "đang/vừa ngập"
SESSION_HOURS = float(os.getenv("SESSION_HOURS", "12"))
RETENTION_DAYS = float(os.getenv("RETENTION_DAYS", "30"))   # xoá video kết quả cũ hơn; 0 = giữ mãi
SECRET_KEY = os.getenv("SECRET_KEY", "")                  # trống: sinh ngẫu nhiên và lưu ở storage/secret.key
DEVICE_API_KEY = os.getenv("DEVICE_API_KEY", "")          # thiết bị ngoài gửi X-API-Key; trống: tắt
NO_MODEL_MSG = "Chưa có model: đặt models/best.pt (TorchScript) vào thư mục models/, cách tải xem README"


@dataclass
class ModelConfig:
    path: str = "best.pt"               # model QUYẾT ĐỊNH ngập (TorchScript: p = m(khung BGR))
    seg_path: str = "seg.pt"            # model segment CHỈ để tô vùng nước, không ảnh hưởng %
    device: str = "auto"                # best.pt chạy ở: auto (cuda nếu có) | cpu | cuda
    water_classes: str = "water|flood|nuoc"
    water_conf: float = 0.4             # segment (tô): ngưỡng tin cậy của mảng nước
    draw_width: int = 960               # segment (tô): thu khung rộng hơn giá trị này về đúng chiều rộng này trước khi suy luận; 0 = không thu
    flood_on: float = 0.9               # bật NGẬP khi p trung bình >= giá trị này
    flood_off: float = 0.6              # tắt khi p trung bình < giá trị này
    window_s: float = 1.0               # cửa sổ trung bình p (giây)
    segment_s: float = 2.0
    infer_fps: float = 5.0


def _yaml(models_dir: Path) -> dict:
    f = models_dir / "model.yaml"
    if not f.exists():
        return {}
    data = yaml.safe_load(f.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{f} phải là một mapping YAML")
    return {str(k).lower(): v for k, v in data.items()}


def load_model_config(models_dir: Path | None = None) -> ModelConfig:
    models_dir = Path(models_dir or MODELS_DIR)
    cfg = ModelConfig()
    known = {f.name: f.type for f in fields(ModelConfig)}
    for k, v in _yaml(models_dir).items():
        if k not in known:
            raise ValueError(f"model.yaml: khoá không hợp lệ '{k}' (hợp lệ: {', '.join(known)})")
        setattr(cfg, k, type(getattr(cfg, k))(v))
    for name in known:                       # biến môi trường: MODEL_PATH, MODEL_SEG_PATH, MODEL_DEVICE, FLOOD_ON, WINDOW_S...
        v = os.getenv(f"MODEL_{name.upper()}" if name in ("path", "seg_path", "device") else name.upper())
        if v not in (None, ""):
            setattr(cfg, name, type(getattr(cfg, name))(v))
    if cfg.device not in ("auto", "cpu", "cuda"):
        raise ValueError(f"device không hợp lệ: '{cfg.device}' (auto | cpu | cuda)")
    if not 0 <= cfg.flood_off <= cfg.flood_on <= 1:
        raise ValueError("cần 0 <= flood_off <= flood_on <= 1")
    return cfg
