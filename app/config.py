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
    path: str = "best.pt"               # model QUYẾT ĐỊNH ngập trên GPU (TorchScript: p = m(khung BGR)); L-14 fp16
    cpu_path: str = "best_cpu.pt"       # model quyết định khi KHÔNG dùng GPU (B-32 fp32, nhanh trên CPU); không có file này thì dùng `path`
    seg_path: str = "seg.pt"            # model segment CHỈ để tô vùng nước, không ảnh hưởng %
    device: str = "auto"                # best.pt chạy ở: auto (cuda nếu có) | cpu | cuda
    water_classes: str = "water|flood|nuoc"
    water_conf: float = 0.4             # segment (tô): ngưỡng tin cậy của mảng nước
    draw_width: int = 960               # segment (tô): thu khung rộng hơn giá trị này về đúng chiều rộng này trước khi suy luận; 0 = không thu
    flood_on: float = 0.9               # bật NGẬP khi p trung bình >= giá trị này
    flood_off: float = 0.3              # tắt khi p trung bình < giá trị này
    window_s: float = 1.0               # cửa sổ trung bình p (giây)
    min_hold_s: float = 10.0            # sau khi bật NGẬP, giữ tối thiểu từng ấy giây rồi mới áp luật tắt (chống chớp tắt); 0 = không giữ
    crop_black_border: bool = True      # cắt viền đen (letterbox/pillarbox) của video trước khi đưa vào best.pt và seg.pt
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


def _cast(current, v):
    """Ép giá trị YAML/biến môi trường về kiểu của khoá; bool đọc đúng chuỗi '0'/'false' (bool('0') của Python là True)."""
    if isinstance(current, bool):
        if isinstance(v, str):
            t = v.strip().lower()
            if t in ("1", "true", "yes", "on"):
                return True
            if t in ("0", "false", "no", "off"):
                return False
            raise ValueError(f"giá trị đúng/sai không hợp lệ: '{v}' (1/0, true/false)")
        return bool(v)
    return type(current)(v)


def cuda_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def resolve_decider_path(cfg: ModelConfig, models_dir: Path) -> Path:
    """File model quyết định sẽ dùng: GPU (device=cuda, hoặc auto và có CUDA) -> `path`; ngược lại -> `cpu_path` nếu có file, không thì `path`."""
    models_dir = Path(models_dir)
    gpu = cfg.device == "cuda" or (cfg.device == "auto" and cuda_available())
    first, second = (cfg.path, cfg.cpu_path) if gpu else (cfg.cpu_path, cfg.path)
    return models_dir / (first if (models_dir / first).exists() else second)


RULE_KEYS = {"window_s": "window_s", "on": "flood_on", "off": "flood_off", "min_hold_s": "min_hold_s", "infer_fps": "infer_fps", "segment_s": "segment_s"}


def embedded_rule(path: Path) -> dict:
    """Luật nhúng trong config.json của file TorchScript (đọc thẳng từ zip, không nạp model). Không đọc được thì {}."""
    import json
    import zipfile
    try:
        with zipfile.ZipFile(path) as z:
            name = next((n for n in z.namelist() if n.endswith("extra/config.json")), None)
            rule = json.loads(z.read(name))["rule"] if name else {}
        return {RULE_KEYS[k]: float(v) for k, v in rule.items() if k in RULE_KEYS}
    except Exception:
        return {}


def load_model_config(models_dir: Path | None = None) -> ModelConfig:
    """Thứ tự ưu tiên: mặc định < luật nhúng trong file model đang dùng (best.pt hoặc best_cpu.pt) < models/model.yaml < biến môi trường."""
    models_dir = Path(models_dir or MODELS_DIR)
    known = {f.name: f.type for f in fields(ModelConfig)}
    over = {}
    for k, v in _yaml(models_dir).items():
        if k not in known:
            raise ValueError(f"model.yaml: khoá không hợp lệ '{k}' (hợp lệ: {', '.join(known)})")
        over[k] = v
    for name in known:                       # biến môi trường: MODEL_PATH, MODEL_CPU_PATH, MODEL_SEG_PATH, MODEL_DEVICE, FLOOD_ON, WINDOW_S...
        v = os.getenv(f"MODEL_{name.upper()}" if name in ("path", "cpu_path", "seg_path", "device") else name.upper())
        if v not in (None, ""):
            over[name] = v
    cfg = ModelConfig()
    for k in ("path", "cpu_path", "device"):
        if k in over:
            setattr(cfg, k, _cast(getattr(cfg, k), over[k]))
    if cfg.device not in ("auto", "cpu", "cuda"):
        raise ValueError(f"device không hợp lệ: '{cfg.device}' (auto | cpu | cuda)")
    for k, v in embedded_rule(resolve_decider_path(cfg, models_dir)).items():
        setattr(cfg, k, _cast(getattr(cfg, k), v))
    for k, v in over.items():
        setattr(cfg, k, _cast(getattr(cfg, k), v))
    if not 0 <= cfg.flood_off <= cfg.flood_on <= 1:
        raise ValueError("cần 0 <= flood_off <= flood_on <= 1")
    if cfg.min_hold_s < 0:
        raise ValueError("min_hold_s phải >= 0")
    return cfg
