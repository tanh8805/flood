"""Tạo sự cố ngập: MỘT hàm dùng chung cho POST /api/incidents và phân tích video."""
import shutil
from dataclasses import dataclass
from datetime import datetime

import cv2
import numpy as np
from sqlalchemy.orm import Session

from . import config
from .models import Area, Evidence, Incident

MAX_FRAMES = 12
MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_CLIP_BYTES = 100 * 1024 * 1024


class IncidentError(ValueError):
    pass


@dataclass
class FrameIn:
    original: bytes
    overlay: bytes | None = None
    crop: bytes | None = None        # None: dùng lại ảnh gốc
    t_s: float | None = None
    p: float | None = None


def _check_image(data, name):
    if not data or len(data) > MAX_IMAGE_BYTES:
        raise IncidentError(f"Ảnh {name} rỗng hoặc quá 10 MB")
    if cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR) is None:
        raise IncidentError(f"Ảnh {name} không giải mã được (cần JPEG hoặc PNG)")


def create_incident(db: Session, *, area_id: int, p_max: float, frames: list[FrameIn], clip: bytes | None = None,
                    recorded_at: datetime | None = None, video_start_s=None, video_end_s=None, uploader_id=None,
                    model_name="", source="device", job_id=None, check_id=None) -> Incident:
    area = db.get(Area, area_id)
    if not area or not area.active:
        raise IncidentError(f"Khu vực {area_id} không tồn tại hoặc đã tắt")
    if not 0.0 <= p_max <= 1.0:
        raise IncidentError("p phải nằm trong [0, 1]")
    if video_start_s is not None and video_end_s is not None and video_end_s < video_start_s:
        raise IncidentError("t_end < t_start")
    if not frames:
        raise IncidentError("Cần ít nhất một ảnh minh chứng")
    if len(frames) > MAX_FRAMES:
        raise IncidentError(f"Tối đa {MAX_FRAMES} ảnh")
    for n, f in enumerate(frames):
        _check_image(f.original, f"gốc #{n + 1}")
        if f.overlay is not None:
            _check_image(f.overlay, f"overlay #{n + 1}")
        if f.crop is not None:
            _check_image(f.crop, f"cắt #{n + 1}")
    if clip is not None and (not clip or len(clip) > MAX_CLIP_BYTES):
        raise IncidentError("Đoạn video rỗng hoặc quá lớn")
    inc = Incident(area_id=area_id, p_max=p_max, recorded_at=recorded_at, video_start_s=video_start_s, video_end_s=video_end_s,
                   uploader_id=uploader_id, model_name=model_name[:128], source=source, job_id=job_id,
                   check_id=check_id)
    db.add(inc)
    db.flush()
    rel = f"evidence/{inc.id}"
    folder = config.STORAGE_DIR / rel
    try:
        folder.mkdir(parents=True, exist_ok=True)

        def save(name, data):
            (folder / name).write_bytes(data)
            return f"{rel}/{name}"
        for n, f in enumerate(frames):
            orig = save(f"{n}_original.jpg", f.original)
            inc.evidence.append(Evidence(original_path=orig, overlay_path=save(f"{n}_overlay.jpg", f.overlay) if f.overlay else None,
                                         crop_path=save(f"{n}_crop.jpg", f.crop) if f.crop else orig, t_s=f.t_s, p=f.p))
        if clip:
            inc.clip_path = save("clip.mp4", clip)
        db.commit()
    except Exception:
        db.rollback()
        shutil.rmtree(folder, ignore_errors=True)
        raise
    return inc


def delete_incident(db: Session, inc: Incident):
    folder = config.STORAGE_DIR / "evidence" / str(inc.id)
    db.delete(inc)
    db.commit()
    shutil.rmtree(folder, ignore_errors=True)


def frames_from_result(res) -> list[FrameIn]:
    """IncidentResult của analyzer -> danh sách FrameIn."""
    return [FrameIn(e.original, e.overlay, e.crop, e.t, e.p) for e in res.evidence]
