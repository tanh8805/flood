"""Xoá video kết quả cũ hơn RETENTION_DAYS; giữ bản ghi lần kiểm tra, dãy t-p, sự cố và ảnh minh chứng."""
import logging
import threading
from datetime import timedelta

from sqlalchemy import select

from . import config
from .db import SessionLocal
from .models import Check, utcnow

log = logging.getLogger("retention")
EVERY_S = 6 * 3600


def purge_old_videos(db, days=None, now=None) -> int:
    days = config.RETENTION_DAYS if days is None else days
    if days <= 0:
        return 0
    cutoff = (now or utcnow()) - timedelta(days=days)
    n = 0
    for c in db.scalars(select(Check).where(Check.video_path.is_not(None), Check.created_at < cutoff)):
        (config.STORAGE_DIR / c.video_path).unlink(missing_ok=True)
        c.video_path = None
        n += 1
    db.commit()
    return n


def start_background(stop: threading.Event):
    def loop():
        while True:
            try:
                with SessionLocal() as db:
                    n = purge_old_videos(db)
                if n:
                    log.warning("Đã xoá %d video kết quả cũ hơn %s ngày", n, config.RETENTION_DAYS)
            except Exception:
                log.exception("dọn video cũ lỗi")
            if stop.wait(EVERY_S):
                return
    t = threading.Thread(target=loop, daemon=True, name="retention")
    t.start()
    return t
