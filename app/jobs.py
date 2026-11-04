"""Hàng đợi job nền: mỗi lần một video."""
import json
import logging
import queue
import threading
import time
import uuid
from pathlib import Path

from . import config
from .analyzer import analyze_video
from .db import SessionLocal
from .incidents import create_incident, frames_from_result
from .models import Check, Job, utcnow

log = logging.getLogger("jobs")


class JobRunner:
    def __init__(self, holder, session_factory=None):
        self.holder, self.Session = holder, session_factory or SessionLocal
        self.q: queue.Queue = queue.Queue()
        self.thread = None

    def start(self):
        with self.Session() as db:        # job còn dang dở từ lần chạy trước: video tạm đã bị bỏ, đánh dấu lỗi
            for j in db.query(Job).filter(Job.status.in_(("queued", "running"))):
                j.status, j.error, j.finished_at = "failed", "Bị gián đoạn do khởi động lại", utcnow()
            db.commit()
        self.thread = threading.Thread(target=self._loop, daemon=True, name="job-runner")
        self.thread.start()

    def submit(self, user_id, area_id, filename, path) -> str:
        jid = uuid.uuid4().hex
        with self.Session() as db:
            db.add(Job(id=jid, user_id=user_id, area_id=area_id, filename=filename[:250], message="Đang chờ"))
            db.commit()
        self.q.put((jid, Path(path)))
        return jid

    def _loop(self):
        while True:
            jid, path = self.q.get()
            try:
                self.run(jid, path)
            except Exception:
                log.exception("job %s lỗi ngoài dự kiến", jid)

    def run(self, jid, path: Path):
        db = self.Session()
        try:
            job = db.get(Job, jid)
            job.status, job.message = "running", "Bắt đầu"
            db.commit()
            last = [0.0]

            def progress(frac, msg):
                if time.time() - last[0] < 0.5 and frac < 1:
                    return
                last[0] = time.time()
                job.progress, job.message = round(frac, 3), msg
                db.commit()

            self.holder.refresh()                      # best.pt mới chép vào thì lần phân tích này dùng luôn
            model, cfg, err = self.holder.snapshot()
            if model is None:
                raise RuntimeError(err or config.NO_MODEL_MSG)
            rel = f"checks/{jid}.mp4"
            out = config.STORAGE_DIR / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            results, info = analyze_video(path, model, cfg, progress, video_out=out)
            name = getattr(model, "name", model.kind)
            check = Check(area_id=job.area_id, uploader_id=job.user_id, job_id=jid, flooded=bool(results), p_max=info["p_max"],
                          duration_s=info["duration"], model_name=name[:128], video_path=rel if info["video"] else None,
                          series_json=json.dumps(info["series"], separators=(",", ":")))
            db.add(check)
            db.commit()                                # lần kiểm tra có trước, sự cố gắn vào nó
            for r in results:
                create_incident(db, area_id=job.area_id, p_max=r.p_max, frames=frames_from_result(r), uploader_id=job.user_id,
                                video_start_s=r.t_start, video_end_s=r.t_end, model_name=name, source="video", job_id=jid, check_id=check.id)
            job.status, job.progress, job.incidents_found = "done", 1.0, len(results)
            job.message = "Phát hiện ngập" if results else "Không phát hiện ngập"
        except Exception as e:
            db.rollback()
            job = db.get(Job, jid)
            job.status, job.error, job.message = "failed", str(e) or e.__class__.__name__, "Lỗi"
            log.warning("job %s thất bại: %s", jid, e)
        finally:
            job.finished_at = utcnow()
            db.commit()
            db.close()
            if not config.KEEP_VIDEO:
                path.unlink(missing_ok=True)
