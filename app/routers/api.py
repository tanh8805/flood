"""API chung: sự cố, khu vực, job upload, trạng thái model."""
import json
import uuid
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.orm import Session, joinedload

from .. import config, incidents as svc
from ..db import get_db
from ..models import STATUSES, Area, Check, Evidence, Incident, Job, User, iso, utcnow
from ..security import current_user, device_key_ok, optional_user, require
from ..series import p_note

router = APIRouter(prefix="/api")
ALLOWED_VIDEO = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".mpg", ".mpeg", ".ts", ".3gp", ".flv", ".wmv"}
EVENT_AT = func.coalesce(Incident.recorded_at, Incident.created_at)


def ev_out(e: Evidence):
    u = f"/api/evidence/{e.id}"
    return {"id": e.id, "t_s": e.t_s, "p": e.p, "original": f"{u}/original", "overlay": f"{u}/overlay" if e.overlay_path else None,
            "crop": f"{u}/crop" if e.crop_path != e.original_path else None}      # không có mask: ảnh cắt = ảnh gốc, không hiện lặp


def inc_out(i: Incident, detail=True, series_prefix=None):
    out = {"id": i.id, "area_id": i.area_id, "area": i.area.name, "uploader": i.uploader.username if i.uploader else None,
           "created_at": iso(i.created_at), "recorded_at": iso(i.recorded_at), "video_start_s": i.video_start_s,
           "video_end_s": i.video_end_s, "p_max": round(i.p_max, 4), "p_note": p_note(series_prefix) if i.check_id else None, "model_name": i.model_name, "status": i.status, "source": i.source,
           "thumb": f"/api/evidence/{i.evidence[0].id}/crop" if i.evidence else None, "check_id": i.check_id}
    if detail:
        c = i.check
        # sự cố từ video: xem lại video kết quả của lần kiểm tra, nhảy tới lúc bắt đầu; sự cố do thiết bị gửi: clip kèm theo (không có %)
        has_video = bool(c and c.video_path)
        out["clip"] = f"/api/checks/{c.id}/video" if has_video else (f"/api/incidents/{i.id}/clip" if i.clip_path else None)
        out["clip_start"] = i.video_start_s if has_video else None
        out["series"] = json.loads(c.series_json) if c else None
        out["evidence"] = [ev_out(e) for e in i.evidence]
    return out


@router.get("/me")
def me(u: User = Depends(current_user)):
    return {"id": u.id, "username": u.username, "role": u.role}


# ---- khu vực: có sự cố (chưa báo nhầm) trong RECENT_HOURS giờ qua thì "đang ngập"
def area_out(db: Session, a: Area, since):
    latest = db.scalars(select(Incident).options(joinedload(Incident.evidence)).where(Incident.area_id == a.id, Incident.status != "false_alarm")
                        .order_by(EVENT_AT.desc(), Incident.id.desc()).limit(1)).unique().first()
    n = db.scalar(select(func.count()).select_from(Incident).where(Incident.area_id == a.id, Incident.status != "false_alarm", EVENT_AT >= since))
    last_check = db.scalar(select(func.max(Check.created_at)).where(Check.area_id == a.id))
    return {"id": a.id, "name": a.name, "description": a.description, "flooded": n > 0, "recent_count": n, "last_check_at": iso(last_check),
            "last_at": iso(latest.event_at) if latest else None,
            "thumb": f"/api/evidence/{latest.evidence[0].id}/crop" if latest and latest.evidence else None}


@router.get("/areas")
def areas(db: Session = Depends(get_db), _: User = Depends(current_user)):
    since = utcnow() - timedelta(hours=config.RECENT_HOURS)
    out = [area_out(db, a, since) for a in db.scalars(select(Area).where(Area.active))]
    out.sort(key=lambda x: (not x["flooded"], -(datetime.fromisoformat(x["last_at"].rstrip("Z")).timestamp() if x["last_at"] else 0), x["name"].lower()))
    return {"recent_hours": config.RECENT_HOURS, "areas": out}


@router.get("/areas/{aid}")
def area_detail(aid: int, db: Session = Depends(get_db), _: User = Depends(current_user)):
    a = db.get(Area, aid)
    if not a or not a.active:
        raise HTTPException(404, "Không có khu vực này")
    return {"recent_hours": config.RECENT_HOURS, **area_out(db, a, utcnow() - timedelta(hours=config.RECENT_HOURS))}


# ---- lần kiểm tra (mỗi lần phân tích video, ngập hay không)
def check_out(c: Check, detail=False):
    out = {"id": c.id, "area_id": c.area_id, "area": c.area.name, "uploader": c.uploader.username if c.uploader else None,
           "created_at": iso(c.created_at), "flooded": c.flooded, "p_max": round(c.p_max, 4), "p_note": p_note(c.series_json[:80]),
           "duration_s": c.duration_s,
           "model_name": c.model_name, "has_video": bool(c.video_path)}
    if detail:
        out["video"] = f"/api/checks/{c.id}/video" if c.video_path else None
        out["series"] = json.loads(c.series_json)
        out["incident_ids"] = [i.id for i in c.incidents]
    return out


@router.get("/checks")
def list_checks(area_id: int | None = None, limit: int = 30, db: Session = Depends(get_db), _: User = Depends(current_user)):
    q = select(Check).options(joinedload(Check.area), joinedload(Check.uploader)).order_by(Check.created_at.desc(), Check.id.desc())
    if area_id:
        q = q.where(Check.area_id == area_id)
    return [check_out(c) for c in db.scalars(q.limit(min(max(limit, 1), 200))).unique()]


def _check(db, cid):
    c = db.get(Check, cid)
    if not c:
        raise HTTPException(404, "Không có lần kiểm tra này")
    return c


@router.get("/checks/{cid}")
def get_check(cid: int, db: Session = Depends(get_db), _: User = Depends(current_user)):
    return check_out(_check(db, cid), detail=True)


@router.get("/checks/{cid}/video")
def check_video(cid: int, db: Session = Depends(get_db), _: User = Depends(current_user)):
    c = _check(db, cid)
    if not c.video_path:
        raise HTTPException(404, "Video kết quả không còn (đã quá thời gian lưu hoặc dựng lỗi)")
    return _file(c.video_path, "video/mp4")


# ---- sự cố
@router.get("/incidents")
def list_incidents(area_id: int | None = None, recent: bool = False, limit: int = 50, db: Session = Depends(get_db), _: User = Depends(current_user)):
    q = (select(Incident, func.substr(Check.series_json, 1, 80))
         .outerjoin(Check, Check.id == Incident.check_id)
         .options(joinedload(Incident.area), joinedload(Incident.uploader), joinedload(Incident.evidence))
         .order_by(EVENT_AT.desc(), Incident.id.desc()))
    if area_id:
        q = q.where(Incident.area_id == area_id)
    if recent:
        q = q.where(EVENT_AT >= utcnow() - timedelta(hours=config.RECENT_HOURS))
    return [inc_out(i, detail=False, series_prefix=pre) for i, pre in db.execute(q.limit(min(max(limit, 1), 200))).unique()]


def _get(db, iid):
    i = db.get(Incident, iid)
    if not i:
        raise HTTPException(404, "Không có sự cố này")
    return i


@router.get("/incidents/{iid}")
def get_incident(iid: int, db: Session = Depends(get_db), _: User = Depends(current_user)):
    i = _get(db, iid)
    return inc_out(i, series_prefix=i.check.series_json[:80] if i.check else None)


class StatusIn(BaseModel):
    status: str


@router.patch("/incidents/{iid}")
def set_status(iid: int, body: StatusIn, db: Session = Depends(get_db), _: User = Depends(require("admin"))):
    if body.status not in STATUSES:
        raise HTTPException(422, f"status phải là một trong {STATUSES}")
    i = _get(db, iid)
    i.status = body.status
    db.commit()
    return inc_out(i, series_prefix=i.check.series_json[:80] if i.check else None)


def _parse_time(s: str | None):
    if not s:
        return None
    try:
        d = datetime.fromisoformat(s.strip().replace("Z", "+00:00"))
    except ValueError:
        raise HTTPException(422, "recorded_at phải là ISO 8601, ví dụ 2026-10-29T08:30:00Z")
    return d.astimezone(timezone.utc).replace(tzinfo=None) if d.tzinfo else d      # không có tz: coi là UTC


@router.post("/incidents", status_code=201)
def create(request: Request, area_id: int = Form(...), p: float = Form(...), recorded_at: str | None = Form(None),
           video_start_s: float | None = Form(None), video_end_s: float | None = Form(None), model_name: str = Form("device"),
           image: list[UploadFile] = File(default=[]), overlay: list[UploadFile] = File(default=[]),
           crop: list[UploadFile] = File(default=[]), clip: UploadFile | None = File(None),
           user: User | None = Depends(optional_user), db: Session = Depends(get_db)):
    """Dùng chung cho thiết bị ngoài (header X-API-Key) và người dùng uploader/admin (cookie phiên).
    image/overlay/crop là danh sách file; phần tử thứ i của ba danh sách thuộc cùng một khung."""
    if not device_key_ok(request):
        if not user:
            raise HTTPException(401, "Cần đăng nhập hoặc X-API-Key")
        if user.role not in ("uploader", "admin"):
            raise HTTPException(403, "Không đủ quyền")
    if len(overlay) > len(image) or len(crop) > len(image):
        raise HTTPException(422, "overlay/crop không được nhiều hơn image")
    rd = lambda f: f.file.read(svc.MAX_IMAGE_BYTES + 1)
    frames = [svc.FrameIn(rd(f), rd(overlay[n]) if n < len(overlay) else None, rd(crop[n]) if n < len(crop) else None)
              for n, f in enumerate(image)]
    try:
        inc = svc.create_incident(db, area_id=area_id, p_max=p, frames=frames, clip=clip.file.read(svc.MAX_CLIP_BYTES + 1) if clip else None,
                                  recorded_at=_parse_time(recorded_at), video_start_s=video_start_s, video_end_s=video_end_s,
                                  uploader_id=user.id if user else None, model_name=model_name, source="device")
    except svc.IncidentError as e:
        raise HTTPException(422, str(e))
    return inc_out(inc)


@router.delete("/incidents/{iid}", status_code=204)
def delete(iid: int, db: Session = Depends(get_db), _: User = Depends(require("admin"))):
    svc.delete_incident(db, _get(db, iid))


def _file(rel, media):
    f = config.STORAGE_DIR / rel if rel else None
    if not f or not f.is_file():
        raise HTTPException(404, "Không có file")
    return FileResponse(f, media_type=media, headers={"Cache-Control": "private, max-age=3600"})


@router.get("/evidence/{eid}/{kind}")
def evidence_file(eid: int, kind: str, db: Session = Depends(get_db), _: User = Depends(current_user)):
    if kind not in ("original", "overlay", "crop"):
        raise HTTPException(404, "Không có file")
    e = db.get(Evidence, eid)
    return _file(getattr(e, f"{kind}_path") if e else None, "image/jpeg")


@router.get("/incidents/{iid}/clip")
def incident_clip(iid: int, db: Session = Depends(get_db), _: User = Depends(current_user)):
    return _file(_get(db, iid).clip_path, "video/mp4")


# ---- model + job
@router.get("/model")
def model_status(request: Request, _: User = Depends(current_user)):
    request.app.state.holder.refresh()
    s = request.app.state.holder.status()
    return {"loaded": s["loaded"], "error": s["error"]}


def job_out(j: Job, db: Session | None = None):
    ids = [i for (i,) in db.execute(select(Incident.id).where(Incident.job_id == j.id).order_by(Incident.id))] if db else []
    check_id = db.scalar(select(Check.id).where(Check.job_id == j.id)) if db else None
    return {"incident_ids": ids, "check_id": check_id, "id": j.id, "area": j.area.name, "filename": j.filename, "status": j.status, "progress": j.progress, "message": j.message,
            "error": j.error, "incidents_found": j.incidents_found, "created_at": iso(j.created_at), "finished_at": iso(j.finished_at)}


@router.post("/jobs", status_code=202)
def upload_video(request: Request, area_id: int = Form(...), video: UploadFile = File(...), db: Session = Depends(get_db),
                 user: User = Depends(require("uploader", "admin"))):
    holder = request.app.state.holder
    holder.refresh()
    if holder.model is None:
        raise HTTPException(503, holder.error or config.NO_MODEL_MSG)
    area = db.get(Area, area_id)
    if not area or not area.active:
        raise HTTPException(422, "Khu vực không tồn tại hoặc đã tắt")
    ext = ("." + (video.filename or "").rsplit(".", 1)[-1].lower()) if "." in (video.filename or "") else ""
    if ext not in ALLOWED_VIDEO:
        raise HTTPException(422, f"Định dạng video không hỗ trợ ({ext or 'không có đuôi'})")
    folder = config.STORAGE_DIR / "videos"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{uuid.uuid4().hex}{ext}"
    limit, size = config.MAX_UPLOAD_MB * 1024 * 1024, 0
    try:
        with open(path, "wb") as out:
            while chunk := video.file.read(1 << 20):
                size += len(chunk)
                if size > limit:
                    raise HTTPException(413, f"Video lớn hơn {config.MAX_UPLOAD_MB} MB")
                out.write(chunk)
        if size == 0:
            raise HTTPException(422, "File rỗng")
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    jid = request.app.state.runner.submit(user.id, area_id, video.filename or path.name, path)
    return {"id": jid}


@router.get("/jobs/{jid}")
def get_job(jid: str, db: Session = Depends(get_db), user: User = Depends(require("uploader", "admin"))):
    j = db.get(Job, jid)
    if not j or (j.user_id != user.id and user.role != "admin"):
        raise HTTPException(404, "Không có job này")
    return job_out(j, db)


@router.get("/jobs")
def my_jobs(db: Session = Depends(get_db), user: User = Depends(require("uploader", "admin"))):
    q = select(Job).options(joinedload(Job.area)).order_by(Job.created_at.desc()).limit(20)
    if user.role != "admin":
        q = q.where(Job.user_id == user.id)
    return [job_out(j) for j in db.scalars(q)]
