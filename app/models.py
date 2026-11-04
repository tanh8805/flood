from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .db import Base


def utcnow():
    """UTC không kèm tzinfo (SQLite không lưu tz); luôn xuất ra API dạng ISO có 'Z'."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


def iso(dt):
    return dt.isoformat(timespec="seconds") + "Z" if dt else None


class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(primary_key=True)
    username: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(256))
    role: Mapped[str] = mapped_column(String(16), default="viewer")      # admin | uploader | viewer
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class Area(Base):
    __tablename__ = "areas"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(128), unique=True)
    description: Mapped[str] = mapped_column(Text, default="")
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


STATUSES = ("new", "confirmed", "false_alarm")      # mới / đã xác nhận / báo nhầm


class Incident(Base):
    __tablename__ = "incidents"
    id: Mapped[int] = mapped_column(primary_key=True)
    area_id: Mapped[int] = mapped_column(ForeignKey("areas.id"), index=True)
    uploader_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)   # None: thiết bị ngoài (API key)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, index=True)
    recorded_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)           # thời gian quay, nếu biết
    video_start_s: Mapped[float | None] = mapped_column(Float, nullable=True)
    video_end_s: Mapped[float | None] = mapped_column(Float, nullable=True)
    p_max: Mapped[float] = mapped_column(Float)
    model_name: Mapped[str] = mapped_column(String(128), default="")
    status: Mapped[str] = mapped_column(String(16), default="new")
    clip_path: Mapped[str | None] = mapped_column(String(256), nullable=True)               # video do thiết bị ngoài gửi kèm (không có % theo khung)
    check_id: Mapped[int | None] = mapped_column(ForeignKey("checks.id", ondelete="SET NULL"), nullable=True, index=True)   # lần kiểm tra sinh ra sự cố
    source: Mapped[str] = mapped_column(String(16), default="device")                       # video | device
    job_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    area: Mapped[Area] = relationship()
    uploader: Mapped[User | None] = relationship()
    check: Mapped["Check | None"] = relationship()
    evidence: Mapped[list["Evidence"]] = relationship(cascade="all, delete-orphan", order_by="Evidence.id")

    @property
    def event_at(self):
        return self.recorded_at or self.created_at


class Check(Base):
    """Một lần phân tích video (ngập hay không): kết quả, p cao nhất, video kết quả và dãy [t, p, ngập] của mọi lần suy luận."""
    __tablename__ = "checks"
    id: Mapped[int] = mapped_column(primary_key=True)
    area_id: Mapped[int] = mapped_column(ForeignKey("areas.id"), index=True)
    uploader_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    job_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, index=True)
    flooded: Mapped[bool] = mapped_column(Boolean, default=False)
    p_max: Mapped[float] = mapped_column(Float)
    duration_s: Mapped[float] = mapped_column(Float, default=0.0)
    model_name: Mapped[str] = mapped_column(String(128), default="")
    video_path: Mapped[str | None] = mapped_column(String(256), nullable=True)               # None: dựng lỗi hoặc đã quá RETENTION_DAYS
    series_json: Mapped[str] = mapped_column(Text, default="[]")
    area: Mapped[Area] = relationship()
    uploader: Mapped[User | None] = relationship()
    incidents: Mapped[list[Incident]] = relationship(viewonly=True, order_by="Incident.video_start_s")


class Evidence(Base):
    """Một khung minh chứng: ảnh gốc, overlay (nếu có mask), ảnh cắt."""
    __tablename__ = "evidence"
    id: Mapped[int] = mapped_column(primary_key=True)
    incident_id: Mapped[int] = mapped_column(ForeignKey("incidents.id", ondelete="CASCADE"), index=True)
    original_path: Mapped[str] = mapped_column(String(256))
    overlay_path: Mapped[str | None] = mapped_column(String(256), nullable=True)
    crop_path: Mapped[str | None] = mapped_column(String(256), nullable=True)
    t_s: Mapped[float | None] = mapped_column(Float, nullable=True)
    p: Mapped[float | None] = mapped_column(Float, nullable=True)


class Job(Base):
    __tablename__ = "jobs"
    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    area_id: Mapped[int] = mapped_column(ForeignKey("areas.id"))
    filename: Mapped[str] = mapped_column(String(256))
    status: Mapped[str] = mapped_column(String(16), default="queued")     # queued | running | done | failed
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    message: Mapped[str] = mapped_column(String(256), default="")
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    incidents_found: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    area: Mapped[Area] = relationship()
