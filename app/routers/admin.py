"""Quản trị: khu vực, tài khoản."""
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from ..db import get_db
from ..models import Area, Incident, User, iso
from ..security import ROLES, hash_password, require

router = APIRouter(prefix="/api/admin", dependencies=[Depends(require("admin"))])


class AreaIn(BaseModel):
    name: str = Field(min_length=1, max_length=128)
    description: str = ""
    active: bool = True


class UserIn(BaseModel):
    username: str = Field(min_length=3, max_length=64, pattern=r"^[A-Za-z0-9_.\-]+$")
    password: str = Field(min_length=8, max_length=128)
    role: str = "viewer"


class UserPatch(BaseModel):
    password: str | None = Field(None, min_length=8, max_length=128)
    role: str | None = None
    active: bool | None = None


def area_out(a: Area, n: int = 0):
    return {"id": a.id, "name": a.name, "description": a.description, "active": a.active, "incidents": n}


@router.get("/areas")
def list_areas(db: Session = Depends(get_db)):
    counts = dict(db.execute(select(Incident.area_id, func.count()).group_by(Incident.area_id)).all())
    return [area_out(a, counts.get(a.id, 0)) for a in db.scalars(select(Area).order_by(Area.name))]


@router.post("/areas", status_code=201)
def add_area(body: AreaIn, db: Session = Depends(get_db)):
    a = Area(**body.model_dump())
    db.add(a)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(409, "Tên khu vực đã tồn tại")
    return area_out(a)


@router.put("/areas/{aid}")
def edit_area(aid: int, body: AreaIn, db: Session = Depends(get_db)):
    a = db.get(Area, aid)
    if not a:
        raise HTTPException(404, "Không có khu vực này")
    for k, v in body.model_dump().items():
        setattr(a, k, v)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(409, "Tên khu vực đã tồn tại")
    return area_out(a)


@router.delete("/areas/{aid}", status_code=204)
def del_area(aid: int, db: Session = Depends(get_db)):
    a = db.get(Area, aid)
    if not a:
        raise HTTPException(404, "Không có khu vực này")
    if db.scalar(select(func.count()).select_from(Incident).where(Incident.area_id == aid)):
        raise HTTPException(409, "Khu vực đã có sự cố: hãy tắt (active=false) thay vì xoá")
    try:
        db.delete(a)
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(409, "Khu vực còn được tham chiếu (job): hãy tắt thay vì xoá")


def user_out(u: User):
    return {"id": u.id, "username": u.username, "role": u.role, "active": u.active, "created_at": iso(u.created_at)}


@router.get("/users")
def list_users(db: Session = Depends(get_db)):
    return [user_out(u) for u in db.scalars(select(User).order_by(User.username))]


@router.post("/users", status_code=201)
def add_user(body: UserIn, db: Session = Depends(get_db)):
    if body.role not in ROLES:
        raise HTTPException(422, f"role phải là một trong {ROLES}")
    u = User(username=body.username, password_hash=hash_password(body.password), role=body.role)
    db.add(u)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(409, "Tên đăng nhập đã tồn tại")
    return user_out(u)


@router.patch("/users/{uid}")
def edit_user(uid: int, body: UserPatch, me: User = Depends(require("admin")), db: Session = Depends(get_db)):
    u = db.get(User, uid)
    if not u:
        raise HTTPException(404, "Không có tài khoản này")
    if body.role is not None:
        if body.role not in ROLES:
            raise HTTPException(422, f"role phải là một trong {ROLES}")
        u.role = body.role
    if body.active is not None:
        u.active = body.active
    if body.password:
        u.password_hash = hash_password(body.password)
    if u.id == me.id and (u.role != "admin" or not u.active):
        db.rollback()
        raise HTTPException(409, "Không thể tự hạ quyền hoặc khoá chính mình")
    db.commit()
    return user_out(u)


@router.get("/model")
def model_status(request: Request):
    """Chỉ đọc: model nào đang đảm nhiệm vai trò nào. Không có nút/API nạp lại: đổi file là tự nạp (mtime)."""
    request.app.state.holder.refresh()
    return request.app.state.holder.status()
