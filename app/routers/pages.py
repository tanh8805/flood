"""Trang HTML (Jinja2) + đăng nhập/đăng xuất."""
from fastapi import APIRouter, Depends, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import select
from sqlalchemy.orm import Session

from .. import config
from ..db import get_db
from ..models import Area, Check, Incident, User
from ..security import COOKIE, current_user, make_token, optional_user, require, throttle, verify_password

router = APIRouter()
templates = Jinja2Templates(directory=str(config.ROOT / "app" / "templates"))
HOME = {"admin": "/overview", "uploader": "/upload", "viewer": "/areas"}     # trang đầu sau đăng nhập theo vai trò


def page(request: Request, name: str, user: User | None, **ctx):
    return templates.TemplateResponse(request, name, {"user": user, **ctx})


@router.get("/", include_in_schema=False)
def index(user: User | None = Depends(optional_user)):
    return RedirectResponse(HOME[user.role] if user else "/login", 303)


@router.get("/login", response_class=HTMLResponse, include_in_schema=False)
def login_form(request: Request, user: User | None = Depends(optional_user)):
    if user:
        return RedirectResponse(HOME[user.role], 303)
    return page(request, "login.html", None, error=None)


@router.post("/login", include_in_schema=False)
def login(request: Request, username: str = Form(...), password: str = Form(...), db: Session = Depends(get_db)):
    key = (username.lower(), request.client.host if request.client else "")
    if throttle.blocked(key):
        r = page(request, "login.html", None, error="Sai quá nhiều lần, thử lại sau vài phút")
        r.status_code = 429
        return r
    u = db.scalar(select(User).where(User.username == username))
    if not u or not u.active or not verify_password(password, u.password_hash):
        throttle.fail(key)
        r = page(request, "login.html", None, error="Sai tên đăng nhập hoặc mật khẩu")
        r.status_code = 401
        return r
    throttle.ok(key)
    r = RedirectResponse(HOME[u.role], 303)
    r.set_cookie(COOKIE, make_token(u.id), max_age=int(config.SESSION_HOURS * 3600), httponly=True, samesite="lax",
                 secure=request.url.scheme == "https")
    return r


@router.post("/logout", include_in_schema=False)
def logout():
    r = RedirectResponse("/login", 303)
    r.delete_cookie(COOKIE)
    return r


@router.get("/areas", response_class=HTMLResponse, include_in_schema=False)
def areas_page(request: Request, user: User = Depends(current_user)):
    return page(request, "areas.html", user, recent_hours=config.RECENT_HOURS)


@router.get("/areas/{aid}", response_class=HTMLResponse, include_in_schema=False)
def area_page(aid: int, request: Request, user: User = Depends(current_user), db: Session = Depends(get_db)):
    if not db.get(Area, aid):
        raise HTTPException(404, "Không có khu vực này")
    return page(request, "area.html", user, aid=aid, recent_hours=config.RECENT_HOURS)


@router.get("/checks/{cid}", response_class=HTMLResponse, include_in_schema=False)
def check_page(cid: int, request: Request, user: User = Depends(current_user), db: Session = Depends(get_db)):
    c = db.get(Check, cid)
    if not c:
        raise HTTPException(404, "Không có lần kiểm tra này")
    if user.role != "admin" and not (user.role == "uploader" and c.uploader_id == user.id):
        raise HTTPException(403, "Không đủ quyền xem lần kiểm tra này")
    return page(request, "check.html", user, cid=cid)


@router.get("/incidents/{iid}", response_class=HTMLResponse, include_in_schema=False)
def incident_page(iid: int, request: Request, user: User = Depends(current_user), db: Session = Depends(get_db)):
    if not db.get(Incident, iid):
        raise HTTPException(404, "Không có sự cố này")
    return page(request, "incident.html", user, iid=iid)


@router.get("/upload", response_class=HTMLResponse, include_in_schema=False)
def upload_page(request: Request, user: User = Depends(require("uploader", "admin"))):
    return page(request, "upload.html", user, max_mb=config.MAX_UPLOAD_MB)


@router.get("/admin", response_class=HTMLResponse, include_in_schema=False)
def admin_page(request: Request, user: User = Depends(require("admin"))):
    return page(request, "admin.html", user)


@router.get("/overview", response_class=HTMLResponse, include_in_schema=False)
def overview_page(request: Request, user: User = Depends(require("admin"))):
    return page(request, "overview.html", user, recent_hours=config.RECENT_HOURS)


@router.get("/incidents", response_class=HTMLResponse, include_in_schema=False)
def incidents_page(request: Request, user: User = Depends(require("admin"))):
    return page(request, "incidents.html", user)
