import logging
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import select

from . import config
from .adapters import ModelHolder
from .db import Base, SessionLocal, engine
from . import retention
from .series import recompute_legacy
from .jobs import JobRunner
from .models import User
from .routers import admin, api, pages

log = logging.getLogger("app")


def describe_models(st):
    """Một dòng log: model nào quyết định, model nào vẽ."""
    if not st["loaded"]:
        return st["error"]
    d, w = st["decide"], st["draw"]
    return f"quyết định: {d['name']}; vẽ vùng nước: {w['name'] if w['loaded'] else (w['error'] or 'không có seg.pt')}"


def warn_if_no_users():
    with SessionLocal() as db:
        if not db.scalar(select(User).limit(1)):
            log.warning("Chưa có tài khoản nào. Tạo admin đầu tiên: python manage.py create-admin")


@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(engine)
    config.STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    warn_if_no_users()
    app.state.holder = ModelHolder()
    app.state.holder.refresh()
    with SessionLocal() as db:                         # bản ghi cũ chỉ có p thô: tính lại p làm mượt nếu được
        r = recompute_legacy(db, config.load_model_config())
        if r["recomputed"] or r["aligned"] or r["kept_raw"]:
            log.warning("p làm mượt: tính lại %d, căn chỉnh p_max %d, giữ p thô %d lần kiểm tra cũ", r["recomputed"], r["aligned"], r["kept_raw"])
    st = app.state.holder.status()
    log.warning("Model: %s", describe_models(st))
    app.state.runner = JobRunner(app.state.holder)
    app.state.runner.start()
    stop = threading.Event()
    retention.start_background(stop)
    yield
    stop.set()


app = FastAPI(title="Cảnh báo ngập đường phố", lifespan=lifespan, docs_url="/api/docs", openapi_url="/api/openapi.json")
app.include_router(api.router)
app.include_router(admin.router)
app.include_router(pages.router)
app.mount("/static", StaticFiles(directory=config.ROOT / "app" / "static"), name="static")


@app.exception_handler(HTTPException)
async def http_exc(request, exc: HTTPException):
    if exc.status_code == 401 and not request.url.path.startswith("/api/"):
        return RedirectResponse("/login", 303)
    return JSONResponse({"detail": exc.detail}, exc.status_code)
