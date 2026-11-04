from sqlalchemy import create_engine, event
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from . import config


class Base(DeclarativeBase):
    pass


def make_engine(url=None):
    url = url or config.DATABASE_URL
    kw = {}
    if url.startswith("sqlite"):
        config.STORAGE_DIR.mkdir(parents=True, exist_ok=True)
        kw["connect_args"] = {"check_same_thread": False}
    eng = create_engine(url, pool_pre_ping=True, **kw)
    if url.startswith("sqlite"):
        @event.listens_for(eng, "connect")
        def _fk(conn, _):
            conn.cursor().execute("PRAGMA foreign_keys=ON")
    return eng


engine = make_engine()
SessionLocal = sessionmaker(engine, expire_on_commit=False)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
