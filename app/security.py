"""Mật khẩu (scrypt) và phiên đăng nhập bằng cookie ký HMAC. Chỉ dùng thư viện chuẩn."""
import base64
import hashlib
import hmac
import json
import secrets
import time

from fastapi import Depends, HTTPException, Request
from sqlalchemy.orm import Session

from . import config
from .db import get_db
from .models import User

COOKIE = "session"
ROLES = ("admin", "uploader", "viewer")
_N, _R, _P = 2 ** 14, 8, 1


def hash_password(pw: str) -> str:
    salt = secrets.token_bytes(16)
    h = hashlib.scrypt(pw.encode(), salt=salt, n=_N, r=_R, p=_P)
    return f"scrypt${salt.hex()}${h.hex()}"


def verify_password(pw: str, stored: str) -> bool:
    try:
        _, salt, h = stored.split("$")
        got = hashlib.scrypt(pw.encode(), salt=bytes.fromhex(salt), n=_N, r=_R, p=_P)
        return hmac.compare_digest(got, bytes.fromhex(h))
    except Exception:
        return False


def secret_key() -> bytes:
    if config.SECRET_KEY:
        return config.SECRET_KEY.encode()
    f = config.STORAGE_DIR / "secret.key"
    if not f.exists():
        config.STORAGE_DIR.mkdir(parents=True, exist_ok=True)
        f.write_text(secrets.token_hex(32))
        f.chmod(0o600)
    return f.read_text().strip().encode()


def make_token(uid: int, key: bytes | None = None, now: float | None = None) -> str:
    body = base64.urlsafe_b64encode(json.dumps({"uid": uid, "exp": (now or time.time()) + config.SESSION_HOURS * 3600}).encode()).decode()
    sig = hmac.new(key or secret_key(), body.encode(), hashlib.sha256).hexdigest()
    return f"{body}.{sig}"


def read_token(token: str, key: bytes | None = None, now: float | None = None) -> int | None:
    try:
        body, sig = token.rsplit(".", 1)
        good = hmac.new(key or secret_key(), body.encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(sig, good):
            return None
        d = json.loads(base64.urlsafe_b64decode(body))
        return d["uid"] if d["exp"] > (now or time.time()) else None
    except Exception:
        return None


def optional_user(request: Request, db: Session = Depends(get_db)) -> User | None:
    tok = request.cookies.get(COOKIE)
    uid = read_token(tok) if tok else None
    u = db.get(User, uid) if uid else None
    return u if u and u.active else None


def current_user(u: User | None = Depends(optional_user)) -> User:
    if not u:
        raise HTTPException(401, "Chưa đăng nhập")
    return u


def require(*roles):
    def dep(u: User = Depends(current_user)) -> User:
        if u.role not in roles:
            raise HTTPException(403, "Không đủ quyền")
        return u
    return dep


def device_key_ok(request: Request) -> bool:
    k = request.headers.get("x-api-key", "")
    return bool(config.DEVICE_API_KEY) and hmac.compare_digest(k, config.DEVICE_API_KEY)


class LoginThrottle:
    """Chặn dò mật khẩu: tối đa `limit` lần sai trong `window` giây cho mỗi (username, ip)."""

    def __init__(self, limit=5, window=300):
        self.limit, self.window, self.fails = limit, window, {}

    def blocked(self, key):
        now = time.time()
        self.fails[key] = [t for t in self.fails.get(key, []) if now - t < self.window]
        return len(self.fails[key]) >= self.limit

    def fail(self, key):
        self.fails.setdefault(key, []).append(time.time())

    def ok(self, key):
        self.fails.pop(key, None)


throttle = LoginThrottle()
