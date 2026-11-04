import os
import secrets
import tempfile

# Mật khẩu thử lấy từ biến môi trường (TEST_ADMIN_PASSWORD, TEST_USER_PASSWORD); không đặt thì sinh ngẫu nhiên mỗi lần chạy. Không ghi cứng trong repo.
ADMIN_PW = os.getenv("TEST_ADMIN_PASSWORD") or secrets.token_urlsafe(12)
USER_PW = os.getenv("TEST_USER_PASSWORD") or secrets.token_urlsafe(12)

_tmp = tempfile.mkdtemp(prefix="flood-test-")
os.environ["STORAGE_DIR"] = _tmp
os.environ["MODELS_DIR"] = os.path.join(_tmp, "models")        # rỗng: không có model thật
os.environ["DEVICE_API_KEY"] = "devkey"

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import manage  # noqa: E402
from app.main import app  # noqa: E402

manage.create_admin("admin", ADMIN_PW)


def jpg(v=128):
    ok, b = cv2.imencode(".jpg", np.full((40, 60, 3), v, np.uint8))
    return b.tobytes()


class BrightModel:
    kind, name, description = "fake", "fake.pt", "model giả"

    def predict(self, frame):
        p = float(frame.mean()) / 255
        mask = None
        if p > 0.5:
            mask = np.zeros(frame.shape[:2], bool)
            mask[frame.shape[0] // 2:, :] = True
        return {"p_ngap": p, "mask": mask}


@pytest.fixture(scope="session")
def client():
    with TestClient(app) as c:
        yield c


def login(client, user, pw):
    c = TestClient(app)
    r = c.post("/login", data={"username": user, "password": pw}, follow_redirects=False)
    assert r.status_code == 303, r.text
    return c


@pytest.fixture(scope="session")
def admin(client):
    return login(client, "admin", ADMIN_PW)


@pytest.fixture(scope="session")
def users(admin):
    out = {}
    for role in ("uploader", "viewer"):
        r = admin.post("/api/admin/users", json={"username": f"u_{role}", "password": USER_PW, "role": role})
        assert r.status_code == 201, r.text
        out[role] = login(admin, f"u_{role}", USER_PW)
    return out


@pytest.fixture(scope="session")
def area(admin):
    r = admin.post("/api/admin/areas", json={"name": "Nguyễn Hữu Cảnh", "description": "Ven kênh"})
    assert r.status_code == 201
    return r.json()
