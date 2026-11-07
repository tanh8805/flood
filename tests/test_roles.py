"""Phân quyền theo vai trò (admin / uploader / viewer): chuyển trang sau đăng nhập, menu, trang HTML và API (chặn ở backend, không chỉ ẩn nút)."""
import json

import pytest
from fastapi.testclient import TestClient

from app.main import app
from tests.conftest import ADMIN_PW, USER_PW, login


@pytest.fixture(scope="module")
def world(admin, users):
    """Hai uploader (up1 = u_uploader, up2), 2 khu vực, 2 lần kiểm tra (mỗi uploader 1), 1 sự cố mới + 1 sự cố đã báo nhầm."""
    from app.db import SessionLocal
    from app.models import Area, Check, Incident, User
    assert admin.post("/api/admin/users", json={"username": "u_uploader2", "password": USER_PW, "role": "uploader"}).status_code == 201
    a1 = admin.post("/api/admin/areas", json={"name": "Phân quyền A"}).json()
    a2 = admin.post("/api/admin/areas", json={"name": "Phân quyền B"}).json()
    with SessionLocal() as db:
        up1 = db.query(User).filter_by(username="u_uploader").one().id
        up2 = db.query(User).filter_by(username="u_uploader2").one().id
        c1 = Check(area_id=a1["id"], uploader_id=up1, flooded=True, p_max=0.95, series_json=json.dumps([[0, 0.9, 1, 0.9]]), duration_s=1)
        c2 = Check(area_id=a2["id"], uploader_id=up2, flooded=False, p_max=0.1, series_json=json.dumps([[0, 0.1, 0, 0.1]]), duration_s=1)
        db.add_all([c1, c2]); db.flush()
        i_new = Incident(area_id=a1["id"], p_max=0.95, video_start_s=0, video_end_s=2, check_id=c1.id, source="video", status="new")
        i_bad = Incident(area_id=a1["id"], p_max=0.9, video_start_s=0, video_end_s=2, check_id=c2.id, source="video", status="false_alarm")
        db.add_all([i_new, i_bad]); db.commit()
        w = {"a1": a1["id"], "a2": a2["id"], "c1": c1.id, "c2": c2.id, "i_new": i_new.id, "i_bad": i_bad.id}
    w["up2"] = login(admin, "u_uploader2", USER_PW)
    return w


def test_login_redirects_by_role(client, users, admin):
    for who, user, pw, home in (("admin", "admin", ADMIN_PW, "/overview"), ("uploader", "u_uploader", USER_PW, "/upload"), ("viewer", "u_viewer", USER_PW, "/areas")):
        c = TestClient(app)
        r = c.post("/login", data={"username": user, "password": pw}, follow_redirects=False)
        assert r.status_code == 303 and r.headers["location"] == home, who
        assert c.get("/", follow_redirects=False).headers["location"] == home
        assert c.get("/login", follow_redirects=False).headers["location"] == home
    assert TestClient(app).get("/", follow_redirects=False).headers["location"] == "/login"


def _links(html):
    import re
    nav = re.search(r"<nav.*?</nav>", html, re.S).group(0)
    return re.findall(r'href="([^"]+)"', nav)


def test_menu_per_role(users, admin, world):
    assert _links(admin.get("/overview").text) == ["/overview", "/incidents", "/admin", "/upload"]
    assert _links(users["uploader"].get("/upload").text) == ["/upload", "/areas"]
    assert _links(users["viewer"].get("/areas").text) == ["/areas"]


@pytest.mark.parametrize("path,allowed", [("/overview", {"admin"}), ("/incidents", {"admin"}), ("/admin", {"admin"}), ("/upload", {"admin", "uploader"}), ("/areas", {"admin", "uploader", "viewer"})])
def test_html_pages_by_role(users, admin, world, path, allowed):
    for role in ("admin", "uploader", "viewer"):
        assert users[role].get(path, follow_redirects=False).status_code == (200 if role in allowed else 403), (role, path)
    r = TestClient(app).get(path, follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/login"            # chưa đăng nhập: về trang đăng nhập


def test_checks_pages_and_data_are_scoped(users, admin, world):
    up1, up2, viewer = users["uploader"], world["up2"], users["viewer"]
    # trang chi tiết lần kiểm tra
    assert admin.get(f"/checks/{world['c1']}").status_code == 200 and admin.get(f"/checks/{world['c2']}").status_code == 200
    assert up1.get(f"/checks/{world['c1']}").status_code == 200 and up1.get(f"/checks/{world['c2']}").status_code == 403
    assert up2.get(f"/checks/{world['c2']}").status_code == 200 and up2.get(f"/checks/{world['c1']}").status_code == 403
    assert viewer.get(f"/checks/{world['c1']}").status_code == 403
    # API
    assert up1.get(f"/api/checks/{world['c1']}").status_code == 200 and up1.get(f"/api/checks/{world['c2']}").status_code == 403
    assert viewer.get(f"/api/checks/{world['c1']}").status_code == 403 and viewer.get("/api/checks").status_code == 403
    mine = {c["id"] for c in up1.get("/api/checks?limit=200").json()}
    assert world["c1"] in mine and world["c2"] not in mine                                   # uploader chỉ thấy lần kiểm tra của mình
    assert {world["c1"], world["c2"]} <= {c["id"] for c in admin.get("/api/checks?limit=200").json()}
    assert world["c2"] not in {c["id"] for c in up1.get(f"/api/checks?area_id={world['a2']}&limit=200").json()}


def test_check_video_permissions(users, admin, world):
    up1, viewer = users["uploader"], users["viewer"]
    ok = lambda r: r.status_code != 403                                                      # qua cửa quyền (file không có thì 404)
    assert ok(admin.get(f"/api/checks/{world['c1']}/video")) and ok(admin.get(f"/api/checks/{world['c2']}/video"))
    assert ok(up1.get(f"/api/checks/{world['c1']}/video")) and up1.get(f"/api/checks/{world['c2']}/video").status_code == 403
    assert ok(viewer.get(f"/api/checks/{world['c1']}/video"))                                # có sự cố chưa báo nhầm: viewer xem được (trang sự cố cần)
    assert viewer.get(f"/api/checks/{world['c2']}/video").status_code == 403                 # chỉ có sự cố báo nhầm: không


def test_status_change_and_delete_are_admin_only(users, admin, world):
    for role in ("uploader", "viewer"):
        assert users[role].patch(f"/api/incidents/{world['i_new']}", json={"status": "confirmed"}).status_code == 403
        assert users[role].delete(f"/api/incidents/{world['i_new']}").status_code == 403
    assert admin.get(f"/api/incidents/{world['i_new']}").json()["status"] == "new"
    for role in ("uploader", "viewer"):                                                      # xem chi tiết sự cố thì được
        assert users[role].get(f"/api/incidents/{world['i_new']}").status_code == 200
        assert users[role].get(f"/incidents/{world['i_new']}").status_code == 200


def test_incident_page_buttons_only_for_admin(users, admin, world):
    assert 'id="confirm"' in admin.get(f"/incidents/{world['i_new']}").text
    for role in ("uploader", "viewer"):
        assert 'id="confirm"' not in users[role].get(f"/incidents/{world['i_new']}").text


def test_area_page_checks_section(users, admin, world):
    assert 'id="checks"' in admin.get(f"/areas/{world['a1']}").text and 'id="checks"' in users["uploader"].get(f"/areas/{world['a1']}").text
    assert 'id="checks"' not in users["viewer"].get(f"/areas/{world['a1']}").text


def test_model_and_jobs_and_admin_api_blocked_for_viewer(users, admin, world):
    v = users["viewer"]
    assert v.get("/api/model").status_code == 403 and v.get("/api/jobs").status_code == 403
    assert v.post("/api/jobs", data={"area_id": world["a1"]}, files={"video": ("x.mp4", b"x", "video/mp4")}).status_code == 403
    for path in ("/api/admin/overview", "/api/admin/users", "/api/admin/areas", "/api/admin/model"):
        assert v.get(path).status_code == 403 and users["uploader"].get(path).status_code == 403, path
        assert admin.get(path).status_code == 200, path
    assert users["uploader"].get("/api/model").status_code == 200 and users["uploader"].get("/api/jobs").status_code == 200
    assert users["uploader"].post("/api/admin/areas", json={"name": "x"}).status_code == 403


def test_my_jobs_only_own(users, admin, world):
    from app.db import SessionLocal
    from app.models import Job, User
    with SessionLocal() as db:
        up2 = db.query(User).filter_by(username="u_uploader2").one().id
        db.add(Job(id="roletest0000000000000000000000a", user_id=up2, area_id=world["a2"], filename="job-cua-up2.mp4", status="done", message="xong"))
        db.commit()
    names = lambda c: {j["filename"] for j in c.get("/api/jobs").json()}
    assert "job-cua-up2.mp4" in names(world["up2"]) and "job-cua-up2.mp4" not in names(users["uploader"]) and "job-cua-up2.mp4" in names(admin)


def test_admin_overview_content_and_actions(admin, world):
    o = admin.get("/api/admin/overview").json()
    assert set(o) == {"recent_hours", "areas_total", "flooded_areas", "new_incidents_total", "new_incidents", "recent_checks", "model"}
    ids = [i["id"] for i in o["new_incidents"]]
    assert world["i_new"] in ids and world["i_bad"] not in ids and o["new_incidents_total"] >= len(ids) >= 1
    assert all(i["status"] == "new" for i in o["new_incidents"])
    assert {world["c1"], world["c2"]} <= {c["id"] for c in o["recent_checks"]} and {"id", "area", "uploader", "flooded", "created_at", "p_max"} <= set(o["recent_checks"][0])
    assert world["a1"] in [a["id"] for a in o["flooded_areas"]] and "loaded" in o["model"]
    r = admin.patch(f"/api/incidents/{world['i_new']}", json={"status": "confirmed"})           # Xác nhận ngay tại chỗ
    assert r.status_code == 200
    o2 = admin.get("/api/admin/overview").json()
    assert world["i_new"] not in [i["id"] for i in o2["new_incidents"]] and o2["new_incidents_total"] == o["new_incidents_total"] - 1
    assert admin.patch(f"/api/incidents/{world['i_new']}", json={"status": "new"}).status_code == 200
