import time

import pytest

import cv2
import numpy as np
from fastapi.testclient import TestClient

from app.main import app
from tests.conftest import USER_PW, BrightModel, jpg, login


def test_pages_redirect_to_login_when_anonymous(client):
    r = TestClient(app).get("/areas", follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/login"
    assert TestClient(app).get("/api/areas").status_code == 401


def test_wrong_password_and_throttle(client):
    c = TestClient(app)
    for _ in range(5):
        assert c.post("/login", data={"username": "nobody", "password": "x"}).status_code == 401
    assert c.post("/login", data={"username": "nobody", "password": "x"}).status_code == 429


def test_roles(admin, users, area):
    v, u = users["viewer"], users["uploader"]
    assert v.get("/api/areas").status_code == 200
    assert v.get("/upload", follow_redirects=False).status_code == 403
    assert v.get("/admin").status_code == 403
    assert u.get("/admin").status_code == 403
    assert u.get("/upload").status_code == 200
    assert v.post("/api/admin/areas", json={"name": "x"}).status_code == 403
    assert admin.get("/admin").status_code == 200


def test_no_model_blocks_upload_but_app_runs(client, users, area):
    assert client.app.state.holder.model is None
    r = users["viewer"].get("/api/model").json()
    assert not r["loaded"] and "Chưa có model" in r["error"]
    r = users["uploader"].post("/api/jobs", data={"area_id": area["id"]}, files={"video": ("a.mp4", b"x", "video/mp4")})
    assert r.status_code == 503 and "Chưa có model" in r.json()["detail"]
    assert "Chưa có model" in client.app.state.holder.status()["error"]


def test_create_incident_via_device_key_and_view(client, users, area):
    files = [("image", ("a.jpg", jpg(), "image/jpeg")), ("image", ("b.jpg", jpg(90), "image/jpeg")), ("crop", ("c.jpg", jpg(), "image/jpeg"))]
    data = {"area_id": area["id"], "p": "0.93", "recorded_at": "2020-05-01T08:30:00Z"}
    assert TestClient(app).post("/api/incidents", data=data, files=files).status_code == 401
    r = TestClient(app).post("/api/incidents", data=data, files=files, headers={"X-API-Key": "wrong"})
    assert r.status_code == 401
    r = TestClient(app).post("/api/incidents", data=data, files=files, headers={"X-API-Key": "devkey"})
    assert r.status_code == 201, r.text
    inc = r.json()
    assert inc["recorded_at"] == "2020-05-01T08:30:00Z" and len(inc["evidence"]) == 2 and inc["status"] == "new"
    got = users["viewer"].get(f"/api/incidents/{inc['id']}").json()
    assert got["area"] == "Nguyễn Hữu Cảnh"
    img = users["viewer"].get(got["evidence"][0]["original"])
    assert img.status_code == 200 and img.headers["content-type"] == "image/jpeg"
    assert TestClient(app).get(got["evidence"][0]["original"]).status_code == 401      # ảnh cũng cần đăng nhập


def test_create_incident_validation(users, area):
    u = users["uploader"]
    ok = [("image", ("a.jpg", jpg(), "image/jpeg"))]
    assert users["viewer"].post("/api/incidents", data={"area_id": area["id"], "p": "0.9"}, files=ok).status_code == 403
    assert u.post("/api/incidents", data={"area_id": area["id"], "p": "1.5"}, files=ok).status_code == 422
    assert u.post("/api/incidents", data={"area_id": 9999, "p": "0.9"}, files=ok).status_code == 422
    assert u.post("/api/incidents", data={"area_id": area["id"], "p": "0.9"}, files=[("image", ("a.jpg", b"junk", "image/jpeg"))]).status_code == 422
    assert u.post("/api/incidents", data={"area_id": area["id"], "p": "0.9", "recorded_at": "hôm qua"}, files=ok).status_code == 422
    r = u.post("/api/incidents", data={"area_id": area["id"], "p": "0.9"})
    assert r.status_code == 422


def test_area_flooded_status_and_delete_rules(admin, users, area):
    a = next(x for x in users["viewer"].get("/api/areas").json()["areas"] if x["id"] == area["id"])
    assert a["flooded"] is False                    # sự cố ở test trước có occurred_at năm 2020, quá RECENT_HOURS
    r = users["uploader"].post("/api/incidents", data={"area_id": area["id"], "p": "0.8"}, files=[("image", ("a.jpg", jpg(), "image/jpeg"))])
    assert r.status_code == 201
    a = next(x for x in users["viewer"].get("/api/areas").json()["areas"] if x["id"] == area["id"])
    assert a["flooded"] and a["recent_count"] == 1
    assert admin.delete(f"/api/admin/areas/{area['id']}").status_code == 409
    assert users["uploader"].delete(f"/api/incidents/{r.json()['id']}").status_code == 403
    assert admin.delete(f"/api/incidents/{r.json()['id']}").status_code == 204


def test_admin_user_rules(admin):
    me = next(u for u in admin.get("/api/admin/users").json() if u["username"] == "admin")
    assert admin.patch(f"/api/admin/users/{me['id']}", json={"role": "viewer"}).status_code == 409
    assert admin.post("/api/admin/users", json={"username": "u_viewer", "password": USER_PW}).status_code == 409
    assert admin.post("/api/admin/users", json={"username": "ab", "password": USER_PW}).status_code == 422
    assert admin.post("/api/admin/users", json={"username": "okname", "password": "short"}).status_code == 422
    other = admin.post("/api/admin/users", json={"username": "locked", "password": USER_PW}).json()
    admin.patch(f"/api/admin/users/{other['id']}", json={"active": False})
    r = TestClient(app).post("/login", data={"username": "locked", "password": USER_PW})
    assert r.status_code == 401


def test_video_job_end_to_end(client, users, area, tmp_path, monkeypatch):
    client.app.state.holder.model = BrightModel()
    from app import config
    client.app.state.holder.cfg = config.ModelConfig()
    v = tmp_path / "v.mp4"
    w = cv2.VideoWriter(str(v), cv2.VideoWriter_fourcc(*"mp4v"), 10, (160, 120))
    for i in range(140):
        w.write(np.full((120, 160, 3), 235 if 40 <= i < 100 else 20, np.uint8))
    w.release()
    u = users["uploader"]
    r = u.post("/api/jobs", data={"area_id": area["id"]}, files={"video": ("clip.mp4", v.read_bytes(), "video/mp4")})
    assert r.status_code == 202, r.text
    jid = r.json()["id"]
    for _ in range(100):
        j = u.get(f"/api/jobs/{jid}").json()
        if j["status"] in ("done", "failed"):
            break
        time.sleep(0.2)
    assert j["status"] == "done" and j["incidents_found"] == 1, j
    incs = [i for i in u.get(f"/api/incidents?area_id={area['id']}").json() if i["source"] == "video"]
    assert len(incs) == 1
    detail = u.get(f"/api/incidents/{incs[0]['id']}").json()
    e0 = detail["evidence"][0]
    assert e0["overlay"] and e0["t_s"] is not None and u.get(e0["overlay"]).status_code == 200
    assert detail["video_start_s"] is not None and detail["model_name"] == "fake.pt" and detail["uploader"] == "u_uploader"
    assert detail["clip"] and u.get(detail["clip"]).status_code == 200
    assert detail["series"] and detail["series"][0][0] < 0.5 and len(detail["series"][0]) == 4
    assert not list((config.STORAGE_DIR / "videos").glob("*"))          # video gốc đã xoá
    assert login(client, "u_viewer", USER_PW).get(f"/api/jobs/{jid}").status_code == 403
    assert u.get("/api/jobs").json()[0]["id"] == jid
    r = u.post("/api/jobs", data={"area_id": area["id"]}, files={"video": ("notes.txt", b"hello", "text/plain")})
    assert r.status_code == 422
    client.app.state.holder.model = None


def test_bad_video_fails_job_cleanly(client, users, area):
    client.app.state.holder.model = BrightModel()
    from app import config
    client.app.state.holder.cfg = config.ModelConfig()
    u = users["uploader"]
    jid = u.post("/api/jobs", data={"area_id": area["id"]}, files={"video": ("bad.mp4", b"not a video at all", "video/mp4")}).json()["id"]
    for _ in range(50):
        j = u.get(f"/api/jobs/{jid}").json()
        if j["status"] in ("done", "failed"):
            break
        time.sleep(0.2)
    assert j["status"] == "failed" and "Không đọc được video" in j["error"]
    client.app.state.holder.model = None


def test_html_pages_render(admin, area):
    for path in ("/areas", f"/areas/{area['id']}", "/upload", "/admin", "/login", "/"):
        assert admin.get(path).status_code == 200, path


def test_status_change_admin_only_and_false_alarm_clears_area(admin, users, area):
    r = users["uploader"].post("/api/incidents", data={"area_id": area["id"], "p": "0.7"}, files=[("image", ("a.jpg", jpg(), "image/jpeg"))])
    iid = r.json()["id"]
    assert users["uploader"].patch(f"/api/incidents/{iid}", json={"status": "confirmed"}).status_code == 403
    assert admin.patch(f"/api/incidents/{iid}", json={"status": "bogus"}).status_code == 422
    assert admin.patch(f"/api/incidents/{iid}", json={"status": "confirmed"}).json()["status"] == "confirmed"
    flooded = lambda: next(x for x in users["viewer"].get("/api/areas").json()["areas"] if x["id"] == area["id"])["flooded"]
    assert flooded()
    for inc in users["viewer"].get(f"/api/incidents?area_id={area['id']}").json():
        admin.patch(f"/api/incidents/{inc['id']}", json={"status": "false_alarm"})
    assert not flooded()                         # báo nhầm thì khu vực không còn đỏ


def test_create_admin_command(admin):
    import manage
    manage.create_admin("second_admin", "longenough1")
    assert any(u["username"] == "second_admin" and u["role"] == "admin" for u in admin.get("/api/admin/users").json())
    import pytest
    with pytest.raises(SystemExit):
        manage.create_admin("second_admin", "longenough1")
    with pytest.raises(SystemExit):
        manage.create_admin("third", "short")


def test_areas_list_flooded_first_with_thumb_and_no_coordinates(admin, users, area):
    quiet = admin.post("/api/admin/areas", json={"name": "A Khu yên tĩnh"}).json()
    assert "lat" not in quiet and "lng" not in quiet
    assert admin.post("/api/admin/areas", json={"name": "Khu cũ", "description": "x"}).status_code == 201
    r = users["uploader"].post("/api/incidents", data={"area_id": area["id"], "p": "0.9"}, files=[("image", ("a.jpg", jpg(), "image/jpeg"))])
    assert r.status_code == 201
    areas = users["viewer"].get("/api/areas").json()["areas"]
    assert areas[0]["id"] == area["id"] and areas[0]["flooded"] and areas[0]["recent_count"] >= 1
    assert areas[0]["thumb"] and users["viewer"].get(areas[0]["thumb"]).status_code == 200
    assert areas[0]["last_at"]
    rest = [a for a in areas if not a["flooded"]]
    assert rest and all(a["thumb"] is None and a["last_at"] is None for a in rest if a["name"] == "A Khu yên tĩnh")
    one = users["viewer"].get(f"/api/areas/{area['id']}").json()
    assert one["flooded"] and users["viewer"].get("/api/areas/9999").status_code == 404
    assert users["viewer"].get("/map", follow_redirects=False).status_code == 404


def test_old_incident_shows_normal_but_keeps_last_time(admin, users):
    a = admin.post("/api/admin/areas", json={"name": "Khu từng ngập"}).json()
    ok = users["uploader"].post("/api/incidents", data={"area_id": a["id"], "p": "0.9", "recorded_at": "2020-01-01T00:00:00Z"},
                                files=[("image", ("a.jpg", jpg(), "image/jpeg"))])
    assert ok.status_code == 201
    row = next(x for x in users["viewer"].get("/api/areas").json()["areas"] if x["id"] == a["id"])
    assert not row["flooded"] and row["recent_count"] == 0 and row["last_at"] == "2020-01-01T00:00:00Z" and row["thumb"]


def test_admin_model_card_is_read_only_and_names_both_roles(admin, users):
    assert admin.post("/api/admin/model/reload").status_code in (404, 405)           # không có nút/API nạp lại
    st = admin.get("/api/admin/model").json()
    assert set(st) >= {"decide", "draw", "loaded", "error", "config"} and "name" in st["decide"] and "name" in st["draw"]
    assert users["viewer"].get("/api/admin/model").status_code == 403
    html = admin.get("/admin").text
    assert "Nạp lại model" not in html and "Quyết định ngập" in html and "Tô vùng nước" in html


def test_upload_picks_up_new_decider_without_restart(client, users, area, monkeypatch):
    """Có model quyết định sau khi app đã chạy (chép best.pt classify): lần kiểm tra kế tiếp dùng được ngay."""
    import os
    from app import config
    holder = client.app.state.holder
    assert holder.model is None
    r = users["viewer"].get("/api/model").json()
    assert not r["loaded"] and "Chưa có model" in r["error"] and set(r) == {"loaded", "error"}

    def fake_build(cfg, d):
        if not (d / "best.pt").exists():
            raise FileNotFoundError(config.NO_MODEL_MSG)
        return BrightModel()
    monkeypatch.setattr(holder, "decider_builder", fake_build)
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    (config.MODELS_DIR / "best.pt").write_bytes(b"x")
    os.utime(config.MODELS_DIR / "best.pt", (1_900_000_000, 1_900_000_000))
    try:
        assert users["viewer"].get("/api/model").json()["loaded"] is True
        assert holder.model is not None
    finally:
        (config.MODELS_DIR / "best.pt").unlink()
        holder.refresh()
        assert holder.model is None


def _upload_and_wait(client, user, area_id, video_bytes, name="v.mp4"):
    client.app.state.holder.model = BrightModel()
    from app import config
    client.app.state.holder.cfg = config.ModelConfig()
    jid = user.post("/api/jobs", data={"area_id": area_id}, files={"video": (name, video_bytes, "video/mp4")}).json()["id"]
    for _ in range(100):
        j = user.get(f"/api/jobs/{jid}").json()
        if j["status"] in ("done", "failed"):
            return j
        time.sleep(0.2)
    raise AssertionError("job không xong")


def _video(path, bright=(), secs=10):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (160, 120))
    for i in range(secs * 10):
        w.write(np.full((120, 160, 3), 235 if any(a <= i / 10 < b for a, b in bright) else 20, np.uint8))
    w.release()
    return path.read_bytes()


def test_check_recorded_for_flooded_and_dry_video(client, users, area, admin, tmp_path):
    u, viewer = users["uploader"], users["viewer"]
    dry = admin.post("/api/admin/areas", json={"name": "Khu kiểm tra khô"}).json()
    j = _upload_and_wait(client, u, dry["id"], _video(tmp_path / "dry.mp4"))
    assert j["status"] == "done" and j["incidents_found"] == 0 and j["incident_ids"] == [] and j["check_id"]
    c = viewer.get(f"/api/checks/{j['check_id']}").json()
    assert c["flooded"] is False and c["incident_ids"] == [] and c["uploader"] == "u_uploader" and c["area"] == "Khu kiểm tra khô"
    assert c["p_max"] < 0.2 and c["series"] and all(row[2] == 0 for row in c["series"]) and all(len(row) == 4 for row in c["series"]) and c["model_name"] == "fake.pt"
    v = viewer.get(c["video"])
    assert v.status_code == 200 and v.headers["content-type"] == "video/mp4" and len(v.content) > 500
    assert TestClient(app).get(c["video"]).status_code == 401                 # video chỉ cho người đã đăng nhập
    assert viewer.get(c["video"], headers={"Range": "bytes=0-99"}).status_code == 206

    j2 = _upload_and_wait(client, u, dry["id"], _video(tmp_path / "wet.mp4", bright=((3, 8),)))
    assert j2["incidents_found"] == 1
    c2 = viewer.get(f"/api/checks/{j2['check_id']}").json()
    assert c2["flooded"] is True and c2["incident_ids"] == j2["incident_ids"] and c2["p_max"] > 0.8
    inc = viewer.get(f"/api/incidents/{c2['incident_ids'][0]}").json()
    assert inc["check_id"] == c2["id"] and inc["clip"] == c2["video"] and inc["series"] == c2["series"] and inc["clip_start"] >= 2
    checks = viewer.get(f"/api/checks?area_id={dry['id']}").json()
    assert [x["id"] for x in checks] == [c2["id"], c["id"]] and [x["flooded"] for x in checks] == [True, False]   # mới nhất trước, cả ngập lẫn không
    a = next(x for x in viewer.get("/api/areas").json()["areas"] if x["id"] == dry["id"])
    assert a["last_check_at"] and a["flooded"] is True
    client.app.state.holder.model = None
    assert viewer.get("/api/checks/99999").status_code == 404
    assert users["viewer"].get("/api/jobs/" + j["id"]).status_code == 403


def test_retention_deletes_old_result_videos_but_keeps_records(client, users, area, admin, tmp_path):
    from datetime import timedelta
    from app import config, retention
    from app.db import SessionLocal
    from app.models import Check, utcnow
    ar = admin.post("/api/admin/areas", json={"name": "Khu retention"}).json()
    j = _upload_and_wait(client, users["uploader"], ar["id"], _video(tmp_path / "w.mp4", bright=((2, 8),)))
    client.app.state.holder.model = None
    cid, iid = j["check_id"], j["incident_ids"][0]
    video = config.STORAGE_DIR / f"checks/{j['id']}.mp4"
    assert video.exists()
    with SessionLocal() as db:
        assert retention.purge_old_videos(db, days=30) == 0 and video.exists()          # còn mới: giữ
        db.get(Check, cid).created_at = utcnow() - timedelta(days=31)
        db.commit()
        assert retention.purge_old_videos(db, days=0) == 0                              # days=0: tắt dọn dẹp
        assert retention.purge_old_videos(db, days=30) == 1
    assert not video.exists()
    c = users["viewer"].get(f"/api/checks/{cid}").json()
    assert c["video"] is None and c["series"] and c["flooded"]                         # bản ghi + dãy t-p còn
    assert users["viewer"].get(f"/api/checks/{cid}/video").status_code == 404
    inc = users["viewer"].get(f"/api/incidents/{iid}").json()
    assert inc["clip"] is None and inc["evidence"] and users["viewer"].get(inc["evidence"][0]["original"]).status_code == 200


def test_check_pages_render(admin, area):
    existing = admin.get("/api/checks?limit=1").json()
    if existing:
        assert admin.get(f"/checks/{existing[0]['id']}").status_code == 200
    assert admin.get("/checks/99999").status_code == 404


def _mk_check(db, area_id, rows, p_max, incidents=()):
    """Tạo lần kiểm tra 'cũ' trực tiếp trong DB: rows = series đã lưu (3 hoặc 4 cột)."""
    import json
    from app.models import Check, Incident
    c = Check(area_id=area_id, flooded=bool(incidents), p_max=p_max, series_json=json.dumps(rows), duration_s=len(rows) / 5)
    db.add(c)
    db.flush()
    for (a, b, p) in incidents:
        db.add(Incident(area_id=area_id, p_max=p, video_start_s=a, video_end_s=b, check_id=c.id, source="video"))
    db.commit()
    return c


def _legacy_rows(raw):
    """Dãy p thô -> [[t, p, ngập]] đúng như bản cũ đã lưu (trạng thái theo luật hiện tại)."""
    from app.rules import Hysteresis
    h = Hysteresis(2, 5, 0.8, 0.6)
    return [[round(i * 0.2, 2), p, int(h.update(p))] for i, p in enumerate(raw)]


def test_legacy_records_get_smoothed_p_recomputed_or_noted(client, users, area, admin):
    from app import config
    from app.db import SessionLocal
    from app.models import Check, Incident
    from app.rules import Hysteresis
    from app.series import P_NOTE, recompute_legacy
    cfg = config.ModelConfig(flood_on=0.8, window_s=2.0)          # luật cũ của bản ghi giả
    raw = [0.1] * 10 + [0.99, 0.3, 0.98, 0.97, 0.96] + [0.95] * 30 + [0.2] * 10            # có đỉnh/trũng lẻ để thấy khác biệt
    rows = _legacy_rows(raw)
    h = Hysteresis(2, 5, 0.8, 0.6)
    expect = [round((h.update(p), h.avg)[1], 3) for p in raw]
    flooded = [r for r in rows if r[2]]
    a, b = flooded[0][0], flooded[-1][0] + 0.2
    with SessionLocal() as db:
        ok = _mk_check(db, area["id"], rows, p_max=max(raw), incidents=[(a, b, max(raw))])
        bad_rows = [list(r) for r in rows]
        bad_rows[20][2] = 1 - bad_rows[20][2]                                              # trạng thái đã lưu không khớp luật hiện tại
        bad = _mk_check(db, area["id"], bad_rows, p_max=0.99, incidents=[(a, b, 0.99)])
        empty = _mk_check(db, area["id"], [], p_max=0.7)
        done_rows = [r + [0.5] for r in rows[:3]]
        new = _mk_check(db, area["id"], done_rows, p_max=0.5)
        ids = (ok.id, bad.id, empty.id, new.id)
    with SessionLocal() as db:
        r1 = recompute_legacy(db, cfg)
    assert r1["recomputed"] >= 1 and r1["kept_raw"] >= 1 and set(r1) == {"recomputed", "aligned", "kept_raw"}
    v = users["viewer"]
    c_ok = v.get(f"/api/checks/{ids[0]}").json()
    assert all(len(r) == 4 for r in c_ok["series"]) and [r[3] for r in c_ok["series"]] == expect and [r[1] for r in c_ok["series"]] == raw   # thô còn nguyên
    assert c_ok["p_max"] == pytest.approx(max(expect), abs=1e-3) and c_ok["p_max"] < max(raw) and c_ok["p_note"] is None
    inc = v.get(f"/api/incidents?area_id={area['id']}").json()
    mine = [i for i in inc if i["check_id"] == ids[0]][0]
    assert mine["p_max"] == pytest.approx(max(r[3] for r in c_ok["series"] if a <= r[0] < b), abs=1e-3) and mine["p_note"] is None
    c_bad = v.get(f"/api/checks/{ids[1]}").json()
    assert all(len(r) == 3 for r in c_bad["series"]) and c_bad["p_max"] == 0.99 and c_bad["p_note"] == P_NOTE                       # để nguyên + ghi chú
    bad_inc = [i for i in inc if i["check_id"] == ids[1]][0]
    assert bad_inc["p_max"] == 0.99 and bad_inc["p_note"] == P_NOTE
    assert v.get(f"/api/incidents/{bad_inc['id']}").json()["p_note"] == P_NOTE
    assert v.get(f"/api/checks/{ids[2]}").json()["p_note"] == P_NOTE and v.get(f"/api/checks/{ids[2]}").json()["p_max"] == 0.7      # rỗng: không làm gì
    assert v.get(f"/api/checks/{ids[3]}").json()["p_max"] == 0.5                                                                     # đã có p làm mượt và p_max khớp: giữ nguyên
    with SessionLocal() as db:
        snap = [(c.id, c.series_json, c.p_max) for c in db.query(Check).order_by(Check.id)] + [(i.id, i.p_max) for i in db.query(Incident).order_by(Incident.id)]
        recompute_legacy(db, cfg)
        again = [(c.id, c.series_json, c.p_max) for c in db.query(Check).order_by(Check.id)] + [(i.id, i.p_max) for i in db.query(Incident).order_by(Incident.id)]
    assert snap == again                                                                                                           # chạy lại không đổi gì
    inc_dev = users["uploader"].post("/api/incidents", data={"area_id": area["id"], "p": "0.77"}, files=[("image", ("a.jpg", jpg(), "image/jpeg"))]).json()
    assert inc_dev["p_max"] == 0.77 and inc_dev["p_note"] is None                                                                    # thiết bị: p do thiết bị gửi, không ghi chú


def test_new_analysis_stores_smoothed_p_max(client, users, area, admin, tmp_path):
    ar = admin.post("/api/admin/areas", json={"name": "Khu p làm mượt"}).json()
    w = cv2.VideoWriter(str(tmp_path / "s.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 10, (160, 120))
    for i in range(100):
        t = i / 10
        w.write(np.full((120, 160, 3), 255 if 4.0 <= t < 4.2 else (200 if 2 <= t < 9 else 20), np.uint8))
    w.release()
    j = _upload_and_wait(client, users["uploader"], ar["id"], (tmp_path / "s.mp4").read_bytes())
    client.app.state.holder.model = None
    c = users["viewer"].get(f"/api/checks/{j['check_id']}").json()
    assert all(len(r) == 4 for r in c["series"]) and c["p_note"] is None
    assert c["p_max"] == pytest.approx(max(r[3] for r in c["series"]), abs=1e-3) and c["p_max"] < max(r[1] for r in c["series"])
    for iid in j["incident_ids"]:
        inc = users["viewer"].get(f"/api/incidents/{iid}").json()
        assert inc["p_note"] is None and inc["p_max"] <= c["p_max"] + 1e-6


def test_checks_from_previous_build_get_p_max_aligned_to_smoothed(client, users, area):
    """Bản trước đã lưu dãy 4 cột nhưng p_max vẫn là p thô: căn chỉnh về p làm mượt lớn nhất, cả sự cố thuộc nó."""
    from app import config
    from app.db import SessionLocal
    from app.series import recompute_legacy
    rows = [[round(i * 0.2, 2), 0.99 if i == 5 else 0.5, 0, 0.4 + (0.01 * i if i < 12 else 0.12)] for i in range(20)]
    with SessionLocal() as db:
        c = _mk_check(db, area["id"], rows, p_max=0.99, incidents=[(0.0, 2.0, 0.99)])
        cid = c.id
        r = recompute_legacy(db, config.ModelConfig())
    assert r["aligned"] >= 1
    got = users["viewer"].get(f"/api/checks/{cid}").json()
    assert got["p_max"] == pytest.approx(max(x[3] for x in rows), abs=1e-3) and got["p_note"] is None
    inc = [i for i in users["viewer"].get(f"/api/incidents?area_id={area['id']}").json() if i["check_id"] == cid][0]
    assert inc["p_max"] == pytest.approx(max(x[3] for x in rows if x[0] < 2.0), abs=1e-3)
