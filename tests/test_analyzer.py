import cv2
import numpy as np
import pytest

from app import config
from app.analyzer import analyze_video, crop_to_mask, draw_overlay
from app.rules import Hysteresis, merge_spans, pick_evidence_times, segment_flags
from app.video import Video


def test_hysteresis_exposes_smoothed_value():
    h = Hysteresis(2, 5, 0.8, 0.6)                                           # cửa sổ 10 mẫu
    assert h.update(1.0) is True and h.avg == pytest.approx(1.0)            # mẫu đầu: trung bình 1 mẫu, 1.0 >= 0.8 nên bật
    h.update(0.0)
    assert h.avg == pytest.approx(0.5) and h.state is False                  # 0.5 < tắt 0.6 -> tắt; avg là giá trị đã so với ngưỡng
    for _ in range(20):
        h.update(0.2)
    assert h.avg == pytest.approx(0.2) and len(h.buf) == 10


def test_hysteresis_on_off():
    h = Hysteresis(2, 5, 0.8, 0.6)             # cửa sổ 10 mẫu
    for _ in range(10):
        assert not h.update(0.7)               # 0.7 < 0.8 chưa bật
    states = [h.update(0.95) for _ in range(10)]
    assert states[-1] and not states[0]
    h2 = Hysteresis(2, 5, 0.8, 0.6)
    for _ in range(10):
        h2.update(0.9)
    assert h2.update(0.65) and h2.state        # trung bình ~0.875 -> vẫn bật
    for _ in range(10):
        h2.update(0.1)
    assert not h2.state


def test_segments_and_merge():
    times = [i * 0.2 for i in range(50)]       # 10 s, 5 đoạn
    states = [2 <= t < 6 or t >= 8 for t in times]
    flags = segment_flags(times, states, 2.0)
    assert flags == {0: False, 1: True, 2: True, 3: False, 4: True}
    spans = merge_spans(flags, 2.0, 10.0)
    assert [(s.t_start, s.t_end) for s in spans] == [(2.0, 6.0), (8.0, 10.0)]
    assert merge_spans({0: False}, 2.0, 2.0) == []
    assert merge_spans({0: True, 1: True}, 2.0, 3.5)[0].t_end == 3.5


def test_pick_evidence_min_gap_and_max3():
    samples = [(i * 0.2, 0.5 + (0.01 if i % 2 else 0)) for i in range(50)]
    samples[10] = (2.0, 0.99)
    samples[12] = (2.4, 0.98)      # cách 2.0 chỉ 0.4 s -> bị loại
    picked = pick_evidence_times(samples, 0, 10)
    assert len(picked) <= 3 and (2.0, 0.99) in picked and (2.4, 0.98) not in picked
    assert picked == sorted(picked)
    assert all(b[0] - a[0] >= 1.0 - 1e-9 for a, b in zip(picked, picked[1:]))


def test_crop_and_overlay():
    f = np.full((100, 200, 3), 50, np.uint8)
    assert crop_to_mask(f, None) is f
    m = np.zeros((100, 200), bool)
    m[40:60, 100:140] = True                    # 20x40 -> nới 10% = 2 và 4 px
    c = crop_to_mask(f, m)
    assert c.shape[:2] == (24, 48)
    o = draw_overlay(f, m)
    assert not np.array_equal(o, f)


class BrightModel:
    """Giả: p = độ sáng khung; mask = nửa dưới khi sáng."""
    kind = "fake"

    def predict(self, frame):
        p = float(frame.mean()) / 255
        mask = None
        if p > 0.5:
            mask = np.zeros(frame.shape[:2], bool)
            mask[frame.shape[0] // 2:, :] = True
        return {"p_ngap": p, "mask": mask}


def make_video(path, bright=((4, 10),), secs=14, fps=10, size=(160, 120)):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    for i in range(secs * fps):
        t = i / fps
        v = 235 if any(a <= t < b for a, b in bright) else 20
        w.write(np.full((size[1], size[0], 3), v, np.uint8))
    w.release()


def test_analyze_video_one_incident(tmp_path):
    v = tmp_path / "a.mp4"
    make_video(v)
    res, info = analyze_video(v, BrightModel(), config.ModelConfig())
    assert len(res) == 1
    r = res[0]
    assert 4 <= r.t_start <= 6 and 8 <= r.t_end <= 12
    assert r.p_max == pytest.approx(235 / 255, abs=0.03)
    assert 1 <= len(r.evidence) <= 3
    e = r.evidence[0]
    assert e.overlay is not None and cv2.imdecode(np.frombuffer(e.crop, np.uint8), 1).shape[0] < 120


def test_analyze_video_dry(tmp_path):
    v = tmp_path / "d.mp4"
    make_video(v, bright=())
    res, info = analyze_video(v, BrightModel(), config.ModelConfig())
    assert res == [] and info["flooded_segments"] == 0


def test_two_incidents(tmp_path):
    v = tmp_path / "b.mp4"
    make_video(v, bright=((2, 6), (10, 16)), secs=18)
    res, _ = analyze_video(v, BrightModel(), config.ModelConfig())
    assert len(res) == 2


def test_unreadable_video_raises_value_error(tmp_path):
    f = tmp_path / "bad.mp4"
    f.write_bytes(b"not a video")
    with pytest.raises(ValueError, match="Không đọc được video"):
        Video(f)


def _frames(path):
    cap = cv2.VideoCapture(str(path))
    out = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        out.append(fr)
    return out


class NoMaskModel(BrightModel):
    def predict(self, frame):
        return {"p_ngap": float(frame.mean()) / 255, "mask": None}


def test_result_video_covers_whole_video_and_draws_water_mask_but_no_text(tmp_path):
    v, out = tmp_path / "a.mp4", tmp_path / "result.mp4"
    make_video(v)                                                # 14 s, ngập ở 4-10 s
    res, info = analyze_video(v, BrightModel(), config.ModelConfig(), video_out=out)
    assert info["video"] and out.exists()
    frames = _frames(out)
    assert len(frames) >= 130 and frames[0].shape[1] <= 854      # toàn bộ video, không chỉ đoạn ngập
    mid = frames[70]                                             # 7 s: đang ngập, model có mask nửa dưới
    h = mid.shape[0]
    top, bottom = mid[: h // 4].astype(int), mid[3 * h // 4:].astype(int)
    assert abs(top.mean() - bottom.mean()) > 20
    assert top.std() < 10 and bottom[:, :, 0].std() < 10         # nền phẳng: không có chữ/banner
    dry = frames[10].astype(int)                                 # 1 s: không ngập, không mask -> không vẽ gì
    assert dry.std() < 8 and abs(dry[: h // 4].mean() - dry[3 * h // 4:].mean()) < 8


def test_result_video_unchanged_for_model_without_mask(tmp_path):
    v, out = tmp_path / "a.mp4", tmp_path / "result.mp4"
    make_video(v)
    analyze_video(v, NoMaskModel(), config.ModelConfig(), video_out=out)
    assert _frames(out)[70].astype(int).std() < 8


def test_no_flood_still_gives_video_and_series(tmp_path):
    v, out = tmp_path / "d.mp4", tmp_path / "result.mp4"
    make_video(v, bright=())
    res, info = analyze_video(v, BrightModel(), config.ModelConfig(), video_out=out)
    assert res == [] and info["video"] and out.exists() and len(_frames(out)) >= 130
    assert info["p_max"] < 0.2 and all(row[2] == 0 for row in info["series"])


def test_series_covers_every_inference_with_raw_and_smoothed(tmp_path):
    v = tmp_path / "a.mp4"
    make_video(v)
    cfg = config.ModelConfig()
    _, info = analyze_video(v, BrightModel(), cfg)
    s = info["series"]
    assert all(len(row) == 4 for row in s)                                   # [t, p thô, ngập, p làm mượt]
    assert s[0][0] == 0 and all(b[0] > a[0] for a, b in zip(s, s[1:]))
    assert all(0 <= row[1] <= 1 and row[2] in (0, 1) and 0 <= row[3] <= 1 for row in s)
    assert len(s) == pytest.approx(14 * 5, abs=3)                            # 5 lần suy luận mỗi giây, cả video
    assert s[0][2] == 0 and s[-1][2] == 0 and any(row[2] for row in s)
    assert info["p_max"] == pytest.approx(max(row[1] for row in s), abs=0.001)   # p_max vẫn là p thô lớn nhất
    n = int(cfg.window_s * cfg.infer_fps)
    for i, row in enumerate(s):                                              # p làm mượt = trung bình trượt n mẫu p thô gần nhất
        win = [r[1] for r in s[max(0, i - n + 1): i + 1]]
        assert row[3] == pytest.approx(sum(win) / len(win), abs=0.004)
    assert any(abs(row[1] - row[3]) > 0.05 for row in s)                     # làm mượt có tác dụng thật (khác p thô)


def test_smoothed_value_is_what_the_on_off_rule_sees(tmp_path):
    """Nhãn luôn nhất quán với số hiển thị: p làm mượt >= bật -> Ngập; < tắt -> Không ngập; giữa hai ngưỡng -> giữ nguyên trạng thái trước."""
    v = tmp_path / "a.mp4"
    make_video(v)
    cfg = config.ModelConfig()
    _, info = analyze_video(v, BrightModel(), cfg)
    prev = 0
    for t, raw, on, smooth in info["series"]:
        if smooth >= cfg.flood_on + 0.002:
            assert on == 1
        elif smooth < cfg.flood_off - 0.002:
            assert on == 0
        else:
            assert on == prev or abs(smooth - cfg.flood_on) <= 0.002 or abs(smooth - cfg.flood_off) <= 0.002
        prev = on


def test_partial_file_removed_when_model_fails(tmp_path):
    class Boom:
        def predict(self, f):
            raise RuntimeError("model hỏng")
    v, out = tmp_path / "a.mp4", tmp_path / "result.mp4"
    make_video(v, secs=3)
    with pytest.raises(RuntimeError):
        analyze_video(v, Boom(), config.ModelConfig(), video_out=out)
    assert not out.exists()


def test_mask_shared_between_inferences_but_never_older_than_half_a_second(tmp_path):
    v, out = tmp_path / "a.mp4", tmp_path / "result.mp4"
    make_video(v)                                              # 10 fps, ngập (mask nửa dưới) từ 4 s
    analyze_video(v, BrightModel(), config.ModelConfig(infer_fps=1.0), video_out=out)   # suy luận mỗi 1 s: giữa hai lần có khoảng trống 1 s
    frames = _frames(out)

    def drawn(i):
        h = frames[i].shape[0]
        return abs(frames[i][: h // 4].astype(int).mean() - frames[i][3 * h // 4:].astype(int).mean()) > 20
    assert drawn(60) and drawn(62) and drawn(64)               # 6.0 s suy luận; 6.2-6.4 s dùng chung mask đó
    assert not drawn(57) and not drawn(58)                     # 5.7-5.8 s: mask của 5.0 s đã cũ hơn 0.5 s -> không vẽ
    assert drawn(70)                                           # 7.0 s: lần suy luận mới


def make_levels(path, levels, secs=14, fps=10, size=(160, 120)):
    """levels: [(t0, t1, độ sáng)]; nền 20."""
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    for i in range(secs * fps):
        t = i / fps
        v = next((val for a, b, val in levels if a <= t < b), 20)
        w.write(np.full((size[1], size[0], 3), v, np.uint8))
    w.release()


def test_p_max_is_smoothed_not_raw_spike(tmp_path):
    v = tmp_path / "spike.mp4"
    make_levels(v, [(5.0, 5.2, 255)])                                    # một cú nhảy 0.2 s giữa nền tối
    res, info = analyze_video(v, BrightModel(), config.ModelConfig())
    raw = [r[1] for r in info["series"]]
    sm = [r[3] for r in info["series"]]
    assert info["p_raw_max"] == pytest.approx(max(raw), abs=1e-3) and max(raw) > 0.85          # p thô có đỉnh cao
    assert info["p_max"] == pytest.approx(max(sm), abs=1e-3) and info["p_max"] < 0.3    # p_max hiển thị là p làm mượt: đỉnh bị dàn mỏng
    assert res == []


def test_incident_confidence_is_max_smoothed_inside_the_span(tmp_path):
    v = tmp_path / "wet.mp4"
    make_levels(v, [(7.0, 7.4, 255), (4.0, 10.0, 215)])                  # ngập ổn định (p~0.84) + một đoạn chói ngắn
    res, info = analyze_video(v, BrightModel(), config.ModelConfig(flood_on=0.8, window_s=2.0))
    assert len(res) == 1
    r, s = res[0], info["series"]
    inside = [row for row in s if r.t_start <= row[0] < r.t_end]
    assert r.p_max == pytest.approx(max(row[3] for row in inside), abs=1e-3)
    assert max(row[1] for row in inside) > r.p_max + 0.01                    # p thô cao hơn p làm mượt lớn nhất
    assert info["p_max"] == pytest.approx(max(row[3] for row in s), abs=1e-3)


def test_hysteresis_peek_matches_update_without_changing_state():
    from app.rules import Hysteresis
    h = Hysteresis(1.0, 5, 0.8, 0.6)
    for p in (0.1, 0.9, 1.0, 1.0, 1.0, 0.2, 0.2, 0.9, 0.9, 0.1, 1.0):
        want = h.peek(p)
        before = (list(h.buf), h.state, h.avg)
        assert h.peek(p) == want and (list(h.buf), h.state, h.avg) == before
        assert h.update(p) == want


def test_mask_smoother_blends_and_holds_with_hysteresis_and_resets():
    from app.rules import MaskSmoother
    s = MaskSmoother(0.6)
    on, off = np.ones((4, 4), bool), np.zeros((4, 4), bool)
    assert not s.update(off).any()
    assert not s.update(on).any()                  # 0.4 <= 0.5: một mẫu lẻ chưa bật (không nhấp nháy)
    assert s.update(on).all()                      # 0.64 > 0.5
    assert s.update(off).all()                     # 0.38 > 0.25: giữ
    assert not s.update(off).any()                 # 0.23 < 0.25: tắt
    s.update(on); s.update(on); s.reset()
    assert not s.update(off).any()                 # reset: không dính lịch sử (không còn giữ mask cũ)


class GatedModel:
    """Giống Composite: p = độ sáng khung; chỉ chạy "model vẽ" khi draw_if đồng ý. Đếm số lần vẽ."""
    gated_mask = True

    def __init__(self):
        self.drawn = 0

    def predict(self, frame, draw_if=None):
        p = float(frame.mean()) / 255
        mask = None
        if draw_if is None or draw_if(p):
            self.drawn += 1
            mask = np.zeros(frame.shape[:2], bool)
            mask[frame.shape[0] // 2:, :] = True
        return {"p_ngap": p, "mask": mask}


def test_water_drawn_only_while_flooded_and_drawer_skipped_otherwise(tmp_path):
    v, out = tmp_path / "a.mp4", tmp_path / "result.mp4"
    make_video(v)                                                # 14 s, sáng (ngập) 4–10 s
    m = GatedModel()
    analyze_video(v, m, config.ModelConfig(), video_out=out)
    frames = _frames(out)

    def drawn(i):
        h = frames[i].shape[0]
        return abs(frames[i][: h // 4].astype(int).mean() - frames[i][3 * h // 4:].astype(int).mean()) > 20
    assert not any(drawn(i) for i in range(0, 38))               # 0–3,8 s: chưa ngập, không vẽ
    assert drawn(70)                                             # 7 s: đang ngập
    assert not any(drawn(i) for i in range(125, len(frames)))    # sau khi hết ngập: không vẽ
    assert m.drawn < 14 * 5 * 0.6                                # không chạy model vẽ trên đoạn không ngập (14 s x 5 mẫu/s)
