import cv2
import numpy as np
import pytest

from app import config
from app.analyzer import analyze_video
from app.crop import box_from_frames, crop_to_box, detect_black_border


def scene(h=120, w=200, seed=0, level=120):
    rng = np.random.default_rng(seed)
    return np.clip(rng.normal(level, 40, (h, w, 3)), 0, 255).astype(np.uint8)


def with_bars(f, left=0, right=0, top=0, bottom=0):
    g = f.copy()
    if left: g[:, :left] = 0
    if right: g[:, -right:] = 0
    if top: g[:top] = 0
    if bottom: g[-bottom:] = 0
    return g


def test_pillarbox_detected_and_cropped():
    fs = [with_bars(scene(seed=i), left=30, right=20) for i in range(4)]
    assert box_from_frames(fs) == (30, 0, 180, 120)
    assert crop_to_box(fs[0], (30, 0, 180, 120)).shape[:2] == (120, 150)


def test_letterbox_both_axes():
    fs = [with_bars(scene(seed=i), left=10, right=10, top=12, bottom=8) for i in range(3)]
    assert box_from_frames(fs) == (10, 12, 190, 112)


def test_no_border_none():
    assert box_from_frames([scene(seed=i) for i in range(4)]) is None


def test_tiny_border_ignored():
    assert box_from_frames([with_bars(scene(seed=i), left=2) for i in range(3)]) is None      # 1% < 2%


def test_dark_night_scene_not_cropped():
    # cảnh ban đêm: nền rất tối nhưng có nhiễu cảm biến và điểm sáng -> không phải viền đen thuần
    rng = np.random.default_rng(1)
    fs = []
    for i in range(5):
        f = np.clip(rng.normal(6, 5, (120, 200, 3)), 0, 255).astype(np.uint8)
        f[40:50, 100:110] = 220
        fs.append(f)
    assert box_from_frames(fs) is None


def test_all_black_not_cropped():
    assert box_from_frames([np.zeros((120, 200, 3), np.uint8)] * 3) is None


def test_border_must_be_black_in_every_frame():
    # cột trái đen ở khung 1 nhưng có nội dung ở khung 2 (cảnh cắt ghép khác nhau) -> không coi là viền
    a = with_bars(scene(seed=1), left=30)
    b = scene(seed=2)
    assert box_from_frames([a, b]) is None


def test_content_too_small_not_cropped():
    assert box_from_frames([with_bars(scene(seed=i), left=60, right=60) for i in range(3)]) is None   # còn 40% chiều rộng


def _video(path, left=0, right=0, secs=6, fps=10):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (200, 120))
    for i in range(secs * fps):
        w.write(with_bars(scene(seed=i, level=200 if 2 <= i / fps < 5 else 30), left=left, right=right))
    w.release()


class SeeEdges:
    """Ghi lại kích thước khung model nhận; p = độ sáng."""
    kind = "fake"

    def __init__(self):
        self.shapes = set()

    def predict(self, frame):
        self.shapes.add(frame.shape[:2])
        return {"p_ngap": float(frame.mean()) / 255, "mask": None}


def test_analyzer_crops_before_model_and_reports_box(tmp_path):
    v = tmp_path / "bars.mp4"
    _video(v, left=40, right=30)
    box = detect_black_border(v)
    # nén mp4v làm lem tới một khối ~16 px sát nội dung: hộp có thể chừa vài px viền, không bao giờ cắt vào nội dung
    assert box is not None and 40 <= box[0] <= 56 and 170 <= box[2] <= 186 and box[1] == 0 and box[3] == 120
    m = SeeEdges()
    out = tmp_path / "res.mp4"
    _, info = analyze_video(v, m, config.ModelConfig(min_hold_s=0.0), video_out=out)
    assert info["crop"] == box
    assert m.shapes == {(120, box[2] - box[0])}                       # model chỉ thấy phần đã cắt
    cap = cv2.VideoCapture(str(out)); ok, fr = cap.read()
    assert ok and fr.shape[1] <= (box[2] - box[0]) and fr.shape[0] <= 120


def test_analyzer_crop_can_be_disabled(tmp_path):
    v = tmp_path / "bars.mp4"
    _video(v, left=40, right=30)
    m = SeeEdges()
    _, info = analyze_video(v, m, config.ModelConfig(min_hold_s=0.0, crop_black_border=False))
    assert info["crop"] is None and m.shapes == {(120, 200)}
