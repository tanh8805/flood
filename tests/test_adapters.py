import os

import cv2
import numpy as np
import pytest
import torch

import app.adapters as ad
from app import config
from app.adapters import Composite, ModelHolder, build_decider, build_drawer
from app.adapters.base import ModelError
from app.adapters.best import BestAdapter, pick_device
from app.adapters.seg import SegAdapter
from tests.fakes import FakeResult, FakeYolo

FRAME = np.zeros((100, 200, 3), np.uint8)
CFG = config.ModelConfig()


# ---- best.pt: TorchScript, p = m(khung BGR) ----
class _Mean(torch.nn.Module):
    """Giống hợp đồng của best.pt: nhận tensor uint8 HxWx3 (BGR), trả p (scalar) trong 0..1."""
    def forward(self, frame):
        return frame.float().mean() / 255.0


def _save_best(path, module=None):
    torch.jit.script(module or _Mean()).save(str(path))


def test_best_adapter_loads_torchscript_and_feeds_raw_bgr_frame(tmp_path):
    _save_best(tmp_path / "best.pt")
    a = BestAdapter(tmp_path / "best.pt", device="cpu")
    assert a.kind == "best" and "best.pt" in a.name and a.device == "cpu"
    f = np.full((37, 91, 3), 51, np.uint8)                              # kích thước bất kỳ, không resize ở ngoài
    r = a.predict(f)
    assert r["p_ngap"] == pytest.approx(0.2) and r["mask"] is None
    assert a.predict(np.asfortranarray(f))["p_ngap"] == pytest.approx(0.2)    # khung không liên tục vẫn chạy


def test_best_adapter_clamps_p_and_rejects_non_torchscript(tmp_path):
    class Big(torch.nn.Module):
        def forward(self, frame):
            return frame.float().mean() * 100
    _save_best(tmp_path / "big.pt", Big())
    assert BestAdapter(tmp_path / "big.pt", device="cpu").predict(np.ones((4, 4, 3), np.uint8))["p_ngap"] == 1.0
    (tmp_path / "bad.pt").write_text("không phải TorchScript")
    with pytest.raises(ModelError, match="bad.pt"):
        BestAdapter(tmp_path / "bad.pt", device="cpu")


def test_best_adapter_with_injected_module():
    a = BestAdapter("models/best.pt", module=lambda t: torch.tensor(0.75))
    assert a.predict(FRAME)["p_ngap"] == pytest.approx(0.75)


def test_pick_device():
    assert pick_device("auto", True) == "cuda" and pick_device("auto", False) == "cpu"
    assert pick_device("cpu", True) == "cpu" and pick_device("cuda", True) == "cuda"
    with pytest.raises(ModelError, match="CUDA"):
        pick_device("cuda", False)


def test_build_decider_and_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="Chưa có model"):
        build_decider(config.ModelConfig(), tmp_path)
    _save_best(tmp_path / "best.pt")
    assert build_decider(config.ModelConfig(device="cpu"), tmp_path).kind == "best"


# ---- seg.pt: chỉ tô vùng nước ----
def _rect(shape, y0, y1, x0, x1):
    m = np.zeros(shape, np.float32)
    m[y0:y1, x0:x1] = 1.0
    return m


def test_segment_mask_only_water_class():
    water = _rect((100, 200), 0, 50, 0, 100)                           # 25% khung, cùng kích thước khung (như retina_masks)
    car = _rect((100, 200), 50, 100, 100, 200)
    m = FakeYolo("segment", {0: "water", 1: "car"}, FakeResult(masks=np.stack([water, car]), cls=[0, 1]))
    r = SegAdapter(CFG, model=m).predict(FRAME)
    assert r["mask"].shape == (100, 200) and r["mask"].dtype == bool
    assert 0.24 < r["mask"].mean() < 0.26                              # chỉ lấy class nước, bỏ class xe
    assert r["p_ngap"] == 0.0                                          # seg.pt không bao giờ quyết định: p luôn 0
    assert not r["mask"][50:, 100:].any()


def test_segment_maps_letterboxed_mask_back_to_frame():
    """Suy luận ở 96x128 với khung 100x200: tỉ lệ 0.64, nội dung 64x128 nằm giữa, viền 16 hàng trên/dưới."""
    data = np.zeros((96, 128), np.float32)
    data[16:48, :] = 1.0                                               # nửa trên của vùng nội dung = nửa trên của khung gốc
    m = FakeYolo("segment", {0: "water"}, FakeResult(masks=data[None], cls=[0]))
    r = SegAdapter(CFG, model=m).predict(FRAME)
    mask = r["mask"]
    assert mask.shape == (100, 200)
    assert mask[:48].all() and not mask[54:].any()                     # ~nửa trên; viền letterbox không lọt vào
    assert 0.47 < mask.mean() < 0.53


def test_segment_separate_patches_are_not_bridged():
    """Hai mảng nước rời nhau: không được có gì nối giữa chúng (lỗi cũ của masks.xy: đường nối thẳng)."""
    data = _rect((100, 200), 20, 80, 5, 60) + _rect((100, 200), 20, 80, 140, 195)
    m = FakeYolo("segment", {0: "water"}, FakeResult(masks=data[None], cls=[0]))
    mask = SegAdapter(CFG, model=m).predict(FRAME)["mask"]
    assert not mask[:, 70:130].any()                                   # vùng giữa hai mảng hoàn toàn trống
    n, _ = cv2.connectedComponents(mask.astype(np.uint8))
    assert n - 1 == 2
    from app.analyzer import draw_overlay
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    assert len(cnts) == 2                                              # mỗi mảng một contour riêng
    out = draw_overlay(np.full((100, 200, 3), 50, np.uint8), mask)
    assert (out[:, 70:130] == 50).all()                                # viền không kéo qua khoảng giữa


def test_segment_cleans_noise_and_tiny_blobs():
    data = _rect((100, 200), 20, 80, 10, 150)                          # mảng chính
    data[2:4, 170:172] = 1.0                                           # đốm nhiễu 4 px
    data[90:93, 190:196] = 1.0                                         # mảng nhỏ ~0.1% khung (< 1%)
    data[50, 150:170] = 1.0                                            # cầu mảnh 1 px nối mảng chính ra ngoài
    m = FakeYolo("segment", {0: "water"}, FakeResult(masks=data[None], cls=[0]))
    mask = SegAdapter(CFG, model=m).predict(FRAME)["mask"]
    assert not mask[:10, 160:].any() and not mask[85:, 185:].any() and not mask[48:53, 155:].any()
    assert mask[30:70, 20:140].all()


def test_segment_all_blobs_too_small_gives_no_mask():
    data = np.zeros((100, 200), np.float32)
    data[40:44, 40:44] = 1.0                                           # 16 px = 0.08% khung
    m = FakeYolo("segment", {0: "water"}, FakeResult(masks=data[None], cls=[0]))
    assert SegAdapter(CFG, model=m).predict(FRAME) == {"p_ngap": 0.0, "mask": None}


def test_segment_empty_or_other_class_gives_no_mask():
    big = np.ones((1, 100, 200), np.float32)
    empty = FakeYolo("segment", {0: "water"}, FakeResult(masks=np.zeros((0, 100, 200), np.float32), cls=[]))
    assert SegAdapter(CFG, model=empty).predict(FRAME) == {"p_ngap": 0.0, "mask": None}
    other = FakeYolo("segment", {0: "water", 1: "car"}, FakeResult(masks=big, cls=[1]))
    assert SegAdapter(CFG, model=other).predict(FRAME) == {"p_ngap": 0.0, "mask": None}



def test_segment_model_must_be_segment_task_and_have_water_class():
    with pytest.raises(ModelError, match="segment"):
        SegAdapter(CFG, model=FakeYolo("classify", {0: "water"}, None))
    with pytest.raises(ModelError, match="detect|segment"):
        SegAdapter(CFG, model=FakeYolo("detect", {0: "water"}, None))
    with pytest.raises(ModelError) as e:
        SegAdapter(CFG, model=FakeYolo("segment", {0: "cat", 1: "dog"}, None))
    assert "cat" in str(e.value) and "dog" in str(e.value)
    cfg = config.ModelConfig(water_classes="^lu$")
    assert SegAdapter(cfg, model=FakeYolo("segment", {0: "lu", 1: "ok"}, None)).ids == [0]


def test_build_drawer_is_optional_and_rejects_bad_file(tmp_path):
    assert build_drawer(config.ModelConfig(), tmp_path) is None
    (tmp_path / "seg.pt").write_text("hỏng")
    with pytest.raises(ModelError, match="seg.pt"):
        build_drawer(config.ModelConfig(), tmp_path)


def test_segment_uses_draw_params_and_maps_mask_back_to_original_size():
    """Khung 1920x1080 -> thu về rộng 960 trước khi suy luận, conf 0.4, retina_masks; mask trả về đúng cỡ khung gốc."""
    big = np.zeros((1080, 1920, 3), np.uint8)
    seen = {}

    def result(f):
        seen["shape"] = f.shape[:2]
        return FakeResult(masks=_rect(f.shape[:2], 270, 540, 0, 960)[None], cls=[0])      # nửa dưới-trên, cùng cỡ khung đã thu (retina)
    m = FakeYolo("segment", {0: "water"}, result)
    r = SegAdapter(CFG, model=m).predict(big)
    assert seen["shape"] == (540, 960)
    assert m.calls[0][1] == {"conf": 0.4, "retina_masks": True}
    assert r["mask"].shape == (1080, 1920) and 0.49 < r["mask"].mean() < 0.51


def test_segment_small_frame_not_resized_and_config_overrides():
    cfg = config.ModelConfig(water_conf=0.55, draw_width=0)
    m = FakeYolo("segment", {0: "water"}, lambda f: FakeResult(masks=_rect(f.shape[:2], 0, 50, 0, 100)[None], cls=[0]))
    SegAdapter(cfg, model=m).predict(np.zeros((1080, 1920, 3), np.uint8))
    assert m.calls[0][0] == (1080, 1920) and m.calls[0][1]["conf"] == 0.55
    m2 = FakeYolo("segment", {0: "water"}, lambda f: FakeResult(masks=_rect(f.shape[:2], 0, 50, 0, 100)[None], cls=[0]))
    SegAdapter(CFG, model=m2).predict(FRAME)                                     # 200 px < 960: giữ nguyên
    assert m2.calls[0][0] == (100, 200)




# ---- ghép hai vai trò ----
def test_composite_percent_from_decider_mask_from_drawer():
    mask = np.ones((4, 4), bool)

    class Dec:
        kind, name, description = "best", "dec", "d"
        predict = lambda self, f: {"p_ngap": 0.1, "mask": None}

    class Draw:
        name = "seg"
        predict = lambda self, f: {"p_ngap": 1.0, "mask": mask}        # segment báo p=1: KHÔNG được ảnh hưởng %

    r = Composite(Dec(), Draw()).predict(FRAME)
    assert r["p_ngap"] == pytest.approx(0.1) and r["mask"] is mask
    assert Composite(Dec(), None).predict(FRAME) == {"p_ngap": 0.1, "mask": None}

    class Boom:
        name = "seg"

        def predict(self, f):
            raise RuntimeError("hỏng")
    assert Composite(Dec(), Boom()).predict(FRAME) == {"p_ngap": 0.1, "mask": None}      # vẽ lỗi không làm mất quyết định
    assert "vẽ vùng nước: seg" in Composite(Dec(), Draw()).name


def test_composite_draws_only_when_gate_allows():
    from app.adapters import Composite
    calls = []

    class Dec:
        kind = "fake"
        name = description = "d"
        predict = lambda self, f: {"p_ngap": 0.3, "mask": None}

    class Draw:
        name = "x"
        def predict(self, f):
            calls.append(1)
            return {"p_ngap": 1.0, "mask": np.ones((100, 200), bool)}
    c = Composite(Dec(), Draw())
    assert c.predict(FRAME, draw_if=lambda p: p >= 0.5)["mask"] is None and calls == []      # không ngập: không chạy model vẽ
    assert c.predict(FRAME, draw_if=lambda p: p >= 0.2)["mask"] is not None and calls == [1]
    assert c.predict(FRAME)["mask"] is not None                                              # không có cổng = luôn vẽ (ảnh minh chứng)


# ---- cấu hình ----
def test_defaults_are_the_best_pt_rule(tmp_path):
    c = config.load_model_config(tmp_path)
    assert (c.window_s, c.infer_fps, c.flood_on, c.flood_off, c.segment_s) == (1.0, 5.0, 0.9, 0.6, 2.0)
    assert (c.water_conf, c.draw_width, c.path, c.seg_path, c.device) == (0.4, 960, "best.pt", "seg.pt", "auto")


def test_config_yaml_and_env(tmp_path, monkeypatch):
    (tmp_path / "model.yaml").write_text("flood_on: 0.85\nwindow_s: 3\ndevice: cpu\n")
    c = config.load_model_config(tmp_path)
    assert (c.flood_on, c.flood_off, c.window_s, c.device) == (0.85, 0.6, 3.0, "cpu")
    monkeypatch.setenv("FLOOD_OFF", "0.5")
    monkeypatch.setenv("MODEL_DEVICE", "auto")
    c = config.load_model_config(tmp_path)
    assert (c.flood_off, c.device) == (0.5, "auto")
    monkeypatch.delenv("MODEL_DEVICE")
    (tmp_path / "model.yaml").write_text("device: tpu\n")
    with pytest.raises(ValueError, match="device"):
        config.load_model_config(tmp_path)


def test_old_kind_key_is_rejected(tmp_path):
    (tmp_path / "model.yaml").write_text("kind: v5\n")
    with pytest.raises(ValueError, match="kind"):
        config.load_model_config(tmp_path)


# ---- holder ----
def test_holder_without_model_does_not_crash(tmp_path):
    h = ModelHolder(tmp_path)
    st = h.reload()
    assert not st["loaded"] and "Chưa có model" in st["error"] and h.snapshot()[0] is None


def test_holder_bad_yaml_reports_error(tmp_path):
    (tmp_path / "model.yaml").write_text("bogus: 1\n")
    st = ModelHolder(tmp_path).reload()
    assert not st["loaded"] and "bogus" in st["error"]


class _FakeBest:
    kind, description = "best", "fake"

    def __init__(self, path):
        text = path.read_text()
        if text == "bad":
            raise ModelError("file hỏng")
        self.name = f"{path.name} ({text})"

    def predict(self, frame):
        return {"p_ngap": 0.1, "mask": None}


class _FakeSeg:
    name, classes = "seg.pt (segment)", ["water"]

    def __init__(self, path):
        if path.read_text() == "bad":
            raise ModelError("file hỏng")

    def predict(self, frame):
        return {"p_ngap": 0.0, "mask": np.ones((100, 200), bool)}


class _Counts:
    def __init__(self, fn):
        self.fn, self.calls = fn, 0

    def __call__(self, cfg, d):
        self.calls += 1
        return self.fn(cfg, d)


def _dec(cfg, d):
    p = d / cfg.path
    if not p.exists():
        raise FileNotFoundError(config.NO_MODEL_MSG)
    return _FakeBest(p)


def _draw(cfg, d):
    p = d / cfg.seg_path
    return _FakeSeg(p) if p.exists() else None


def _bump(path, ns):
    os.utime(path, ns=(ns, ns))


def test_holder_reloads_each_role_when_its_file_changes(tmp_path):
    dec, draw = _Counts(_dec), _Counts(_draw)
    best, seg = tmp_path / "best.pt", tmp_path / "seg.pt"
    best.write_text("v1"); _bump(best, 1_000_000_000_000_000_000)
    h = ModelHolder(tmp_path, decider_builder=dec, drawer_builder=draw)
    h.refresh(); h.refresh()
    st = h.status()
    assert (dec.calls, draw.calls) == (1, 1) and st["decide"]["name"] == "best.pt (v1)" and not st["draw"]["loaded"] and st["draw"]["error"] is None
    assert h.model is not None and h.model.drawer is None
    seg.write_text("x"); _bump(seg, 1_000_000_000_000_000_000)
    h.refresh(); h.refresh()
    assert (dec.calls, draw.calls) == (1, 2) and h.status()["draw"]["loaded"] and h.model.drawer is not None     # chỉ model tô nạp lại
    best.write_text("v2"); _bump(best, 1_000_000_001_000_000_000)
    h.refresh()
    assert (dec.calls, draw.calls) == (2, 2) and h.status()["decide"]["name"] == "best.pt (v2)" and h.model.drawer is not None
    best.unlink(); seg.unlink()
    h.refresh()
    st = h.status()
    assert not st["loaded"] and "Chưa có model" in st["error"] and not st["draw"]["loaded"]


def test_holder_bad_seg_keeps_decision_working_and_is_not_retried(tmp_path):
    best, seg = tmp_path / "best.pt", tmp_path / "seg.pt"
    best.write_text("v1")
    draw = _Counts(_draw)
    h = ModelHolder(tmp_path, decider_builder=_dec, drawer_builder=draw)
    seg.write_text("bad"); _bump(seg, 1_000_000_000_000_000_000)
    h.refresh(); h.refresh()
    st = h.status()
    assert draw.calls == 1 and st["loaded"] and not st["draw"]["loaded"] and "file hỏng" in st["draw"]["error"]
    assert h.model.predict(FRAME)["p_ngap"] == pytest.approx(0.1)


def test_holder_bad_best_reports_error_and_recovers(tmp_path):
    best = tmp_path / "best.pt"
    best.write_text("bad"); _bump(best, 1_000_000_000_000_000_000)
    dec = _Counts(_dec)
    h = ModelHolder(tmp_path, decider_builder=dec, drawer_builder=_draw)
    h.refresh(); h.refresh()
    assert dec.calls == 1 and not h.status()["loaded"] and "file hỏng" in h.status()["error"]
    best.write_text("v2"); _bump(best, 1_000_000_001_000_000_000)
    h.refresh()
    assert h.status()["loaded"] and dec.calls == 2


def test_startup_log_line_for_loaded_and_missing_models(tmp_path):
    from app.main import describe_models
    (tmp_path / "best.pt").write_text("v1")
    h = ModelHolder(tmp_path, decider_builder=_dec, drawer_builder=_draw)
    h.refresh()
    assert describe_models(h.status()) == "quyết định: best.pt (v1); vẽ vùng nước: không có seg.pt"
    (tmp_path / "seg.pt").write_text("x")
    h.reload()
    assert "vẽ vùng nước: seg.pt (segment)" in describe_models(h.status())
    assert "Chưa có model" in describe_models(ModelHolder(tmp_path / "nope").status())


def test_shipped_seg_pt_is_a_water_segment_model():
    """models/seg.pt có trong repo: phải nạp được, là segment, có lớp nước (bỏ qua nếu thiếu ultralytics)."""
    pytest.importorskip("ultralytics")
    seg = config.ROOT / "models" / "seg.pt"
    if not seg.exists():
        pytest.skip("không có models/seg.pt")
    a = build_drawer(config.ModelConfig(), seg.parent)
    assert a.task == "segment" and a.classes == ["water"]
