"""Video so sánh để TRÌNH BÀY (không thuộc app): ghép ngang 2 cách tô vùng nước trên cùng một video, cùng một quyết định ngập.
  trái  = seg.pt của nhóm (models/seg.pt, yolo11s-seg)
  phải  = V-FloodNet pretrained (Liang et al. 2023; LinkNet + EfficientNet-B4, ảnh), trọng số ở ~/models_ngoai/vfloodnet/records/link_efficientb4_model.pth (KHÔNG nằm trong repo)
Quyết định ngập lấy từ best.pt của app (GPU) hoặc best_cpu.pt, luật nhúng trong model, đúng như analyze_video; chỉ tô khi đang NGẬP; mask làm mượt như app (MaskSmoother).
Phía trên video ghi "Xác suất ngập (p làm mượt)" và trạng thái. Mỗi video xuất kèm 1 ảnh chụp (giữa đợt báo NGẬP đầu tiên).
Cần: segmentation-models-pytorch (>=0.5), efficientnet-pytorch, Pillow, ffmpeg (các thư viện này KHÔNG có trong requirements.txt của app).
Dùng (từ thư mục flood-alert):  python tools/so_sanh_seg.py [clip ...] [--demo-dir DIR] [--out DIR]
Mặc định clip = demo_01 demo_02 demo_03 demo_05; --demo-dir mặc định ~/Videos/ngap/demo; --out mặc định ~/Videos/ngap/so_sanh."""
import argparse
import io
import pickle
import pickletools
import subprocess
import sys
import types
import zipfile
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app import config
from app.adapters import SegAdapter, build_decider
from app.analyzer import CLIP_WIDTH, MASK_MAX_AGE, MASK_SMOOTH, ResultWriter, _scaled, draw_overlay
from app.crop import crop_to_box, detect_black_border
from app.rules import Hysteresis, MaskSmoother
from app.video import Video

HOME = Path.home()
VFN = HOME / "models_ngoai/vfloodnet/records/link_efficientb4_model.pth"
FONT = "/usr/share/fonts/liberation-sans-fonts/LiberationSans-Bold.ttf"
SAFE = ("torch", "segmentation_models_pytorch", "efficientnet_pytorch", "collections", "timm", "pretrainedmodels", "functools", "numpy", "_codecs", "builtins", "__builtin__", "typing")
BAD = {"eval", "exec", "system", "popen", "compile", "open", "__import__", "getattr", "setattr", "execfile", "input"}
PW = 640           # chiều rộng mỗi nửa
BAR = 86           # thanh chữ phía trên


# ---- V-FloodNet: file .pth là pickle nguyên model của smp cũ; quét an toàn rồi nạp state_dict vào smp.Linknet hiện hành ----
def check_pickle(path):
    with zipfile.ZipFile(path) as z:
        data = z.read(next(n for n in z.namelist() if n.endswith("data.pkl")))
    strings = []
    for op, arg, _ in pickletools.genops(io.BytesIO(data)):
        if op.name in ("SHORT_BINUNICODE", "BINUNICODE", "UNICODE"):
            strings.append(arg)
        mod = nm = None
        if op.name in ("GLOBAL", "INST") and arg:
            mod, nm = arg.split(" ", 1) if " " in arg else (arg, "")
        elif op.name == "STACK_GLOBAL" and len(strings) >= 2:
            mod, nm = strings[-2], strings[-1]
        if mod and (not mod.startswith(SAFE) or nm in BAD):
            raise RuntimeError(f"pickle gọi {mod}.{nm}: từ chối nạp {path}")


def load_vfloodnet(path, device):
    import segmentation_models_pytorch as smp
    check_pickle(path)

    class Mod(torch.nn.Module):
        def __init__(self, *a, **k):
            super().__init__()

    class Obj:
        def __init__(self, *a, **k):
            pass

        def __setstate__(self, st):
            self.__dict__.update(st if isinstance(st, dict) else {})

    mods = {"MBConvBlock", "Conv2dStaticSamePadding", "Identity", "MemoryEfficientSwish", "SegmentationHead", "Activation", "Conv2dReLU", "EfficientNetEncoder", "DecoderBlock", "LinknetDecoder", "TransposeX2", "Linknet"}

    class U(pickle.Unpickler):
        def find_class(self, mod, name):
            if mod.startswith(("segmentation_models_pytorch", "efficientnet_pytorch")):
                return Mod if name in mods else Obj
            return super().find_class(mod, name)
    pm = types.ModuleType("stubpickle")
    pm.Unpickler, pm.UnpicklingError, pm.load, pm.Pickler, pm.HIGHEST_PROTOCOL = U, pickle.UnpicklingError, pickle.load, pickle.Pickler, pickle.HIGHEST_PROTOCOL
    sd = torch.load(str(path), map_location="cpu", weights_only=False, pickle_module=pm).state_dict()
    m = smp.Linknet(encoder_name="efficientnet-b4", encoder_depth=5, encoder_weights=None, in_channels=3, classes=1, activation="sigmoid")
    m.load_state_dict(sd, strict=True)
    return m.eval().to(device)


def vfloodnet_mask(model, frame_bgr, device):
    """Như test_image_seg.py của V-FloodNet: RGB, 416x416, chuẩn hoá ImageNet, làm tròn 0/1, giữ mảng nước liên thông lớn nhất, phóng về cỡ khung."""
    h, w = frame_bgr.shape[:2]
    x = cv2.resize(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB), (416, 416), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
    x = ((x - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]).astype(np.float32)
    with torch.no_grad():
        p = model.predict(torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0).to(device))
    m = p.squeeze().cpu().numpy().round().astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n > 1:
        m = (lab == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)
    return cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)


def header(width, left, right, prob, state):
    img = Image.new("RGB", (width, BAR), (24, 24, 28))
    d = ImageDraw.Draw(img)
    f1, f2 = ImageFont.truetype(FONT, 30), ImageFont.truetype(FONT, 22)
    txt = f"Xác suất ngập: {prob * 100:.0f}%  ({'NGẬP' if state else 'không ngập'})"
    d.text((width // 2, 6), txt, font=f1, fill=(255, 90, 90) if state else (110, 220, 130), anchor="ma")
    d.text((PW // 2, BAR - 8), left, font=f2, fill=(255, 255, 255), anchor="ms")
    d.text((PW + PW // 2, BAR - 8), right, font=f2, fill=(255, 255, 255), anchor="ms")
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def run(path, out_mp4, decider, seg, vfn, cfg, dev):
    with Video(path) as vid:
        hyst = Hysteresis(cfg.window_s, cfg.infer_fps, cfg.flood_on, cfg.flood_off, cfg.min_hold_s)
        box = detect_black_border(vid.path) if cfg.crop_black_border else None
        period, next_t = 1.0 / cfg.infer_fps, 0.0
        sm = [MaskSmoother(MASK_SMOOTH), MaskSmoother(MASK_SMOOTH)]
        masks, mt, prob, state, writer, on_times, idx = [None, None], 0.0, 0.0, False, None, [], -1
        while True:
            ok, frame = vid.cap.read()
            if not ok:
                break
            frame = crop_to_box(frame, box)
            idx += 1
            t = idx / vid.fps
            h, w = frame.shape[:2]
            ph = max(2, round(h * PW / w) // 2 * 2)
            if writer is None:
                writer = ResultWriter(out_mp4, (2 * PW, BAR + ph), vid.fps)
            if t + 1e-9 >= next_t:
                p = decider.predict(frame)["p_ngap"]
                state = hyst.update(p); prob = hyst.avg
                if state:
                    on_times.append(t)
                    size = (PW, ph)
                    for i, m in enumerate((seg.predict(frame)["mask"], vfloodnet_mask(vfn, frame, dev))):
                        raw = _scaled(m, size) if m is not None and m.any() else np.zeros(size[::-1], bool)
                        s = sm[i].update(raw)
                        masks[i] = s if s.any() else None
                else:
                    for s in sm:
                        s.reset()
                    masks = [None, None]
                mt = t
                next_t = max(next_t + period, t + period / 2)
            f = cv2.resize(frame, (PW, ph), interpolation=cv2.INTER_AREA)
            fresh = state and t - mt <= MASK_MAX_AGE + 1e-6
            panels = [draw_overlay(f, m) if (fresh and m is not None) else f for m in masks]
            writer.write(np.vstack([header(2 * PW, "seg.pt (của nhóm)", "V-FloodNet (pretrained, Liang et al. 2023)", prob, state), np.hstack(panels)]))
        ok = writer.close()
        return ok, on_times, vid.duration


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clips", nargs="*", default=["demo_01", "demo_02", "demo_03", "demo_05"])
    ap.add_argument("--demo-dir", default=None)
    ap.add_argument("--out", default=str(HOME / "Videos/ngap/so_sanh"))
    a = ap.parse_args()
    d = Path(a.demo_dir) if a.demo_dir else HOME / "Videos/ngap/demo"
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    cfg = config.load_model_config()
    dec = build_decider(cfg, config.MODELS_DIR)
    seg = SegAdapter(cfg, path=config.MODELS_DIR / cfg.seg_path)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    vfn = load_vfloodnet(VFN, dev)
    print("quyết định:", dec.name, "| luật", cfg.window_s, cfg.flood_on, cfg.flood_off, cfg.min_hold_s, "| vẽ trái:", seg.name, "| phải: V-FloodNet", dev, flush=True)
    for c in a.clips:
        mp4 = out / f"{c}_so_sanh.mp4"
        ok, on, dur = run(d / f"{c}.mp4", mp4, dec, seg, vfn, cfg, dev)
        shot = (on[0] + on[-1]) / 2 if on else dur / 2                                      # giữa khoảng đang báo NGẬP
        shot = min(shot, dur - 0.5)
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-ss", f"{shot:.2f}", "-i", str(mp4), "-frames:v", "1", str(out / f"{c}_so_sanh.jpg")], check=False)
        print(f"{c}: {'ok' if ok else 'LỖI ghi video'} -> {mp4} | bật NGẬP {'từ %.1f s' % on[0] if on else 'không bao giờ'}, {len(on)} mẫu bật | ảnh chụp giây {shot:.1f}", flush=True)


if __name__ == "__main__":
    main()
