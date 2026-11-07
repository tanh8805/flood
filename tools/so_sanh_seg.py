"""So sánh cách tô vùng nước trên một video (để trình bày, không thuộc app). Quyết định ngập luôn lấy từ best.pt của app (GPU) hoặc best_cpu.pt (CPU),
luật nhúng trong model, đúng như analyze_video; chỉ tô khi đang NGẬP; mask làm mượt như app. Phía trên video ghi "Xác suất ngập (p làm mượt)" và trạng thái.

  python tools/so_sanh_seg.py video.mp4 --model seg         -> video chỉ tô bằng seg.pt (của nhóm)
  python tools/so_sanh_seg.py video.mp4 --model vfloodnet   -> video chỉ tô bằng V-FloodNet (pretrained, Liang et al. 2023)
  python tools/so_sanh_seg.py video.mp4 --model both        -> ghép ngang 2 bên (mặc định)

Tham số: nhiều video cùng lúc được; --out thư mục ra (mặc định ~/Videos/ngap/so_sanh/). Mỗi video ra <tên>_seg.mp4 / <tên>_vfloodnet.mp4 / <tên>_so_sanh.mp4 (H.264) kèm 1 ảnh .jpg chụp giữa đợt báo NGẬP.
V-FloodNet cần trọng số tải riêng (để NGOÀI repo, đường dẫn bằng biến VFLOODNET_PATH) và `pip install -r requirements-so-sanh.txt`; thiếu thì script báo cách tải rồi dừng, không crash. Xem README, mục "So sánh với V-FloodNet"."""
import argparse
import io
import os
import pickle
import pickletools
import subprocess
import sys
import types
import zipfile
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app import config
from app.adapters import SegAdapter, build_decider
from app.analyzer import MASK_MAX_AGE, MASK_SMOOTH, ResultWriter, _scaled, draw_overlay
from app.crop import crop_to_box, detect_black_border
from app.rules import Hysteresis, MaskSmoother
from app.video import Video

HOME = Path.home()
DEFAULT_VFN = "~/models_ngoai/vfloodnet/records/link_efficientb4_model.pth"
FONT_PATHS = ("/usr/share/fonts/liberation-sans-fonts/LiberationSans-Bold.ttf", "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans-Bold.ttf", "DejaVuSans-Bold.ttf", "arialbd.ttf")
TITLES = {"seg": "seg.pt (của nhóm)", "vfloodnet": "V-FloodNet (pretrained, Liang et al. 2023)"}
SAFE = ("torch", "segmentation_models_pytorch", "efficientnet_pytorch", "collections", "timm", "pretrainedmodels", "functools", "numpy", "_codecs", "builtins", "__builtin__", "typing")
BAD = {"eval", "exec", "system", "popen", "compile", "open", "__import__", "getattr", "setattr", "execfile", "input"}
BAR = 86           # thanh chữ phía trên

HOW_TO_DOWNLOAD = """Không dùng được V-FloodNet: {why}
Cách tải trọng số (để ngoài repo):
  1. Mở https://huggingface.co/xmlyqing00/V-FloodNet/tree/main và tải records.zip (~800 MB)
     (hoặc thư mục Google Drive trong README của https://github.com/xmlyqing00/V-FloodNet)
  2. Giải nén lấy records/link_efficientb4_model.pth, ví dụ đặt ở {default}
  3. Trỏ đường dẫn: export VFLOODNET_PATH=/đường/dẫn/link_efficientb4_model.pth   (mặc định {default})
  4. pip install -r requirements-so-sanh.txt
Chỉ tô bằng seg.pt thì dùng: --model seg"""


class NoVFloodNet(Exception):
    pass


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
    try:
        import segmentation_models_pytorch as smp
    except ImportError as e:
        raise NoVFloodNet(f"chưa cài segmentation-models-pytorch / efficientnet-pytorch ({e})") from e
    if not Path(path).is_file():
        raise NoVFloodNet(f"không thấy file trọng số {path}")
    try:
        check_pickle(path)
    except zipfile.BadZipFile as e:
        raise NoVFloodNet(f"{path} không phải file .pth hợp lệ ({e})") from e

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


def font(size):
    for p in FONT_PATHS:
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            continue
    return ImageFont.load_default()


def header(width, titles, pw, prob, state):
    img = Image.new("RGB", (width, BAR), (24, 24, 28))
    d = ImageDraw.Draw(img)
    txt = f"Xác suất ngập: {prob * 100:.0f}%  ({'NGẬP' if state else 'không ngập'})"
    d.text((width // 2, 6), txt, font=font(30), fill=(255, 90, 90) if state else (110, 220, 130), anchor="ma")
    for i, t in enumerate(titles):
        d.text((pw * i + pw // 2, BAR - 8), t, font=font(22), fill=(255, 255, 255), anchor="ms")
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def run(path, out_mp4, decider, drawers, titles, pw, cfg):
    """drawers: danh sách hàm frame -> mask bool (hoặc None); mỗi hàm là một nửa video."""
    with Video(path) as vid:
        hyst = Hysteresis(cfg.window_s, cfg.infer_fps, cfg.flood_on, cfg.flood_off, cfg.min_hold_s)
        box = detect_black_border(vid.path) if cfg.crop_black_border else None
        period, next_t = 1.0 / cfg.infer_fps, 0.0
        sm = [MaskSmoother(MASK_SMOOTH) for _ in drawers]
        masks, mt, prob, state, writer, on_times, idx = [None] * len(drawers), 0.0, 0.0, False, None, [], -1
        while True:
            ok, frame = vid.cap.read()
            if not ok:
                break
            frame = crop_to_box(frame, box)
            idx += 1
            t = idx / vid.fps
            h, w = frame.shape[:2]
            ph = max(2, round(h * pw / w) // 2 * 2)
            if writer is None:
                writer = ResultWriter(out_mp4, (pw * len(drawers), BAR + ph), vid.fps)
            if t + 1e-9 >= next_t:
                p = decider.predict(frame)["p_ngap"]
                state = hyst.update(p); prob = hyst.avg
                if state:
                    on_times.append(t)
                    for i, fn in enumerate(drawers):
                        m = fn(frame)
                        raw = _scaled(m, (pw, ph)) if m is not None and m.any() else np.zeros((ph, pw), bool)
                        s = sm[i].update(raw)
                        masks[i] = s if s.any() else None
                else:
                    for s in sm:
                        s.reset()
                    masks = [None] * len(drawers)
                mt = t
                next_t = max(next_t + period, t + period / 2)
            f = cv2.resize(frame, (pw, ph), interpolation=cv2.INTER_AREA)
            fresh = state and t - mt <= MASK_MAX_AGE + 1e-6
            panels = [draw_overlay(f, m) if (fresh and m is not None) else f for m in masks]
            writer.write(np.vstack([header(pw * len(drawers), titles, pw, prob, state), np.hstack(panels)]))
        ok = writer.close()
        return ok, on_times, vid.duration


def main():
    ap = argparse.ArgumentParser(description="So sánh cách tô vùng nước (seg.pt / V-FloodNet) trên video, cùng quyết định ngập của best.pt")
    ap.add_argument("videos", nargs="+", help="đường dẫn video (mp4, ...)")
    ap.add_argument("--model", choices=("seg", "vfloodnet", "both"), default="both", help="seg: chỉ seg.pt; vfloodnet: chỉ V-FloodNet; both: ghép ngang 2 bên (mặc định)")
    ap.add_argument("--out", default=str(HOME / "Videos/ngap/so_sanh"), help="thư mục ra (mặc định ~/Videos/ngap/so_sanh/)")
    a = ap.parse_args()
    vids = [Path(v).expanduser() for v in a.videos]
    missing = [str(v) for v in vids if not v.is_file()]
    if missing:
        sys.exit("Không thấy video: " + ", ".join(missing))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    vfn = None
    if a.model in ("vfloodnet", "both"):
        path = Path(os.environ.get("VFLOODNET_PATH") or DEFAULT_VFN).expanduser()
        try:
            vfn = load_vfloodnet(path, dev)
        except NoVFloodNet as e:
            sys.exit(HOW_TO_DOWNLOAD.format(why=e, default=DEFAULT_VFN))
    cfg = config.load_model_config()
    try:
        dec = build_decider(cfg, config.MODELS_DIR)
    except FileNotFoundError as e:
        sys.exit(f"{e}\nTải best.pt / best_cpu.pt từ Release (xem README, mục Chạy app).")
    seg = SegAdapter(cfg, path=config.MODELS_DIR / cfg.seg_path) if a.model in ("seg", "both") else None
    out = Path(a.out).expanduser(); out.mkdir(parents=True, exist_ok=True)
    fns = {"seg": (lambda f: seg.predict(f)["mask"]) if seg else None, "vfloodnet": (lambda f: vfloodnet_mask(vfn, f, dev)) if vfn else None}
    keys = ["seg", "vfloodnet"] if a.model == "both" else [a.model]
    pw = 640 if len(keys) == 2 else 960
    suffix = {"seg": "seg", "vfloodnet": "vfloodnet", "both": "so_sanh"}[a.model]
    print(f"quyết định: {dec.name} | luật cửa sổ {cfg.window_s:g} s, bật {cfg.flood_on}, tắt {cfg.flood_off}, giữ {cfg.min_hold_s:g} s | vẽ: {', '.join(keys)}", flush=True)
    for v in vids:
        mp4 = out / f"{v.stem}_{suffix}.mp4"
        ok, on, dur = run(v, mp4, dec, [fns[k] for k in keys], [TITLES[k] for k in keys], pw, cfg)
        shot = min((on[0] + on[-1]) / 2 if on else dur / 2, max(0.0, dur - 0.5))      # giữa khoảng đang báo NGẬP
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-nostdin", "-v", "error", "-y", "-ss", f"{shot:.2f}", "-i", str(mp4), "-frames:v", "1", str(mp4.with_suffix(".jpg"))], check=False)
        print(f"{v.name}: {'ok' if ok else 'LỖI ghi video'} -> {mp4} | bật NGẬP {'từ %.1f s' % on[0] if on else 'không bao giờ'} | ảnh chụp giây {shot:.1f}", flush=True)


if __name__ == "__main__":
    main()
