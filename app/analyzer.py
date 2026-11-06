"""Video -> sự cố ngập + minh chứng. Chỉ biết interface predict(frame) của model, không biết model là gì."""
import logging
import subprocess
from dataclasses import dataclass, field

import cv2
import imageio_ffmpeg
import numpy as np

from .crop import crop_to_box, detect_black_border
from .rules import Hysteresis, MaskSmoother, merge_spans, pick_evidence_times, segment_flags
from .video import Video

OVERLAY_COLOR = (255, 160, 0)   # BGR
CROP_PAD = 0.10
JPEG_QUALITY = 88
CLIP_WIDTH = 854
MASK_SMOOTH = 0.6        # trọng số lịch sử khi làm mượt mask nước giữa các lần suy luận (WaterSmoother 0,6)
MASK_MAX_AGE = 0.5      # giây: không vẽ mask của lần suy luận cũ hơn thế lên khung hiện tại


@dataclass
class Evidence:
    t: float
    p: float
    original: bytes
    overlay: bytes | None
    crop: bytes


@dataclass
class IncidentResult:
    t_start: float
    t_end: float
    p_max: float
    evidence: list[Evidence] = field(default_factory=list)


def jpeg(img):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    return buf.tobytes()


def draw_overlay(frame, mask, thickness=2):
    """Tô màu bán trong suốt + viền lên vùng mask. Không vẽ chữ."""
    out = frame.copy()
    out[mask] = (0.55 * out[mask] + 0.45 * np.array(OVERLAY_COLOR)).astype(np.uint8)
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, cnts, -1, OVERLAY_COLOR, thickness)
    return out


def crop_to_mask(frame, mask):
    """Cắt theo bbox của mask nới 10% mỗi chiều; không có mask thì trả ảnh gốc."""
    if mask is None or not mask.any():
        return frame
    ys, xs = np.where(mask)
    h, w = mask.shape
    pad_x, pad_y = int((xs.max() - xs.min() + 1) * CROP_PAD), int((ys.max() - ys.min() + 1) * CROP_PAD)
    x0, x1 = max(0, xs.min() - pad_x), min(w, xs.max() + 1 + pad_x)
    y0, y1 = max(0, ys.min() - pad_y), min(h, ys.max() + 1 + pad_y)
    return frame[y0:y1, x0:x1]


def make_evidence(frame, t, p, mask) -> Evidence:
    has = mask is not None and mask.any()
    return Evidence(t, p, jpeg(frame), jpeg(draw_overlay(frame, mask)) if has else None, jpeg(crop_to_mask(frame, mask if has else None)))


class ResultWriter:
    """Ghi video kết quả H.264 qua ffmpeg (pipe rawvideo). Lỗi ghi không làm hỏng phân tích: ok=False và file bị xoá."""

    def __init__(self, out_path, size, fps):
        self.out, self.size, self.ok = out_path, size, True
        cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{size[0]}x{size[1]}",
               "-r", f"{fps:.3f}", "-i", "-", "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "24", "-pix_fmt", "yuv420p",
               "-movflags", "+faststart", str(out_path)]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

    def write(self, frame):
        if self.ok:
            try:
                self.proc.stdin.write(frame.tobytes())
            except OSError:
                self.ok = False

    def close(self, keep=True):
        try:
            self.proc.stdin.close()
        except OSError:
            self.ok = False
        self.ok = self.proc.wait() == 0 and self.ok and keep
        if not self.ok:
            self.out.unlink(missing_ok=True)
        return self.ok


def _scaled(mask, size):
    """Mask bool của khung gốc -> bool theo kích thước khung xuất (viền mượt)."""
    return cv2.resize(mask.astype(np.uint8) * 255, size, interpolation=cv2.INTER_AREA) > 127


def analyze_video(path, model, cfg, progress=None, video_out=None):
    """Trả (danh sách IncidentResult, info). info: duration, p_max (p LÀM MƯỢT lớn nhất, giá trị hiển thị), p_raw_max (p thô lớn nhất, chỉ để lưu), series [[t, p thô, ngập 0/1, p làm mượt]] của MỌI lần suy luận, video (đã ghi hay chưa).
    Nếu có `video_out`: ghi video kết quả của TOÀN BỘ video; model có mask thì vẽ vùng nước (mask của lần suy luận gần nhất <= khung),
    không vẽ chữ. progress(frac 0..1, thông báo)."""
    progress = progress or (lambda f, m: None)
    with Video(path) as vid:
        hyst = Hysteresis(cfg.window_s, cfg.infer_fps, cfg.flood_on, cfg.flood_off, getattr(cfg, "min_hold_s", 0.0))
        # viền đen (letterbox/pillarbox): cắt trước cả best.pt và seg.pt; hộp tính một lần cho cả video; info["crop"] = hộp (x0, y0, x1, y1) hoặc None
        box = detect_black_border(vid.path) if getattr(cfg, "crop_black_border", False) else None
        period = 1.0 / cfg.infer_fps
        samples, states, smooth, next_t, idx = [], [], [], 0.0, -1
        writer, mask_small, mask_t, size = None, None, 0.0, None
        smoother = MaskSmoother(MASK_SMOOTH)
        try:
            while True:
                ok, frame = vid.cap.read()
                if not ok:
                    break
                frame = crop_to_box(frame, box)
                idx += 1
                t = idx / vid.fps
                if video_out is not None and writer is None:
                    h, w = frame.shape[:2]
                    k = min(1.0, CLIP_WIDTH / w)
                    size = (max(2, int(w * k) // 2 * 2), max(2, int(h * k) // 2 * 2))
                    writer = ResultWriter(video_out, size, vid.fps)
                if t + 1e-9 >= next_t:
                    # chỉ chạy model vẽ khi đang NGẬP (Composite nhận draw_if); model khác (không có draw_if) vẫn trả mask, analyzer lọc bên dưới
                    r = model.predict(frame, draw_if=hyst.peek) if getattr(model, "gated_mask", False) else model.predict(frame)
                    samples.append((t, r["p_ngap"]))
                    states.append(hyst.update(r["p_ngap"]))
                    smooth.append(hyst.avg)                      # p làm mượt: đúng giá trị luật bật/tắt so với ngưỡng
                    if writer:                          # mask dùng chung cho mọi khung tới lần suy luận kế tiếp
                        if states[-1]:                  # CHỈ vẽ khi đang NGẬP; mask làm mượt theo thời gian (WaterSmoother 0,6)
                            raw = _scaled(r["mask"], size) if r["mask"] is not None and r["mask"].any() else np.zeros(size[::-1], bool)
                            sm = smoother.update(raw)
                            mask_small = sm if sm.any() else None
                        else:
                            smoother.reset()
                            mask_small = None
                        mask_t = t
                    next_t = max(next_t + period, t + period / 2)
                    if vid.frames and len(samples) % 10 == 0:
                        progress(min(0.9, 0.9 * idx / vid.frames), f"Đang phân tích {t:.0f}s")
                if writer:
                    f = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
                    fresh = mask_small is not None and t - mask_t <= MASK_MAX_AGE + 1e-6
                    writer.write(draw_overlay(f, mask_small) if fresh else f)
        except BaseException:
            if writer:
                writer.close(keep=False)
            raise
        video_ok = writer.close() if writer else False
        if not samples:
            if video_out is not None:
                video_out.unlink(missing_ok=True)
            raise ValueError("Video không có khung hình nào")
        duration = max(vid.duration, samples[-1][0] + period)
        flags = segment_flags([x[0] for x in samples], states, cfg.segment_s)
        spans = merge_spans(flags, cfg.segment_s, duration)
        # [t, p thô, ngập 0/1, p làm mượt]: trang web hiển thị p làm mượt để % và nhãn luôn nhất quán
        series = [[round(t, 2), round(p, 3), int(st), round(sm, 3)] for (t, p), st, sm in zip(samples, states, smooth)]
        info = {"duration": round(duration, 2), "p_max": max(smooth), "p_raw_max": max(p for _, p in samples), "series": series, "video": video_ok,
                "segments": len(flags), "flooded_segments": sum(flags.values()), "samples": len(samples), "crop": box}

        progress(0.92, "Cắt ảnh minh chứng")
        results = []
        for sp in spans:
            inside = [x for x in samples if sp.t_start <= x[0] < sp.t_end]
            smooth_in = [sm for (t, _), sm in zip(samples, smooth) if sp.t_start <= t < sp.t_end]
            res = IncidentResult(sp.t_start, sp.t_end, max(smooth_in))       # độ tin cậy = p làm mượt lớn nhất trong đoạn
            for t, p in pick_evidence_times(inside, sp.t_start, sp.t_end):
                frame = crop_to_box(vid.frame_at(t), box)
                if frame is not None:
                    res.evidence.append(make_evidence(frame, t, p, model.predict(frame)["mask"]))
            results.append(res)
        progress(1.0, "Xong")
        return results, info
