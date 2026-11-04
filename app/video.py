"""Đọc video bằng OpenCV; không đọc được thì chuyển H.264 bằng ffmpeg của imageio-ffmpeg rồi đọc lại."""
import subprocess
import tempfile
from pathlib import Path

import cv2
import imageio_ffmpeg

UNREADABLE = "Không đọc được video (định dạng không hỗ trợ hoặc file hỏng)"


def _readable(path):
    cap = cv2.VideoCapture(str(path))
    try:
        return cap.isOpened() and cap.grab()
    finally:
        cap.release()


def transcode_h264(src, dst):
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-v", "error", "-i", str(src), "-an", "-c:v", "libx264", "-preset", "veryfast",
           "-crf", "23", "-pix_fmt", "yuv420p", "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", str(dst)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 or not Path(dst).exists():
        raise ValueError(UNREADABLE)
    return Path(dst)


class Video:
    """Context manager: mở video (có chuyển H.264 nếu cần), dọn file tạm khi đóng."""

    def __init__(self, path):
        self.src, self.path, self._tmp = Path(path), Path(path), None
        if not _readable(self.path):
            self._tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
            self._tmp.close()
            try:
                self.path = transcode_h264(self.src, self._tmp.name)
                if not _readable(self.path):
                    raise ValueError(UNREADABLE)
            except Exception:
                self.close()
                raise
        self.cap = cv2.VideoCapture(str(self.path))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if not self.fps or self.fps != self.fps or self.fps > 240:
            self.fps = 25.0
        self.frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.duration = self.frames / self.fps if self.frames > 0 else 0.0

    def frame_at(self, t):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, round(t * self.fps)))
        ok, f = self.cap.read()
        return f if ok else None

    def close(self):
        if getattr(self, "cap", None):
            self.cap.release()
        if self._tmp:
            Path(self._tmp.name).unlink(missing_ok=True)
            self._tmp = None

    __enter__ = lambda self: self

    def __exit__(self, *a):
        self.close()
