"""Cắt viền đen (letterbox / pillarbox) của video trước khi đưa vào best.pt và seg.pt.

Hộp cắt tính MỘT lần cho cả video từ vài khung rải đều: một cột/hàng chỉ bị coi là viền khi nó đen thuần (trung bình thấp VÀ gần như không đổi trên chiều kia)
ở MỌI khung lấy mẫu, và nằm liền từ mép khung. Nhờ vậy cảnh tối/ban đêm (có nhiễu, có điểm sáng) không bị cắt nhầm; cảnh đen hoàn toàn thì không cắt.
Không cắt nếu mỗi trục viền < MIN_BORDER (2%) hoặc phần nội dung còn lại nhỏ hơn MIN_CONTENT (50%) mỗi chiều.
Thuần numpy/cv2, không phụ thuộc phần còn lại của app (handoff/crop_black_border.py là bản sao)."""
import cv2
import numpy as np

BLACK_MEAN = 10.0     # độ sáng trung bình (0-255) của cột/hàng viền phải thấp hơn
BLACK_STD = 3.0       # và độ lệch chuẩn dọc theo cột/hàng phải thấp hơn (viền đen thuần, không có họa tiết/nhiễu)
MIN_BORDER = 0.02     # viền mỗi trục (tổng hai phía) dưới tỉ lệ này thì bỏ qua
MIN_CONTENT = 0.5     # phần nội dung còn lại phải >= tỉ lệ này mỗi chiều
N_SAMPLES = 9


def _stats(frame):
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return g.mean(0), g.std(0), g.mean(1), g.std(1)       # cột: mean, std; hàng: mean, std


def box_from_frames(frames):
    """frames: danh sách khung BGR cùng kích thước. Trả (x0, y0, x1, y1) (x1, y1 loại trừ) hoặc None nếu không cần cắt."""
    frames = [f for f in frames if f is not None]
    if not frames:
        return None
    h, w = frames[0].shape[:2]
    st = [_stats(f) for f in frames if f.shape[:2] == (h, w)]
    if not st:
        return None
    col_black = (np.max([s[0] for s in st], 0) < BLACK_MEAN) & (np.max([s[1] for s in st], 0) < BLACK_STD)
    row_black = (np.max([s[2] for s in st], 0) < BLACK_MEAN) & (np.max([s[3] for s in st], 0) < BLACK_STD)

    def run(mask):
        """Số phần tử đen liền từ đầu và từ cuối."""
        n = len(mask)
        a = 0
        while a < n and mask[a]:
            a += 1
        if a == n:
            return n, 0
        b = 0
        while b < n and mask[n - 1 - b]:
            b += 1
        return a, b

    l, r = run(col_black)
    t, b = run(row_black)
    if l + r >= w or t + b >= h:
        return None                                       # cảnh đen hoàn toàn
    if (l + r) < MIN_BORDER * w:
        l = r = 0
    if (t + b) < MIN_BORDER * h:
        t = b = 0
    x0, x1, y0, y1 = l, w - r, t, h - b
    if (x1 - x0) < MIN_CONTENT * w or (y1 - y0) < MIN_CONTENT * h:
        return None
    if (x0, y0, x1, y1) == (0, 0, w, h):
        return None
    return int(x0), int(y0), int(x1), int(y1)


def detect_black_border(path, n=N_SAMPLES):
    """Mở video `path`, lấy n khung rải đều (bỏ khung đầu/cuối 2%), trả hộp cắt hoặc None. Không đụng tới con trỏ đọc của nơi khác."""
    cap = cv2.VideoCapture(str(path))
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        idx = sorted({int(total * (0.02 + 0.96 * i / max(1, n - 1))) for i in range(n)}) if total > 0 else [0]
        frames = []
        for i in idx:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, f = cap.read()
            if ok:
                frames.append(f)
        return box_from_frames(frames)
    finally:
        cap.release()


def crop_to_box(frame, box):
    """Cắt khung theo hộp (None = giữ nguyên). Trả view, không sao chép."""
    if frame is None or box is None:
        return frame
    x0, y0, x1, y1 = box
    return frame[y0:y1, x0:x1]
