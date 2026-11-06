"""Luật thời gian: p trung bình trượt, trễ bật/tắt, nhãn đoạn, gộp đoạn ngập liền nhau thành sự cố. Thuần hàm, không cần model."""
from collections import deque
from dataclasses import dataclass

import numpy as np

MIN_FRAME_FRAC = 0.5      # đoạn NGẬP khi >= 50% lần suy luận trong đoạn ở trạng thái NGẬP


class Hysteresis:
    """p trung bình trượt n mẫu gần nhất; bật khi >= on, giữ cho tới khi < off.
    hold_s: sau khi bật, giữ NGẬP tối thiểu từng ấy giây (min-hold) rồi mới áp luật tắt; 0 = không giữ."""

    def __init__(self, window_s, infer_fps, on, off, hold_s=0.0):
        self.buf = deque(maxlen=max(1, int(round(window_s * infer_fps))))
        self.on, self.off, self.state, self.avg = on, off, False, 0.0
        self.hold_n = int(round(hold_s * infer_fps))
        self.held = 0                      # số mẫu liên tiếp đã ở trạng thái NGẬP (tính cả mẫu vừa bật)

    def _next(self, avg):
        new = avg >= (self.off if self.state else self.on)
        if self.state and not new and self.held < self.hold_n:
            new = True                     # còn trong thời gian giữ tối thiểu
        return new

    def update(self, p):
        """Nhận p thô của một lần suy luận; cập nhật `avg` (p làm mượt, đúng giá trị so với ngưỡng) và trả trạng thái ngập."""
        self.buf.append(p)
        self.avg = sum(self.buf) / len(self.buf)
        new = self._next(self.avg)
        self.held = self.held + 1 if new else 0
        self.state = new
        return self.state

    def peek(self, p):
        """Trạng thái NGẬP nếu mẫu tiếp theo có p thô này (không đổi trạng thái)."""
        buf = list(self.buf)[-(self.buf.maxlen - 1):] if self.buf.maxlen > 1 else []
        return self._next(sum(buf + [p]) / (len(buf) + 1))


class MaskSmoother:
    """Làm mượt mask nước theo thời gian (kiểu WaterSmoother 0,6): trung bình trượt mũ (smooth = trọng số lịch sử),
    bật điểm khi > on, đã bật thì giữ tới khi < off. Mỗi lần ngập mới tạo bộ mới (reset) để không dính mask cũ."""

    def __init__(self, smooth=0.6, on=0.5, off=0.25):
        self.smooth, self.on, self.off = smooth, on, off
        self.cur = self.state = None

    def update(self, mask):
        cur = mask.astype(np.float32)
        if self.smooth > 0 and self.cur is not None and self.cur.shape == cur.shape:
            cur = (1 - self.smooth) * cur + self.smooth * self.cur
        self.cur = cur
        out = cur > self.on
        if self.smooth > 0 and self.state is not None and self.state.shape == out.shape:
            out |= self.state & (cur > self.off)
        self.state = out
        return out

    def reset(self):
        self.cur = self.state = None


@dataclass
class Span:
    t_start: float
    t_end: float
    segments: list


def segment_flags(times, states, segment_s, min_frac=MIN_FRAME_FRAC):
    """{chỉ số đoạn: bool}. Mỗi đoạn NGẬP khi tỉ lệ mẫu NGẬP >= min_frac."""
    count, flooded = {}, {}
    for t, s in zip(times, states):
        k = int(t // segment_s)
        count[k] = count.get(k, 0) + 1
        flooded[k] = flooded.get(k, 0) + bool(s)
    return {k: flooded[k] / count[k] >= min_frac for k in sorted(count)}


def merge_spans(flags, segment_s, duration):
    """Các đoạn ngập liền nhau -> danh sách Span (giây trong video)."""
    spans, run = [], []
    for k in sorted(flags) + [None]:
        if k is not None and flags[k] and (not run or k == run[-1] + 1):
            run.append(k)
            continue
        if run:
            spans.append(Span(run[0] * segment_s, min((run[-1] + 1) * segment_s, duration), run))
        run = [k] if k is not None and flags[k] else []
    return spans


def pick_evidence_times(samples, t_start, t_end, n=3, min_gap=1.0):
    """samples: [(t, p)]. Tối đa n khung p cao nhất trong [t_start, t_end], cách nhau >= min_gap giây."""
    cand = sorted((s for s in samples if t_start <= s[0] <= t_end), key=lambda s: (-s[1], s[0]))
    chosen = []
    for t, p in cand:
        if all(abs(t - c[0]) >= min_gap for c in chosen):
            chosen.append((t, p))
        if len(chosen) == n:
            break
    return sorted(chosen)
