"""Dãy [t, p thô, ngập, p làm mượt]: nhận dạng định dạng và tính lại p làm mượt cho bản ghi cũ (chỉ có [t, p thô, ngập])."""
import json
import logging

from sqlalchemy import select

from .models import Check
from .rules import Hysteresis

log = logging.getLogger("series")
P_NOTE = "p thô (bản ghi cũ, chưa tính lại được)"


def kind_of_prefix(prefix: str | None) -> str:
    """'smoothed' (hàng đầu có 4 cột) | 'raw' (3 cột hoặc rỗng/không đọc được). Chỉ cần ~80 ký tự đầu của series_json."""
    if not prefix or not prefix.startswith("[["):
        return "raw"
    row = prefix[1:prefix.find("]") + 1] if "]" in prefix else prefix[1:]
    return "smoothed" if row.count(",") >= 3 else "raw"


def p_note(prefix: str | None) -> str | None:
    return None if kind_of_prefix(prefix) == "smoothed" else P_NOTE


def _align_p_max(c, rows) -> bool:
    """Đặt p_max của lần kiểm tra và của các sự cố thuộc nó = p làm mượt lớn nhất trong dãy `rows` (cột 4). True nếu có đổi."""
    changed = False
    top = max(r[3] for r in rows)
    if abs(c.p_max - top) > 5e-4:
        c.p_max, changed = top, True
    for inc in c.incidents:
        if inc.video_start_s is None or inc.video_end_s is None:
            continue
        inside = [r[3] for r in rows if inc.video_start_s <= r[0] < inc.video_end_s]
        if inside and abs(inc.p_max - max(inside)) > 5e-4:
            inc.p_max, changed = max(inside), True
    return changed


def recompute_legacy(db, cfg) -> dict:
    """Đưa p_max (lần kiểm tra + sự cố thuộc nó) về p LÀM MƯỢT lớn nhất.
    - Dãy cũ chỉ có [t, p thô, ngập]: tính lại p làm mượt từ p thô đã lưu, ghi dãy 4 cột. Chỉ làm khi chạy luật hiện tại trên dãy p thô
      cho ra đúng dãy trạng thái đã lưu (cùng tham số lúc phân tích); không thì để nguyên (giao diện ghi chú p thô).
    - Dãy đã có p làm mượt nhưng p_max còn là p thô (bản trước): chỉ căn chỉnh p_max.
    Chạy lại nhiều lần không đổi gì thêm."""
    converted = aligned = kept_raw = 0
    for c in db.scalars(select(Check)):
        try:
            series = json.loads(c.series_json or "[]")
        except ValueError:
            series = []
        if not series:
            continue
        if len(series[0]) >= 4:
            aligned += _align_p_max(c, series)
            continue
        h, new = Hysteresis(cfg.window_s, cfg.infer_fps, cfg.flood_on, cfg.flood_off), []
        for t, p, on in series:
            if int(h.update(p)) != on:               # tham số khác lúc phân tích: không dựng lại được
                new = None
                break
            new.append([t, p, on, round(h.avg, 3)])
        if new is None:
            kept_raw += 1
            log.warning("lần kiểm tra #%s: không tính lại được p làm mượt, giữ p thô", c.id)
            continue
        c.series_json = json.dumps(new, separators=(",", ":"))
        _align_p_max(c, new)
        converted += 1
    db.commit()
    return {"recomputed": converted, "aligned": aligned, "kept_raw": kept_raw}
