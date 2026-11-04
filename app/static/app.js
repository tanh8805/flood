async function api(url, opts = {}) {
  const r = await fetch(url, { credentials: 'same-origin', ...opts });
  if (r.status === 401) { location.href = '/login'; throw new Error('Chưa đăng nhập'); }
  const text = await r.text();
  let data = null; try { data = text ? JSON.parse(text) : null; } catch (e) { data = { detail: text }; }
  if (!r.ok) {
    let d = data && data.detail;
    if (Array.isArray(d)) d = d.map(x => (x.loc ? x.loc.slice(1).join('.') + ': ' : '') + x.msg).join('; ');
    throw new Error(d || ('Lỗi ' + r.status));
  }
  return data;
}
const jsonOpts = (method, body) => ({ method, headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
function esc(s) { return String(s ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c])); }
function fmtTime(iso) { return iso ? new Date(iso).toLocaleString('vi-VN') : '—'; }
function fmtSec(s) { if (s == null) return '—'; const m = Math.floor(s / 60); return m + ':' + String(Math.round(s % 60)).padStart(2, '0'); }
function banner(el, kind, text) { el.className = 'banner ' + kind; el.textContent = text; el.hidden = !text; }

function relTime(iso) {
  if (!iso) return '—';
  const s = Math.max(0, (Date.now() - new Date(iso).getTime()) / 1000);
  if (s < 60) return 'vừa xong';
  if (s < 3600) return Math.floor(s / 60) + ' phút trước';
  if (s < 86400) return Math.floor(s / 3600) + ' giờ trước';
  return Math.floor(s / 86400) + ' ngày trước';
}
const STATUS_VI = { new: 'Mới', confirmed: 'Đã xác nhận', false_alarm: 'Báo nhầm' };

/* Thanh xác suất phía trên video: lấy lần suy luận gần nhất <= thời điểm đang phát. series = [[t, p thô, ngập 0/1, p làm mượt], ...] */
function videoBox(src, series, startAt) {
  const box = document.createElement('div');
  box.className = 'vbox';
  box.innerHTML = `${series && series.length ? '<div class="pbar" aria-live="off"><b title="Xác suất đã làm mượt (trung bình trượt), cùng giá trị luật bật/tắt dùng để quyết định Ngập/Không ngập">Xác suất ngập: <span class="pv">—</span></b><span class="pill pl"></span></div>' : ''}
    <video controls preload="metadata" playsinline></video>`;
  const v = box.querySelector('video'); v.src = src;
  if (startAt) v.addEventListener('loadedmetadata', () => { v.currentTime = startAt; }, { once: true });
  if (series && series.length) {
    const pv = box.querySelector('.pv'), pl = box.querySelector('.pl');
    const update = () => {
      let lo = 0, hi = series.length - 1;            // chỉ số lớn nhất có t <= currentTime
      while (lo < hi) { const mid = (lo + hi + 1) >> 1; if (series[mid][0] <= v.currentTime + 1e-6) lo = mid; else hi = mid - 1; }
      // series = [t, p thô, ngập, p làm mượt]: hiển thị p LÀM MƯỢT (giá trị luật bật/tắt so với ngưỡng) để khớp nhãn. Bản ghi cũ (3 cột) chỉ có p thô.
      const s = series[lo], on = s[2], p = s.length > 3 ? s[3] : s[1];
      box.dataset.idx = lo;                           // mẫu đang hiển thị (để kiểm tra/gỡ lỗi)
      pv.textContent = Math.round(p * 100) + '%';
      pl.textContent = on ? 'Ngập' : 'Không ngập';
      pl.className = 'pill pl ' + (on ? 'flood' : 'ok');
    };
    for (const ev of ['timeupdate', 'seeked', 'loadeddata', 'play', 'playing']) v.addEventListener(ev, update);
    update();
  }
  return box;
}
