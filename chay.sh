#!/usr/bin/env bash
# Chạy app một lệnh (Linux): tự tạo .venv + cài requirements nếu chưa có, tải model từ Release nếu thiếu, hỏi mật khẩu tạo admin ở lần đầu,
# chạy web ở cổng 8000 (bận thì 8001), đợi server sẵn sàng rồi mở trình duyệt. Ctrl+C để tắt.
#   ./chay.sh            hoặc bấm biểu tượng "Cảnh báo ngập" (cài bằng tools/cai_bieu_tuong.sh)
# Tuỳ chọn (biến môi trường): ADMIN_PASSWORD (không đặt thì hỏi), NO_BROWSER=1, SKIP_MODELS=1, PYTHON=python3.12,
#   FLOOD_VENV=/đường/dẫn/venv (dùng venv có sẵn, không tạo .venv)   FLOOD_SYSTEM_SITE=1 (tạo .venv với --system-site-packages, dùng torch có sẵn)
set -euo pipefail
cd "$(dirname "$0")"
PYTHON="${PYTHON:-python3}"
VENV="${FLOOD_VENV:-$PWD/.venv}"
MODELS="${MODELS_DIR:-$PWD/models}"
RELEASE="v1.3"; REPO="tanh8805/flood"
export PIP_USER=0 PIP_DISABLE_PIP_VERSION_CHECK=1

# 1. venv + thư viện
if [ -n "${FLOOD_VENV:-}" ]; then
  [ -x "$VENV/bin/python" ] || { echo "FLOOD_VENV=$VENV không có bin/python"; exit 1; }
elif [ ! -x "$VENV/bin/python" ]; then
  echo ">> Tạo $VENV"
  "$PYTHON" -m venv ${FLOOD_SYSTEM_SITE:+--system-site-packages} "$VENV"
fi
PY="$VENV/bin/python"
if ! "$PY" -c "import fastapi, uvicorn, sqlalchemy, ultralytics, cv2, imageio_ffmpeg" 2>/dev/null; then
  echo ">> Cài thư viện (requirements.txt), lần đầu có thể mất vài phút"
  "$PY" -m pip install -q -r requirements.txt
fi

# 2. model: thiếu best.pt hoặc best_cpu.pt (hoặc seg.pt) thì tải từ Release
if [ -z "${SKIP_MODELS:-}" ]; then
  mkdir -p "$MODELS"
  for f in best.pt best_cpu.pt seg.pt; do
    [ -s "$MODELS/$f" ] && continue
    echo ">> Tải $f từ Release $RELEASE"
    if command -v gh >/dev/null 2>&1; then gh release download "$RELEASE" -R "$REPO" -p "$f" -D "$MODELS" --clobber
    else curl -fL --progress-bar -o "$MODELS/$f" "https://github.com/$REPO/releases/download/$RELEASE/$f"; fi
  done
fi
[ "${1:-}" = "--setup" ] && { echo "Đã chuẩn bị xong."; exit 0; }

# 3. chưa có admin thì hỏi mật khẩu để tạo
if ! "$PY" -c "import sys; from sqlalchemy import select; from app.db import Base, SessionLocal, engine; from app.models import User; Base.metadata.create_all(engine); sys.exit(0 if SessionLocal().scalar(select(User).where(User.role == 'admin')) else 1)"; then
  echo ">> Chưa có tài khoản admin: tạo 'admin' (mật khẩu tối thiểu 8 ký tự)"
  "$PY" manage.py create-admin
fi

# 4. cổng 8000, bận thì 8001
busy() { "$PY" -c "import socket,sys; s=socket.socket(); s.settimeout(1); sys.exit(0 if s.connect_ex(('127.0.0.1',$1))==0 else 1)"; }
PORT=8000
if busy 8000; then PORT=8001; echo ">> Cổng 8000 đang bận, dùng 8001"; fi
if busy "$PORT"; then echo "Cổng $PORT cũng bận. Tắt chương trình đang dùng cổng đó rồi chạy lại."; exit 1; fi
URL="http://localhost:$PORT"

# 5. đợi server sẵn sàng rồi mở trình duyệt (chạy nền), sau đó chạy web (Ctrl+C để tắt)
(
  for _ in $(seq 1 120); do
    if "$PY" -c "import urllib.request as u; u.urlopen('http://127.0.0.1:$PORT/login', timeout=1)" 2>/dev/null; then
      echo ">> Sẵn sàng: $URL"
      [ -n "${NO_BROWSER:-}" ] || xdg-open "$URL" >/dev/null 2>&1 || echo "(không mở được trình duyệt tự động, hãy mở $URL)"
      exit 0
    fi
    sleep 1
  done
) &
echo ">> Đang khởi động web ($URL), đăng nhập bằng tài khoản admin. Ctrl+C để tắt."
exec "$PY" -m uvicorn app.main:app --port "$PORT"
