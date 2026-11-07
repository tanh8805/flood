#!/usr/bin/env bash
# Chọn video bằng hộp thoại (zenity nếu có, không thì nhập đường dẫn), chạy tools/so_sanh_seg.py --model both (seg.pt | V-FloodNet ghép ngang),
# xong thì mở video kết quả. Có thể đưa sẵn video:  ./so_sanh.sh video.mp4
# Cần trọng số V-FloodNet ngoài repo (VFLOODNET_PATH, xem README, mục "So sánh với V-FloodNet"). Biến FLOOD_VENV / FLOOD_SYSTEM_SITE như chay.sh.
set -euo pipefail
cd "$(dirname "$0")"
VIDEO="${1:-}"
if [ -z "$VIDEO" ]; then
  START="$HOME/Videos/ngap/demo/"; [ -d "$START" ] || START="$HOME/Videos/"; [ -d "$START" ] || START="$HOME/"
  if command -v zenity >/dev/null 2>&1; then
    VIDEO="$(zenity --file-selection --title="Chọn video để so sánh cách tô vùng nước" --filename="$START" --file-filter='Video | *.mp4 *.mov *.avi *.mkv *.webm *.m4v' 2>/dev/null)" || { echo "Chưa chọn video."; exit 1; }
  else
    read -r -p "Đường dẫn video (mp4, mov, ...): " VIDEO
  fi
fi
VIDEO="${VIDEO/#\~/$HOME}"
[ -f "$VIDEO" ] || { echo "Không thấy video: $VIDEO"; exit 1; }
./chay.sh --setup >/dev/null            # venv + thư viện + model (không chạy web)
PY="${FLOOD_VENV:-$PWD/.venv}/bin/python"
export PIP_USER=0 PIP_DISABLE_PIP_VERSION_CHECK=1
"$PY" -c "import segmentation_models_pytorch, efficientnet_pytorch" 2>/dev/null || { echo ">> Cài requirements-so-sanh.txt"; "$PY" -m pip install -q -r requirements-so-sanh.txt; }
OUT="${SO_SANH_OUT:-$HOME/Videos/ngap/so_sanh}"
"$PY" tools/so_sanh_seg.py "$VIDEO" --model both --out "$OUT"
RESULT="$OUT/$(basename "${VIDEO%.*}")_so_sanh.mp4"
if [ -f "$RESULT" ]; then
  echo ">> Kết quả: $RESULT"
  [ -n "${NO_OPEN:-}" ] || xdg-open "$RESULT" >/dev/null 2>&1 || true
fi
