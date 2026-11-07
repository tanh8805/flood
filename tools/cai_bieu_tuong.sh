#!/usr/bin/env bash
# Cài biểu tượng "Cảnh báo ngập" vào menu ứng dụng (Linux): tạo ~/.local/share/applications/canh-bao-ngap.desktop (Terminal=true, Exec trỏ chay.sh).
#   tools/cai_bieu_tuong.sh [đường_dẫn_ảnh_biểu_tượng]    ảnh tuỳ chọn (svg/png); mặc định app/static/icon.svg
# Bỏ biểu tượng: rm ~/.local/share/applications/canh-bao-ngap.desktop
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ICON="${1:-$ROOT/app/static/icon.svg}"
[ -f "$ICON" ] || { echo "Không thấy ảnh biểu tượng: $ICON (bỏ trống đối số để dùng mặc định)"; exit 1; }
chmod +x "$ROOT/chay.sh"
DIR="${XDG_DATA_HOME:-$HOME/.local/share}/applications"; mkdir -p "$DIR"
F="$DIR/canh-bao-ngap.desktop"
cat > "$F" <<DESK
[Desktop Entry]
Type=Application
Name=Cảnh báo ngập
Comment=Chạy web app cảnh báo ngập đường phố (Ctrl+C để tắt)
Exec="$ROOT/chay.sh"
Path=$ROOT
Icon=$ICON
Terminal=true
Categories=Utility;
DESK
chmod +x "$F"
command -v update-desktop-database >/dev/null 2>&1 && update-desktop-database "$DIR" >/dev/null 2>&1 || true
echo "Đã tạo $F"
echo "Tìm 'Cảnh báo ngập' trong menu ứng dụng; bấm vào là app tự chạy và mở trình duyệt."
