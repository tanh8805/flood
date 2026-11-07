"""Dùng cho chay.bat: đợi web sẵn sàng ở cổng PORT rồi mở trình duyệt. python tools/mo_trinh_duyet.py PORT"""
import sys
import time
import urllib.request
import webbrowser

port = sys.argv[1] if len(sys.argv) > 1 else "8000"
for _ in range(120):
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{port}/login", timeout=1)
        print(f">> Sẵn sàng: http://localhost:{port}", flush=True)
        webbrowser.open(f"http://localhost:{port}")
        break
    except Exception:
        time.sleep(1)
