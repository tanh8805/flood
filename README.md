# Cảnh báo ngập đường phố (khung web app, model cắm-rút)

Uploader gửi video theo khu vực → hệ thống phân tích → nếu ngập thì lưu **sự cố ngập** kèm ảnh minh chứng.
Viewer xem danh sách khu vực (khu vực đang ngập lên đầu) và bấm vào xem minh chứng. Admin quản lý khu vực, tài khoản và model.

## Kiến trúc
```
 trình duyệt ──► FastAPI (app/main.py) ──► routers/pages  (Jinja2: đăng nhập, tổng quan, danh sách khu vực, upload, chi tiết, quản trị)
 thiết bị ngoài ─ X-API-Key ─►            routers/api    (sự cố, lần kiểm tra, khu vực, job, ảnh/video có kiểm tra đăng nhập)
                                          routers/admin  (khu vực, tài khoản, model)
        upload video ─► jobs.JobRunner (1 luồng, 1 video một lúc)
                          └► analyzer.analyze_video ─► rules (làm mượt, trễ bật/tắt, đoạn 2 s, gộp sự cố, chọn 3 khung)
                                  │  predict(frame) -> {p_ngap, mask|None}
                                  ▼
                          adapters/  quyết định: best.pt (TorchScript)   ·   tô vùng nước: seg.pt   ← models/
                          └► incidents.create_incident  (hàm DUY NHẤT ghi sự cố, POST /api/incidents cũng gọi)
                                  ▼
                          SQLAlchemy (SQLite | PostgreSQL)  +  storage/ (ảnh, video kết quả, video tạm, secret.key)
```

## Chạy nhanh (một click)
- **Linux:** chạy `tools/cai_bieu_tuong.sh` một lần để cài biểu tượng, rồi bấm biểu tượng **"Cảnh báo ngập"** trong menu ứng dụng. Hoặc chạy thẳng `./chay.sh`. Lần đầu script tự tạo `.venv`, cài `requirements.txt`, tải `best.pt` và `best_cpu.pt` từ Release v1.3 (bằng `gh`, không có thì `curl`) nếu thiếu, hỏi mật khẩu để tạo tài khoản `admin`; sau đó chạy web ở cổng 8000 (bận thì 8001), đợi sẵn sàng rồi tự mở trình duyệt. Ctrl+C để tắt.
- **Windows:** double-click `chay.bat` (cần Python 3.12 với `py` launcher; làm tương đương: `.venv`, cài thư viện, tải model, tạo admin, chạy web, mở trình duyệt).
- **So sánh cách tô vùng nước:** `./so_sanh.sh` mở hộp thoại chọn video (zenity, hoặc nhập đường dẫn), chạy `tools/so_sanh_seg.py --model both` rồi mở video kết quả (cần trọng số V-FloodNet, xem mục "So sánh với V-FloodNet").

## Chạy app
Cần Python 3.12 (hoặc Docker, xem dưới). Bốn lệnh từ clone tới trình duyệt:
```
git clone https://github.com/tanh8805/flood.git && cd flood
python -m venv venv && . venv/bin/activate && pip install -r requirements.txt     # có GPU: cài torch bản CUDA trước
gh release download v1.3 -R tanh8805/flood -p best.pt -p best_cpu.pt -D models/ --clobber
ADMIN_PASSWORD='<mật khẩu ≥ 8 ký tự>' python manage.py create-admin && uvicorn app.main:app --port 8000     # rồi mở http://localhost:8000
```
Đăng nhập bằng tài khoản `admin` với mật khẩu vừa đặt. Test: `pip install pytest httpx && python -m pytest tests -q`.

Các file model nằm ở `models/` (xem `models/README.md`):

| file | trong git? | vai trò |
|---|---|---|
| `models/seg.pt` (~20 MB) | có | chỉ tô vùng nước |
| `models/best.pt` (~580 MB, CLIP ViT-L/14 fp16, > 100 MB nên không đẩy lên git) | **không**, đính kèm **Release v1.3** | quyết định ngập khi có GPU |
| `models/best_cpu.pt` (~335 MB, CLIP ViT-B/32 fp32) | **không**, đính kèm **Release v1.3** | quyết định ngập khi không có GPU (app tự chọn) |

Không có `gh` thì tải bằng curl từ `https://github.com/tanh8805/flood/releases/download/v1.3/<tên file>`. **Chọn model theo thiết bị:** có GPU (CUDA) thì dùng `best.pt` (L-14, fp16); không có GPU (hoặc `MODEL_DEVICE=cpu`) thì dùng `best_cpu.pt` (B-32 fp32, ~74 ms/khung trên CPU, trong khi L-14 trên CPU ~0,9 s/khung). Thiếu `best_cpu.pt` thì dùng `best.pt` cho cả hai (rất chậm trên CPU). **Luật bật/tắt nằm trong chính file model** (mục `rule` của `config.json` nhúng) nên mỗi model dùng luật riêng. Thiếu cả hai thì app vẫn chạy nhưng trang Gửi video báo "Chưa có model".

### Chạy bằng Docker
```
docker compose up -d --build
docker compose exec -e ADMIN_PASSWORD='<mật khẩu của bạn>' app python manage.py create-admin
```
`models/` được mount từ máy chủ (nên đã tải model ở trên); ảnh và DB nằm trong volume `flood-storage`. Image dùng torch bản CPU (tự dùng `best_cpu.pt`). Đặt `DEVICE_API_KEY` (và `SECRET_KEY` nếu muốn) trong môi trường hoặc file `.env` (không đưa vào git).

### Dùng
Đăng nhập bằng admin (vào trang **Tổng quan**), mở **Khu vực & tài khoản** để tạo khu vực (chỉ cần tên và mô tả) và tài khoản. Mật khẩu luôn do bạn đặt: repo không có mật khẩu mẫu nào. Test dùng mật khẩu ngẫu nhiên mỗi lần chạy (đặt `TEST_ADMIN_PASSWORD`, `TEST_USER_PASSWORD` nếu muốn cố định). Không dùng bản đồ nên app chạy được hoàn toàn offline.

**Trang Khu vực** là danh sách khu vực: khu vực có sự cố trong `RECENT_HOURS` giờ qua lên đầu với nhãn đỏ "Đang ngập", còn lại nhãn xám "Bình thường". Mỗi dòng có lần ngập gần nhất ("12 phút trước"), số sự cố trong khoảng đó và một ảnh minh chứng nhỏ của sự cố mới nhất. Có ô lọc theo tên và tự làm mới mỗi 30 giây. Bấm khu vực để xem danh sách sự cố, bấm tiếp để xem chi tiết.

## Tài khoản và vai trò
Mỗi vai trò vào một trang riêng sau khi đăng nhập và bị chặn ở backend (API trả 403), không chỉ ẩn nút.

| role | trang đầu | làm được |
|---|---|---|
| `admin` | Tổng quan | menu: Tổng quan (số khu vực đang ngập, sự cố MỚI cần xử lý với nút Xác nhận / Báo nhầm, lần kiểm tra gần đây, trạng thái model), Sự cố, Khu vực & tài khoản, Gửi video; đổi trạng thái / xoá sự cố; xem mọi lần kiểm tra |
| `uploader` | Gửi video | Gửi video, lịch sử lần gửi và lần kiểm tra của chính mình, Khu vực và chi tiết sự cố |
| `viewer` | Khu vực | chỉ Khu vực (danh sách cảnh báo) và chi tiết sự cố; không thấy Gửi video, Khu vực & tài khoản, nút đổi trạng thái |

Sự cố "báo nhầm" không làm khu vực "Đang ngập".

Tạo tài khoản: admin đầu tiên bằng `python manage.py create-admin`; các tài khoản khác ở trang **Khu vực & tài khoản** hoặc `POST /api/admin/users` (JSON `{"username", "password" (≥ 8 ký tự), "role"}`, cần đăng nhập admin). Thiết bị ngoài không cần tài khoản: gửi `X-API-Key` (xem mục API).

## Cấu trúc thư mục
```
app/                    web app (best.pt quyết định, seg.pt tô, chỉ tô khi NGẬP)
models/                 best.pt, best_cpu.pt, seg.pt (+ README.md: tải từ Release)
tools/so_sanh_seg.py    video so sánh cách tô vùng nước (xem bên dưới), không thuộc app
docs/ket_qua.md         bảng các thí nghiệm, bảng 4 cách tô, hạn chế
tests/                  pytest (phân quyền, adapter, phân tích video, API)
requirements.txt        app;  requirements-so-sanh.txt: thêm cho V-FloodNet
chay.sh  chay.bat  so_sanh.sh   chạy nhanh (xem trên);  tools/cai_bieu_tuong.sh: biểu tượng menu Linux
manage.py  Dockerfile  docker-compose.yml
```

## So sánh với V-FloodNet
`tools/so_sanh_seg.py` dựng video so sánh cách tô vùng nước của nhóm (`seg.pt`) với **V-FloodNet** (model ảnh pretrained, LinkNet + EfficientNet-B4). Cả hai dùng chung quyết định ngập của `best.pt`; chỉ tô khi đang NGẬP; % ngập ghi phía trên video. Không thuộc app, app không dùng V-FloodNet.

**Trọng số V-FloodNet (không nằm trong repo, tải từ nguồn gốc):** tải `records.zip` (~800 MB) ở https://huggingface.co/xmlyqing00/V-FloodNet/tree/main (hoặc thư mục Google Drive trong README của https://github.com/xmlyqing00/V-FloodNet), giải nén lấy `records/link_efficientb4_model.pth`, đặt ngoài repo và trỏ bằng biến môi trường:
```
export VFLOODNET_PATH=~/models_ngoai/vfloodnet/records/link_efficientb4_model.pth     # mặc định chính đường dẫn này
pip install -r requirements-so-sanh.txt
```
Ba lệnh (`--out` = thư mục ra, mặc định `~/Videos/ngap/so_sanh/`; mỗi video ra kèm 1 ảnh `.jpg`):
```
python tools/so_sanh_seg.py video.mp4 --model seg         # video chỉ tô bằng seg.pt        -> video_seg.mp4
python tools/so_sanh_seg.py video.mp4 --model vfloodnet   # video chỉ tô bằng V-FloodNet    -> video_vfloodnet.mp4
python tools/so_sanh_seg.py video.mp4 --model both        # ghép ngang 2 bên (mặc định)     -> video_so_sanh.mp4
```
Thiếu trọng số hoặc thư viện thì script báo rõ cách tải rồi dừng (không crash); `--model seg` không cần V-FloodNet. Kết quả đo (IoU, % tô nhầm) nằm ở `docs/ket_qua.md`, mục 4: V-FloodNet tô nhầm nhiều hơn `seg.pt` trên bộ khung của nhóm.

**Trích dẫn.** Liang, Y., Li, X., Tsai, B., Chen, Q., Jafari, N. (2023). *V-FloodNet: A video segmentation system for urban flood detection and quantification.* Environmental Modelling & Software, 160, 105586.
```
@article{liang2023v,
  title={V-FloodNet: A video segmentation system for urban flood detection and quantification},
  author={Liang, Yongqing and Li, Xin and Tsai, Brian and Chen, Qin and Jafari, Navid},
  journal={Environmental Modelling \& Software},
  volume={160},
  pages={105586},
  year={2023},
  publisher={Elsevier}
}
```
Mã và trọng số thuộc tác giả (README gốc ghi "All rights are reserved"), nên repo này không phân phối lại trọng số.

## Hai model, hai vai trò tách bạch
| Vai trò | File | Ảnh hưởng |
|---|---|---|
| **Quyết định ngập (tính %)** | `models/best.pt` (GPU, L-14) hoặc `models/best_cpu.pt` (CPU, B-32): TorchScript, `p = m(khung BGR)` (khung bất kỳ cỡ nào, model tự resize/chuẩn hoá; **không** tự đổi RGB hay resize ở ngoài) | p, luật bật/tắt, nhãn Ngập/Không ngập, sự cố |
| **Tô vùng nước** | `models/seg.pt` (Ultralytics **segment**, lớp `water`) | Chỉ tô vùng nước lên video và ảnh minh chứng; không ảnh hưởng %. Không có `seg.pt` thì video giữ khung gốc. |

Không cần `open_clip`: `best.pt` chứa sẵn backbone. Lý do tách: model nước đo diện tích nước nên báo nhầm đường ướt ban đêm; `best.pt` phân biệt được chuyện đó.

- **Luật ngập (mặc định, đọc từ model đang dùng).** 5 khung/giây; **L-14 (`best.pt`): p trung bình trong 1 giây, bật ≥ 0,9, tắt < 0,3. B-32 (`best_cpu.pt`): trong 2 giây, bật ≥ 0,9, tắt < 0,3**; **sau khi bật, giữ NGẬP tối thiểu 10 giây** (`min_hold_s`, chống chớp tắt) rồi mới áp luật tắt; báo theo đoạn 2 giây (đoạn ngập khi ≥ 50% mẫu đang bật), các đoạn ngập liền nhau gộp thành 1 sự cố. Luật chọn bằng kiểm chéo lồng trên dữ liệu train (precision ≥ 0,98 trước, rồi recall tối đa); số liệu trong `docs/ket_qua.md`.
- **Cắt viền đen.** Trước cả `best.pt` và `seg.pt`, viền đen letterbox/pillarbox được cắt (`crop_black_border`, mặc định bật; `app/crop.py`). Hộp cắt tính một lần từ 9 khung rải đều; cột/hàng chỉ là viền khi đen thuần ở mọi khung và liền từ mép khung, nên cảnh ban đêm không bị cắt nhầm; không cắt nếu viền < 2% hoặc phần còn lại < 50% mỗi chiều. Video kết quả và ảnh minh chứng là phần đã cắt; `info["crop"]` ghi hộp cắt.
- **Cách tô vùng nước.** Chỉ tô khi đang **NGẬP**; không ngập thì `seg.pt` không chạy và video giữ khung gốc. Suy luận: `conf=0.4` (`water_conf`), `retina_masks=True`, khung rộng hơn 960 px được thu về rộng 960 (`draw_width`, 0 = không thu). Mask được làm sạch (open/close, bỏ mảng nhỏ hơn 1% khung) rồi làm mượt theo thời gian như `WaterSmoother(0.6)` (không nhấp nháy, reset mỗi lần ngập mới). Ảnh minh chứng dùng mask chưa làm mượt của chính khung đó.
- **Đổi model = chép file.** Mỗi model tự nạp lại khi file của nó đổi (mtime), không cần khởi động lại và không có nút/API nạp lại. Chép `seg.pt` mới thì chỉ model tô nạp lại, chép `best.pt` thì chỉ model quyết định nạp lại. Video đang phân tích dở vẫn dùng model cũ. File lỗi chỉ bị nạp lại khi file đổi tiếp. Lỗi ở `seg.pt` không làm mất việc quyết định.
- **Trang Khu vực & tài khoản** hiển thị (chỉ đọc) model nào đang đảm nhiệm vai trò nào và các ngưỡng.
- **Thêm adapter mới:** viết lớp có `predict(frame_bgr) -> {"p_ngap": float, "mask": bool HxW | None}` và đăng ký trong `app/adapters/__init__.py` (`build_decider` / `build_drawer`). Phần còn lại của app chỉ dùng interface này (qua `Composite`).

### Tham số (`models/model.yaml`, tuỳ chọn)
File YAML phẳng, mọi khoá đều tuỳ chọn: `path`, `cpu_path`, `seg_path`, `device` (auto/cpu/cuda), `water_classes`, `water_conf`, `draw_width`, `flood_on`, `flood_off`, `window_s`, `min_hold_s`, `crop_black_border`, `segment_s`, `infer_fps`. Ví dụ: `flood_on: 0.95`.
Thứ tự ưu tiên: mặc định < luật nhúng trong file model đang dùng < `model.yaml` < biến môi trường (`MODEL_PATH`, `MODEL_CPU_PATH`, `MODEL_SEG_PATH`, `MODEL_DEVICE`, `FLOOD_ON`, `FLOOD_OFF`, `WINDOW_S`, `MIN_HOLD_S`, `CROP_BLACK_BORDER`, `SEGMENT_S`, `INFER_FPS`, `WATER_CLASSES`, `WATER_CONF`, `DRAW_WIDTH`).
Mặc định: 5 khung/giây, p trung bình trượt (L-14: 1 s, B-32: 2 s), bật khi p ≥ 0,9, tắt khi p < 0,3, giữ tối thiểu 10 s, cắt viền đen bật, đoạn 2 s.

## Phân tích video
OpenCV đọc video; không đọc được thì đổi sang H.264 bằng ffmpeg của `imageio-ffmpeg` rồi đọc lại. Các đoạn ngập liền nhau gộp thành 1 sự cố (bắt đầu/kết thúc trong video, độ tin cậy = p làm mượt lớn nhất trong đoạn).
Minh chứng mỗi sự cố: tối đa 3 khung có p thô cao nhất, cách nhau ≥ 1 s; lưu ảnh gốc JPEG, và nếu có mask thì thêm overlay và ảnh cắt (bbox mask nới 10%; không có mask thì ảnh cắt = ảnh gốc).

**Lần kiểm tra và video kết quả.** Mỗi lần phân tích video tạo một bản ghi **lần kiểm tra** (`checks`: khu vực, người gửi, thời gian, ngập/không ngập, xác suất cao nhất (p làm mượt), đường dẫn video kết quả, dãy `[t, p, ngập]`), kể cả khi không ngập. Sự cố ngập vẫn tạo như cũ và gắn vào lần kiểm tra đó (`incidents.check_id`). Video kết quả là H.264, rộng ≤ 854 px, dựng cho **toàn bộ** video gốc. Có `seg.pt`: vùng nước được tô bán trong suốt kèm viền (mask của lần suy luận gần nhất ≤ khung); không có: giữ khung gốc. Không vẽ chữ, % hay banner lên video. Video chỉ tải được khi đã đăng nhập (`/api/checks/{id}/video`, có hỗ trợ tua/Range).

**Xác suất theo khung.** Dãy `[t, p thô, ngập, p làm mượt]` của mọi lần suy luận được lưu (`checks.series_json`): `p thô` là xác suất của từng khung, `p làm mượt` là trung bình trượt (cửa sổ `window_s`) chính là giá trị mà luật bật/tắt so với ngưỡng, `ngập` là trạng thái sau luật trễ bật/tắt. Phía trên video, trang web hiển thị "Xác suất ngập: 87%" bằng **p làm mượt**, không phải p thô, kèm nhãn Ngập (đỏ) / Không ngập (xanh), nên số và nhãn luôn nhất quán: nhãn Ngập khi số ≥ ngưỡng bật, Không ngập khi < ngưỡng tắt, giữa hai ngưỡng giữ trạng thái trước. Cập nhật theo `timeupdate`/`seeked`, lấy lần suy luận gần nhất ≤ thời điểm đang phát. **"Xác suất cao nhất" / "độ tin cậy"** (trang lần kiểm tra, trang khu vực, trang sự cố, `p_max` trong API) là **p làm mượt lớn nhất** (của cả video với lần kiểm tra, của đoạn ngập với sự cố), nên không bao giờ vượt quá số hiển thị trên thanh phía trên video. p thô vẫn nằm trong dãy lưu và trong `evidence.p` của từng khung minh chứng nhưng giao diện không hiển thị.

**Bản ghi cũ** (tạo khi `p_max` còn là p thô, dãy chỉ có 3 cột): lúc khởi động app tính lại p làm mượt từ dãy p thô đã lưu, rồi cập nhật dãy (thành 4 cột), `p_max` của lần kiểm tra và của các sự cố thuộc nó. Chỉ làm khi chạy luật hiện tại trên dãy p thô cho ra đúng dãy trạng thái Ngập/Không ngập đã lưu (tức tham số lúc phân tích giống hiện tại); không thì giữ nguyên p thô và giao diện/API ghi chú `p_note` = "p thô (bản ghi cũ, chưa tính lại được)". Dãy đã có p làm mượt nhưng `p_max` còn là p thô (bản trước) thì chỉ căn chỉnh `p_max`. Chạy lại nhiều lần không đổi gì thêm. Sự cố do thiết bị ngoài gửi (không có lần kiểm tra) giữ nguyên p do thiết bị gửi, không ghi chú. Nên sao lưu `storage/app.db` trước khi lên bản này nếu cần giữ số cũ.

**Nơi xem.** Trang upload luôn hiện video đã phân tích kèm %; không ngập thì ghi "Không phát hiện ngập" phía trên video. Trang khu vực liệt kê các lần kiểm tra gần đây (cả ngập lẫn không) và sự cố ngập; trang chi tiết sự cố mở video của lần kiểm tra, nhảy tới lúc sự cố bắt đầu. Danh sách khu vực có "Kiểm tra lần cuối". Sự cố do thiết bị ngoài gửi (không qua phân tích video) chỉ có clip đính kèm, không có % theo khung.

**Lưu trữ.** `RETENTION_DAYS` (mặc định 30; 0 = giữ mãi): video kết quả cũ hơn bị xoá (kiểm tra lúc khởi động và mỗi 6 giờ), bản ghi lần kiểm tra, dãy t-p, sự cố và ảnh minh chứng được giữ; trang báo "video không còn"

## API cho thiết bị ngoài
`POST /api/incidents` (multipart), xác thực bằng header `X-API-Key: $DEVICE_API_KEY` (đặt biến này để bật) hoặc cookie phiên uploader/admin:

| trường | |
|---|---|
| `area_id`, `p` (0..1) | bắt buộc |
| `image` (≥ 1 file JPEG/PNG, tối đa 12) | ảnh gốc, mỗi file là một khung |
| `overlay`, `crop` | tuỳ chọn; phần tử thứ i khớp `image` thứ i (không gửi `crop` thì ảnh cắt = ảnh gốc) |
| `recorded_at` | thời gian quay, ISO 8601 (không có múi giờ: coi là UTC); không gửi thì chỉ có thời gian nhận |
| `video_start_s`, `video_end_s`, `clip` (mp4), `model_name` | tuỳ chọn |

```
curl -H "X-API-Key: $KEY" -F area_id=1 -F p=0.92 -F image=@anh.jpg localhost:8000/api/incidents
```
Phân tích video gọi nội bộ cùng hàm `app.incidents.create_incident`. Tài liệu API đầy đủ: `/api/docs`.

## Biến môi trường khác
`DATABASE_URL` (mặc định SQLite `storage/app.db`; PostgreSQL: `postgresql+psycopg://…`, cần cài driver), `STORAGE_DIR`, `MODELS_DIR`, `MAX_UPLOAD_MB` (500), `RETENTION_DAYS` (30, xoá video kết quả cũ), `RECENT_HOURS` (6: khu vực có sự cố trong khoảng này được coi là đang/vừa ngập), `SESSION_HOURS` (12), `SECRET_KEY` (mặc định sinh ngẫu nhiên, lưu `storage/secret.key`).

## Bảo mật và giới hạn
- Mật khẩu băm scrypt; phiên bằng cookie HttpOnly, SameSite=Lax, ký HMAC; chặn đăng nhập sai quá 5 lần / 5 phút. Nếu chạy sau HTTPS, cookie tự thêm `Secure`.
- Ảnh và video minh chứng chỉ tải được khi đã đăng nhập.
- Hàng đợi job nằm trong bộ nhớ: khởi động lại thì job đang dở bị đánh dấu lỗi (video tạm đã bị bỏ).
- Chạy một worker uvicorn (model và hàng đợi nằm trong tiến trình).
- Chưa có migration: bảng tạo bằng `create_all`; đổi cấu trúc cần xoá DB hoặc tự migrate.
