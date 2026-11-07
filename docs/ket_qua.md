# Kết quả và hạn chế (Release v1.3)

## Đang dùng
| thành phần | file | ghi chú |
|---|---|---|
| Quyết định ngập, có GPU | `models/best.pt` = CLIP ViT-L/14 (visual fp16) + logistic, train với clip mới | luật nhúng: cửa sổ **1 s**, bật **0,9**, tắt **0,3**, giữ NGẬP tối thiểu **10 s** |
| Quyết định ngập, không GPU | `models/best_cpu.pt` = CLIP ViT-B/32 (fp32) + logistic, train với clip mới | luật nhúng: cửa sổ **2 s**, bật **0,9**, tắt **0,3**, giữ **10 s** |
| Tô vùng nước | `models/seg.pt` (yolo11s-seg, bản cũ), chỉ tô khi đang NGẬP | không ảnh hưởng quyết định |

Chung: 5 khung/giây, đoạn 2 s ngập khi ≥ 50% mẫu đang bật. App tự chọn `best.pt` (GPU) hoặc `best_cpu.pt` (CPU) và đọc luật từ `config.json` nhúng trong file đó (ưu tiên: mặc định < luật nhúng < `model.yaml` < biến môi trường).

## 1. Vì sao L-14 trên GPU, B-32 trên CPU (tiêu chí đặt trước)
Tiêu chí chốt trước khi chạy: kiểm chéo lồng chọn luật theo "precision ≥ 0,98 trước, rồi recall tối đa", giữ 10 s. Dữ liệu: v6 + 8 clip mới (46 clip, 729 đoạn 2 s, 5 fold theo nhóm clip).

| | B-32 | L-14 |
|---|---|---|
| Lồng, gộp (recall / precision / F1) | 0,887 / **0,972** / 0,927 (FP 12, FN 53) | 0,934 / **0,984** / 0,958 (FP 7, FN 31) |
| Đạt precision ≥ 0,98? | không | có |
| Chỉ clip mới (189 đoạn) | 0,616 / 0,945 / 0,746 | 0,768 / 1,000 / 0,869 |
| Chỉ clip cũ | 0,972 / 0,977 / 0,975 | 0,986 / 0,980 / 0,983 |
| Luật cuối (chọn trên toàn OOF, không lồng) | W 2 s, 0,9/0,3: 0,887 / 0,993 / 0,937 | W 1 s, 0,9/0,3: 0,929 / 0,995 / 0,961 |
| FP đêm; chớp tắt/clip | 0/137; 0,174 | 0/137; 0,065 |
| CPU (best, khung 720p, trung vị) | **74 ms** | 914 ms |

Chỉ L-14 đạt ngưỡng đặt trước, nên L-14 là model quyết định khi có GPU. B-32 nhanh gấp ~12 lần trên CPU nên dùng làm `best_cpu.pt` khi không có GPU; model này **không đạt** precision 0,98 (0,972) và recall thấp hơn rõ trên clip mới (0,616).
Luật chọn khác nhau giữa các fold (nhất là L-14: fold 3 chọn bật 0,6, fold 4 chọn cửa sổ 3 s), nên số lồng chỉ mang tính tham khảo. Chỉ có 8 clip mới (189 đoạn) nên số "riêng clip mới" nhiễu. Số liệu gốc: `results/v7_new_cv.json`, `results/v7_cpu_speed.json` trong repo nghiên cứu.

## 2. Lịch sử chọn model (đính chính)
- `best.pt` của **Release v1.0 là B-32** (visual 512 chiều), không phải L-14; hai lần chấm demo_test đầu (lần 1, lần 2) chạy B-32 v1.0.
- **v1.1** là L-14 (luật 1 s, 0,9/0,6). Trên demo_01 L-14 báo NGẬP lúc 14,0 s trong khi nước tràn lòng đường lúc 48 s (precision 0,815).
- **v1.2** đổi về B-32 (luật 2 s, 0,9/0,4) vì báo sớm đó và vì L-14 chậm trên CPU. Quyết định này **có nhìn demo_test** (một lần chạy L-14 trên demo_01) và đi ngược thứ tự OOF.
- **v1.3** chọn lại theo tiêu chí đặt trước ở mục 1 sau khi thêm clip mới, nên đảo lại v1.2: L-14 cho GPU, B-32 cho CPU. Vì đã xem demo_test ở v1.1/v1.2, demo_test không còn là bộ kiểm độc lập cho quyết định chọn backbone.

## 3. demo_test (chấm theo đoạn 2 s)
| | recall | precision | F1 | ghi chú |
|---|---|---|---|---|
| Lần 1 (B-32 v1.0, không giữ) | 0,707 | 0,986 | 0,824 | 4 clip |
| Lần 2 (B-32 v1.0, giữ 10 s) | 0,727 | 0,968 | 0,830 | demo_01 64/66, 0 FP, bật 52,6 s (nhãn 48 s); demo_02 FP 2→5; demo_03 15/56; demo_05 10/18 |
| L-14 v1.1, chỉ demo_01 | 1,000 | 0,815 | 0,898 | 66/66, 15 FP, bật 14,0 s (sớm 34 s) |
| B-32 v1.2 luật 2 s/0,4, chỉ demo_01 | 0,955 | 1,000 | 0,977 | 63/66, 0 FP, bật 53,2 s |
| **v1.3 (L-14, GPU), chỉ demo_01** | 1,000 | 0,786 | 0,880 | 66/66 đoạn ngập, 18 FP, bật lúc 13,0 s (**sớm 35 s** so với nhãn 48 s); không tắt trước hết clip |

Thời gian `analyze_video(demo_01)` (180 s video, GPU RTX 3050, L-14 + seg.pt): 55,6 s khi không ghi video kết quả, 167,8 s khi ghi video kết quả như app (ghi video là phần tốn nhất).
Chưa chấm lại demo_02, demo_03, demo_05 với model v1.3 (chỉ demo_01, một lần, chỉ để ghi lại, không dùng để chọn lại).

## 4. Model tô vùng nước: so 4 cách, giữ seg.pt cũ
So trên 664 khung đánh giá (388 khung ngập có mask Đạt, 276 khung không ngập mask rỗng; 5 fold theo `folds_v7.json`), bằng chính adapter của app, GPU RTX 3050.

| cách tô | IoU khung ngập (TB / trung vị) | % diện tích tô nhầm, khung không ngập (TB; khung nhầm > 5%) | ms/khung |
|---|---|---|---|
| `seg.pt` cũ (**đang dùng**) | 0,834 / 0,915 | 18,5%; 48,6% | 14 |
| yolo11s-seg train lần 2 (OOF) | **0,544** / 0,817 | 2,65%; 5,1% | 14 |
| `seg.pt` cũ + veto SegFormer-B2 ADE20K (sky/building/house/tree/plant/wall) | 0,840 / 0,917 | 17,3%; 48,6% | 76 mỗi khung (~27 nếu chạy 1 lần/giây) |
| V-FloodNet ảnh pretrained (LinkNet + EfficientNet-B4, Liang et al. 2023) | 0,827 / 0,943 | 20,5%; 64,5% | 26 |

- Lần 2 tô nhầm ít nhất nhưng IoU 0,544 thấp hơn ngưỡng 0,75 đặt trước (fold 1 và 2 gần như không tô gì); bản "all" có IoU 0,96 chỉ vì đã học chính các khung này.
- Veto gần như không giảm tô nhầm (phần nhầm chính là đường ướt ban đêm, bê tông ướt, không thuộc các lớp bị veto).
- V-FloodNet cho IoU tương đương nhưng tô nhầm nhiều nhất (khung đêm ướt tô 100%), trên clip mới IoU chỉ 0,69.
- Chốt: giữ `seg.pt` cũ, chỉ tô khi đang NGẬP. Code thử nghiệm (yolo_v2, veto, V-FloodNet, biến `SEG_MODEL`) nằm ở nhánh `thu-nghiem-seg` của repo local, không có trên main. Video so sánh seg.pt với V-FloodNet để trình bày: `tools/so_sanh_seg.py`.
- seg.pt cũ có thể đã thấy một phần các khung này nên số của nó nghiêng về phía có lợi cho nó.

## 5. Hạn chế đã biết
1. **Bật sớm khi đường rất ướt.** Với v1.3 vẫn còn: demo_01 báo từ 13 s trong khi nước tràn lòng đường lúc 48 s (precision 0,786). Đây là điểm yếu đã biết, chưa khắc phục. Đường ướt/mưa to chưa tràn lòng đường vẫn có thể cho p ≥ 0,9; giữ 10 s kéo dài cảnh báo sai đó (demo_02: FP 2→5 đoạn khi thêm giữ 10 s; demo_01 với L-14 v1.1: bật từ 14 s trong khi nước tràn lòng đường ở 48 s).
2. **Camera nhìn đường quá xa / nước đục.** Trên clip mới recall chỉ 0,768 (L-14) và 0,616 (B-32); demo_03: chỉ 15/56 đoạn ngập được báo (B-32 v1.0), bật lúc 125,4 s so với nhãn 68 s.
3. **Sát ngưỡng.** p trung bình quanh 0,9 làm thời điểm bật lệch nhiều giây chỉ vì sai số cỡ 3·10⁻⁴ (fp16 so với fp32, cách giải mã video). `best.pt` L-14 fp16 lệch p so với pipeline open_clip fp16 tới 0,0077 trên 48 khung kiểm (đa số ≤ 0,001) và có thể đổi thời điểm bật vài giây.
4. **Tốc độ CPU.** L-14 ~0,9 s/khung trên CPU (nên cần GPU); B-32 ~74 ms/khung.
5. **seg.pt tô nhầm** đường ướt/bê tông ướt khi đang báo NGẬP (mục 4); chưa có cách nào trong 4 cách đã thử vừa đủ IoU vừa ít tô nhầm.
