# Kết quả và hạn chế (best.pt v1.2 = CLIP ViT-B/32)

**Đang dùng (Release v1.2):** `best.pt` = CLIP ViT-B/32 (cùng file với v1.0, sha256 `73eb5254…`), luật chọn ở bước A bằng kiểm chéo lồng với tiêu chí "tối đa recall với precision ≥ 0,97" + giữ 10 s: cửa sổ **2 s**, bật **0,9**, tắt **0,4**, giữ NGẬP tối thiểu **10 s**, 5 khung/s, đoạn 2 s ngập khi ≥ 50% mẫu đang bật. Luật này cho B-32 OOF (540 đoạn): recall 0,977, precision 0,991, F1 0,984 (FP 3, FN 8).
Release v1.1 (L-14) vẫn còn trên GitHub nhưng **không dùng**.

## 0. Vì sao đổi từ L-14 (v1.1) về B-32 (v1.2) — và việc đã xem demo_test
- **Lý do:** (1) trên demo_01, L-14 báo NGẬP lúc 14,0 s trong khi nước tràn lòng đường lúc 48 s (sớm 34 s, precision theo đoạn 0,815, 15 đoạn báo sai); (2) L-14 chạy CPU ~1 s/khung, B-32 ~55 ms.
- **Đây là quyết định có nhìn demo_test.** Lý do (1) đến từ một lần chạy L-14 trên demo_01, một clip của bộ demo_test, nên demo_test không còn là bộ kiểm độc lập cho riêng quyết định chọn backbone này. Quyết định cũng **đi ngược thứ tự OOF** (L-14: recall 0,986 / precision 0,992; B-32: 0,977 / 0,991 với luật cuối, cả hai trên dữ liệu train v6), vì chênh OOF nhỏ (khoảng 3 đoạn) còn chênh về báo sớm và tốc độ CPU thì lớn. Lần chạy L-14 trên demo_01 chỉ có một clip, không đủ để nói L-14 kém hơn nói chung.
- Lần chạy B-32 với luật mới trên demo_01 (sau khi đã đổi, không dùng để chọn gì): 63/66 đoạn ngập (recall 0,955), 0 báo sai (precision 1,000), bật lúc 53,2 s (nhãn 48 s). Chưa chạy lại trên demo_02/03/05 với luật mới.
- Bước train lại bộ phân loại với clip mới sẽ so B-32 và L-14 bằng OOF, ưu tiên precision ≥ 0,98 rồi mới đến recall, kèm tốc độ CPU; kết quả đó mới là căn cứ cho lựa chọn tiếp theo.

## (Ghi lại từ v1.1) Luật L-14 đã thử

## 1. Backbone thật của từng số (đính chính)
- `best.pt` của **Release v1.0 là B-32** (visual ra 512 chiều, 87,8 M tham số), không phải L-14. v1.1 là L-14 (768 chiều). **v1.2 = file v1.0 (B-32)** với luật mới.
- Bảng OOF dưới đây tính trên embedding cache của **từng backbone** (hàng L-14 = L-14 thật).
- Hai lần chấm demo_test (lần 1, lần 2) chạy **best.pt B-32 v1.0**. Dòng "L-14 v1.1" trong mục 3 là lần chạy riêng, ghi rõ.

## 2. Kiểm chéo lồng trên dữ liệu train v6 (540 đoạn 2 s, 5 fold theo nhóm clip)
| bản | luật | recall | precision | F1 |
|---|---|---|---|---|
| B-32 v6 | không giữ | 0,961 | 0,991 | 0,976 |
| L-14 v6 | không giữ | 0,977 | 0,991 | 0,984 |
| B-32 | giữ 10 s, tiêu chí sim5 | 0,972 | 0,991 | 0,982 |
| L-14 | giữ 10 s, tiêu chí sim5 (luật của v1.1, không dùng nữa) | 0,986 | 0,992 | 0,989 (FP 3, FN 5, chớp 0,05/clip) |
| B-32 | giữ 10 s, tiêu chí "tối đa recall, precision ≥ 0,97" | 0,969 | 0,977 | 0,973 |
| L-14 | giữ 10 s, tiêu chí "tối đa recall, precision ≥ 0,97" | 0,986 | 0,941 | 0,963 |

Tiêu chí "precision ≥ 0,97" không chọn được luật đạt ngưỡng cho L-14 trong CV lồng (0,941), nên luật L-14 của v1.1 là luật sim5 (W 1 s, 0,9/0,6, giữ 10 s). B-32 đạt (nested 0,977, luật cuối W 2 s, 0,9/0,4, giữ 10 s) nên v1.2 dùng luật này.

## 3. demo_test (4 clip + demo_02 bản 194 s), chấm theo đoạn 2 s
| | recall | precision | F1 | ghi chú |
|---|---|---|---|---|
| Lần 1 (B-32, không giữ) | 0,707 | 0,986 | 0,824 | |
| Lần 2 (B-32, giữ 10 s) | 0,727 | 0,968 | 0,830 | demo_01 64/66 đoạn ngập, 0 FP, bật lúc 52,6 s (nhãn 48 s); demo_02 FP 2→5; demo_03 15/56; demo_05 10/18 |
| L-14 v1.1, chỉ demo_01 (app, 1 lần) | 1,000 | 0,815 | 0,898 | 66/66 đoạn ngập, nhưng 15 đoạn FP: bật lúc 14,0 s, **34 s trước** nhãn 48 s; có 1 lần tắt 36–40 s |

demo_01 với L-14 không còn bỏ sót nhưng báo sớm khi đường rất ướt (hạn chế 1). Chưa chấm L-14 trên demo_02/03/05.

## 4. Hạn chế đã biết
1. **Bật sớm khi đường rất ướt.** Đường ướt/mưa to chưa tràn lòng đường vẫn có thể cho p ≥ 0,9; min-hold 10 s kéo dài cảnh báo sai đó (demo_02: FP 2→5 đoạn khi thêm giữ 10 s; demo_01 với L-14: bật từ 14 s trong khi nước tràn lòng đường ở 48 s).
2. **Camera nhìn đường quá xa.** Nước chiếm vài pixel thì model thấy ít (demo_03: chỉ 15/56 đoạn ngập được báo, bật lúc 125,4 s so với nhãn 68 s).
3. **Sát ngưỡng.** p trung bình 1 s quanh 0,9 làm thời điểm bật lệch nhiều giây chỉ vì sai số 3·10⁻⁴ (fp16 so với fp32 hay cách giải mã video khác nhau). Gói `best.pt` L-14 fp16 lệch p so với pipeline open_clip fp16 tới 0,0077 trên 48 khung kiểm (đa số ≤ 0,001), và trên demo_01 đoạn 14–19,6 s quyết định lệch vì đúng lý do này; đã chấp nhận.
4. **Tốc độ CPU.** B-32 ~55 ms/khung (CPU) và ~10 ms (GPU). Bản L-14 v1.1 chạy CPU ~1 s/khung và phải `.eval().float()`.
5. Hạn chế 1 và 2 ở trên được ghi từ lần chạy L-14 và demo_02/demo_03 của B-32 (v1.0); với B-32 W 2 s/0,4 chưa chấm lại demo_02, demo_03, demo_05.
