# models/

| file | trong git? | vai trò |
|---|---|---|
| `seg.pt` (~20 MB) | có | tô vùng nước (yolo11s-seg), chỉ tô khi đang NGẬP |
| `best.pt` (~580 MB) | không (> 100 MB) | quyết định ngập khi có GPU: CLIP ViT-L/14 fp16 |
| `best_cpu.pt` (~335 MB) | không | quyết định ngập khi không có GPU: CLIP ViT-B/32 fp32 |

Tải `best.pt` và `best_cpu.pt` từ [Release v1.3](https://github.com/tanh8805/flood/releases/tag/v1.3) (kèm cả `seg.pt` nếu cần):

```
gh release download v1.3 -R tanh8805/flood -p best.pt -p best_cpu.pt -D models/ --clobber
# hoặc: curl -L -o models/best.pt     https://github.com/tanh8805/flood/releases/download/v1.3/best.pt
#       curl -L -o models/best_cpu.pt https://github.com/tanh8805/flood/releases/download/v1.3/best_cpu.pt
```

App tự chọn `best.pt` (có GPU) hoặc `best_cpu.pt` (không có GPU), và đọc luật bật/tắt từ `config.json` nhúng trong file đó. Tuỳ chọn cấu hình: `models/model.yaml` (xem README, mục Tham số).
sha256: `best.pt` 6bb304f5…, `best_cpu.pt` c5cc5643…, `seg.pt` dcc1e619….
