"""Model QUYẾT ĐỊNH ngập: models/best.pt, TorchScript, p = m(frame BGR uint8 HxWx3) với kích thước khung bất kỳ.
Model tự resize/chuẩn hoá bên trong nên KHÔNG tự đổi kích thước hay đổi RGB ở ngoài. Không cần open_clip."""
from pathlib import Path

import numpy as np

from .base import ModelError, result


def pick_device(want: str, cuda_ok: bool) -> str:
    """auto: cuda nếu có, không thì cpu. cpu / cuda: đúng như yêu cầu (cuda mà không có GPU thì báo lỗi)."""
    if want == "auto":
        return "cuda" if cuda_ok else "cpu"
    if want == "cuda" and not cuda_ok:
        raise ModelError("device=cuda nhưng không có GPU CUDA")
    return want


class BestAdapter:
    kind = "best"
    classes = ["p(ngập) do best.pt trả về"]

    def __init__(self, path, device="auto", module=None):
        """`module` cho phép tiêm module giả (test); mặc định nạp TorchScript từ `path`."""
        self.path = Path(path)
        self.device = "cpu"
        if module is None:
            try:
                import torch
                self.device = pick_device(device, torch.cuda.is_available())
                module = torch.jit.load(str(self.path), map_location=self.device).eval()
            except ModelError:
                raise
            except Exception as e:
                raise ModelError(f"Không nạp được {self.path.name} (cần file TorchScript): {e}") from e
        self.module = module
        self.name = f"{self.path.name} (TorchScript, {self.device})"
        self.description = f"{self.path.name}: TorchScript, p = m(khung BGR) [{self.device}]"

    def predict(self, frame_bgr):
        import torch
        with torch.no_grad():
            return result(float(self.module(torch.from_numpy(np.ascontiguousarray(frame_bgr)))))
