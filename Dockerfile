FROM python:3.12-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
# ultralytics kéo theo opencv-python (bản có GUI) nên cần các thư viện hệ thống này
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
# torch bản CPU cho nhẹ; muốn GPU thì đổi dòng này sang bản CUDA tương ứng
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY app ./app
COPY manage.py .
# models/ được mount từ ngoài (best.pt, seg.pt), storage/ giữ DB + ảnh minh chứng
VOLUME ["/app/storage"]
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
