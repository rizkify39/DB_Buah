FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OPENCV_DISABLE_GUI=1 \
    QT_QPA_PLATFORM=offscreen \
    MPLBACKEND=Agg

WORKDIR /app

# System deps OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade pip

# 🔥 Torch CPU (PIN KERAS)
RUN pip install --no-cache-dir \
    torch==2.0.1+cpu \
    torchvision==0.15.2+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# 🔒 NUMPY HARUS SEBELUM ULTRALYTICS
RUN pip install --no-cache-dir numpy==1.26.4

# 🔥 ULTRALYTICS TANPA DEPENDENCIES
RUN pip install --no-cache-dir --no-deps ultralytics==8.0.196

# Install dependency app (Flask, matplotlib, dll)
RUN pip install --no-cache-dir -r requirements.txt

# 🔨 PAKSA NUMPY SEKALI LAGI (ANTI DITIMPA)
RUN pip uninstall -y numpy || true && \
    pip install --no-cache-dir numpy==1.26.4

COPY . .

EXPOSE 8080

CMD ["gunicorn", "--workers", "1", "--threads", "2", "--timeout", "120", "--bind", "0.0.0.0:8080", "app:app"]
