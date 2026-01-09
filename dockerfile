FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OPENCV_DISABLE_GUI=1 \
    QT_QPA_PLATFORM=offscreen \
    MPLBACKEND=Agg

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade pip

# 1. PASANG PONDASI: Numpy Versi 1.x (PENTING!)
RUN pip install "numpy<2.0.0"

# 2. PASANG MESIN: Torch CPU
RUN pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cpu

# 3. PASANG SISANYA (Ultralytics Baru)
RUN pip install -r requirements.txt

# 4. CEK ULANG (Safety Net)
RUN pip uninstall -y numpy && pip install "numpy<2.0.0"

COPY . .

EXPOSE 8080

CMD ["gunicorn", "--workers", "1", "--threads", "2", "--timeout", "120", "--bind", "0.0.0.0:8080", "app:app"]
