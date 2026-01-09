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

# 1. Install Numpy 1.x DULUAN (Penting!)
RUN pip install "numpy<2.0.0"

# 2. Install Torch CPU (Versi 2.2.2)
RUN pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cpu

# 3. Install sisanya (Ultralytics baru, Flask, dll)
RUN pip install -r requirements.txt

# 4. Final Check: Pastikan Numpy tidak terupdate ke 2.0 secara diam-diam
RUN pip uninstall -y numpy && pip install "numpy<2.0.0"

COPY . .

EXPOSE 8080

CMD ["gunicorn", "--workers", "1", "--threads", "2", "--timeout", "120", "--bind", "0.0.0.0:8080", "app:app"]
