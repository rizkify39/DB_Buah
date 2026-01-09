# Gunakan base image Python yang sangat ringan (Debian Slim)
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OPENCV_DISABLE_GUI=1 \
    QT_QPA_PLATFORM=offscreen \
    MPLBACKEND=Agg \
    LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/lib:/lib

WORKDIR /app

# 1. INSTALL SYSTEM LIBRARY (Wajib buat OpenCV)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 2. COPY Requirements
COPY requirements.txt .

# 3. INSTALL PYTHON PACKAGES (Bertahap biar aman)
RUN pip install --no-cache-dir --upgrade pip

# Install Torch CPU dulu
RUN pip install --no-cache-dir torch==2.0.1+cpu torchvision==0.15.2+cpu --index-url https://download.pytorch.org/whl/cpu

# Install Ultralytics TANPA dependencies dulu (biar ga narik sampah)
RUN pip install --no-cache-dir --no-deps ultralytics==8.0.196

# Install requirements sisanya (Flask, Pillow, dll)
RUN pip install --no-cache-dir -r requirements.txt

# [FIX UTAMA] Paksa Install OpenCV Headless secara eksplisit di layer terpisah
# Ini memastikan modul 'cv2' benar-benar ada
RUN pip uninstall -y opencv-python || true
RUN pip install --no-cache-dir --force-reinstall opencv-python-headless==4.8.0.74

# Install dependency pendukung Ultralytics yang penting-penting aja
RUN pip install --no-cache-dir matplotlib>=3.3.0 scipy>=1.4.1 tqdm>=4.64.0 pyyaml>=5.3.1 psutil py-cpuinfo thop pandas seaborn

# ===============================
# CACHE BUST (WAJIB UNTUK RAILWAY)
# ===============================
ARG CACHE_BUST=1
RUN echo "Cache bust version: ${CACHE_BUST}"

# Copy aplikasi terakhir (biar kalau ganti code doang, ga perlu install ulang library)
COPY . .

EXPOSE 8080

CMD ["gunicorn", "--workers", "1", "--threads", "2", "--timeout", "120", "--bind", "0.0.0.0:8080", "app:app"]
