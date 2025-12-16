# Gunakan base image Python yang sangat ringan (Debian Slim)
# Ini jauh lebih stabil daripada Nixpacks buat urusan library Linux
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OPENCV_DISABLE_GUI=1 \
    QT_QPA_PLATFORM=offscreen \
    MPLBACKEND=Agg \
    LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/lib:/lib

# Set work directory
WORKDIR /app

# 1. INSTALL SYSTEM DEPENDENCIES (FIX LIBGL)
# Kita install libgl1 langsung di level OS Debian. Dijamin works.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# 2. INSTALL PYTHON PACKAGES (HEMAT MEMORI)
# Gabung dalam satu layer biar image size kecil
RUN pip install --no-cache-dir --upgrade pip && \
    # Install Torch CPU Only (Wajib)
    pip install --no-cache-dir torch==2.0.1+cpu torchvision==0.15.2+cpu --index-url https://download.pytorch.org/whl/cpu && \
    # Install sisa requirements
    pip install --no-cache-dir -r requirements.txt && \
    # Hapus OpenCV biasa (jika kebawa) dan paksa Headless
    pip uninstall -y opencv-python || true && \
    pip install --no-cache-dir opencv-python-headless==4.8.0.74 && \
    # Buang sampah NVIDIA yang bikin berat
    pip uninstall -y nvidia-cudnn-cu12 nvidia-cublas-cu12 nvidia-cuda-runtime-cu12 ultralytics-thop || true && \
    # Bersihkan cache
    rm -rf /root/.cache/pip

# Copy aplikasi
COPY . .

# Expose port (Railway butuh ini)
EXPOSE 8080

# Command start
CMD ["gunicorn", "--workers", "1", "--threads", "2", "--timeout", "120", "--bind", "0.0.0.0:8080", "app:app"]
