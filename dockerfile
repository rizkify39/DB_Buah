# ===============================
# BASE IMAGE
# ===============================
FROM python:3.10-slim

# ===============================
# ENVIRONMENT VARIABLES
# ===============================
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OPENCV_DISABLE_GUI=1 \
    QT_QPA_PLATFORM=offscreen \
    MPLBACKEND=Agg \
    LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/lib:/lib

WORKDIR /app

# ===============================
# SYSTEM DEPENDENCIES (OPEN-CV)
# ===============================
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ===============================
# COPY REQUIREMENTS
# ===============================
COPY requirements.txt .

# ===============================
# PYTHON DEPENDENCIES (URUTAN KRITIS)
# ===============================
RUN pip install --no-cache-dir --upgrade pip

# 1️⃣ TORCH CPU (HARUS PALING AWAL)
RUN pip install --no-cache-dir torch==2.0.1+cpu torchvision==0.15.2+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# 2️⃣ 🔥 HANCURKAN & KUNCI NUMPY (ANTI NUMPY 2.x)
RUN pip uninstall -y numpy || true
RUN pip install --no-cache-dir numpy==1.26.4

# 3️⃣ ULTRALYTICS TANPA DEPENDENCY (BIAR NGGAK NARIK NUMPY LAIN)
RUN pip install --no-cache-dir --no-deps ultralytics==8.0.196

# 4️⃣ INSTALL DEPENDENCY PROJECT
RUN pip install --no-cache-dir -r requirements.txt

# 5️⃣ PAKSA OPENCV HEADLESS
RUN pip uninstall -y opencv-python || true
RUN pip install --no-cache-dir --force-reinstall opencv-python-headless==4.8.0.74

# 6️⃣ DEPENDENCY TAMBAHAN YANG AMAN
RUN pip install --no-cache-dir \
    matplotlib>=3.3.0 \
    scipy>=1.4.1 \
    tqdm>=4.64.0 \
    pyyaml>=5.3.1 \
    psutil \
    py-cpuinfo \
    thop \
    pandas \
    seaborn

# ===============================
# CACHE BUST (RAILWAY WAJIB)
# ===============================
ARG CACHE_BUST=1
RUN echo "CACHE_BUST=${CACHE_BUST}"

# ===============================
# COPY APP SOURCE
# ===============================
COPY . .

# ===============================
# EXPOSE & RUN
# ===============================
EXPOSE 8080
CMD ["gunicorn", "--workers", "1", "--threads", "2", "--timeout", "120", "--bind", "0.0.0.0:8080", "app:app"]
