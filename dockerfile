FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OPENCV_DISABLE_GUI=1 \
    QT_QPA_PLATFORM=offscreen \
    MPLBACKEND=Agg

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install dependencies dengan urutan yang BENAR untuk menghindari konflik
RUN pip install --upgrade pip && \
    # 1. Install Numpy DULUAN versi 1.26.4 (Wajib < 2.0)
    pip install "numpy==1.26.4" && \
    # 2. Install Torch CPU secara eksplisit agar tidak download versi GPU yang besar
    pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cpu && \
    # 3. Install sisanya, TAPI jangan upgrade numpy lagi (--no-deps untuk ultralytics sementara)
    pip install -r requirements.txt

# Copy source code
COPY . .

# Expose port
EXPOSE 8080

# Run command
CMD ["gunicorn", "--workers", "1", "--threads", "2", "--timeout", "120", "--bind", "0.0.0.0:8080", "app:app"]
