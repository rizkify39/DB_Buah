#!/bin/bash
# Workaround untuk libGL.so.1
# Cari libGL di Nix store dan tambahkan ke LD_LIBRARY_PATH
MESA_LIB=$(find /nix/store -name "libGL.so.1" -type f 2>/dev/null | head -1)
if [ -n "$MESA_LIB" ]; then
  export LD_LIBRARY_PATH=$(dirname "$MESA_LIB"):$LD_LIBRARY_PATH
fi

# Set environment variables untuk OpenCV
export OPENCV_DISABLE_GUI=1
export QT_QPA_PLATFORM=offscreen
export DISPLAY=

# Start gunicorn
exec gunicorn --workers 1 --threads 2 --timeout 120 app:app

