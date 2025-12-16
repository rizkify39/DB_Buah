#!/bin/bash
# Workaround untuk libGL.so.1
# Cari libGL di berbagai lokasi
MESA_LIB=$(find /nix/store -name "libGL.so.1" -type f 2>/dev/null | head -1)
if [ -n "$MESA_LIB" ]; then
  export LD_LIBRARY_PATH=$(dirname "$MESA_LIB"):$LD_LIBRARY_PATH
fi

# Cari di lokasi standar
if [ -f "/usr/lib/x86_64-linux-gnu/libGL.so.1" ]; then
  export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
fi

# Gunakan dummy library yang sudah dibuat di fase install
if [ -f "/app/libgl_dummy/libGL.so.1" ]; then
  export LD_LIBRARY_PATH=/app/libgl_dummy:$LD_LIBRARY_PATH
  export LD_PRELOAD=/app/libgl_dummy/libGL.so.1:$LD_PRELOAD
fi

# Fallback: buat dummy library jika tidak ditemukan
if [ ! -f "/app/libgl_dummy/libGL.so.1" ]; then
  mkdir -p /tmp/libgl_dummy
  cat > /tmp/libgl_dummy/libgl.c << 'EOF'
#include <dlfcn.h>
void glBegin(unsigned int mode) {}
void glEnd(void) {}
void glVertex3f(float x, float y, float z) {}
void glClear(unsigned int mask) {}
void glColor3f(float r, float g, float b) {}
void glLoadIdentity(void) {}
void glMatrixMode(unsigned int mode) {}
void glOrtho(double left, double right, double bottom, double top, double near, double far) {}
void glViewport(int x, int y, int width, int height) {}
EOF
  gcc -shared -fPIC -o /tmp/libgl_dummy/libGL.so.1 /tmp/libgl_dummy/libgl.c 2>/dev/null || true
  if [ -f "/tmp/libgl_dummy/libGL.so.1" ]; then
    export LD_LIBRARY_PATH=/tmp/libgl_dummy:$LD_LIBRARY_PATH
    export LD_PRELOAD=/tmp/libgl_dummy/libGL.so.1:$LD_PRELOAD
  fi
fi

# Set environment variables untuk OpenCV
export OPENCV_DISABLE_GUI=1
export QT_QPA_PLATFORM=offscreen
export DISPLAY=

# Debug: print LD_LIBRARY_PATH dan LD_PRELOAD
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH" >&2
echo "LD_PRELOAD=$LD_PRELOAD" >&2
ls -la /tmp/libgl_dummy/ >&2 || true

# Start gunicorn
exec gunicorn --workers 1 --threads 2 --timeout 120 app:app

