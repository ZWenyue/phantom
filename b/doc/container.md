容器中使用，需要先export

```bash
export LD_LIBRARY_PATH=/usr/local/nvidia/lib64:${LD_LIBRARY_PATH:-}
export __EGL_VENDOR_LIBRARY_DIRS=$HOME/.local/share/glvnd/egl_vendor.d
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
```