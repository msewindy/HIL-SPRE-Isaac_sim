#!/bin/bash
# Setup CUDA environment variables for JAX/TensorFlow
# This script should be sourced before running training
#
# Usage:
#   source setup_cuda_env.sh
#   python examples/train_rlpd.py ...
#
# NOTE: This script uses CUDA 12.9 for JAX compatibility (or CUDA 12.4 if 12.9 not available).
#       If no CUDA 12.x is installed, it will fall back to CUDA 13.0.
#       For CUDA 13.0, use setup_cuda_env_cuda13.sh

# 检查 CUDA 12.9 是否存在，如果存在则使用，否则检查 CUDA 12.4，最后回退到 CUDA 13.0
if [ -d "/usr/local/cuda-12.9" ]; then
    # CUDA 12.9 路径设置（用于 JAX，匹配 jaxlib 0.4.29+cuda12.cudnn91）
    export CUDA_HOME=/usr/local/cuda-12.9
    CUDA_VERSION="12.9"
elif [ -d "/usr/local/cuda-12.4" ]; then
    # CUDA 12.4 路径设置（备选）
    export CUDA_HOME=/usr/local/cuda-12.4
    CUDA_VERSION="12.4"
else
    # 回退到 CUDA 13.0
    export CUDA_HOME=/usr/local/cuda-13.0
    CUDA_VERSION="13.0"
    echo "[WARNING] CUDA 12.x not found, using CUDA 13.0 (may not work with JAX)"
fi
# 重要：确保包含系统库路径（cuDNN 和 libcuda.so 通常在这里）
# 顺序很重要：系统库路径在前，然后是 CUDA 路径
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:${CUDA_HOME}/lib64:${CUDA_HOME}/targets/x86_64-linux/lib:$LD_LIBRARY_PATH
export PATH=${CUDA_HOME}/bin:$PATH

# JAX/XLA GPU memory settings (for distributed deployment)
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.8  # 80% for Learner (Isaac Sim on different machine)

# JAX platform setting (optional, JAX will auto-detect)
# export JAX_PLATFORMS=cuda

# Skip CUDA version constraints check (only needed for CUDA 13.0)
# This allows using CUDA 13.0 libraries with jaxlib 0.4.29+cuda12.cudnn91
# WARNING: Only use if you're sure CUDA 13.0 is compatible
if [ "$CUDA_VERSION" = "13.0" ]; then
    export JAX_SKIP_CUDA_CONSTRAINTS_CHECK=1
fi

echo "[INFO] CUDA environment variables set:"
echo "  CUDA_HOME: $CUDA_HOME (CUDA $CUDA_VERSION)"
echo "  LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo "  XLA_PYTHON_CLIENT_MEM_FRACTION: $XLA_PYTHON_CLIENT_MEM_FRACTION"
echo ""
if [ "$CUDA_VERSION" = "12.9" ] || [ "$CUDA_VERSION" = "12.4" ]; then
    echo "[INFO] Using CUDA $CUDA_VERSION for JAX compatibility"
else
    echo "[WARNING] Using CUDA 13.0 - JAX may not work. Install CUDA 12.x for JAX support."
fi
echo ""
echo "[INFO] To verify GPU availability, run:"
echo "  python3 -c \"import jax; print('Devices:', jax.devices())\""

