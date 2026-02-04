#!/bin/bash
# 创建 cuDNN 符号链接到 CUDA 12.9 目录
# 需要 sudo 权限执行

echo "[INFO] 创建 cuDNN 符号链接到 CUDA 12.9 目录..."

# 创建主要的 cuDNN 库符号链接
sudo ln -sf /usr/lib/x86_64-linux-gnu/libcudnn.so.9 /usr/local/cuda-12.9/lib64/libcudnn.so.9
sudo ln -sf /usr/lib/x86_64-linux-gnu/libcudnn.so.9.18.1 /usr/local/cuda-12.9/lib64/libcudnn.so.9.18.1

# 创建其他 cuDNN 库的符号链接
sudo ln -sf /usr/lib/x86_64-linux-gnu/libcudnn_ops.so.9 /usr/local/cuda-12.9/lib64/libcudnn_ops.so.9
sudo ln -sf /usr/lib/x86_64-linux-gnu/libcudnn_cnn.so.9 /usr/local/cuda-12.9/lib64/libcudnn_cnn.so.9
sudo ln -sf /usr/lib/x86_64-linux-gnu/libcudnn_adv.so.9 /usr/local/cuda-12.9/lib64/libcudnn_adv.so.9

# 更新库缓存
sudo ldconfig

echo "[INFO] 符号链接创建完成！"
echo ""
echo "[INFO] 验证符号链接："
ls -la /usr/local/cuda-12.9/lib64/libcudnn* | head -5

