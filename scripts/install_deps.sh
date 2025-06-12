#!/bin/bash
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "🔍 Detecting device..."

# 检测设备
if python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
    DEVICE="cuda"
elif ls /dev/davinci* > /dev/null 2>&1; then
    DEVICE="npu"
else
    DEVICE="cpu"
fi

echo "📦 Installing for device: $DEVICE"

# 安装通用依赖
pip install -r requirements.txt

# 安装设备专用依赖
if [ -f "requirements-${DEVICE}.txt" ]; then
    pip install -r "requirements-${DEVICE}.txt"
    echo "✅ $DEVICE dependencies installed"
fi

echo "🎉 Installation completed!"