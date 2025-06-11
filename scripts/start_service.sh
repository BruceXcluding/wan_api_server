#!/bin/bash
"""
通用智能启动脚本 - 优化版
自动检测硬件环境并启动最优配置
"""

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# 项目根目录
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo -e "${BLUE}🚀 FastAPI Multi-GPU I2V Service - General Launcher${NC}"
echo "=================================================="

# 默认配置
export MODEL_CKPT_DIR="${MODEL_CKPT_DIR:-/data/models/modelscope/hub/Wan-AI/Wan2.1-I2V-14B-720P}"
export T5_CPU="${T5_CPU:-true}"
export DIT_FSDP="${DIT_FSDP:-true}"
export T5_FSDP="${T5_FSDP:-false}"
export VAE_PARALLEL="${VAE_PARALLEL:-true}"
export CFG_SIZE="${CFG_SIZE:-1}"
export ULYSSES_SIZE="${ULYSSES_SIZE:-1}"
export RING_SIZE="${RING_SIZE:-1}" 
export USE_ATTENTION_CACHE="${USE_ATTENTION_CACHE:-false}"
export CACHE_START_STEP="${CACHE_START_STEP:-12}"
export CACHE_INTERVAL="${CACHE_INTERVAL:-4}"
export CACHE_END_STEP="${CACHE_END_STEP:-37}"
export MAX_CONCURRENT_TASKS="${MAX_CONCURRENT_TASKS:-2}"
export TASK_TIMEOUT="${TASK_TIMEOUT:-1800}"
export SERVER_HOST="${SERVER_HOST:-0.0.0.0}"
export SERVER_PORT="${SERVER_PORT:-8088}"

# 分布式配置
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29500}"

# 系统优化
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-16}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-16}"

# Python 路径 - 添加 wan 模块路径
WAN_PROJECT_ROOT="$(dirname "$PROJECT_ROOT")"  # 上一级目录，即 /workspace/Wan2.1

# 设置 PYTHONPATH，确保 wan 模块可被找到
export PYTHONPATH="$WAN_PROJECT_ROOT:$PROJECT_ROOT:$PROJECT_ROOT/src:$PROJECT_ROOT/utils:${PYTHONPATH:-}"

echo -e "${BLUE}📋 Python Path Configuration:${NC}"
echo "  - WAN_PROJECT_ROOT: $WAN_PROJECT_ROOT"
echo "  - PROJECT_ROOT: $PROJECT_ROOT"
echo "  - PYTHONPATH (first 5 paths):"
echo "$PYTHONPATH" | tr ':' '\n' | head -5 | sed 's/^/    /'

# 验证 wan 模块
echo -e "${BLUE}🔍 Verifying wan module...${NC}"
if [ -d "$WAN_PROJECT_ROOT/wan" ]; then
    echo -e "${GREEN}✅ wan module found at: $WAN_PROJECT_ROOT/wan${NC}"
else
    echo -e "${YELLOW}⚠️  wan module not found at: $WAN_PROJECT_ROOT/wan${NC}"
fi

# Python 路径 - 修改：确保 utils 目录直接可见

echo -e "${BLUE}📋 General Configuration:${NC}"
echo "  - Project Root: $PROJECT_ROOT"
echo "  - Model Path: $MODEL_CKPT_DIR"
echo "  - T5 CPU Mode: $T5_CPU"
echo "  - DIT FSDP: $DIT_FSDP"
echo "  - VAE Parallel: $VAE_PARALLEL"
echo "  - Ulysses Size: $ULYSSES_SIZE"
echo "  - Ring Size: $RING_SIZE"
echo "  - CFG Size: $CFG_SIZE"
echo "  - Max Concurrent: $MAX_CONCURRENT_TASKS"
echo "  - Timeout: ${TASK_TIMEOUT}s"
echo "  - Server: $SERVER_HOST:$SERVER_PORT"

# 环境信息检查
echo -e "${BLUE}🔍 Environment Information:${NC}"
echo "  - Current Directory: $(pwd)"
echo "  - Project Root: $PROJECT_ROOT"
echo "  - Python Version: $(python3 --version)"

# 检查项目结构
echo -e "${BLUE}📁 Project Structure Check:${NC}"
[ -d "$PROJECT_ROOT/src" ] && echo "  ✅ src/" || echo "  ❌ src/"
[ -d "$PROJECT_ROOT/utils" ] && echo "  ✅ utils/" || echo "  ❌ utils/"
[ -f "$PROJECT_ROOT/utils/device_detector.py" ] && echo "  ✅ utils/device_detector.py" || echo "  ❌ utils/device_detector.py"
[ -f "$PROJECT_ROOT/utils/__init__.py" ] && echo "  ✅ utils/__init__.py" || echo "  ❌ utils/__init__.py"
[ -d "$PROJECT_ROOT/src/schemas" ] && echo "  ✅ src/schemas/" || echo "  ❌ src/schemas/"
[ -d "$PROJECT_ROOT/src/pipelines" ] && echo "  ✅ src/pipelines/" || echo "  ❌ src/pipelines/"

# 检查模型路径
if [ ! -d "$MODEL_CKPT_DIR" ]; then
    echo -e "${YELLOW}⚠️  Model directory not found: $MODEL_CKPT_DIR${NC}"
    echo -e "${YELLOW}   Continuing anyway (model will be downloaded if needed)${NC}"
fi

# 验证detect_device模块 - 修改：简化验证逻辑
echo -e "${BLUE}📦 Verifying detect_device...${NC}"
python3 -c "
import sys
import os

# 设置路径
project_root = '$PROJECT_ROOT'
paths = [project_root, os.path.join(project_root, 'src'), os.path.join(project_root, 'utils')]
for p in paths:
    if p not in sys.path:
        sys.path.insert(0, p)

print(f'Project root: {project_root}')

try:
    from utils.device_detector import detect_device
    print('✅ detect_device import successful')
    device_type, device_count, backend = detect_device()
    print(f'Device detected: {device_type}:{device_count}, backend: {backend}')
except Exception as e:
    print(f'❌ detect_device import failed: {e}')
    exit(1)
"

# 如果detect_device验证失败，退出
if [[ $? -ne 0 ]]; then
    echo -e "${RED}❌ Device detector verification failed!${NC}"
    exit 1
fi

# 自动设备检测
echo -e "${BLUE}🔍 Auto-detecting hardware environment...${NC}"
DETECTED_DEVICE=$(python3 -c "
import sys
import os
# 设置路径
project_root = '$PROJECT_ROOT'
paths = [project_root, os.path.join(project_root, 'src'), os.path.join(project_root, 'utils')]
for p in paths:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from utils.device_detector import detect_device
    device_type, device_count, backend = detect_device()
    print(f'{device_type}:{device_count}:{backend}')
except Exception as e:
    print(f'cpu:1:gloo')
" 2>/dev/null)

IFS=':' read -r DEVICE_TYPE DEVICE_COUNT BACKEND <<< "$DETECTED_DEVICE"

echo -e "${GREEN}✅ Detected: $DEVICE_TYPE with $DEVICE_COUNT device(s), backend: $BACKEND${NC}"

# 🔧 添加：自动计算分布式推理参数
if [ "$DEVICE_COUNT" -gt 1 ]; then
    # 如果没有手动设置，则自动计算最优配置
    if [ "$ULYSSES_SIZE" = "1" ] && [ "$RING_SIZE" = "1" ]; then
        if [ "$DEVICE_COUNT" -le 8 ]; then
            export ULYSSES_SIZE="$DEVICE_COUNT"
            export RING_SIZE="1"
        else
            # 对于更大的设备数，进行因子分解
            ULYSSES_SIZE=$(python3 -c "
import math
device_count = $DEVICE_COUNT
ulysses_size = int(math.sqrt(device_count))
for u in range(ulysses_size, 0, -1):
    if device_count % u == 0:
        print(u)
        break
else:
    print(device_count)
")
            export RING_SIZE=$((DEVICE_COUNT / ULYSSES_SIZE))
        fi
        echo -e "${BLUE}🔗 Auto-calculated distributed config: Ulysses=${ULYSSES_SIZE}, Ring=${RING_SIZE}${NC}"
    else
        echo -e "${BLUE}🔗 Using manual distributed config: Ulysses=${ULYSSES_SIZE}, Ring=${RING_SIZE}${NC}"
    fi
    
    # 验证配置
    PRODUCT=$((ULYSSES_SIZE * RING_SIZE))
    if [ "$PRODUCT" -ne "$DEVICE_COUNT" ]; then
        echo -e "${YELLOW}⚠️  Warning: ulysses_size($ULYSSES_SIZE) * ring_size($RING_SIZE) = $PRODUCT != device_count($DEVICE_COUNT)${NC}"
        echo -e "${YELLOW}   Adjusting to: ulysses_size=$DEVICE_COUNT, ring_size=1${NC}"
        export ULYSSES_SIZE="$DEVICE_COUNT"
        export RING_SIZE="1"
    fi
else
    # 单设备强制设置为1
    export ULYSSES_SIZE="1"
    export RING_SIZE="1"
fi

# 设置设备相关环境变量
if [ "$DEVICE_TYPE" = "npu" ]; then
    export NPU_VISIBLE_DEVICES="${NPU_VISIBLE_DEVICES:-$(seq -s, 0 $((DEVICE_COUNT-1)))}"
    export ASCEND_LAUNCH_BLOCKING="0"
    export ALGO=0
    export HCCL_TIMEOUT="0"
    export HCCL_BUFFSIZE="1024"
    export HCCL_CONNECT_TIMEOUT="600"
    export HCCL_EXEC_TIMEOUT="0"
    export HCCL_HEARTBEAT_TIMEOUT="0"
    export HCCL_ASYNC_ERROR_HANDLING="0"
    export RANK_TABLE_FILE=""
    export ASCEND_GLOBAL_LOG_LEVEL="3"
    export ASCEND_SLOG_PRINT_TO_STDOUT="0"
    export ASCEND_GLOBAL_EVENT_ENABLE="0"
    export ASCEND_DEVICE_ID="0"
    export HCCL_WHITELIST_DISABLE=1
    export HCCL_SECURITY_ENABLE=0
    export HCCL_OVER_OFI=0
    export TASK_QUEUE_ENABLE=1
    export PTCOPY_ENABLE=1
    export COMBINED_ENABLE=1
    echo -e "${BLUE}📱 NPU Configuration:${NC}"
    echo "  - NPU Devices: $NPU_VISIBLE_DEVICES"
    echo "  - HCCL_TIMEOUT: $HCCL_TIMEOUT"
    echo "  - HCCL_CONNECT_TIMEOUT: $HCCL_CONNECT_TIMEOUT"
    echo "  - HCCL_WHITELIST_DISABLE: $HCCL_WHITELIST_DISABLE"
    echo "  - ALGO: $ALGO"
elif [ "$DEVICE_TYPE" = "cuda" ]; then
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$(seq -s, 0 $((DEVICE_COUNT-1)))}"
    export NCCL_TIMEOUT="${NCCL_TIMEOUT:-1800}"
    export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-0}"
    echo -e "${BLUE}🎮 CUDA Configuration:${NC}"
    echo "  - CUDA Devices: $CUDA_VISIBLE_DEVICES"
    echo "  - NCCL Timeout: $NCCL_TIMEOUT"
elif [ "$DEVICE_TYPE" = "cpu" ]; then
    export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
    export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
    export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-}"
    export GLOO_DEVICE_TRANSPORT="${GLOO_DEVICE_TRANSPORT:-tcp}"
    echo -e "${BLUE}💻 CPU Configuration:${NC}"
    echo "  - Threads: $OMP_NUM_THREADS"
    echo "  - Gloo Socket: ${GLOO_SOCKET_IFNAME:-auto}"
    mem_total=$(free -g | awk '/^Mem:/{print $2}')
    echo "  - System Memory: ${mem_total}GB"
    if [ $mem_total -lt 32 ]; then
        echo -e "${YELLOW}⚠️  Warning: System has less than 32GB RAM, performance may be limited${NC}"
    fi
fi

# 验证 Python 环境 - 修改：简化验证逻辑
echo -e "${BLUE}🐍 Checking Python environment...${NC}"
python3 -c "
import sys
import os

project_root = '$PROJECT_ROOT'
paths = [project_root, os.path.join(project_root, 'src'), os.path.join(project_root, 'utils')]
for p in paths:
    if p not in sys.path:
        sys.path.insert(0, p)

print(f'Python: {sys.version}')

try:
    import torch
    print(f'PyTorch: {torch.__version__}')
    if '$DEVICE_TYPE' == 'npu':
        import torch_npu
        print(f'torch_npu available: {torch_npu.npu.is_available()}')
        print(f'NPU device count: {torch_npu.npu.device_count()}')
    elif '$DEVICE_TYPE' == 'cuda':
        print(f'CUDA available: {torch.cuda.is_available()}')
        print(f'CUDA device count: {torch.cuda.device_count()}')
    from schemas import VideoSubmitRequest
    from utils.device_detector import detect_device
    print('✅ All project modules imported successfully')
    # 如有 get_available_pipelines 可用则加上
    try:
        from pipelines import get_available_pipelines
        print(f'Available pipelines: {get_available_pipelines()}')
    except Exception as e:
        print('pipelines module import ok (no get_available_pipelines)')
except ImportError as e:
    print(f'❌ Import failed: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
except Exception as e:
    print(f'⚠️  Environment check warning: {e}')
    import traceback
    traceback.print_exc()
"

if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Python environment check failed!${NC}"
    echo -e "${YELLOW}💡 Please check your PyTorch installation and project dependencies${NC}"
    exit 1
fi

# 清理设备缓存 - 修改：简化缓存清理逻辑
echo -e "${BLUE}🗑️  Clearing device cache...${NC}"
python3 -c "
import sys
import os

project_root = '$PROJECT_ROOT'
paths = [project_root, os.path.join(project_root, 'src'), os.path.join(project_root, 'utils')]
for p in paths:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from utils.device_detector import detect_device
    device_type, _, _ = detect_device()
    if device_type == 'npu':
        import torch_npu
        torch_npu.npu.empty_cache()
        print('✅ NPU cache cleared')
    elif device_type == 'cuda':
        import torch
        torch.cuda.empty_cache()
        print('✅ CUDA cache cleared')
    else:
        print('✅ No device cache to clear')
except Exception as e:
    print(f'⚠️  Cache clear warning: {e}')
"

# 清理旧进程 - 增强版
echo -e "${BLUE}🧹 Cleaning up old processes...${NC}"

check_and_free_port() {
    local port=$1
    if lsof -ti:$port > /dev/null 2>&1; then
        echo -e "${YELLOW}⚠️  Port $port is in use, killing processes...${NC}"
        lsof -ti:$port | xargs kill -9 2>/dev/null || true
        sleep 2
        echo -e "${GREEN}✅ Port $port freed${NC}"
    fi
}

pkill -f "i2v_api.py" || true
pkill -f "torchrun.*i2v_api" || true
pkill -f "python.*i2v.*api" || true
sleep 3

check_and_free_port ${MASTER_PORT}
check_and_free_port ${SERVER_PORT}

if [ "$DEVICE_TYPE" = "npu" ]; then
    echo -e "${BLUE}🔧 NPU specific cleanup...${NC}"
    pkill -f "python.*torch_npu" 2>/dev/null || true
    ipcs -m | grep $(whoami) | awk '{print $2}' | xargs -I {} ipcrm -m {} 2>/dev/null || true
    sync
    echo -e "${GREEN}✅ NPU cleanup completed${NC}"
fi

mkdir -p generated_videos
mkdir -p logs

trap 'echo -e "${YELLOW}🛑 Stopping service...${NC}"; pkill -f "torchrun.*i2v_api"; pkill -f "python.*i2v_api"; exit 0' INT TERM

echo -e "${BLUE}🔍 Pre-launch final check...${NC}"

if [ "$DEVICE_COUNT" -gt 1 ]; then
    echo "  - World Size: $DEVICE_COUNT"
    echo "  - Master: $MASTER_ADDR:$MASTER_PORT"
    echo "  - Distributed Config: Ulysses=$ULYSSES_SIZE, Ring=$RING_SIZE"
    PRODUCT=$((ULYSSES_SIZE * RING_SIZE))
    if [ "$PRODUCT" -ne "$DEVICE_COUNT" ]; then
        echo -e "${RED}❌ Error: $ULYSSES_SIZE * $RING_SIZE = $PRODUCT ≠ $DEVICE_COUNT${NC}"
        exit 1
    fi
    echo -e "${GREEN}✅ Distributed config verified${NC}"
fi

if [ ! -d "$MODEL_CKPT_DIR" ]; then
    echo -e "${YELLOW}⚠️  Model path check: $MODEL_CKPT_DIR not found${NC}"
    echo -e "${YELLOW}   Will attempt to download model on first use${NC}"
else
    echo -e "${GREEN}✅ Model path exists: $MODEL_CKPT_DIR${NC}"
fi

echo -e "${BLUE}📋 Final Environment Summary:${NC}"
echo "  - DEVICE_TYPE: $DEVICE_TYPE"
echo "  - DEVICE_COUNT: $DEVICE_COUNT"
echo "  - ULYSSES_SIZE: $ULYSSES_SIZE"
echo "  - RING_SIZE: $RING_SIZE"
if [ "$DEVICE_TYPE" = "npu" ]; then
    echo "  - NPU_VISIBLE_DEVICES: $NPU_VISIBLE_DEVICES"
    echo "  - HCCL_TIMEOUT: $HCCL_TIMEOUT"
    echo "  - HCCL_CONNECT_TIMEOUT: $HCCL_CONNECT_TIMEOUT"
    echo "  - HCCL_WHITELIST_DISABLE: $HCCL_WHITELIST_DISABLE"
    echo "  - ALGO: $ALGO"
elif [ "$DEVICE_TYPE" = "cuda" ]; then
    echo "  - CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
    echo "  - NCCL_TIMEOUT: $NCCL_TIMEOUT"
fi

if [ "$DEVICE_COUNT" -gt 1 ]; then
    echo -e "${GREEN}🚀 Starting $DEVICE_COUNT-process distributed service on $DEVICE_TYPE...${NC}"
    LOG_FILE="logs/${DEVICE_TYPE}_distributed_$(date +%Y%m%d_%H%M%S).log"
    
    # 🔥 关键修改：使用torchrun启动分布式
    torchrun \
        --nproc_per_node=$DEVICE_COUNT \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        src/i2v_api.py 2>&1 | tee "$LOG_FILE"
else
    LOG_FILE="logs/${DEVICE_TYPE}_single_$(date +%Y%m%d_%H%M%S).log"
    python3 src/i2v_api.py 2>&1 | tee "$LOG_FILE"
fi

echo -e "${YELLOW}Service stopped.${NC}"