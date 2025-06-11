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

# 🔥 新增：检查命令行参数
FORCE_SINGLE_DEVICE=false
if [[ "$1" == "--single" ]] || [[ "$1" == "-s" ]]; then
    FORCE_SINGLE_DEVICE=true
    echo -e "${YELLOW}🎯 Force single-device mode enabled${NC}"
fi

# 项目根目录
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo -e "${BLUE}🚀 Wan2.1 I2V Multi-Device API Server${NC}"
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

# Python 路径设置
WAN_PROJECT_ROOT="$(dirname "$PROJECT_ROOT")"
export PYTHONPATH="$WAN_PROJECT_ROOT:$PROJECT_ROOT:$PROJECT_ROOT/src:$PROJECT_ROOT/utils:${PYTHONPATH:-}"

echo -e "${BLUE}📋 General Configuration:${NC}"
echo "  - Project Root: $PROJECT_ROOT"
echo "  - WAN Module: $WAN_PROJECT_ROOT"
echo "  - Model Path: $MODEL_CKPT_DIR"
echo "  - T5 CPU Mode: $T5_CPU"
echo "  - Max Concurrent: $MAX_CONCURRENT_TASKS"
echo "  - Server: $SERVER_HOST:$SERVER_PORT"

# 验证项目结构
echo -e "${BLUE}📁 Project Structure Check:${NC}"
[ -d "$PROJECT_ROOT/src" ] && echo "  ✅ src/" || echo "  ❌ src/"
[ -d "$PROJECT_ROOT/utils" ] && echo "  ✅ utils/" || echo "  ❌ utils/"
[ -f "$PROJECT_ROOT/utils/device_detector.py" ] && echo "  ✅ device_detector.py" || echo "  ❌ device_detector.py"
[ -d "$WAN_PROJECT_ROOT/wan" ] && echo "  ✅ wan module" || echo "  ⚠️  wan module"

# 验证设备检测模块
echo -e "${BLUE}📦 Verifying device detection...${NC}"
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
    print('✅ device_detector import successful')
    device_type, device_count, backend = detect_device()
    print(f'✅ Device detected: {device_type}:{device_count}, backend: {backend}')
except Exception as e:
    print(f'❌ device_detector failed: {e}')
    exit(1)
"

if [[ $? -ne 0 ]]; then
    echo -e "${RED}❌ Device detector verification failed!${NC}"
    exit 1
fi

# 自动设备检测
echo -e "${BLUE}🔍 Auto-detecting hardware environment...${NC}"
DETECTED_DEVICE=$(python3 -c "
import sys
import os
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

# 🔥 新增：单卡模式强制设置
if [ "$FORCE_SINGLE_DEVICE" = true ]; then
    echo -e "${YELLOW}🎯 Forcing single-device mode...${NC}"
    DEVICE_COUNT=1
    export WORLD_SIZE=1
    export RANK=0
    export LOCAL_RANK=0
    
    # 单卡专用环境变量
    if [ "$DEVICE_TYPE" = "npu" ]; then
        export NPU_VISIBLE_DEVICES="0"
    elif [ "$DEVICE_TYPE" = "cuda" ]; then
        export CUDA_VISIBLE_DEVICES="0"
    fi
    
    echo -e "${GREEN}✅ Single-device mode: Using only ${DEVICE_TYPE}:0${NC}"
else
    echo -e "${GREEN}✅ Detected: $DEVICE_TYPE with $DEVICE_COUNT device(s), backend: $BACKEND${NC}"
fi

# 自动计算分布式推理参数
if [ "$DEVICE_COUNT" -gt 1 ]; then
    if [ "$ULYSSES_SIZE" = "1" ] && [ "$RING_SIZE" = "1" ]; then
        export ULYSSES_SIZE="$DEVICE_COUNT"
        export RING_SIZE="1"
        echo -e "${BLUE}🔗 Auto-calculated: Ulysses=${ULYSSES_SIZE}, Ring=${RING_SIZE}${NC}"
    fi
    
    # 验证配置
    PRODUCT=$((ULYSSES_SIZE * RING_SIZE))
    if [ "$PRODUCT" -ne "$DEVICE_COUNT" ]; then
        echo -e "${YELLOW}⚠️  Adjusting: ulysses_size=$DEVICE_COUNT, ring_size=1${NC}"
        export ULYSSES_SIZE="$DEVICE_COUNT"
        export RING_SIZE="1"
    fi
else
    export ULYSSES_SIZE="1"
    export RING_SIZE="1"
    
    # 🔥 新增：单卡模式禁用分布式特性
    if [ "$FORCE_SINGLE_DEVICE" = true ]; then
        export T5_FSDP="false"
        export DIT_FSDP="false"
        export VAE_PARALLEL="false"
        echo -e "${BLUE}🎯 Single-device: FSDP and parallel features disabled${NC}"
    fi
fi

# 🔥 设备特定环境变量
if [ "$DEVICE_TYPE" = "npu" ]; then
    export NPU_VISIBLE_DEVICES="${NPU_VISIBLE_DEVICES:-$(seq -s, 0 $((DEVICE_COUNT-1)))}"
    
    # 🔥 添加本地成功配置的关键环境变量
    export ALGO="0"                                           # 🔥 关键：算法配置
    export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"  # 🔥 关键：NPU内存配置
    export CPU_AFFINITY_CONF="1"                              # 🔥 关键：CPU亲和性配置
    export TOKENIZERS_PARALLELISM="false"
    
    # 🔥 NPU分布式通信修复
    export ASCEND_LAUNCH_BLOCKING="0"       # 调试模式，获取准确错误
    export HCCL_TIMEOUT="1800"              # 增加超时时间
    export HCCL_CONNECT_TIMEOUT="1800"      # 增加连接超时
    export HCCL_EXEC_TIMEOUT="300"          # 增加执行超时
    export HCCL_HEARTBEAT_TIMEOUT="1800"    # 增加心跳超时
    
    # 🔥 禁用安全特性（单机多卡）
    export HCCL_WHITELIST_DISABLE="1"
    export HCCL_SECURITY_ENABLE="0"
    export HCCL_OVER_OFI="0"
    
    # 🔥 单机多卡配置
    export RANK_TABLE_FILE=""               # 单机不需要rank table
    export ASCEND_DEVICE_ID="0"             # 让每个进程自动设置
    
    # 🔥 性能优化
    export TASK_QUEUE_ENABLE="2"
    export PTCOPY_ENABLE="1" 
    export COMBINED_ENABLE="1"
    export HCCL_BUFFSIZE="1024"

    # 🔥 单机多卡模式标识
    export HCCL_SINGLE_NODE="1"
    export HCCL_LOCAL_RANK_NUM="$DEVICE_COUNT"
    
    # 🔥 日志控制
    export ASCEND_GLOBAL_LOG_LEVEL="1"
    export ASCEND_SLOG_PRINT_TO_STDOUT="0"
    export ASCEND_GLOBAL_EVENT_ENABLE="0"
    export HCCL_DEBUG="0"
    
    # 🔥 新增：单卡模式下的简化配置
    if [ "$DEVICE_COUNT" -eq 1 ]; then
        echo -e "${BLUE}📱 NPU Single-Device Configuration:${NC}"
        echo "  - NPU Device: 0"
        echo "  - ALGO: $ALGO"
        echo "  - Memory: expandable_segments"
        echo "  - Mode: Single NPU (no HCCL)"
        
        # 单卡模式禁用HCCL相关设置
        unset HCCL_TIMEOUT
        unset HCCL_CONNECT_TIMEOUT
        unset HCCL_EXEC_TIMEOUT
        unset HCCL_HEARTBEAT_TIMEOUT
        export HCCL_DISABLE="1"
    else
        echo -e "${BLUE}📱 NPU Multi-Device Configuration:${NC}"
        echo "  - NPU Devices: $NPU_VISIBLE_DEVICES"
        echo "  - ALGO: $ALGO"
        echo "  - PYTORCH_NPU_ALLOC_CONF: $PYTORCH_NPU_ALLOC_CONF"
        echo "  - TASK_QUEUE_ENABLE: $TASK_QUEUE_ENABLE"
        echo "  - CPU_AFFINITY_CONF: $CPU_AFFINITY_CONF"
        echo "  - TOKENIZERS_PARALLELISM: $TOKENIZERS_PARALLELISM"
        echo "  - HCCL Timeouts: ${HCCL_TIMEOUT}s"
        echo "  - Single Node Mode: $HCCL_SINGLE_NODE"
    fi
    
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
    
    echo -e "${BLUE}💻 CPU Configuration:${NC}"
    echo "  - Threads: $OMP_NUM_THREADS"
    mem_total=$(free -g | awk '/^Mem:/{print $2}')
    echo "  - System Memory: ${mem_total}GB"
fi

# 🔥 NPU连通性检查（修改：只在多卡时执行）
if [ "$DEVICE_TYPE" = "npu" ] && [ "$DEVICE_COUNT" -gt 1 ]; then
    echo -e "${BLUE}🔍 NPU Connectivity Check...${NC}"
    
    # 检查NPU设备
    for i in $(seq 0 $((DEVICE_COUNT-1))); do
        if [ -c "/dev/davinci$i" ]; then
            echo "  ✅ NPU $i: Device found"
        else
            echo "  ❌ NPU $i: Device not found"
        fi
    done
    
    # 简单通信测试
    python3 -c "
import torch_npu
import torch.distributed as dist
import os
from datetime import timedelta

os.environ['WORLD_SIZE'] = '$DEVICE_COUNT'
os.environ['RANK'] = '0'
os.environ['LOCAL_RANK'] = '0'
os.environ['MASTER_ADDR'] = '127.0.0.1'
os.environ['MASTER_PORT'] = '29501'  # 使用不同端口避免冲突

try:
    torch_npu.npu.set_device(0)
    print('✅ NPU 0 accessible')
    
    if int('$DEVICE_COUNT') > 1:
        dist.init_process_group(
            backend='hccl', 
            init_method='env://', 
            timeout=timedelta(seconds=300)
        )
        print('✅ HCCL communication initialized')
        dist.destroy_process_group()
        print('✅ HCCL test passed')
except Exception as e:
    print(f'⚠️  NPU connectivity warning: {e}')
    print('   Continuing anyway...')
" && echo -e "${GREEN}✅ NPU connectivity verified${NC}" || echo -e "${YELLOW}⚠️  NPU connectivity test had warnings${NC}"

# 🔥 新增：单卡NPU检查
elif [ "$DEVICE_TYPE" = "npu" ] && [ "$DEVICE_COUNT" -eq 1 ]; then
    echo -e "${BLUE}🔍 NPU Single-Device Check...${NC}"
    python3 -c "
import torch_npu
try:
    torch_npu.npu.set_device(0)
    print('✅ NPU 0 accessible')
    print('ℹ️  Single-device mode: No HCCL communication needed')
except Exception as e:
    print(f'❌ NPU 0 access failed: {e}')
    exit(1)
" && echo -e "${GREEN}✅ NPU single-device verified${NC}"
fi

# Python环境验证
echo -e "${BLUE}🐍 Verifying Python environment...${NC}"
python3 -c "
import sys
import os

project_root = '$PROJECT_ROOT'
paths = [project_root, os.path.join(project_root, 'src'), os.path.join(project_root, 'utils')]
for p in paths:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    import torch
    print(f'✅ PyTorch: {torch.__version__}')
    
    if '$DEVICE_TYPE' == 'npu':
        import torch_npu
        print(f'✅ torch_npu: Available={torch_npu.npu.is_available()}, Devices={torch_npu.npu.device_count()}')
    elif '$DEVICE_TYPE' == 'cuda':
        print(f'✅ CUDA: Available={torch.cuda.is_available()}, Devices={torch.cuda.device_count()}')
    
    from schemas import VideoSubmitRequest
    from utils.device_detector import detect_device
    print('✅ Project modules imported successfully')
    
except Exception as e:
    print(f'❌ Import failed: {e}')
    exit(1)
"

if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Python environment check failed!${NC}"
    exit 1
fi

# 清理设备缓存
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

# 清理旧进程
echo -e "${BLUE}🧹 Cleaning up old processes...${NC}"

check_and_free_port() {
    local port=$1
    if lsof -ti:$port > /dev/null 2>&1; then
        echo -e "${YELLOW}⚠️  Port $port in use, freeing...${NC}"
        lsof -ti:$port | xargs kill -9 2>/dev/null || true
        sleep 2
        echo -e "${GREEN}✅ Port $port freed${NC}"
    fi
}

# 终止旧进程
pkill -f "i2v_api.py" 2>/dev/null || true
pkill -f "torchrun.*i2v_api" 2>/dev/null || true
sleep 3

# 释放端口
check_and_free_port ${MASTER_PORT}
check_and_free_port ${SERVER_PORT}

# NPU特殊清理
if [ "$DEVICE_TYPE" = "npu" ]; then
    echo -e "${BLUE}🔧 NPU specific cleanup...${NC}"
    pkill -f "python.*torch_npu" 2>/dev/null || true
    sync
    echo -e "${GREEN}✅ NPU cleanup completed${NC}"
fi

# 创建必要目录
mkdir -p generated_videos
mkdir -p logs

# 设置信号处理
trap 'echo -e "${YELLOW}🛑 Stopping service...${NC}"; pkill -f "torchrun.*i2v_api"; pkill -f "python.*i2v_api"; exit 0' INT TERM

# 最终检查
echo -e "${BLUE}🔍 Pre-launch verification...${NC}"

if [ "$DEVICE_COUNT" -gt 1 ]; then
    echo "  - World Size: $DEVICE_COUNT"
    echo "  - Master: $MASTER_ADDR:$MASTER_PORT"
    echo "  - Distributed: Ulysses=$ULYSSES_SIZE, Ring=$RING_SIZE"
    
    PRODUCT=$((ULYSSES_SIZE * RING_SIZE))
    if [ "$PRODUCT" -ne "$DEVICE_COUNT" ]; then
        echo -e "${RED}❌ Config error: $ULYSSES_SIZE * $RING_SIZE = $PRODUCT ≠ $DEVICE_COUNT${NC}"
        exit 1
    fi
    echo -e "${GREEN}✅ Distributed config verified${NC}"
fi

if [ ! -d "$MODEL_CKPT_DIR" ]; then
    echo -e "${YELLOW}⚠️  Model path not found: $MODEL_CKPT_DIR${NC}"
    echo -e "${YELLOW}   Will download on first use${NC}"
else
    echo -e "${GREEN}✅ Model path exists${NC}"
fi

echo -e "${BLUE}📋 Final Summary:${NC}"
echo "  - Device: $DEVICE_TYPE ($DEVICE_COUNT devices)"
echo "  - Backend: $BACKEND"
echo "  - Mode: $([ "$DEVICE_COUNT" -gt 1 ] && echo "MULTI-DEVICE" || echo "SINGLE-DEVICE")"
echo "  - Distributed: $([ "$DEVICE_COUNT" -gt 1 ] && echo "YES" || echo "NO")"

# 🔥 修改：启动服务（支持单卡选项）
if [ "$DEVICE_COUNT" -gt 1 ]; then
    echo -e "${GREEN}🚀 Starting $DEVICE_COUNT-device distributed service...${NC}"
    LOG_FILE="logs/${DEVICE_TYPE}_distributed_$(date +%Y%m%d_%H%M%S).log"
    
    # NPU使用standalone模式，GPU使用标准模式
    if [ "$DEVICE_TYPE" = "npu" ]; then
        torchrun \
            --standalone \
            --nnodes=1 \
            --nproc_per_node=$DEVICE_COUNT \
            --master_addr=127.0.0.1 \
            --master_port=$MASTER_PORT \
            src/i2v_api.py 2>&1 | tee "$LOG_FILE"
    else
        torchrun \
            --nproc_per_node=$DEVICE_COUNT \
            --master_addr=$MASTER_ADDR \
            --master_port=$MASTER_PORT \
            src/i2v_api.py 2>&1 | tee "$LOG_FILE"
    fi
else
    echo -e "${GREEN}🚀 Starting single-device service...${NC}"
    LOG_FILE="logs/${DEVICE_TYPE}_single_$(date +%Y%m%d_%H%M%S).log"
    
    # 🔥 修改：单卡模式使用直接Python启动
    echo -e "${BLUE}ℹ️  Using direct Python execution (no torchrun)${NC}"
    python3 src/i2v_api.py 2>&1 | tee "$LOG_FILE"
fi

echo -e "${YELLOW}Service stopped.${NC}"