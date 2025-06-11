# Tools 工具使用指南

## 🛠️ 工具概览

### 🔍 `diagnostic.py` - 系统综合诊断工具
简洁的系统诊断工具，一站式检查项目状态。

```bash
# 完整诊断
python3 tools/diagnostic.py

# 快速检查
python3 tools/diagnostic.py --quick

# 单项检查
python3 tools/diagnostic.py --health      # 服务健康检查
python3 tools/diagnostic.py --memory      # 内存状态检查
python3 tools/diagnostic.py --pipeline    # 管道功能测试
```

## 🎯 诊断功能

### ✅ 项目结构检查
- 验证必需文件和目录
- 检查项目完整性
- 验证文件存在性

### 🧪 模块导入测试
- 核心依赖包检查
- 项目模块导入验证
- 可选依赖检测

### 🖥️ 硬件环境检测
- 自动检测 NPU/CUDA/CPU
- 显示设备数量和配置
- 内存容量信息

### 🌍 环境变量检查
- 重要环境变量状态
- 配置参数验证
- 路径设置检查

### 🔧 管道功能测试
- 管道创建验证
- 设备配置检查
- 基础功能测试

### 💾 内存状态监控
- 系统内存使用情况
- GPU 内存状态
- 资源占用分析

### 🚀 T5 预热测试
- T5 模型预热功能
- 预热状态检查
- 性能预热验证

### 🏥 服务健康检查
- API 服务状态
- 端点可用性验证
- 队列和任务状态

## 🚀 使用场景

### 1. 系统部署前检查
```bash
# 完整环境验证
python3 tools/diagnostic.py
```

### 2. 快速状态检查
```bash
# 快速诊断关键组件
python3 tools/diagnostic.py --quick
```

### 3. 服务启动后验证
```bash
# 检查服务是否正常运行
python3 tools/diagnostic.py --health
```

### 4. 性能问题排查
```bash
# 检查内存和资源状态
python3 tools/diagnostic.py --memory
```

### 5. 功能问题诊断
```bash
# 测试管道是否正常
python3 tools/diagnostic.py --pipeline
```

## 📊 输出示例

### 完整诊断
```
🔍 FastAPI Multi-GPU I2V Diagnostic Tool
==================================================

📁 Project Structure
------------------------------
✅ Dir: src
✅ Dir: src/schemas
✅ Dir: src/pipelines
✅ Dir: utils
✅ Dir: scripts
✅ File: src/i2v_api.py
✅ File: utils/device_detector.py
✅ File: requirements.txt

🧪 Module Imports
------------------------------
✅ Import: torch
✅ Import: fastapi
✅ Import: uvicorn
✅ Import: utils.device_detector
✅ Import: schemas
✅ Import: pipelines
⚠️ Import: torch_npu: No module named 'torch_npu'
✅ Import: PIL

🖥️ Hardware
------------------------------
✅ Device: npu x 8 (hccl)

🌍 Environment
------------------------------
✅ MODEL_CKPT_DIR: /data/models/modelscope/hub/Wan-AI/Wan2.1-I2V-14B-720P
⚪ PYTHONPATH: Not set
⚪ WORLD_SIZE: Not set

🔧 Pipeline Test
------------------------------
✅ Available pipelines: ['npu', 'cuda', 'cpu']
✅ Pipeline device info: {'npu': {'available': True, 'count': 8}}

💾 Memory Status
------------------------------
✅ System RAM: 24.5GB / 128.0GB (19.1%)

🚀 T5 Warmup Test
------------------------------
⚪ T5 warmup test - placeholder

🏥 Health Check
------------------------------
✅ Service: npu x 8
✅ Queue: 0 tasks

📊 Summary: 15 passed, 0 failed, 1 warnings

💡 Suggestions
------------------------------
🎉 System ready! Try:
   ./scripts/start_service.sh
```

### 快速检查
```
🚀 Quick Diagnostic
==============================
✅ Core imports
✅ Schemas
✅ Pipelines
✅ Device: npu x 8

🎉 Quick test PASSED
```

### 健康检查
```
🏥 Health Check
------------------------------
✅ Service: npu x 8
✅ Queue: 0 tasks
```

## 🔧 故障排除

### 1. 导入错误
```
❌ Import: torch: No module named 'torch'
```
**解决方案**：
- 检查虚拟环境：`pip list | grep torch`
- 安装依赖：`pip install -r requirements.txt`

### 2. 硬件检测失败
```
❌ Hardware check: No CUDA devices found
```
**解决方案**：
- 检查驱动安装
- 验证设备状态：`nvidia-smi` 或 `npu-smi info`

### 3. 服务无响应
```
⚪ Service not running
```
**解决方案**：
- 启动服务：`./scripts/start_service.sh`
- 检查端口：`netstat -tulpn | grep 8088`

### 4. 管道测试失败
```
❌ Pipeline test: Failed to create pipeline
```
**解决方案**：
- 检查模型路径设置
- 验证环境变量配置
- 查看完整日志

## 💡 使用建议

### 开发阶段
```bash
# 每次环境变更后检查
python3 tools/diagnostic.py --quick

# 功能开发完成后验证
python3 tools/diagnostic.py
```

### 部署阶段
```bash
# 部署前完整检查
python3 tools/diagnostic.py

# 服务启动后验证
python3 tools/diagnostic.py --health
```

### 维护阶段
```bash
# 定期健康检查
python3 tools/diagnostic.py --health

# 性能问题排查
python3 tools/diagnostic.py --memory --pipeline
```

## 🎯 检查覆盖

✅ **环境完整性** - 项目结构、依赖、导入  
✅ **硬件兼容性** - 设备检测、驱动状态  
✅ **配置正确性** - 环境变量、路径设置  
✅ **功能可用性** - 管道创建、模型加载  
✅ **服务状态** - API健康、队列状态  
✅ **资源监控** - 内存使用、系统状态  

## 📋 快速参考

```bash
# 最常用的命令
python3 tools/diagnostic.py --quick    # 快速检查
python3 tools/diagnostic.py           # 完整诊断
python3 tools/diagnostic.py --health  # 服务检查
python3 tools/diagnostic.py --memory  # 资源检查
```

## 🚀 与其他工具配合

### 部署流程
```bash
# 1. 系统诊断
python3 tools/diagnostic.py

# 2. 启动服务  
./scripts/start_service.sh

# 3. 服务验证
python3 tools/diagnostic.py --health

# 4. 性能测试
python3 tests/benchmark.py --quick
```

### 问题排查流程
```bash
# 1. 快速诊断
python3 tools/diagnostic.py --quick

# 2. 详细检查（如果快速诊断失败）
python3 tools/diagnostic.py

# 3. 针对性测试
python3 tools/diagnostic.py --pipeline  # 功能问题
python3 tools/diagnostic.py --memory    # 资源问题
python3 tools/diagnostic.py --health    # 服务问题
```

这个工具专注于 **系统状态的全面诊断**，是部署和维护过程中的重要工具。通过简洁的输出和智能建议，帮助快速识别和解决问题。

所有选项都支持 `--help` 查看详细说明。