# Wan2.1 Multi-Device Distributed Service

基于 Wan2.1 模型的多卡分布式视频生成 API 服务，目前支持图像到视频（Image-to-Video）生成。采用模块化架构设计，支持华为昇腾 NPU 和 NVIDIA GPU 多卡分布式推理。

## 📋 功能支持

> **🎯 当前版本**: 专注于图像到视频（Image-to-Video, I2V）生成  
> **🚀 后续规划**: 将支持文本到视频（Text-to-Video, T2V）生成  
> **🔄 架构设计**: 模块化架构已为多模态扩展做好准备

### 当前支持
- ✅ **I2V 生成**: 基于输入图像生成高质量视频
- ✅ **多卡分布式**: NPU/GPU 8卡并行推理
- ✅ **异步处理**: 完整的任务队列和状态管理
- ✅ **多设备支持**: 华为昇腾 NPU 和 NVIDIA GPU

### 开发中
- 🚧 **T2V 生成**: 纯文本提示词生成视频
- 🚧 **多模态融合**: I2V + T2V 混合生成
- 🚧 **视频编辑**: 基于现有视频的智能编辑

## 🚀 项目特色

- **🎯 多卡分布式**：支持 NPU/GPU 8卡并行推理，自动设备检测
- **🧠 T5 CPU 模式**：支持 T5 文本编码器在 CPU 上运行，节省显存
- **🔄 异步处理**：基于 FastAPI 的异步任务队列和状态管理
- **🧩 模块化架构**：清晰的分层设计，易于维护和扩展
- **⚡ 性能优化**：注意力缓存、VAE并行等多种加速技术
- **📊 任务管理**：完整的任务生命周期管理和队列控制
- **🛡️ 容错机制**：健壮的错误处理和资源清理
- **🔧 简洁工具**：诊断工具和性能测试，专注实用性

## 📁 项目结构

```
wan-api-server/
├── src/                              # 🎯 核心源码
│   ├── i2v_api.py                    # FastAPI 主应用
│   ├── schemas/                      # 📋 数据模型
│   │   ├── __init__.py
│   │   └── video.py                  # 请求/响应模型定义
│   ├── services/                     # 🔧 业务逻辑层
│   │   ├── __init__.py
│   │   └── video_service.py          # 任务管理服务
│   ├── pipelines/                    # 🚀 推理管道
│   │   ├── __init__.py
│   │   ├── base_pipeline.py          # 管道基类
│   │   ├── npu_pipeline.py           # NPU 管道实现
│   │   ├── cuda_pipeline.py          # CUDA 管道实现
│   │   ├── cpu_pipeline.py           # CPU 管道实现
│   │   └── pipeline_factory.py       # 管道工厂
│   └── utils/                        # 🛠️ 内部工具类
│       └── __init__.py
├── utils/                            # 🛠️ 项目级工具
│   ├── __init__.py
│   └── device_detector.py            # 设备自动检测
├── scripts/                          # 📜 启动脚本
│   └── start_service.sh              # 智能启动脚本
├── tests/                            # ✅ 测试工具
│   ├── benchmark.py                  # 性能基准测试
│   └── README.md                     # 测试工具使用指南
├── tools/                            # 🛠️ 诊断工具
│   ├── diagnostic.py                 # 系统诊断工具
│   └── README.md                     # 工具使用指南
├── requirements.txt                  # 依赖清单
└── README.md                         # 项目文档
```

## 🔧 环境要求

### 硬件支持

#### NPU (华为昇腾)
- **设备型号**：910B1/910B2/910B4 等昇腾芯片
- **显存要求**：单卡 24GB+ (T5 CPU 模式) / 32GB+ (标准模式)
- **驱动版本**：CANN 8.0+

#### GPU (NVIDIA)
- **设备型号**：RTX 3090/4090, A100, H100 等
- **显存要求**：单卡 24GB+ (推荐 32GB+)
- **驱动版本**：CUDA 11.8+ / CUDA 12.0+

### 系统要求
- **CPU**：16+ 核心 (T5 CPU 模式建议 32+ 核心)
- **内存**：64GB+ 系统内存 (T5 CPU 模式需要更多)
- **存储**：200GB+ 可用空间 (模型 + 输出视频)
- **操作系统**：Linux (推荐 Ubuntu 20.04+)

### 软件环境
- **Python**：3.10+
- **PyTorch**：2.0+
- **设备扩展**：torch_npu (NPU) / torch (CUDA)

## 🛠️ 快速开始

### 1. 下载 Wan2.1 模型

#### NPU 环境 (华为昇腾)
```bash
# 下载 NPU 优化版本
git clone https://modelers.cn/MindIE/Wan2.1.git
cd Wan2.1

# 安装依赖
pip install -r requirements.txt
pip install -e .

# 验证安装
python -c "import wan; print('✅ Wan2.1 NPU version installed')"
```

#### GPU 环境 (NVIDIA)
```bash
# 下载标准版本
git clone https://github.com/Wan-Video/Wan2.1.git
cd Wan2.1

# 安装依赖
pip install -r requirements.txt
pip install -e .

# 验证安装
python -c "import wan; print('✅ Wan2.1 GPU version installed')"
```

### 2. 下载本项目并启动服务

```bash
# 下载本项目
git clone <your-repository-url>
cd wan-multigpu-i2v

# 系统诊断 (推荐首次运行)
python3 tools/diagnostic.py

# 快速检查环境
python3 tools/diagnostic.py --quick
```

### 3. 环境配置

```bash
# 安装API服务依赖
pip install -r requirements.txt

# 设置模型路径 (指向步骤1下载的模型)
export MODEL_CKPT_DIR="/path/to/Wan2.1/checkpoints/i2v-14B"

# T5 CPU 模式 (推荐，节省显存)
export T5_CPU=true
export MAX_CONCURRENT_TASKS=2

# 验证配置
python3 tools/diagnostic.py --health
```

### 4. 启动服务

```bash
# 智能启动 (自动检测设备)
chmod +x scripts/start_service.sh
./scripts/start_service.sh
```

### 5. 服务验证

```bash
# 健康检查
curl http://localhost:8088/health

# API 文档
open http://localhost:8088/docs

# 性能测试
python3 tests/benchmark.py --quick
```

## 🧪 核心工具

### 🔍 系统诊断工具
```bash
# 完整诊断
python3 tools/diagnostic.py

# 快速检查
python3 tools/diagnostic.py --quick

# 单项检查
python3 tools/diagnostic.py --health      # 服务健康
python3 tools/diagnostic.py --memory      # 内存状态
python3 tools/diagnostic.py --pipeline    # 管道功能
```

**诊断覆盖**：
- ✅ 项目结构检查
- ✅ 模块导入测试
- ✅ 硬件环境检测
- ✅ 环境变量验证
- ✅ 管道功能测试
- ✅ 内存状态监控
- ✅ 服务健康检查

### 🚀 性能基准测试
```bash
# 快速测试 (1个请求)
python3 tests/benchmark.py --quick

# 标准测试 (3个请求)
python3 tests/benchmark.py

# 并发测试
python3 tests/benchmark.py --concurrent --users 2

# 保存结果
python3 tests/benchmark.py --save results.json
```

**测试覆盖**：
- ✅ API 响应时间测试
- ✅ 并发用户负载测试
- ✅ 服务健康状态监控
- ✅ 内存和资源监控

## 🎛️ 配置参数

### 核心配置

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `T5_CPU` | false | T5 编码器是否使用 CPU |
| `DIT_FSDP` | true | DiT 模型是否使用 FSDP 分片 |
| `VAE_PARALLEL` | true | VAE 是否并行编解码 |
| `ULYSSES_SIZE` | 8 | Ulysses 序列并行组数 |
| `MAX_CONCURRENT_TASKS` | 5 | 最大并发任务数 |
| `TASK_TIMEOUT` | 1800 | 任务超时时间(秒) |
| `SERVER_PORT` | 8088 | 服务端口 |
| `MODEL_CKPT_DIR` | - | 模型文件路径 |

### 推荐配置

#### T5 CPU 模式 (节省显存)
```bash
export T5_CPU=true
export MAX_CONCURRENT_TASKS=2
export DIT_FSDP=true
export VAE_PARALLEL=false
```

#### 高性能模式 (大显存环境)
```bash
export T5_CPU=false
export MAX_CONCURRENT_TASKS=5
export DIT_FSDP=true
export VAE_PARALLEL=true
export USE_ATTENTION_CACHE=true
```

#### NPU 专用环境变量
```bash
export ASCEND_LAUNCH_BLOCKING=0
export HCCL_TIMEOUT=2400
export HCCL_BUFFSIZE=512
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
```

## 📚 API 接口

### 核心端点

| 端点 | 方法 | 功能 |
|------|------|------|
| `/submit` | POST | 提交视频生成任务 |
| `/status/{task_id}` | GET | 查询任务状态 |
| `/health` | GET | 服务健康检查 |
| `/metrics` | GET | 性能监控指标 |
| `/docs` | GET | API 文档 |

### 请求示例

```bash
# 提交任务
curl -X POST http://localhost:8088/submit \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A cat walking in the garden",
    "image_url": "https://picsum.photos/512/512",
    "image_size": "512*512",
    "num_frames": 41
  }'

# 查询状态
curl http://localhost:8088/status/your-task-id

# 健康检查
curl http://localhost:8088/health
```

### 请求参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `prompt` | string | - | 视频描述提示词 |
| `image_url` | string | - | 输入图像地址 |
| `image_size` | string | "512*512" | 输出视频分辨率 |
| `num_frames` | int | 41 | 视频帧数 |
| `guidance_scale` | float | 3.0 | CFG 引导系数 |
| `infer_steps` | int | 30 | 推理步数 |
| `seed` | int | null | 随机数种子 |

## 📊 性能参考

### NPU 8卡环境 (T5 CPU 模式)
- **单请求时间**: 40-60秒 (41帧视频)
- **并发2用户成功率**: > 90%
- **吞吐量**: 0.03-0.05 req/s

### CUDA 多卡环境
- **单请求时间**: 30-50秒 (41帧视频)
- **并发2用户成功率**: > 95%
- **吞吐量**: 0.04-0.06 req/s

### CPU 环境
- **单请求时间**: 120-300秒 (41帧视频)
- **建议**: 仅用于测试

## 🔧 故障排除

### 常见问题

#### 1. 启动失败
```bash
# 检查项目结构
python3 tools/diagnostic.py

# 检查设备环境
python3 tools/diagnostic.py --pipeline

# 检查模型路径
ls -la $MODEL_CKPT_DIR
```

#### 2. 内存不足
```bash
# 启用 T5 CPU 模式
export T5_CPU=true
export MAX_CONCURRENT_TASKS=1

# 检查内存状态
python3 tools/diagnostic.py --memory
```

#### 3. 服务无响应
```bash
# 检查服务状态
python3 tools/diagnostic.py --health

# 重启服务
pkill -f i2v_api.py
./scripts/start_service.sh
```

#### 4. 性能问题
```bash
# 性能测试
python3 tests/benchmark.py --quick

# 优化配置
export T5_CPU=true
export DIT_FSDP=true
export VAE_PARALLEL=false
```

### 诊断流程

```bash
# 1. 快速诊断
python3 tools/diagnostic.py --quick

# 2. 详细检查（如果问题持续）
python3 tools/diagnostic.py

# 3. 针对性测试
python3 tools/diagnostic.py --pipeline  # 功能问题
python3 tools/diagnostic.py --memory    # 资源问题
python3 tools/diagnostic.py --health    # 服务问题

# 4. 性能验证
python3 tests/benchmark.py --quick
```

## 🎯 最佳实践

### 部署建议
1. **首次部署**: 运行完整诊断 `python3 tools/diagnostic.py`
2. **启动服务**: 使用智能启动脚本 `./scripts/start_service.sh`
3. **服务验证**: 运行快速测试 `python3 tests/benchmark.py --quick`
4. **监控维护**: 定期健康检查 `python3 tools/diagnostic.py --health`

### 性能优化
- **显存受限**: 启用 `T5_CPU=true`，降低 `MAX_CONCURRENT_TASKS`
- **高性能**: 禁用 `T5_CPU=false`，启用 `USE_ATTENTION_CACHE=true`
- **稳定性**: 使用保守的并发设置，增加超时时间

### 环境配置
- **开发环境**: T5_CPU=true, MAX_CONCURRENT_TASKS=1
- **生产环境**: 根据硬件资源调整并发数
- **测试环境**: 使用快速模式减少测试时间

## 🤝 贡献指南

### 开发流程
```bash
# 1. 项目验证
python3 tools/diagnostic.py

# 2. 代码开发
# ... 你的代码 ...

# 3. 测试验证
python3 tests/benchmark.py --quick

# 4. 提交代码
git commit -m "feat: your amazing feature"
```

### 代码规范
- 保持工具的简洁性
- 添加适当的错误处理
- 更新相应的文档

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件。

## 🙏 致谢

- **🧠 Wan AI Team** - 提供 Wan2.1-I2V-14B-720P 基础模型
- **🔥 华为昇腾** - NPU 硬件支持和 CANN 软件栈
- **💚 NVIDIA** - GPU 硬件支持和 CUDA 生态系统
- **⚡ FastAPI Team** - 高性能异步 Web 框架
- **🔥 PyTorch Team** - 强大的深度学习框架

---

## 🚀 快速开始检查清单

- [ ] 运行系统诊断: `python3 tools/diagnostic.py`
- [ ] 设置模型路径: `export MODEL_CKPT_DIR="..."`
- [ ] 启用T5 CPU模式: `export T5_CPU=true`
- [ ] 启动服务: `./scripts/start_service.sh`
- [ ] 验证健康: `curl http://localhost:8088/health`
- [ ] 性能测试: `python3 tests/benchmark.py --quick`

**🌟 开始你的 AI 视频生成之旅！**

需要帮助？查看 [tools/README.md](tools/README.md) 和 [tests/README.md](tests/README.md) 获取详细的工具使用指南。