# Tests Directory

这个目录包含项目的测试脚本。

## 🧪 测试脚本说明

### 🚀 `benchmark.py` - API 性能基准测试
简洁实用的服务性能基准测试工具，测试真实 API 响应。

```bash
# 快速基准测试 (1个请求)
python3 tests/benchmark.py --quick

# 标准基准测试 (3个请求)
python3 tests/benchmark.py

# 自定义请求数量
python3 tests/benchmark.py --count 5

# 并发测试 (2用户，每用户2请求)
python3 tests/benchmark.py --concurrent

# 自定义并发参数
python3 tests/benchmark.py --concurrent --users 3

# 保存结果到文件
python3 tests/benchmark.py --save my_benchmark.json

# 测试远程服务
python3 tests/benchmark.py --url http://remote-server:8088
```

## 🎯 测试功能

### ✅ 健康检查
- 验证服务是否正常运行
- 检查设备配置和状态
- 确认队列和任务状态

### 🚀 响应时间测试
- 提交真实的视频生成任务
- 测量从提交到完成的总时间
- 统计成功率、平均/最小/最大时间

### 🔥 并发测试
- 模拟多个用户同时使用
- 测试系统在负载下的表现
- 计算吞吐量和并发成功率

### 💾 内存监控
- 检查队列大小
- 监控活跃任务数
- 获取系统资源状态

## 📊 输出示例

```
🎯 FastAPI I2V Simple Benchmark
========================================
Target: http://localhost:8088
✅ Service: npu x 8

💾 Memory Check
--------------------
✅ Queue Size:    0
✅ Active Tasks:  0

🚀 Response Time Test (3 requests)
----------------------------------------
Request 1/3... ✅ 45.2s
Request 2/3... ✅ 43.8s  
Request 3/3... ✅ 44.1s

📊 Results:
   Success Rate: 100.0% (3/3)
   Avg Time:     44.4s
   Min Time:     43.8s
   Max Time:     45.2s

🎉 Benchmark completed!
✅ Total successful requests: 3
```

## 🚀 推荐使用流程

### 1. 服务启动后验证
```bash
# 快速检查服务是否正常
python3 tests/benchmark.py --quick
```

### 2. 性能基线测试
```bash
# 建立性能基线 (保存结果)
python3 tests/benchmark.py --count 5 --save baseline.json
```

### 3. 负载测试
```bash
# 测试并发性能
python3 tests/benchmark.py --concurrent --users 2
```

### 4. 定期监控
```bash
# 定期检查性能是否退化
python3 tests/benchmark.py --save daily_$(date +%Y%m%d).json
```

## 📈 性能基线参考

### NPU 8卡环境 (25帧视频)
- **单请求时间**: 40-60秒
- **并发2用户成功率**: > 90%
- **吞吐量**: 0.03-0.05 req/s

### CUDA 多卡环境 (25帧视频)
- **单请求时间**: 30-50秒
- **并发2用户成功率**: > 95%
- **吞吐量**: 0.04-0.06 req/s

### CPU 环境 (25帧视频)
- **单请求时间**: 120-300秒
- **建议**: 仅用于测试

## 🔧 故障排除

### 服务无响应
```bash
# 检查服务状态
curl http://localhost:8088/health

# 检查进程
ps aux | grep i2v_api
```

### 性能问题
1. **响应时间过长**
   - 检查硬件资源使用率
   - 查看服务日志 `logs/*.log`
   - 减少并发用户数

2. **并发失败率高**
   - 降低并发数 `--users 1`
   - 检查内存使用情况
   - 查看队列状态

3. **测试失败**
   - 确认服务正在运行
   - 检查网络连接
   - 验证API端点可访问

### 结果分析
- **成功率 < 90%**: 系统负载过高，需要优化
- **响应时间增长**: 可能存在内存泄漏或资源竞争
- **并发失败**: 队列管理或分布式同步问题

## 📊 测试数据说明

### 保存的结果格式
```json
{
  "timestamp": "2024-01-01 12:00:00",
  "url": "http://localhost:8088",
  "results": [
    {
      "test": "memory",
      "queue_size": 0,
      "active_tasks": 0
    },
    {
      "test": "response_time",
      "requests": 3,
      "success_count": 3,
      "success_rate": 100.0,
      "avg_time": 44.4,
      "min_time": 43.8,
      "max_time": 45.2
    }
  ]
}
```

### 测试参数
- **prompt**: "A cat walking in the garden"
- **frames**: 25 帧 (快速测试)
- **image_size**: "512*512"
- **seed**: 42 (结果可重现)

## 💡 使用建议

### 开发阶段
- 使用 `--quick` 进行快速验证
- 每次代码变更后运行基础测试
- 保存基线结果作为对比

### 生产部署
- 运行完整的并发测试
- 建立性能监控基线
- 定期检查性能退化

### 性能优化
- 对比不同配置的测试结果
- 分析瓶颈所在 (CPU/GPU/内存/网络)
- 逐步调优并验证改进效果

## 🎯 测试覆盖

✅ **API 可用性** - 服务健康检查  
✅ **功能正确性** - 真实任务执行  
✅ **性能指标** - 响应时间统计  
✅ **并发能力** - 多用户负载测试  
✅ **资源监控** - 内存和队列状态  
✅ **结果持久化** - JSON 格式保存

这个测试工具专注于 **API 层面的端到端测试**，确保整个服务链路的可用性和性能。

## 📋 快速参考

```bash
# 最常用的命令
python3 tests/benchmark.py --quick              # 快速检查
python3 tests/benchmark.py                      # 标准测试  
python3 tests/benchmark.py --concurrent         # 并发测试
python3 tests/benchmark.py --save results.json  # 保存结果
```

所有参数都支持 `--help` 查看详细说明。