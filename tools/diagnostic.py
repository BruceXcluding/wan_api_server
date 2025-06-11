#!/usr/bin/env python3
"""
FastAPI Multi-GPU I2V - 综合诊断工具
====================================

简洁的系统诊断工具，包括：
- 📁 项目结构 - 🧪 模块导入 - 🖥️ 硬件检测 - 🌍 环境变量
- 🔧 管道测试 - 💾 内存监控 - 🚀 T5预热 - 🏥 健康检测

用法: python3 tools/diagnostic.py [--quick] [--health] [--memory] [--pipeline]
"""

import sys
import os
import time
import json
import psutil
import requests
from pathlib import Path

# 设置路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

class Diagnostic:
    def __init__(self):
        self.results = {"passed": 0, "failed": 0, "warnings": 0}
        
    def check(self, name, test_func, critical=False):
        """通用检查函数"""
        try:
            result = test_func()
            if result:
                print(f"✅ {name}")
                self.results["passed"] += 1
                return True
            else:
                print(f"❌ {name}")
                self.results["failed"] += 1
                return False
        except Exception as e:
            status = "❌" if critical else "⚠️"
            print(f"{status} {name}: {e}")
            if critical:
                self.results["failed"] += 1
            else:
                self.results["warnings"] += 1
            return False

    def check_structure(self):
        """检查项目结构"""
        print("\n📁 Project Structure")
        print("-" * 30)
        
        dirs = ["src", "src/schemas", "src/pipelines", "utils", "scripts"]
        files = ["src/i2v_api.py", "utils/device_detector.py", "requirements.txt"]
        
        for d in dirs:
            self.check(f"Dir: {d}", lambda: (PROJECT_ROOT / d).exists())
        
        for f in files:
            self.check(f"File: {f}", lambda: (PROJECT_ROOT / f).exists(), critical=True)

    def check_imports(self):
        """检查模块导入"""
        print("\n🧪 Module Imports")
        print("-" * 30)
        
        modules = [
            ("torch", True),
            ("fastapi", True),
            ("uvicorn", True),
            ("utils.device_detector", True),
            ("schemas", True),
            ("pipelines", True),
            ("torch_npu", False),
            ("PIL", False)
        ]
        
        for module, critical in modules:
            def test_import():
                __import__(module)
                return True
            self.check(f"Import: {module}", test_import, critical)

    def check_hardware(self):
        """检查硬件环境"""
        print("\n🖥️ Hardware")
        print("-" * 30)
        
        # 设备检测
        try:
            from utils.device_detector import detect_device
            device_type, count, backend = detect_device()
            print(f"✅ Device: {device_type} x {count} ({backend})")
            
            # 内存信息
            if device_type == "cuda":
                import torch
                if torch.cuda.is_available():
                    for i in range(min(count, 2)):
                        props = torch.cuda.get_device_properties(i)
                        mem_gb = props.total_memory / 1024**3
                        print(f"   GPU {i}: {props.name} ({mem_gb:.1f}GB)")
            
            return True
        except Exception as e:
            print(f"❌ Hardware check: {e}")
            return False

    def check_environment(self):
        """检查环境变量"""
        print("\n🌍 Environment")
        print("-" * 30)
        
        important_vars = ["MODEL_CKPT_DIR", "PYTHONPATH", "WORLD_SIZE"]
        
        for var in important_vars:
            value = os.environ.get(var, "Not set")
            status = "✅" if value != "Not set" else "⚪"
            print(f"{status} {var}: {value}")

    def test_pipeline(self):
        """测试管道功能"""
        print("\n🔧 Pipeline Test")
        print("-" * 30)
        
        try:
            from pipelines.pipeline_factory import get_available_pipelines
            pipelines = get_available_pipelines()
            print(f"✅ Available pipelines: {pipelines}")
            
            # 简单管道创建测试
            from pipelines.pipeline_factory import PipelineFactory
            device_info = PipelineFactory.get_available_devices()
            print(f"✅ Pipeline device info: {device_info}")
            return True
        except Exception as e:
            print(f"❌ Pipeline test: {e}")
            return False

    def check_memory(self):
        """检查内存状态"""
        print("\n💾 Memory Status")
        print("-" * 30)
        
        # 系统内存
        memory = psutil.virtual_memory()
        print(f"✅ System RAM: {memory.used/1024**3:.1f}GB / {memory.total/1024**3:.1f}GB ({memory.percent:.1f}%)")
        
        # GPU内存
        try:
            import torch
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    allocated = torch.cuda.memory_allocated(i) / 1024**3
                    total = torch.cuda.get_device_properties(i).total_memory / 1024**3
                    print(f"✅ GPU {i}: {allocated:.1f}GB / {total:.1f}GB")
        except:
            pass

    def test_t5_warmup(self):
        """T5模型预热测试"""
        print("\n🚀 T5 Warmup Test")
        print("-" * 30)
        
        try:
            # 这里可以添加T5模型预热逻辑
            # 目前只是占位符
            print("⚪ T5 warmup test - placeholder")
            return True
        except Exception as e:
            print(f"❌ T5 warmup: {e}")
            return False

    def check_health(self, url="http://localhost:8088"):
        """检查服务健康状态"""
        print("\n🏥 Health Check")
        print("-" * 30)
        
        try:
            response = requests.get(f"{url}/health", timeout=5)
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Service: {data.get('device_type')} x {data.get('device_count')}")
                print(f"✅ Queue: {data.get('queue_size')} tasks")
                return True
            else:
                print(f"❌ Service not healthy: {response.status_code}")
                return False
        except requests.exceptions.ConnectionError:
            print("⚪ Service not running")
            return False
        except Exception as e:
            print(f"❌ Health check: {e}")
            return False

    def generate_suggestions(self):
        """生成简单建议"""
        print("\n💡 Suggestions")
        print("-" * 30)
        
        if self.results["failed"] > 0:
            print("🔧 Fix critical issues:")
            print("   pip install -r requirements.txt")
            print("   export MODEL_CKPT_DIR=/path/to/models")
        
        if self.results["warnings"] > 0:
            print("⚠️ Optional improvements available")
        
        if self.results["failed"] == 0:
            print("🎉 System ready! Try:")
            print("   ./scripts/start_service.sh")

    def run_full(self):
        """运行完整诊断"""
        print("🔍 FastAPI Multi-GPU I2V Diagnostic Tool")
        print("=" * 50)
        
        self.check_structure()
        self.check_imports()
        self.check_hardware()
        self.check_environment()
        self.test_pipeline()
        self.check_memory()
        self.test_t5_warmup()
        self.check_health()
        
        # 总结
        print(f"\n📊 Summary: {self.results['passed']} passed, {self.results['failed']} failed, {self.results['warnings']} warnings")
        
        self.generate_suggestions()
        
        return self.results["failed"] == 0

    def run_quick(self):
        """快速检查"""
        print("🚀 Quick Diagnostic")
        print("=" * 30)
        
        # 只检查关键项
        self.check("Core imports", lambda: __import__("utils.device_detector"), True)
        self.check("Schemas", lambda: __import__("schemas"), True)
        self.check("Pipelines", lambda: __import__("pipelines"), True)
        
        try:
            from utils.device_detector import detect_device
            device_type, count, _ = detect_device()
            print(f"✅ Device: {device_type} x {count}")
        except Exception as e:
            print(f"❌ Device: {e}")
            self.results["failed"] += 1
        
        success = self.results["failed"] == 0
        print(f"\n{'🎉 Quick test PASSED' if success else '💥 Issues found'}")
        
        if not success:
            print("   Run full diagnostic: python3 tools/diagnostic.py")
        
        return success

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="System Diagnostic Tool")
    parser.add_argument("--quick", action="store_true", help="Quick check only")
    parser.add_argument("--health", action="store_true", help="Health check only")
    parser.add_argument("--memory", action="store_true", help="Memory check only")
    parser.add_argument("--pipeline", action="store_true", help="Pipeline test only")
    
    args = parser.parse_args()
    
    diagnostic = Diagnostic()
    
    try:
        if args.quick:
            success = diagnostic.run_quick()
        elif args.health:
            success = diagnostic.check_health()
        elif args.memory:
            diagnostic.check_memory()
            success = True
        elif args.pipeline:
            success = diagnostic.test_pipeline()
        else:
            success = diagnostic.run_full()
        
        return 0 if success else 1
        
    except KeyboardInterrupt:
        print("\n⏸️ Interrupted")
        return 1
    except Exception as e:
        print(f"\n💥 Error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())