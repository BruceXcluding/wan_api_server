import os
import logging
from typing import Dict, Any

from pipelines.base_pipeline import BasePipeline
# 🔥 修复：使用统一的设备检测
from utils.device_detector import detect_device

logger = logging.getLogger(__name__)

def create_pipeline():
    """创建分布式pipeline"""
    
    # 🔥 修复：使用统一的设备检测函数
    device_type, device_count, backend = detect_device()
    
    # 获取分布式信息
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    # 获取模型路径
    ckpt_dir = os.environ.get("MODEL_CKPT_DIR", "/path/to/your/ckpt")
    
    # 🔥 添加调试日志
    logger.info(f"Rank {rank}: Creating {device_type} Pipeline")
    
    if device_type == "npu":
        from .npu_pipeline import NPUPipeline
        pipeline = NPUPipeline(
            ckpt_dir=ckpt_dir,
            rank=rank,
            world_size=world_size,
            use_distributed=(world_size > 1)
        )
        logger.info(f"Rank {rank}: NPU Pipeline created successfully")
        return pipeline
        
    elif device_type == "cuda":
        from .cuda_pipeline import CUDAPipeline
        pipeline = CUDAPipeline(
            ckpt_dir=ckpt_dir, 
            rank=rank, 
            world_size=world_size
        )
        logger.info(f"Rank {rank}: CUDA Pipeline created successfully")
        return pipeline
        
    else:
        from .cpu_pipeline import CPUPipeline
        pipeline = CPUPipeline(ckpt_dir=ckpt_dir)
        logger.info(f"Rank {rank}: CPU Pipeline created successfully")
        return pipeline

def get_device_info() -> Dict[str, Any]:
    """获取设备信息 - 兼容性函数"""
    device_type, device_count, backend = detect_device()
    return {
        "device_type": device_type,
        "device_count": device_count,
        "backend": backend
    }

__all__ = [
    "create_pipeline",
    "get_device_info"
]