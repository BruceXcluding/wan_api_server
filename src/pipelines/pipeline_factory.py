import os
import logging
from typing import Dict, Any

from pipelines.base_pipeline import BasePipeline
# 🔥 修复：使用统一的设备检测
from utils.device_detector import detect_device

logger = logging.getLogger(__name__)

def create_pipeline():
    """创建合适的pipeline"""
    # 获取环境变量
    ckpt_dir = os.environ.get("MODEL_CKPT_DIR", "/data/models/wan")
    pipeline_type = os.environ.get("PIPELINE_TYPE", "auto")  # 🔥 新增：pipeline类型选择
    
    logger.info(f"Creating pipeline with ckpt_dir: {ckpt_dir}")
    
    # 🔥 如果指定使用diffuser
    if pipeline_type == "diffuser":
        logger.info("Using Diffuser pipeline (forced)")
        from .diffuser_pipeline import DiffuserPipeline
        return DiffuserPipeline(ckpt_dir)
    
    # 检测设备
    device_type, device_count, backend = detect_device()
    logger.info(f"Detected: {device_type} with {device_count} devices, backend: {backend}")
    
    # 🔥 修复：获取分布式参数
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    # 🔥 自动选择pipeline
    if pipeline_type == "auto":
        # 单卡且有diffusers时优先使用diffuser（更高效）
        if device_count == 1 and device_type in ["cuda", "npu"]:
            try:
                from .diffuser_pipeline import DiffuserPipeline
                logger.info("Auto-selected: Diffuser pipeline (single device)")
                return DiffuserPipeline(ckpt_dir)
            except ImportError:
                logger.info("Diffusers not available, falling back to native pipeline")
    
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