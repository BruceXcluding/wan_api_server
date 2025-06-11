import os
import logging
from typing import Dict, Any

from pipelines.base_pipeline import BasePipeline

logger = logging.getLogger(__name__)

def detect_device() -> Dict[str, Any]:
    """自动检测设备类型和数量"""
    try:
        try:
            import torch_npu
            if torch_npu.npu.is_available():
                device_type = "npu"
                device_count = torch_npu.npu.device_count()
                backend = "hccl"
                logger.info(f"Detected device: {device_type}, count: {device_count}, backend: {backend}")
                return {
                    "device_type": device_type,
                    "device_count": device_count,
                    "backend": backend
                }
        except ImportError:
            pass
        import torch
        if torch.cuda.is_available():
            device_type = "cuda"
            device_count = torch.cuda.device_count()
            backend = "nccl"
        else:
            device_type = "cpu"
            device_count = os.cpu_count() or 1
            backend = "gloo"
        logger.info(f"Detected device: {device_type}, count: {device_count}, backend: {backend}")
        return {
            "device_type": device_type,
            "device_count": device_count,
            "backend": backend
        }
    except Exception as e:
        logger.error(f"Device detection failed: {e}")
        return {
            "device_type": "cpu",
            "device_count": 1,
            "backend": "gloo"
        }

def create_pipeline():
    """创建分布式pipeline"""
    import os
    
    # 🔥 修复：正确获取设备信息
    device_info = detect_device()
    device_type = device_info["device_type"]
    
    # 获取分布式信息
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    # 获取模型路径
    ckpt_dir = os.environ.get("MODEL_CKPT_DIR", "/path/to/your/ckpt")
    
    if device_type == "npu":
        from .npu_pipeline import NPUPipeline
        return NPUPipeline(
            ckpt_dir=ckpt_dir,  # 🔥 添加必需参数
            rank=rank,
            world_size=world_size,
            use_distributed=(world_size > 1)
        )
    elif device_type == "cuda":
        from .cuda_pipeline import CUDAPipeline
        return CUDAPipeline(ckpt_dir=ckpt_dir, rank=rank, world_size=world_size)
    else:
        from .cpu_pipeline import CPUPipeline
        return CPUPipeline(ckpt_dir=ckpt_dir)  # 🔥 添加必需参数

def get_device_info() -> Dict[str, Any]:
    return detect_device()

__all__ = [
    "create_pipeline",
    "get_device_info"
]