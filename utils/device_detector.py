import logging

logger = logging.getLogger(__name__)

def detect_device():
    # 优先检测 NPU
    try:
        import torch_npu
        if torch_npu.npu.is_available():
            npu_count = torch_npu.npu.device_count()
            logger.info(f"Detected NPU: {npu_count} device(s)")
            return "npu", npu_count, "hccl"
    except ImportError:
        logger.info("torch_npu not installed, skip NPU detection")
    except Exception as e:
        logger.warning(f"NPU detection failed: {e}")

    # 检测 CUDA
    try:
        import torch
        if torch.cuda.is_available():
            cuda_count = torch.cuda.device_count()
            logger.info(f"Detected CUDA: {cuda_count} device(s)")
            return "cuda", cuda_count, "nccl"
    except Exception as e:
        logger.warning(f"CUDA detection failed: {e}")

    # 默认 CPU
    logger.info("Fallback to CPU")
    return "cpu", 1, "gloo"