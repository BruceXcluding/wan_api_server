"""
推理管道包
支持多硬件后端的视频生成管道
"""
import logging

logger = logging.getLogger(__name__)

from .base_pipeline import BasePipeline
from .pipeline_factory import create_pipeline, get_device_info

__all__ = [
    "BasePipeline",
    "create_pipeline",
    "get_device_info"
]

__version__ = "1.0.0"
__author__ = "BruceXcluding"
__description__ = "Multi-hardware backend pipelines for video generation"

logger.info("Pipelines package initialized.")