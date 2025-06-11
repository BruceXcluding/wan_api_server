"""
数据模型包
"""

from .video import (
    TaskStatus,
    VideoSubmitRequest,
    VideoSubmitResponse,
    VideoStatusRequest,
    VideoStatusResponse,
    VideoCancelRequest,
    VideoCancelResponse,
    VideoResults,
    HealthResponse,
    MetricsResponse,
    ErrorResponse,
)

__all__ = [
    "TaskStatus",
    "VideoSubmitRequest",
    "VideoSubmitResponse",
    "VideoStatusRequest",
    "VideoStatusResponse",
    "VideoCancelRequest",
    "VideoCancelResponse",
    "VideoResults",
    "HealthResponse",
    "MetricsResponse",
    "ErrorResponse",
]

__version__ = "1.0.0"