"""
视频生成相关的数据模型
"""
from enum import Enum
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from datetime import datetime

class TaskStatus(str, Enum):
    """任务状态枚举"""
    PENDING = "pending"
    RUNNING = "running"
    SUCCEED = "succeed"
    FAILED = "failed"
    CANCELLED = "cancelled"

# 请求模型
class VideoSubmitRequest(BaseModel):
    """视频生成提交请求"""
    prompt: str = Field(..., min_length=5, max_length=1000, description="生成提示词")
    image_url: str = Field(..., description="输入图片URL")
    image_size: Optional[str] = Field("1280*720", description="输出视频尺寸")
    num_frames: Optional[int] = Field(81, ge=24, le=121, description="视频帧数")
    guidance_scale: Optional[float] = Field(3.0, ge=1.0, le=10.0, description="引导强度")
    infer_steps: Optional[int] = Field(30, ge=20, le=100, description="推理步数")
    seed: Optional[int] = Field(None, description="随机种子")
    negative_prompt: Optional[str] = Field("", description="负面提示词")
    
    # 分布式参数
    vae_parallel: Optional[bool] = Field(False, description="VAE并行")
    ulysses_size: Optional[int] = Field(1, description="Ulysses并行大小")
    dit_fsdp: Optional[bool] = Field(False, description="DiT FSDP")
    t5_fsdp: Optional[bool] = Field(False, description="T5 FSDP")
    cfg_size: Optional[int] = Field(1, description="CFG并行大小")
    
    # 性能优化参数
    use_attentioncache: Optional[bool] = Field(False, description="使用注意力缓存")
    start_step: Optional[int] = Field(12, description="缓存开始步数")
    attentioncache_interval: Optional[int] = Field(4, description="缓存间隔")
    end_step: Optional[int] = Field(37, description="缓存结束步数")
    sample_solver: Optional[str] = Field("unipc", description="采样求解器")
    sample_shift: Optional[float] = Field(5.0, description="采样偏移")

# 响应模型
class VideoSubmitResponse(BaseModel):
    """视频生成提交响应"""
    requestId: str = Field(..., description="任务ID")
    status: TaskStatus = Field(..., description="任务状态")
    message: str = Field(..., description="状态信息")
    estimated_time: Optional[int] = Field(None, description="预估完成时间(秒)")

class VideoResults(BaseModel):
    """视频生成结果"""
    video_url: str = Field(..., description="视频访问URL")
    video_path: str = Field(..., description="视频文件路径")
    duration: float = Field(..., description="视频时长(秒)")
    frames: int = Field(..., description="视频帧数")
    size: str = Field(..., description="视频尺寸")
    file_size: int = Field(..., description="文件大小(字节)")

class VideoStatusRequest(BaseModel):
    """视频状态查询请求"""
    requestId: str = Field(..., description="任务ID")

class VideoStatusResponse(BaseModel):
    """视频状态查询响应"""
    requestId: str = Field(..., description="任务ID")
    status: TaskStatus = Field(..., description="任务状态")
    progress: int = Field(..., ge=0, le=100, description="完成进度")
    message: str = Field(..., description="状态信息")
    created_at: str = Field(..., description="创建时间")
    updated_at: str = Field(..., description="更新时间")
    results: Optional[VideoResults] = Field(None, description="生成结果")
    reason: Optional[str] = Field(None, description="失败原因")
    elapsed_time: Optional[int] = Field(None, description="已用时间(秒)")

class VideoCancelRequest(BaseModel):
    """视频生成取消请求"""
    requestId: str = Field(..., description="任务ID")

class VideoCancelResponse(BaseModel):
    """视频生成取消响应"""
    requestId: str = Field(..., description="任务ID")
    status: str = Field(..., description="取消状态")
    message: str = Field(..., description="状态信息")

class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str = Field(..., description="服务状态")
    timestamp: float = Field(..., description="检查时间戳")
    uptime: float = Field(..., description="运行时长")
    config: Dict[str, Any] = Field(..., description="配置信息")
    service: Dict[str, Any] = Field(..., description="服务统计")
    resources: Dict[str, Any] = Field(..., description="资源使用")

class MetricsResponse(BaseModel):
    """监控指标响应"""
    timestamp: float = Field(..., description="时间戳")
    system: Dict[str, Any] = Field(..., description="系统指标")
    service: Dict[str, Any] = Field(..., description="服务指标")
    tasks: Dict[str, Any] = Field(..., description="任务指标")
    performance: Dict[str, Any] = Field(..., description="性能指标")

class ErrorResponse(BaseModel):
    """错误响应"""
    error: str = Field(..., description="错误类型")
    message: str = Field(..., description="错误信息")
    timestamp: str = Field(..., description="错误时间")
    request_id: Optional[str] = Field(None, description="请求ID")