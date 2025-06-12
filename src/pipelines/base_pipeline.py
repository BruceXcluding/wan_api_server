import abc
import os
import logging
import torch
import torch.distributed as dist
from pathlib import Path
from typing import Optional, Callable, Dict, Any, List
from PIL import Image
import requests
from io import BytesIO

logger = logging.getLogger(__name__)

class BasePipeline(abc.ABC):
    """分布式推理基础管道"""

    def __init__(self, ckpt_dir: str, **model_args):
        self.ckpt_dir = ckpt_dir
        self.model_args = model_args
        self.model = None
        self.device_type = model_args.get("device_type", "cuda")
        self.rank = int(os.environ.get("RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.t5_cpu = model_args.get('t5_cpu', False)
        logger.info(f"Initializing {self.__class__.__name__} on rank {self.rank}")

        self.ulysses_size = model_args.get('ulysses_size', self.world_size)  # 默认等于world_size
        self.ring_size = model_args.get('ring_size', 1)                      # 默认不使用ring并行
        self.cfg_size = model_args.get('cfg_size', 1)                       # 默认不使用cfg并行
    
        logger.info(f"Initializing {self.__class__.__name__} on rank {self.rank}")
        logger.info(f"Rank {self.rank}: Distributed config - ulysses_size={self.ulysses_size}, ring_size={self.ring_size}")
        
        # 🔥 关键修改：自动初始化分布式和模型
        self.initialize()

    def initialize(self):
        """初始化分布式和模型"""
        # 🔥 修改：只在多卡时初始化分布式
        if self.world_size > 1:
            self._init_distributed()
            # 🔥 新增：设备特定的分布式初始化（默认为空）
            self._setup_device_distributed()
        self.model = self._load_model()
        logger.info(f"Rank {self.rank}: {self.device_type} Pipeline initialized successfully")
    
    def _setup_device_distributed(self):
        """设备特定的分布式初始化（子类可选重写）"""
        # 🔥 默认为空实现，不影响NPU等现有逻辑
        pass
    
    def _init_distributed(self):
        """分布式环境初始化"""
        if not dist.is_initialized():
            backend = self._get_backend()
            # 🔥 修改：使用env://方式，配合torchrun
            dist.init_process_group(backend=backend, init_method="env://")
            logger.info(f"Rank {self.rank}: Distributed initialized with {backend}")

    def sync(self):
        """分布式同步屏障"""
        if self.world_size > 1 and dist.is_initialized():
            try:
                dist.barrier(timeout=timedelta(seconds=30))  # 🔥 添加超时
            except Exception as e:
                logger.warning(f"Rank {self.rank}: Sync barrier failed: {e}")
    
    def generate_video(self, request, task_id: str, progress_callback: Optional[Callable] = None) -> str:
        """生成视频主流程"""
        logger.info(f"Rank {self.rank}: Start video generation for task {task_id}")
        try:
            # 下载图片（只在rank=0）
            image_path = None
            if self.rank == 0:
                image_url = getattr(request, 'image_path', getattr(request, 'image_url', None))
                image_path = self._download_image_sync(image_url, task_id)
            # 广播图片路径
            image_path = self._broadcast_image_path(image_path)
            # 同步
            self.sync()
            # 生成视频
            output_path = self._get_output_path(task_id)
            result_path = self._generate_video_common(request, image_path, output_path, progress_callback)
            logger.info(f"Rank {self.rank}: Video generation completed: {result_path}")
            return result_path
        except Exception as e:
            logger.error(f"Rank {self.rank}: Video generation failed: {str(e)}")
            self._empty_cache()
            raise

    def batch_generate_video(self, requests: List[Any], task_ids: List[str], progress_callback: Optional[Callable] = None) -> list:
        """批量生成视频主流程（可选实现）"""
        results = []
        for request, task_id in zip(requests, task_ids):
            try:
                result = self.generate_video(request, task_id, progress_callback)
                results.append(result)
            except Exception as e:
                logger.error(f"Batch task {task_id} failed: {e}")
                results.append(None)
        return results

    def reload_model(self):
        """热更新模型"""
        logger.info(f"Rank {self.rank}: Reloading model...")
        self.model = self._load_model()
        logger.info(f"Rank {self.rank}: Model reloaded.")

    def _broadcast_image_path(self, image_path: Optional[str]) -> str:
        """用分布式广播同步图片路径"""
        if self.world_size <= 1:
            return image_path

        import torch.distributed as dist
        if not dist.is_initialized():
            return image_path

        # 🔥 真正的分布式广播
        obj_list = [image_path]
        dist.broadcast_object_list(obj_list, src=0)
        logger.info(f"Rank {self.rank}: Image path broadcast completed")
        return obj_list[0]

    def _download_image_sync(self, image_url: str, task_id: str) -> str:
        """同步下载图片"""
        output_dir = Path("generated_videos")
        output_dir.mkdir(exist_ok=True)
        image_path = output_dir / f"{task_id}_input.jpg"
        response = requests.get(image_url, timeout=60)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert("RGB")
        image.save(image_path, quality=95, optimize=True)
        logger.info(f"Image downloaded and saved: {image_path}")
        return str(image_path)

    def _get_output_path(self, task_id: str) -> str:
        output_dir = Path("generated_videos")
        output_dir.mkdir(exist_ok=True)
        return str(output_dir / f"{task_id}.mp4")

    def _generate_video_common(self, request, image_path: str, output_path: str, progress_callback: Optional[Callable] = None) -> str:
        """模板方法：调用设备特定生成逻辑并保存视频"""
        self._log_memory_usage()
        
        # 🔥 所有rank都加载图片
        img = Image.open(image_path).convert("RGB")
        
        # 🔥 关键：所有rank都参与计算
        video_tensor = self._generate_video_device_specific(request, img, progress_callback)
        
        # 🔥 只有rank 0保存视频
        if self.rank == 0 and video_tensor is not None:
            self._save_video(video_tensor, output_path)
            logger.info(f"Video saved to {output_path}")
        
        # 🔥 分布式同步
        if self.world_size > 1:
            import torch.distributed as dist
            if dist.is_initialized():
                dist.barrier()
        
        return f"/videos/{os.path.basename(output_path)}"
        
    @abc.abstractmethod
    def _get_backend(self) -> str:
        """返回分布式后端，如'nccl'或'gloo'"""
        pass

    @abc.abstractmethod
    def _load_model(self):
        """加载模型"""
        pass

    @abc.abstractmethod
    def _generate_video_device_specific(self, request, img, progress_callback: Optional[Callable] = None):
        """设备特定的视频生成逻辑"""
        pass

    @abc.abstractmethod
    def _save_video(self, video_tensor, output_path: str):
        """保存视频"""
        pass

    @abc.abstractmethod
    def _log_memory_usage(self):
        """记录内存使用"""
        pass

    @abc.abstractmethod
    def _empty_cache(self):
        """清理缓存"""
        pass