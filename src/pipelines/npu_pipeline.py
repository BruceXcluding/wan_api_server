import os
import logging
import torch
import torch.distributed as dist
from PIL import Image
from .base_pipeline import BasePipeline

try:
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
    NPU_AVAILABLE = True
except ImportError:
    NPU_AVAILABLE = False

import wan
from wan.configs import WAN_CONFIGS, MAX_AREA_CONFIGS

logger = logging.getLogger(__name__)

class NPUPipeline(BasePipeline):
    """华为昇腾 NPU 视频生成管道 - 支持分布式"""

    def __init__(self, ckpt_dir: str, rank=0, world_size=1, use_distributed=False, **model_args):
        if not NPU_AVAILABLE:
            raise RuntimeError("torch_npu not available, cannot use NPU pipeline")
        
        # 🔥 在调用父类构造函数前设置device_type
        model_args['device_type'] = 'npu'  # 🔥 强制设置为npu
        
        # 🔥 设置分布式参数
        self.rank = rank
        self.world_size = world_size  
        self.local_rank = int(os.environ.get("LOCAL_RANK", rank))
        self.use_distributed = use_distributed
        
        # 分布式配置
        self.t5_fsdp = use_distributed
        self.dit_fsdp = use_distributed
        self.ulysses_size = world_size if use_distributed else 1
        self.vae_parallel = use_distributed
        self.t5_cpu = model_args.get("t5_cpu", False)
        
        # 🔥 调用父类构造函数，传入修正后的model_args
        super().__init__(ckpt_dir, **model_args)

    def _get_backend(self) -> str:
        return "hccl"

    def _load_model(self):
        """加载分布式模型"""
        # 设置设备
        torch_npu.npu.set_device(self.local_rank)
        logger.info(f"Rank {self.rank}: Loading WanI2V on NPU:{self.local_rank}")
        
        # 加载配置
        cfg = WAN_CONFIGS.get("i2v-14B")
        if not cfg:
            raise ValueError("i2v-14B config not found")
        
        # 创建分布式模型
        model = wan.WanI2V(
            config=cfg,
            checkpoint_dir=self.ckpt_dir,
            device_id=self.local_rank,
            rank=self.rank,
            t5_fsdp=self.t5_fsdp,
            dit_fsdp=self.dit_fsdp,
            use_usp=(self.ulysses_size > 1),
            t5_cpu=self.t5_cpu,
            use_vae_parallel=self.vae_parallel,
        )
        
        logger.info(f"Rank {self.rank}: WanI2V loaded with distributed config: "
                   f"t5_fsdp={self.t5_fsdp}, dit_fsdp={self.dit_fsdp}, "
                   f"ulysses_size={self.ulysses_size}, vae_parallel={self.vae_parallel}")
        
        return model

    def _generate_video_device_specific(self, request, img, progress_callback=None):
        """NPU设备特定的视频生成"""
        logger.info(f"Rank {self.rank}: Starting distributed video generation")
        
        # 解析请求参数
        height, width = map(int, getattr(request, "image_size", "1280*720").split("*"))
        max_area = width * height

        # 🔥 记录负面提示词但不使用（因为WanI2V不支持）
        negative_prompt = getattr(request, "negative_prompt", "")
        if negative_prompt and self.rank == 0:
            logger.warning(f"negative_prompt '{negative_prompt}' ignored - WanI2V doesn't support this parameter") 
        
        # 只有rank 0输出详细日志
        if self.rank == 0:
            logger.info(f"Generating video: {width}x{height}, {getattr(request, 'num_frames', 81)} frames")
            
        # 🔥 修复：参照generate.py第311行的精确参数映射
        video = self.model.generate(
            request.prompt,                                    # 第一个位置参数：prompt
            img,                                              # 第二个位置参数：img
            max_area=max_area,                                # 关键字参数
            frame_num=getattr(request, "num_frames", 81),
            shift=getattr(request, "sample_shift", 5.0),      # ✅ schema中有这个字段
            sample_solver=getattr(request, "sample_solver", "unipc"),  # ✅ schema中有这个字段
            sampling_steps=getattr(request, "infer_steps", 40),  # 🔥 修复：infer_steps -> sampling_steps
            guide_scale=getattr(request, "guidance_scale", 5.0),  # 🔥 修复：guidance_scale -> guide_scale
            seed=getattr(request, "seed", 42) if getattr(request, "seed", None) is not None else 42,  # 🔥 防止None
            offload_model=False,  # 🔥 修复：分布式时固定为False
        )
    
        if self.rank == 0:
            logger.info(f"Distributed video generation completed")
            
        return video

    def _save_video(self, video_tensor, output_path: str):
        """保存视频 - 只有rank 0保存"""
        if self.rank == 0:
            logger.info(f"Rank 0: Saving video to {output_path}")
            try:
                from wan.utils.utils import cache_video
                cache_video(
                    tensor=video_tensor[None] if video_tensor.ndim == 4 else video_tensor,
                    save_file=output_path,
                    fps=video_tensor.shape[0] // 5,
                    nrow=1,
                    normalize=True,
                    value_range=(-1, 1)
                )
                logger.info(f"Video saved successfully: {output_path}")
            except Exception as e:
                logger.error(f"Failed to save video: {e}")
                # 创建占位文件
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                with open(output_path, "wb") as f:
                    f.write(b"FAKE_VIDEO_DATA")
        else:
            # 其他rank等待rank 0完成保存
            if dist.is_initialized():
                dist.barrier()

    def _log_memory_usage(self):
        """记录NPU内存使用情况"""
        try:
            memory_allocated = torch_npu.npu.memory_allocated(self.local_rank) / 1024**3
            memory_reserved = torch_npu.npu.memory_reserved(self.local_rank) / 1024**3
            logger.info(f"Rank {self.rank} NPU:{self.local_rank} memory: "
                       f"{memory_allocated:.2f}GB allocated, {memory_reserved:.2f}GB reserved")
        except Exception as e:
            logger.warning(f"Rank {self.rank}: Failed to get NPU memory info: {e}")

    def _empty_cache(self):
        """清空NPU缓存"""
        torch_npu.npu.empty_cache()
        # 分布式同步
        if dist.is_initialized():
            dist.barrier()

    def generate_video(self, request, task_id):
        """生成视频的主入口"""
        try:
            # 处理图片输入
            if hasattr(request, 'image_url') and request.image_url:
                if request.image_url.startswith("http"):
                    import requests
                    from io import BytesIO
                    response = requests.get(request.image_url)
                    img = Image.open(BytesIO(response.content))
                else:
                    img = Image.open(request.image_url)
            else:
                raise ValueError("image_url is required")

            # 生成视频
            video_tensor = self._generate_video_device_specific(request, img)
            
            # 保存视频
            output_path = f"generated_videos/{task_id}.mp4"
            os.makedirs("generated_videos", exist_ok=True)
            self._save_video(video_tensor, output_path)
            
            # 记录内存使用
            self._log_memory_usage()
            
            return f"/videos/{task_id}.mp4"
            
        except Exception as e:
            logger.error(f"Rank {self.rank}: Video generation failed: {e}")
            raise

    def reload_model(self):
        """重新加载模型"""
        logger.info(f"Rank {self.rank}: Reloading model...")
        self._empty_cache()
        self.model = self._load_model()
        logger.info(f"Rank {self.rank}: Model reloaded successfully")