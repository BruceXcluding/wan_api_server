import os
import logging
import torch
from PIL import Image
from .base_pipeline import BasePipeline

import wan
from wan.configs import WAN_CONFIGS, MAX_AREA_CONFIGS, SIZE_CONFIGS

logger = logging.getLogger(__name__)

class CUDAPipeline(BasePipeline):
    """NVIDIA CUDA GPU 视频生成管道"""

    def __init__(self, ckpt_dir: str, rank=0, world_size=1, **model_args):  # 🔥 添加rank, world_size参数
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available, cannot use CUDA pipeline")
        
        # 🔥 设置分布式参数
        self.rank = rank
        self.world_size = world_size
        self.local_rank = int(os.environ.get("LOCAL_RANK", rank))
        
        self.device_type = model_args.get("device_type", "cuda")
        self.t5_cpu = model_args.get("t5_cpu", False)
        self.t5_fsdp = model_args.get("t5_fsdp", world_size > 1)  # 🔥 多卡自动启用
        self.dit_fsdp = model_args.get("dit_fsdp", world_size > 1)  # 🔥 多卡自动启用
        self.cfg_size = model_args.get("cfg_size", 1)
        self.ulysses_size = model_args.get("ulysses_size", world_size)  # 🔥 使用world_size
        self.ring_size = model_args.get("ring_size", 1)
        self.vae_parallel = model_args.get("vae_parallel", world_size > 1)  # 🔥 多卡自动启用
        self.offload_model = model_args.get("offload_model", False)
        
        super().__init__(ckpt_dir, **model_args)

    def _get_backend(self) -> str:
        return "nccl"

    def _load_model(self):
        torch.cuda.set_device(self.local_rank)
        logger.info(f"Rank {self.rank}: Loading WanI2V on CUDA:{self.local_rank} (t5_cpu={self.t5_cpu})")
        cfg = WAN_CONFIGS.get("i2v-14B")  # 你可以根据实际业务动态选择
        model = wan.WanI2V(
            config=cfg,
            checkpoint_dir=self.ckpt_dir,
            device_id=self.local_rank,
            rank=self.rank,
            t5_fsdp=self.t5_fsdp,
            dit_fsdp=self.dit_fsdp,
            use_usp=(self.ulysses_size > 1 or self.ring_size > 1),
            t5_cpu=self.t5_cpu,
            use_vae_parallel=self.vae_parallel,
        )
        return model

    def _generate_video_device_specific(self, request, img, progress_callback=None):
        logger.info(f"Rank {self.rank}: Generating video on CUDA with WanI2V")
        
        # 🔥 记录负面提示词但不使用（因为WanI2V不支持）
        negative_prompt = getattr(request, "negative_prompt", "")
        if negative_prompt and self.rank == 0:
            logger.warning(f"negative_prompt '{negative_prompt}' ignored - WanI2V doesn't support this parameter") 

        video = self.model.generate(
            request.prompt,
            img,
            max_area=MAX_AREA_CONFIGS.get(getattr(request, "image_size", "1280*720"), 1280*720),
            frame_num=getattr(request, "num_frames", 81),
            shift=getattr(request, "sample_shift", 5.0),
            sample_solver=getattr(request, "sample_solver", "unipc"),
            sampling_steps=getattr(request, "infer_steps", 40),
            guide_scale=getattr(request, "guidance_scale", 5.0),
            seed=getattr(request, "seed", 42) if getattr(request, "seed", None) is not None else 42,
            offload_model=getattr(request, "offload_model", self.offload_model),
            # 进度回调
            progress_callback=progress_callback if progress_callback else None,
        )
    
    def _save_video(self, video_tensor, output_path: str):
        logger.info(f"Saving video to {output_path}")
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
        except Exception as e:
            logger.error(f"Failed to save video: {e}")
            with open(output_path, "wb") as f:
                f.write(b"FAKE_VIDEO_DATA")

    def _log_memory_usage(self):
        try:
            memory_allocated = torch.cuda.memory_allocated(self.local_rank) / 1024**3
            memory_reserved = torch.cuda.memory_reserved(self.local_rank) / 1024**3
            logger.info(f"Rank {self.rank} CUDA Memory - Allocated: {memory_allocated:.2f}GB, Reserved: {memory_reserved:.2f}GB")
        except Exception as e:
            logger.warning(f"Failed to get CUDA memory info: {e}")

    def _empty_cache(self):
        torch.cuda.empty_cache()