import os
import torch
import uuid
import numpy as np
from typing import Optional
from PIL import Image
import requests
from io import BytesIO
from threading import Lock

from .base_pipeline import BasePipeline
from ..utils.logger import logger

try:
    from diffusers.utils import export_to_video
    from diffusers import AutoencoderKLWan, WanImageToVideoPipeline
    from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
    from transformers import CLIPVisionModel
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False

class DiffuserPipeline(BasePipeline):
    """基于Diffusers的高效视频生成管道"""

    def __init__(self, ckpt_dir: str, **kwargs):
        if not DIFFUSERS_AVAILABLE:
            raise RuntimeError("diffusers not available, cannot use DiffuserPipeline")
        
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available, DiffuserPipeline requires GPU")
        
        # 单卡设置
        self.rank = 0
        self.world_size = 1
        self.local_rank = 0
        self.device = "cuda"
        
        # 线程锁
        self.model_lock = Lock()
        
        super().__init__(ckpt_dir, **kwargs)

    def _get_backend(self) -> str:
        return "diffusers"

    def _load_model(self):
        """加载Diffusers模型"""
        logger.info(f"Loading Diffusers WanI2V from {self.ckpt_dir}")
        
        # 🔥 加载图像编码器
        image_encoder = CLIPVisionModel.from_pretrained(
            self.ckpt_dir,
            subfolder="image_encoder",
            torch_dtype=torch.float32
        )
        
        # 🔥 加载VAE
        vae = AutoencoderKLWan.from_pretrained(
            self.ckpt_dir,
            subfolder="vae",
            torch_dtype=torch.float32
        )
        
        # 🔥 配置调度器
        scheduler = UniPCMultistepScheduler(
            prediction_type='flow_prediction',
            use_flow_sigmas=True,
            num_train_timesteps=1000,
            flow_shift=3.0
        )
        
        # 🔥 创建管道
        pipeline = WanImageToVideoPipeline.from_pretrained(
            self.ckpt_dir,
            vae=vae,
            image_encoder=image_encoder,
            torch_dtype=torch.bfloat16
        ).to(self.device)
        pipeline.scheduler = scheduler
        
        logger.info("Diffusers WanI2V loaded successfully")
        return pipeline

    def generate_video(self, request, task_id, progress_callback=None):
        """生成视频的主入口"""
        try:
            if progress_callback:
                progress_callback(5, "加载图片")
                
            # 🔥 处理图片输入
            if hasattr(request, 'image_path') and request.image_path:
                if request.image_path.startswith("http"):
                    response = requests.get(request.image_path, timeout=30)
                    if response.status_code != 200:
                        raise ValueError(f"Failed to download image: HTTP {response.status_code}")
                    img = Image.open(BytesIO(response.content)).convert("RGB")
                else:
                    img = Image.open(request.image_path).convert("RGB")
            else:
                raise ValueError("image_path is required")

            if progress_callback:
                progress_callback(10, "预处理图片")

            # 🔥 使用线程锁确保模型安全
            with self.model_lock:
                video_path = self._generate_video_sync(request, task_id, img, progress_callback)
            
            if progress_callback:
                progress_callback(100, "完成")
            
            return f"/videos/{os.path.basename(video_path)}"
            
        except Exception as e:
            logger.error(f"Diffuser video generation failed: {e}")
            raise

    def _generate_video_sync(self, request, task_id: str, image: Image.Image, progress_callback=None):
        """同步生成核心"""
        try:
            if progress_callback:
                progress_callback(15, "解析分辨率")

            # 🔥 解析分辨率 - 兼容两种格式
            mod_value = 16
            image_size = getattr(request, "image_size", "1280*720")  # 🔥 改为默认值而不是auto

            if image_size == "auto":
                # 自动计算逻辑
                aspect_ratio = image.height / image.width
                max_area = 399360  # 模型基础分辨率

                height = round(np.sqrt(max_area * aspect_ratio)) 
                width = round(np.sqrt(max_area / aspect_ratio))

                # 应用模数调整
                height = height // mod_value * mod_value
                width = width // mod_value * mod_value
                logger.info(f"Auto-calculated size: {width}x{height}")
            else:
                # 🔥 兼容两种分隔符: "x" 和 "*"
                if 'x' in image_size:
                    width_str, height_str = image_size.split('x')
                elif '*' in image_size:
                    width_str, height_str = image_size.split('*')
                else:
                    raise ValueError(f"Invalid image_size format: {image_size}. Use 'WIDTHxHEIGHT' or 'WIDTH*HEIGHT'")

                width = int(width_str)
                height = int(height_str)
                logger.info(f"Using specified size: {width}x{height}")

            # 调整图像尺寸
            resized_image = image.resize((width, height))

            if progress_callback:
                progress_callback(25, "设置生成参数")

            # 🔥 设置随机种子
            generator = None
            seed = getattr(request, "seed", None)
            if seed is not None:
                generator = torch.Generator(device=self.device)
                generator.manual_seed(seed)
                logger.info(f"Using random seed: {seed}")

            if progress_callback:
                progress_callback(30, "开始模型推理")

            # 🔥 获取参数 - 只使用sample_steps
            num_frames = getattr(request, "num_frames", 81)
            guidance_scale = getattr(request, "guidance_scale", 3.0)
            num_inference_steps = getattr(request, "sample_steps", 30)  # 🔥 只使用sample_steps
            negative_prompt = getattr(request, "negative_prompt", "") or None

            logger.info(f"Generation params: {width}x{height}, {num_frames} frames, {num_inference_steps} steps, guidance={guidance_scale}")

            # 🔥 执行推理
            output = self.model(
                image=resized_image,
                prompt=request.prompt,
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                num_frames=num_frames,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                generator=generator
            ).frames[0]

            if progress_callback:
                progress_callback(90, "导出视频")

            # 🔥 导出视频
            output_path = f"generated_videos/{task_id}.mp4"  # 🔥 简化文件名
            os.makedirs("generated_videos", exist_ok=True)

            export_to_video(output, output_path, fps=16)

            logger.info(f"Video exported to: {output_path}")
            return output_path

        except Exception as e:
            raise RuntimeError(f"Video generation failed: {str(e)}") from e


    def _log_memory_usage(self):
        try:
            memory_allocated = torch.cuda.memory_allocated() / 1024**3
            memory_reserved = torch.cuda.memory_reserved() / 1024**3
            logger.info(f"CUDA Memory - Allocated: {memory_allocated:.2f}GB, Reserved: {memory_reserved:.2f}GB")
        except Exception as e:
            logger.warning(f"Failed to get CUDA memory info: {e}")

    def _empty_cache(self):
        torch.cuda.empty_cache()