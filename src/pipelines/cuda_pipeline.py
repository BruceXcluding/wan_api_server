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

    def __init__(self, ckpt_dir: str, rank=0, world_size=1, **model_args):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available, cannot use CUDA pipeline")
        
        # 分布式参数
        self.rank = rank
        self.world_size = world_size
        self.local_rank = int(os.environ.get("LOCAL_RANK", rank))
        
        self.device_type = model_args.get("device_type", "cuda")
        self.t5_cpu = model_args.get("t5_cpu", False)
        self.t5_fsdp = model_args.get("t5_fsdp", world_size > 1)
        self.dit_fsdp = model_args.get("dit_fsdp", world_size > 1)
        self.cfg_size = model_args.get("cfg_size", 1)
        self.ulysses_size = model_args.get("ulysses_size", world_size)
        self.ring_size = model_args.get("ring_size", 1)
        self.vae_parallel = model_args.get("vae_parallel", world_size > 1)
        self.offload_model = model_args.get("offload_model", False)
        
        super().__init__(ckpt_dir, **model_args)

    def _get_backend(self) -> str:
        return "nccl"

    def _setup_device_distributed(self):
        """CUDA设备特定的分布式初始化"""
        if self.world_size > 1:
            logger.info(f"Rank {self.rank}: Setting up xfuser distributed environment...")

            try:
                # 🔥 对齐本地generate.py的xfuser初始化
                from xfuser.core.distributed import (
                    init_distributed_environment,
                    initialize_model_parallel,
                )

                # 先初始化分布式环境
                init_distributed_environment(
                    rank=self.rank, 
                    world_size=self.world_size
                )

                # 再初始化模型并行组
                initialize_model_parallel(
                    sequence_parallel_degree=self.world_size,
                    ring_degree=self.ring_size,
                    ulysses_degree=self.ulysses_size,
                )

                logger.info(f"Rank {self.rank}: xfuser model parallel initialized")

            except Exception as e:
                logger.error(f"Rank {self.rank}: xfuser setup failed: {e}")
                raise RuntimeError(f"Cannot initialize xfuser: {e}")

    def _load_model(self):
        torch.cuda.set_device(self.local_rank)
        print(f"[GPU {self.local_rank}] Rank {self.rank}: Loading WanI2V...")
        logger.info(f"Rank {self.rank}: Loading WanI2V on CUDA:{self.local_rank} (t5_cpu={self.t5_cpu})")

        cfg = WAN_CONFIGS.get("i2v-14B")

        model_config = {
            "config": cfg,
            "checkpoint_dir": self.ckpt_dir,
            "device_id": self.local_rank,
            "t5_cpu": self.t5_cpu,
        }

        # 多卡模式配置
        if self.world_size > 1:
            model_config.update({
                "rank": self.rank,
                "t5_fsdp": self.t5_fsdp,
                "dit_fsdp": self.dit_fsdp,
                "use_usp": (self.ulysses_size > 1 or self.ring_size > 1),
            })
            print(f"[GPU {self.local_rank}] Rank {self.rank}: Multi-GPU config enabled")
            logger.info(f"Rank {self.rank}: Multi-GPU config enabled")

        model = wan.WanI2V(**model_config)
        print(f"[GPU {self.local_rank}] Rank {self.rank}: WanI2V loaded!")
        logger.info(f"Rank {self.rank}: WanI2V loaded successfully")

        return model

    def _generate_video_device_specific(self, request, img, progress_callback=None):
        """设备特定的视频生成 - 支持动态调度"""
        print(f"[GPU {self.local_rank}] Rank {self.rank}: STARTING VIDEO GENERATION")
        logger.info(f"Rank {self.rank}: Generating video on CUDA:{self.local_rank}")

        # 🔥 检查当前rank是否应该参与计算
        current_rank = int(os.environ.get("RANK", self.rank))
        expected_world_size = int(os.environ.get("WORLD_SIZE", self.world_size))
        
        if current_rank >= expected_world_size:
            logger.info(f"Rank {current_rank}: Skipping CUDA generation (not in active group, expected_world_size={expected_world_size})")
            return None

        # 🔥 分布式同步点和参数广播（只有参与的rank）
        if expected_world_size > 1:
            import torch.distributed as dist
            if dist.is_initialized():
                print(f"[GPU {self.local_rank}] Rank {self.rank}: Broadcasting parameters...")

                # 🔥 使用动态的rank作为src判断
                dynamic_rank_0 = (current_rank == 0)
                
                if dynamic_rank_0:
                    num_frames = getattr(request, "num_frames", 81)
                    if num_frames != 81:
                        logger.warning(f"num_frames {num_frames} not supported, using 81")
                        num_frames = 81
                
                    image_size = getattr(request, "image_size", "1280*720")
                    
                    # 🔥 确保尺寸合理
                    if '*' in image_size:
                        h_str, w_str = image_size.split('*')
                    elif 'x' in image_size:
                        w_str, h_str = image_size.split('x')
                    else:
                        h_str, w_str = "720", "1280"
                    
                    height, width = int(h_str), int(w_str)
                    
                    # 🔥 确保尺寸是32的倍数（模型要求）
                    height = (height // 32) * 32
                    width = (width // 32) * 32
                    image_size = f"{height}*{width}"
                    
                    logger.info(f"Rank {self.rank}: Validated params - frames: {num_frames}, size: {image_size}")
                    
                    params = {
                        'prompt': request.prompt,
                        'image_size': image_size,
                        'num_frames': num_frames,
                        'sample_shift': getattr(request, "sample_shift", 5.0),
                        'sample_solver': getattr(request, "sample_solver", "unipc"),
                        'sample_steps': getattr(request, "sample_steps", 40),
                        'guidance_scale': getattr(request, "guidance_scale", 5.0),
                        'seed': getattr(request, "seed", 42) if getattr(request, "seed", None) is not None else 42,
                        'offload_model': getattr(request, "offload_model", self.offload_model),
                    }
                    params_list = [params]
                else:
                    params_list = [None]

                # 🔥 只有参与的rank进行广播
                dist.broadcast_object_list(params_list, src=0)
                params = params_list[0]

                print(f"[GPU {self.local_rank}] Rank {self.rank}: Parameters received")
                dist.barrier()
                print(f"[GPU {self.local_rank}] Rank {self.rank}: All ranks synchronized")

                # 使用广播的参数
                prompt = params['prompt']
                image_size = params['image_size']
                num_frames = params['num_frames']
                sample_shift = params['sample_shift']
                sample_solver = params['sample_solver']
                sample_steps = params['sample_steps']
                guidance_scale = params['guidance_scale']
                seed = params['seed']
                offload_model = params['offload_model']
            else:
                # 🔥 非分布式模式的参数处理
                prompt = request.prompt
                image_size = getattr(request, "image_size", "1280*720")
                num_frames = 81 
                sample_shift = getattr(request, "sample_shift", 5.0)
                sample_solver = getattr(request, "sample_solver", "unipc")
                sample_steps = getattr(request, "sample_steps", 40)
                guidance_scale = getattr(request, "guidance_scale", 5.0)
                seed = getattr(request, "seed", 42) if getattr(request, "seed", None) is not None else 42
                offload_model = getattr(request, "offload_model", self.offload_model)
        else:
            # 🔥 单卡模式参数处理
            prompt = request.prompt
            image_size = getattr(request, "image_size", "1280*720")
            num_frames = 81
            sample_shift = getattr(request, "sample_shift", 5.0)
            sample_solver = getattr(request, "sample_solver", "unipc")
            sample_steps = getattr(request, "sample_steps", 40)
            guidance_scale = getattr(request, "guidance_scale", 5.0)
            seed = getattr(request, "seed", 42) if getattr(request, "seed", None) is not None else 42
            offload_model = getattr(request, "offload_model", self.offload_model)

        # 确保图片在正确设备上
        if hasattr(img, 'to'):
            img = img.to(f'cuda:{self.local_rank}')

        # 解析尺寸
        if '*' in image_size:
            height_str, width_str = image_size.split('*')
        elif 'x' in image_size:
            width_str, height_str = image_size.split('x')
        else:
            height_str, width_str = "720", "1280"

        height, width = int(height_str), int(width_str)
        max_area = width * height

        if progress_callback and current_rank == 0:
            progress_callback(15, "模型推理")

        # 🔥 每个参与的rank都调用model.generate
        print(f"[GPU {self.local_rank}] Rank {self.rank}: Calling model.generate()...")
        logger.info(f"Rank {self.rank}: Starting generation - {width}x{height}, {num_frames} frames")

        # 🔥 使用广播的参数进行生成
        video = self.model.generate(
            prompt,
            img,
            max_area=max_area,
            frame_num=num_frames,
            shift=sample_shift,
            sample_solver=sample_solver,
            sampling_steps=sample_steps,
            guide_scale=guidance_scale,
            seed=seed,
            offload_model=offload_model,
        )

        print(f"[GPU {self.local_rank}] Rank {self.rank}: Generation COMPLETED!")
        logger.info(f"Rank {self.rank}: Generation completed on CUDA:{self.local_rank}")

        # 🔥 分布式同步（只有参与的rank）
        if expected_world_size > 1:
            import torch.distributed as dist
            if dist.is_initialized():
                dist.barrier()
                logger.info(f"Rank {self.rank}: Final sync completed")

        return video

    def _save_video(self, video_tensor, output_path: str):
        """保存视频 - 只有rank 0执行"""
        current_rank = int(os.environ.get("RANK", self.rank))
        
        if current_rank == 0:
            logger.info(f"Saving video to {output_path}")
            try:
                from wan.utils.utils import cache_video
                # 🔥 参考generate.py的保存逻辑
                cache_video(
                    tensor=video_tensor[None] if video_tensor.ndim == 4 else video_tensor,
                    save_file=output_path,
                    fps=16,
                    nrow=1,
                    normalize=True,
                    value_range=(-1, 1)
                )
                logger.info(f"Video saved successfully: {output_path}")
            except Exception as e:
                logger.error(f"Failed to save video: {e}")
                # 创建占位符文件
                with open(output_path, "wb") as f:
                    f.write(b"VIDEO_SAVE_FAILED")
                raise

    def _log_memory_usage(self):
        try:
            memory_allocated = torch.cuda.memory_allocated(self.local_rank) / 1024**3
            memory_reserved = torch.cuda.memory_reserved(self.local_rank) / 1024**3
            logger.info(f"Rank {self.rank} CUDA Memory - Allocated: {memory_allocated:.2f}GB, Reserved: {memory_reserved:.2f}GB")
        except Exception as e:
            logger.warning(f"Rank {self.rank}: Failed to get CUDA memory info: {e}")

    def _empty_cache(self):
        torch.cuda.empty_cache()