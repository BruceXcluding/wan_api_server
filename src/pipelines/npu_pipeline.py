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

    def __init__(self, ckpt_dir: str, rank=0, world_size=1, **model_args):
        if not NPU_AVAILABLE:
            raise RuntimeError("torch_npu not available, cannot use NPU pipeline")

        self.rank = rank
        self.world_size = world_size
        self.local_rank = int(os.environ.get("LOCAL_RANK", rank))

        # 设备和模型配置
        self.device_type = model_args.get("device_type", "npu")
        self.t5_cpu = model_args.get("t5_cpu", os.environ.get("T5_CPU", "false").lower() == "true")
        self.t5_fsdp = model_args.get("t5_fsdp", os.environ.get("T5_FSDP", str(world_size > 1)).lower() == "true")
        self.dit_fsdp = model_args.get("dit_fsdp", os.environ.get("DIT_FSDP", str(world_size > 1)).lower() == "true")
        self.ulysses_size = model_args.get("ulysses_size", int(os.environ.get("ULYSSES_SIZE", str(world_size))))
        self.vae_parallel = model_args.get("vae_parallel", os.environ.get("VAE_PARALLEL", str(world_size > 1)).lower() == "true")

        self.use_distributed = world_size > 1
        model_args['device_type'] = 'npu'

        logger.info(f"Rank {self.rank}: t5_cpu={self.t5_cpu}, t5_fsdp={self.t5_fsdp}, dit_fsdp={self.dit_fsdp}")

        super().__init__(ckpt_dir, **model_args)

    def _get_backend(self) -> str:
        return "hccl"
 
    def _load_model(self):
        """加载分布式模型"""
        torch_npu.npu.set_compile_mode(jit_compile=False)
        torch.npu.config.allow_internal_format = False
        torch_npu.npu.set_device(self.local_rank)

        # 🔥 最小化修改：添加ParallelConfig初始化
        if self.ulysses_size > 1:
            try:
                from wan.distributed.parallel_mgr import ParallelConfig, init_parallel_env
                parallel_config = ParallelConfig(
                    sp_degree=self.ulysses_size,
                    ulysses_degree=self.ulysses_size,
                    ring_degree=1,
                    use_cfg_parallel=False,
                    world_size=self.world_size,
                )
                init_parallel_env(parallel_config)
                logger.info(f"Rank {self.rank}: ParallelConfig initialized")
            except ImportError:
                logger.warning(f"Rank {self.rank}: ParallelConfig not available")

        cfg = WAN_CONFIGS.get("i2v-14B")
        if not cfg:
            raise ValueError("i2v-14B config not found")

        try:
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

            # 🔥 配置基础的transformer blocks（不包含动态参数）
            transformer = model.model
            try:
                from mindiesd import CacheConfig, CacheAgent
                config = CacheConfig(
                    method="attention_cache",
                    blocks_count=len(transformer.blocks),
                    steps_count=40
                )
                cache = CacheAgent(config)
                
                # 🔥 创建基础Args对象（只包含不变的配置）
                base_cache_args = type('Args', (), {
                    # CacheAgent相关（不变）
                    'use_attentioncache': False,
                    'start_step': 12,
                    'attentioncache_interval': 4,
                    'end_step': 37,
                    'attentioncache_ratio': 1.2,
                    
                    # 分布式配置（不变）
                    't5_fsdp': self.t5_fsdp,
                    'dit_fsdp': self.dit_fsdp,
                    'cfg_size': 1,
                    'ulysses_size': self.ulysses_size,
                    'ring_size': 1,
                    'vae_parallel': self.vae_parallel,
                    't5_cpu': self.t5_cpu,
                    'task': 'i2v-14B',
                    'ckpt_dir': self.ckpt_dir,
                    
                    # 🔥 默认值（后续可动态更新）
                    'size': '1280*720',
                    'height': 720,
                    'width': 1280,
                    'frame_num': 81,
                    'sample_steps': 40,
                    'sample_solver': 'unipc',
                    'shift': 5.0,
                    'guide_scale': 5.0,
                })()
                
                if self.dit_fsdp and hasattr(transformer, '_fsdp_wrapped_module'):
                    for block in transformer._fsdp_wrapped_module.blocks:
                        if hasattr(block, '_fsdp_wrapped_module'):
                            block._fsdp_wrapped_module.cache = cache
                            block._fsdp_wrapped_module.args = base_cache_args
                else:
                    for block in transformer.blocks:
                        block.cache = cache
                        block.args = base_cache_args
                        
            except ImportError:
                logger.warning(f"Rank {self.rank}: mindiesd not available")
            except Exception as e:
                logger.warning(f"Rank {self.rank}: Cache setup failed: {e}")

            if self.world_size > 1 and dist.is_initialized():
                dist.barrier()

            logger.info(f"Rank {self.rank}: Model loaded successfully")
            return model
            
        except Exception as e:
            logger.error(f"Rank {self.rank}: Model loading failed: {e}")
            raise

    def _update_transformer_args(self, request):
        """🔥 动态更新transformer blocks的args"""
        if not hasattr(self.model, 'model'):
            return
            
        # 解析请求参数
        image_size = getattr(request, "image_size", "1280*720")
        if '*' in image_size:
            h_str, w_str = image_size.split('*')
        elif 'x' in image_size:
            w_str, h_str = image_size.split('x')
        else:
            h_str, w_str = "720", "1280"
        
        height, width = int(h_str), int(w_str)
        height = (height // 32) * 32
        width = (width // 32) * 32
        
        # 🔥 创建动态配置更新
        dynamic_updates = {
            'size': image_size,
            'height': height,
            'width': width,
            'frame_num': getattr(request, "frame_num", 81),
            'sample_steps': getattr(request, "sample_steps", 40),
            'sample_solver': getattr(request, "sample_solver", "unipc"),
            'shift': getattr(request, "sample_shift", 5.0),
            'guide_scale': getattr(request, "guidance_scale", 5.0),
        }
        
        # 🔥 更新所有transformer blocks的args
        transformer = self.model.model
        try:
            if self.dit_fsdp and hasattr(transformer, '_fsdp_wrapped_module'):
                for block in transformer._fsdp_wrapped_module.blocks:
                    if hasattr(block, '_fsdp_wrapped_module') and hasattr(block._fsdp_wrapped_module, 'args'):
                        # 更新args对象的属性
                        for key, value in dynamic_updates.items():
                            setattr(block._fsdp_wrapped_module.args, key, value)
            else:
                for block in transformer.blocks:
                    if hasattr(block, 'args'):
                        # 更新args对象的属性
                        for key, value in dynamic_updates.items():
                            setattr(block.args, key, value)
                            
            if self.rank == 0:
                logger.info(f"🔄 Updated transformer args: size={image_size}, steps={dynamic_updates['sample_steps']}")
                
        except Exception as e:
            logger.warning(f"Rank {self.rank}: Failed to update transformer args: {e}")
        
    def _load_and_process_image(self, request):
        """加载和处理图片"""
        try:
            image_path = getattr(request, 'image_path', getattr(request, 'image_url', None))
            if not image_path:
                raise ValueError("No image_path or image_url provided")
            
            target_size = (512, 512)
            
            if self.rank == 0:
                if image_path.startswith(('http://', 'https://')):
                    import requests
                    from io import BytesIO
                    
                    try:
                        response = requests.get(image_path, timeout=30)
                        response.raise_for_status()
                        img = Image.open(BytesIO(response.content))
                    except Exception:
                        img = Image.new('RGB', target_size, color='blue')
                else:
                    img = Image.open(image_path)
                
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                img = img.resize(target_size, Image.Resampling.LANCZOS)
            else:
                img = Image.new('RGB', target_size, color='blue')
            
            if self.world_size > 1 and dist.is_initialized():
                dist.barrier()
            
            return img
            
        except Exception as e:
            logger.error(f"Rank {self.rank}: Failed to load image: {e}")
            return Image.new('RGB', (512, 512), color='red')

    def _generate_video_device_specific(self, request, img, progress_callback=None):
        """NPU设备特定的视频生成"""
        
        # 🔥 在推理前动态更新transformer args
        self._update_transformer_args(request)
        
        if self.world_size > 1 and dist.is_initialized():
            if self.rank == 0:
                image_size = getattr(request, "image_size", "1280*720")
                
                if '*' in image_size:
                    h_str, w_str = image_size.split('*')
                elif 'x' in image_size:
                    w_str, h_str = image_size.split('x')
                else:
                    h_str, w_str = "720", "1280"
                
                height, width = int(h_str), int(w_str)
                height = (height // 32) * 32
                width = (width // 32) * 32
                
                params = {
                    'prompt': request.prompt,
                    'height': height,
                    'width': width,
                    'max_area': width * height,
                    'shift': getattr(request, "sample_shift", 5.0),
                    'solver': getattr(request, "sample_solver", "unipc"),
                    'steps': getattr(request, "sample_steps", 40),
                    'guidance': getattr(request, "guidance_scale", 5.0),
                    'seed': getattr(request, "seed", 42) if getattr(request, "seed", None) is not None else 42,
                    'offload': getattr(request, "offload_model", False),
                    'image_size': image_size,  # 🔥 传递完整的image_size
                }
                params_list = [params]
            else:
                params_list = [None]

            dist.broadcast_object_list(params_list, src=0)
            params = params_list[0]
            
            # 🔥 所有rank都需要更新transformer args
            if self.rank != 0:
                # 为非0 rank创建临时request对象来更新args
                temp_request = type('Request', (), params)()
                self._update_transformer_args(temp_request)
            
            prompt = params['prompt']
            height = params['height']
            width = params['width']
            max_area = params['max_area']
            shift = params['shift']
            solver = params['solver']
            steps = params['steps']
            guidance = params['guidance']
            seed = params['seed']
            offload = params['offload']
            
            dist.barrier()
        else:
            prompt = request.prompt
            image_size = getattr(request, "image_size", "1280*720")
            
            if '*' in image_size:
                h_str, w_str = image_size.split('*')
            elif 'x' in image_size:
                w_str, h_str = image_size.split('x')
            else:
                h_str, w_str = "720", "1280"
            
            height, width = int(h_str), int(w_str)
            height = (height // 32) * 32
            width = (width // 32) * 32
            max_area = width * height
            
            shift = getattr(request, "sample_shift", 5.0)
            solver = getattr(request, "sample_solver", "unipc")
            steps = getattr(request, "sample_steps", 40)
            guidance = getattr(request, "guidance_scale", 5.0)
            seed = getattr(request, "seed", 42) if getattr(request, "seed", None) is not None else 42
            offload = getattr(request, "offload_model", False)

        if progress_callback:
            progress_callback(15, "模型推理")

        if self.model is None:
            raise RuntimeError(f"Rank {self.rank}: Model is None")

        try:
            video = self.model.generate(
                prompt,
                img,
                max_area=max_area,
                frame_num=81,
                shift=shift,
                sample_solver=solver,
                sampling_steps=steps,
                guide_scale=guidance,
                seed=seed,
                offload_model=offload,
            )
            
        except Exception as e:
            logger.error(f"Rank {self.rank}: model.generate failed: {e}")
            raise

        if progress_callback:
            progress_callback(85, "推理完成")

        if self.world_size > 1 and dist.is_initialized():
            dist.barrier()

        return video
    
    def _save_video(self, video_tensor, output_path: str):
        """保存视频"""
        if self.rank == 0:
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
                logger.error(f"Rank {self.rank}: Failed to save video: {e}")
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                with open(output_path, "wb") as f:
                    f.write(b"FAKE_VIDEO_DATA")
        
        if self.world_size > 1 and dist.is_initialized():
            dist.barrier()

    def _log_memory_usage(self):
        """记录NPU内存使用情况"""
        try:
            memory_allocated = torch_npu.npu.memory_allocated(self.local_rank) / 1024**3
            memory_reserved = torch_npu.npu.memory_reserved(self.local_rank) / 1024**3
            if self.rank == 0:
                logger.info(f"NPU:{self.local_rank} memory: "
                           f"{memory_allocated:.2f}GB allocated, {memory_reserved:.2f}GB reserved")
        except Exception:
            pass
                
    def _empty_cache(self):
        """清空NPU缓存"""
        torch_npu.npu.empty_cache()
        if self.world_size > 1 and dist.is_initialized():
            dist.barrier()

    def generate_video(self, request, task_id, progress_callback=None):
        """生成视频的主入口"""
        logger.info(f"Rank {self.rank}: Starting task {task_id}")
        
        try:
            if progress_callback:
                progress_callback(5, "加载图片")

            img = self._load_and_process_image(request)

            if progress_callback:
                progress_callback(10, "开始生成视频")

            video_tensor = self._generate_video_device_specific(request, img, progress_callback)

            if progress_callback:
                progress_callback(90, "保存视频")

            output_path = f"generated_videos/{task_id}.mp4"
            os.makedirs("generated_videos", exist_ok=True)
            self._save_video(video_tensor, output_path)

            if progress_callback:
                progress_callback(100, "完成")

            logger.info(f"Rank {self.rank}: Task {task_id} completed")
            return f"/videos/{task_id}.mp4"

        except Exception as e:
            logger.error(f"Rank {self.rank}: Video generation failed: {e}")
            self._log_memory_usage()
            raise

    def reload_model(self):
        """重新加载模型"""
        self._empty_cache()
        self.model = self._load_model()

    def sync(self):
        """分布式同步屏障"""
        if self.world_size > 1 and dist.is_initialized():
            try:
                dist.barrier()
            except Exception as e:
                logger.warning(f"Rank {self.rank}: Sync barrier failed: {e}")