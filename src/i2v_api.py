import os
import sys
import torch
import torch.distributed as dist
from datetime import datetime, timedelta

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

import logging
import uuid
import threading
import time
from typing import List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn

from schemas import VideoSubmitRequest, VideoSubmitResponse, VideoStatusResponse, TaskStatus
from pipelines.pipeline_factory import create_pipeline
from utils.device_detector import detect_device
from utils.load_monitor import LoadMonitor
from utils.dynamic_scheduler import DynamicGPUScheduler

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# 全局变量
task_queue = []
status_dict = {}
cancelled_tasks = set()
pipeline = None
load_monitor = None
dynamic_scheduler = None

def init_distributed():
    """初始化分布式环境"""
    try:
        import torch_npu
        npu_available = torch_npu.npu.is_available()
    except ImportError:
        npu_available = False
    
    # 获取分布式信息
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    logger.info(f"Initializing rank {rank}/{world_size}, local_rank {local_rank}")
    
    if world_size > 1:
        # 🔥 先设置设备再初始化分布式
        if npu_available:
            import torch_npu
            torch_npu.npu.set_device(local_rank)
            logger.info(f"Rank {rank}: Set NPU device {local_rank}")
        elif torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            logger.info(f"Rank {rank}: Set CUDA device {local_rank}")
        
        # 🔥 使用正确的分布式初始化
        backend = "hccl" if npu_available else "nccl" if torch.cuda.is_available() else "gloo"
        
        try:
            if not dist.is_initialized():
                dist.init_process_group(
                    backend=backend,
                    timeout=timedelta(seconds=300),
                    init_method='env://'
                )
                logger.info(f"Rank {rank}: Distributed initialized with {backend}")
            else:
                logger.info(f"Rank {rank}: Distributed already initialized")
        except Exception as e:
            logger.error(f"Rank {rank}: Failed to initialize distributed: {e}")
            raise
    else:
        logger.info(f"Rank {rank}: Single device mode")
    
    return rank, local_rank, world_size

def create_app():
    """创建FastAPI应用"""
    app = FastAPI(title="Wan2.1 I2V Multi-Device API Server")
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
    
    os.makedirs("generated_videos", exist_ok=True)
    app.mount("/videos", StaticFiles(directory="generated_videos"), name="videos")

    @app.post("/submit", response_model=VideoSubmitResponse)
    async def submit(request: VideoSubmitRequest):
        task_id = f"req_{uuid.uuid4().hex[:16]}"
        status_dict[task_id] = {
            "status": TaskStatus.PENDING, 
            "result_url": "", 
            "error": "",
            "created_at": datetime.now().isoformat(),
            "progress": 0
        }
        task_queue.append((task_id, request))
        logger.info(f"Task submitted: {task_id}")
        return VideoSubmitResponse(requestId=task_id, status=TaskStatus.PENDING, message="任务已提交", estimated_time=30)

    @app.post("/batch_submit", response_model=List[VideoSubmitResponse])
    async def batch_submit(requests: List[VideoSubmitRequest]):
        responses = []
        for req in requests:
            task_id = f"req_{uuid.uuid4().hex[:16]}"
            status_dict[task_id] = {
                "status": TaskStatus.PENDING, 
                "result_url": "", 
                "error": "",
                "created_at": datetime.now().isoformat(),
                "progress": 0
            }
            task_queue.append((task_id, req))
            responses.append(VideoSubmitResponse(requestId=task_id, status=TaskStatus.PENDING, message="任务已提交", estimated_time=30))
        return responses

    @app.post("/cancel/{task_id}")
    async def cancel_task(task_id: str):
        if task_id not in status_dict:
            raise HTTPException(status_code=404, detail="Task not found")
        
        current_status = status_dict[task_id]["status"]
        if current_status in [TaskStatus.SUCCEED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
            raise HTTPException(status_code=400, detail=f"Task already {current_status.value}")
        
        cancelled_tasks.add(task_id)
        status_dict[task_id]["status"] = TaskStatus.CANCELLED
        status_dict[task_id]["updated_at"] = datetime.now().isoformat()
        
        global task_queue
        task_queue = [(tid, req) for tid, req in task_queue if tid != task_id]
        
        logger.info(f"Task {task_id} cancelled")
        return {"message": f"Task {task_id} cancelled successfully"}

    @app.post("/cancel_all")
    async def cancel_all():
        cancelled_count = 0
        for task_id, task_info in status_dict.items():
            if task_info["status"] in [TaskStatus.PENDING, TaskStatus.RUNNING]:
                cancelled_tasks.add(task_id)
                status_dict[task_id]["status"] = TaskStatus.CANCELLED
                status_dict[task_id]["updated_at"] = datetime.now().isoformat()
                cancelled_count += 1
        
        global task_queue
        task_queue = []
        
        logger.info(f"Cancelled {cancelled_count} tasks")
        return {"message": f"Cancelled {cancelled_count} tasks"}

    @app.get("/status/{task_id}", response_model=VideoStatusResponse)
    async def status(task_id: str):
        s = status_dict.get(task_id)
        if not s:
            raise HTTPException(status_code=404, detail="Task not found")

        message = "任务处理中"
        if s.get("status") == TaskStatus.RUNNING:
            stage = s.get("current_stage", "处理中")
            progress = s.get("progress", 0)
            message = f"{stage} ({progress:.1f}%)"
        elif s.get("status") == TaskStatus.FAILED:
            message = s.get("error", "生成失败")
        elif s.get("status") == TaskStatus.SUCCEED:
            message = "生成完成"
        elif s.get("status") == TaskStatus.CANCELLED:
            message = "任务已取消"

        return VideoStatusResponse(
            requestId=task_id,
            status=s.get("status", TaskStatus.PENDING),
            progress=s.get("progress", 0),
            message=message,
            created_at=s.get("created_at", ""),
            updated_at=s.get("updated_at", ""),
            results=s.get("results"),
            reason=s.get("reason"),
            elapsed_time=s.get("elapsed_time"),
            current_stage=s.get("current_stage"),
        )

    @app.get("/list")
    async def list_tasks():
        """列出所有任务状态"""
        return {
            "total_tasks": len(status_dict),
            "queue_size": len(task_queue),
            "cancelled_tasks": len(cancelled_tasks),
            "tasks": [
                {
                    "id": task_id,
                    "status": info.get("status", "unknown"),
                    "progress": info.get("progress", 0),
                    "created_at": info.get("created_at", ""),
                    "elapsed_time": info.get("elapsed_time", ""),
                    "assigned_gpus": info.get("assigned_gpus", [])
                }
                for task_id, info in status_dict.items()
            ]
        }

    @app.get("/health")
    async def health():
        dtype, dcount, backend = detect_device()
        return {
            "device_type": dtype, 
            "device_count": dcount, 
            "backend": backend, 
            "queue_size": len(task_queue), 
            "total_tasks": len(status_dict),
            "cancelled_tasks": len(cancelled_tasks)
        }
    
    @app.get("/monitor")
    async def monitor():
        """简单的任务监控"""
        running = [{"id": k, "progress": v.get("progress", 0), "stage": v.get("current_stage", "")} 
                   for k, v in status_dict.items() if v.get("status") == TaskStatus.RUNNING]

        return {
            "queue_size": len(task_queue),
            "running_tasks": running,
            "total_tasks": len(status_dict)
        }

    @app.get("/load/status")
    async def get_load_status():
        """获取负载状态"""
        global dynamic_scheduler
        if not dynamic_scheduler:
            return {
                "load_balancing": False,
                "message": "Load balancing not available"
            }
        
        loads = dynamic_scheduler.load_monitor.get_current_loads()
        status = dynamic_scheduler.load_monitor.get_load_status()
        
        return {
            "load_balancing": True,
            "device_count": dynamic_scheduler.world_size,
            "device_type": dynamic_scheduler.device_type,
            "status": status,
            "details": {
                str(rank): {
                    "memory_used_gb": round(load.memory_used_gb, 2),
                    "memory_total_gb": round(load.memory_total_gb, 2),
                    "utilization": round(load.utilization, 3),
                    "load_score": round(load.load_score, 3)
                }
                for rank, load in loads.items()
            },
            "active_tasks": len(status_dict)
        }

    @app.get("/scheduler/status")
    async def get_scheduler_status():
        """获取动态调度器状态"""
        global dynamic_scheduler
        if dynamic_scheduler:
            return dynamic_scheduler.get_scheduler_status()
        else:
            return {"dynamic_scheduling": False, "message": "Dynamic scheduling not enabled"}

    @app.get("/cluster/health")
    async def get_cluster_health():
        """获取集群健康状态"""
        global dynamic_scheduler
        health_info = {
            "status": "healthy",
            "timestamp": time.time(),
            "active_tasks": len(status_dict),
            "queue_size": len(task_queue),
            "load_balancing": dynamic_scheduler is not None
        }
        
        if dynamic_scheduler:
            status = dynamic_scheduler.load_monitor.get_load_status()
            health_info.update({
                "device_status": {
                    "total_devices": dynamic_scheduler.world_size,
                    "available": len(status["available"]),
                    "busy": len(status["busy"]),
                    "overloaded": len(status["overloaded"])
                }
            })
            
            if len(status["overloaded"]) > dynamic_scheduler.world_size * 0.7:
                health_info["status"] = "critical"
            elif len(status["overloaded"]) > dynamic_scheduler.world_size * 0.3:
                health_info["status"] = "warning"
        
        return health_info

    return app

def process_single_task(task_id: str, request):
    """处理单个任务 - 支持动态GPU调度"""
    global dynamic_scheduler

    import torch.distributed as dist
    from datetime import datetime
    
    try:
        logger.info(f"🚀 Processing task with dynamic scheduling: {task_id}")
        
        # 🔥 动态GPU调度
        assigned_ranks = None
        if dynamic_scheduler:
            assigned_ranks = dynamic_scheduler.schedule_task(task_id, request)
            
            if assigned_ranks is None:
                # 任务被加入队列，重新放回队列头部
                logger.info(f"Task {task_id}: Queued, waiting for GPU resources")
                task_queue.insert(0, (task_id, request))
                time.sleep(5)
                return
            
            logger.info(f"Task {task_id}: Assigned to GPUs {assigned_ranks}")
        else:
            # 回退到全GPU模式
            assigned_ranks = list(range(int(os.environ.get("WORLD_SIZE", 1))))
        
        status_dict[task_id]["status"] = TaskStatus.RUNNING
        status_dict[task_id]["updated_at"] = datetime.now().isoformat()
        status_dict[task_id]["start_time"] = datetime.now()
        status_dict[task_id]["assigned_gpus"] = assigned_ranks
        
        # 🔥 广播任务信号
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        if world_size > 1:
            if dynamic_scheduler and len(assigned_ranks) < world_size:
                # 动态子组模式
                task_data = {
                    'type': 'TASK',
                    'request': request, 
                    'task_id': task_id,
                    'assigned_ranks': assigned_ranks,
                    'dynamic_mode': True
                }
            else:
                # 全GPU模式
                task_data = {
                    'type': 'TASK',
                    'request': request, 
                    'task_id': task_id,
                    'dynamic_mode': False
                }
            
            broadcast_data = [task_data]
            dist.broadcast_object_list(broadcast_data, src=0)
            logger.info(f"Task {task_id}: Task signal broadcasted")
        
        def progress_callback(progress, stage="Processing"):
            if task_id in cancelled_tasks:
                raise Exception("Task was cancelled")
            status_dict[task_id]["progress"] = progress
            status_dict[task_id]["current_stage"] = stage
            
            elapsed = datetime.now() - status_dict[task_id]["start_time"]
            logger.info(f"📊 Task {task_id}: {stage} ({progress:.1f}%) - GPUs: {assigned_ranks} - Elapsed: {elapsed}")
            return progress
        
        # 🔥 执行任务 - 动态控制分布式环境（关键修改）
        current_rank = int(os.environ.get("RANK", 0))
        if current_rank in assigned_ranks:
            logger.info(f"Rank {current_rank}: Executing task {task_id}")
            
            # 🔥 保存原始环境变量
            original_world_size = os.environ.get("WORLD_SIZE")
            original_rank = os.environ.get("RANK")
            
            try:
                # 🔥 关键修改：临时调整环境变量让pipeline以为只有assigned_ranks个GPU
                if len(assigned_ranks) < world_size:
                    new_rank = assigned_ranks.index(current_rank)
                    os.environ["WORLD_SIZE"] = str(len(assigned_ranks))
                    os.environ["RANK"] = str(new_rank)
                    logger.info(f"Task {task_id}: Temp env - WORLD_SIZE={len(assigned_ranks)}, RANK={new_rank}")
                
                result = pipeline.generate_video(request, task_id, progress_callback=progress_callback)
                logger.info(f"Rank {current_rank}: Task {task_id} execution completed")
                
            finally:
                # 🔥 恢复环境变量
                if original_world_size:
                    os.environ["WORLD_SIZE"] = original_world_size
                if original_rank:
                    os.environ["RANK"] = original_rank
                    
        else:
            logger.info(f"Rank {current_rank}: Skipping task {task_id} (not assigned)")
            result = None
        
        # 🔥 任务完成处理
        if dynamic_scheduler:
            dynamic_scheduler.release_gpus(task_id)
        
        # 🔥 同步等待
        if world_size > 1 and dist.is_initialized():
            try:
                dist.barrier()  # 🔥 移除 timeout 参数
                logger.info(f"Task {task_id}: All ranks synchronized")
            except Exception as e:
                logger.warning(f"Task {task_id}: Barrier failed: {e}")
                
            # 🔥 发送完成信号
            try:
                completion_signal = [{"type": "TASK_COMPLETED", "task_id": task_id}]
                dist.broadcast_object_list(completion_signal, src=0)
            except Exception as e:
                logger.warning(f"Task {task_id}: Failed to send completion signal: {e}")
        
        # 🔥 最终检查是否取消
        if task_id in cancelled_tasks:
            logger.info(f"Task {task_id} was cancelled during processing")
            return
        
        # 🔥 标记成功
        total_time = datetime.now() - status_dict[task_id]["start_time"]
        status_dict[task_id]["status"] = TaskStatus.SUCCEED
        status_dict[task_id]["result_url"] = result
        status_dict[task_id]["updated_at"] = datetime.now().isoformat()
        status_dict[task_id]["elapsed_time"] = str(total_time)
        logger.info(f"✅ Task {task_id} completed successfully in {total_time}")
        
    except Exception as e:
        # 🔥 异常处理
        if dynamic_scheduler:
            dynamic_scheduler.release_gpus(task_id)
        
        if "start_time" in status_dict[task_id]:
            total_time = datetime.now() - status_dict[task_id]["start_time"]
            status_dict[task_id]["elapsed_time"] = str(total_time)
        
        # 🔥 发送失败信号
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        if world_size > 1 and dist.is_initialized():
            try:
                failure_signal = [{"type": "TASK_FAILED", "task_id": task_id, "error": str(e)}]
                dist.broadcast_object_list(failure_signal, src=0)
            except Exception:
                pass
        
        if task_id in cancelled_tasks:
            logger.info(f"❌ Task {task_id} cancelled during processing")
        else:
            status_dict[task_id]["status"] = TaskStatus.FAILED
            status_dict[task_id]["error"] = str(e)
            status_dict[task_id]["updated_at"] = datetime.now().isoformat()
            logger.error(f"💥 Task {task_id} failed: {e}")

def task_processing_loop():
    """任务处理循环"""
    logger.info("Task processing loop started")
    
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    while True:
        try:
            if task_queue:
                # 🔥 处理单个任务
                task_id, request = task_queue.pop(0)
                
                if task_id in cancelled_tasks:
                    logger.info(f"Task {task_id} was cancelled, skipping")
                    continue
                
                process_single_task(task_id, request)
            else:
                # 🔥 无任务时发送空闲信号
                if world_size > 1:
                    try:
                        import torch.distributed as dist
                        if dist.is_initialized():
                            idle_signal = [{"type": "IDLE"}]
                            dist.broadcast_object_list(idle_signal, src=0)
                            logger.debug("Idle signal sent")
                    except Exception as e:
                        logger.warning(f"Failed to send idle signal: {e}")
                
                time.sleep(1)
                
        except KeyboardInterrupt:
            logger.info("Task processing loop interrupted")
            
            # 🔥 发送关闭信号
            if world_size > 1:
                try:
                    import torch.distributed as dist
                    if dist.is_initialized():
                        shutdown_signal = [{"type": "SHUTDOWN"}]
                        dist.broadcast_object_list(shutdown_signal, src=0)
                        logger.info("Shutdown signal sent to all ranks")
                        
                        try:
                            dist.barrier()  # 🔥 移除 timeout 参数
                            logger.info("All ranks confirmed shutdown")
                        except Exception as e:
                            logger.warning(f"Shutdown barrier failed: {e}")
                except Exception as e:
                    logger.warning(f"Failed to send shutdown signal: {e}")
            break
        except Exception as e:
            logger.error(f"Task processing error: {e}")
            time.sleep(1)

def distributed_worker_loop():
    """支持动态调度的工作循环"""
    rank = int(os.environ.get("RANK", 0))
    
    logger.info(f"Rank {rank}: Worker ready for dynamic scheduling")
    
    if rank == 0:
        return
    
    import torch.distributed as dist
    
    while True:
        try:
            if dist.is_initialized():
                signal = [None]
                dist.broadcast_object_list(signal, src=0)
                
                if signal[0] is not None:
                    signal_data = signal[0]
                    
                    # 🔥 严格检查信号数据类型
                    if isinstance(signal_data, str):
                        logger.warning(f"Rank {rank}: Received unexpected string signal: {signal_data}")
                        continue
                    
                    if not isinstance(signal_data, dict):
                        logger.warning(f"Rank {rank}: Received non-dict signal: {type(signal_data)}")
                        continue
                    
                    signal_type = signal_data.get("type", "UNKNOWN")
                    
                    if signal_type == "SHUTDOWN":
                        logger.info(f"Rank {rank}: Received shutdown signal")
                        break
                    elif signal_type == "IDLE":
                        continue
                    elif signal_type in ["TASK_COMPLETED", "TASK_FAILED"]:
                        task_id = signal_data.get("task_id", "unknown")
                        logger.info(f"Rank {rank}: Task {task_id} {signal_type.lower()}")
                        continue
                    elif signal_type == "TASK":
                        # 🔥 处理任务分配
                        request = signal_data.get('request')
                        task_id = signal_data.get('task_id')
                        dynamic_mode = signal_data.get('dynamic_mode', False)
                        
                        if not request or not task_id:
                            logger.warning(f"Rank {rank}: Invalid task signal - missing request or task_id")
                            continue
                        
                        # 🔥 动态调度逻辑
                        if dynamic_mode:
                            assigned_ranks = signal_data.get('assigned_ranks', [])
                            
                            if rank in assigned_ranks:
                                logger.info(f"Rank {rank}: Participating in dynamic task {task_id}")
                                
                                # 🔥 保存原始环境变量
                                original_world_size = os.environ.get("WORLD_SIZE")
                                original_rank = os.environ.get("RANK")
                                
                                try:
                                    # 🔥 临时调整环境变量
                                    new_rank = assigned_ranks.index(rank)
                                    os.environ["WORLD_SIZE"] = str(len(assigned_ranks))
                                    os.environ["RANK"] = str(new_rank)
                                    
                                    pipeline.generate_video(request, task_id)
                                    logger.info(f"Rank {rank}: Dynamic task {task_id} completed")
                                    
                                finally:
                                    # 🔥 恢复环境变量
                                    if original_world_size:
                                        os.environ["WORLD_SIZE"] = original_world_size
                                    if original_rank:
                                        os.environ["RANK"] = original_rank
                                        
                            else:
                                logger.info(f"Rank {rank}: NOT assigned to task {task_id} - staying IDLE")
                                continue  # 🔥 重要：不参与的rank直接跳过
                        else:
                            # 传统全GPU模式
                            logger.info(f"Rank {rank}: Participating in full-GPU task {task_id}")
                            try:
                                pipeline.generate_video(request, task_id)
                                logger.info(f"Rank {rank}: Full-GPU task {task_id} completed")
                            except Exception as e:
                                logger.error(f"Rank {rank}: Full-GPU task {task_id} failed: {e}")
                        
                        # 同步等待
                        try:
                            dist.barrier()  # 🔥 移除 timeout 参数
                        except Exception as e:
                            logger.warning(f"Rank {rank}: Barrier failed: {e}")
                    else:
                        logger.warning(f"Rank {rank}: Unknown signal type: {signal_type}")
                        continue
                
        except Exception as e:
            logger.warning(f"Rank {rank}: Worker error: {e}")
            time.sleep(1)
    
    logger.info(f"Rank {rank}: Dynamic worker exited")

def main():
    global pipeline, load_monitor, dynamic_scheduler
    rank, local_rank, world_size = init_distributed()
    
    pipeline = create_pipeline()
    logger.info(f"Rank {rank}: Pipeline created successfully")
    
    # 🔥 初始化动态调度器
    if rank == 0 and world_size > 1:
        enable_dynamic = os.environ.get("ENABLE_DYNAMIC_SCHEDULING", "true").lower() == "true"
        if enable_dynamic:
            from utils.device_detector import detect_device
            device_type, _, _ = detect_device()
            dynamic_scheduler = DynamicGPUScheduler(world_size=world_size, device_type=device_type)
            dynamic_scheduler.start()
            logger.info("🎯 Dynamic GPU scheduling enabled")
        else:
            logger.info("Dynamic GPU scheduling disabled")
    
    if rank == 0:
        app = create_app()
        
        # 🔥 启动任务处理线程
        task_thread = threading.Thread(target=task_processing_loop, daemon=True)
        task_thread.start()
        
        try:
            uvicorn.run(app, host="0.0.0.0", port=8088, log_level="info")
        except KeyboardInterrupt:
            logger.info("Shutting down gracefully...")
        finally:
            # 🔥 关闭动态调度器
            if dynamic_scheduler:
                dynamic_scheduler.stop()
            logger.info("FastAPI server stopped")
    else:
        # 🔥 其他rank运行工作循环
        try:
            distributed_worker_loop()
        except KeyboardInterrupt:
            logger.info(f"Rank {rank}: Received interrupt")
        finally:
            logger.info(f"Rank {rank}: Worker stopped")

if __name__ == "__main__":
    main()