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

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# 全局变量
task_queue = []
status_dict = {}
cancelled_tasks = set()
pipeline = None

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
            torch_npu.npu.set_device(local_rank)  # 🔥 使用torch_npu.npu.set_device
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
                    timeout=timedelta(seconds=300),  # 🔥 添加超时
                    init_method='env://'  # 🔥 明确指定
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

    # 🔥 新增：取消任务
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
        
        # 从队列中移除
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

        # 🔥 新增：动态消息
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
            message=message,  # 🔥 修改：使用动态消息
            created_at=s.get("created_at", ""),
            updated_at=s.get("updated_at", ""),
            results=s.get("results"),
            reason=s.get("reason"),
            elapsed_time=s.get("elapsed_time"),
            current_stage=s.get("current_stage"),  # 🔥 新增
        )

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

    return app

def process_tasks():
    """处理任务队列"""
    while task_queue:
        task_id, request = task_queue.pop(0)
        
        if task_id in cancelled_tasks:
            logger.info(f"Task {task_id} was cancelled, skipping")
            continue
            
        try:
            logger.info(f"🚀 Processing task: {task_id}")
            status_dict[task_id]["status"] = TaskStatus.RUNNING
            status_dict[task_id]["updated_at"] = datetime.now().isoformat()
            status_dict[task_id]["start_time"] = datetime.now()
            
            # 🔥 关键修复：广播任务给所有rank
            world_size = int(os.environ.get("WORLD_SIZE", 1))
            if world_size > 1:
                import torch.distributed as dist
                if dist.is_initialized():
                    task_data = [{'request': request, 'task_id': task_id}]
                    dist.broadcast_object_list(task_data, src=0)
                    logger.info(f"Task {task_id} broadcasted to all {world_size} ranks")
            
            def progress_callback(progress, stage="Processing"):
                if task_id in cancelled_tasks:
                    raise Exception("Task was cancelled")
                status_dict[task_id]["progress"] = progress
                status_dict[task_id]["current_stage"] = stage
                
                elapsed = datetime.now() - status_dict[task_id]["start_time"]
                logger.info(f"📊 Task {task_id}: {stage} ({progress:.1f}%) - Elapsed: {elapsed}")
                return progress
            
            # 🔥 现在所有rank都会参与这个调用
            result = pipeline.generate_video(request, task_id, progress_callback=progress_callback)
            
            # 🔥 添加：任务完成后等待所有rank同步
            if world_size > 1:
                import torch.distributed as dist
                if dist.is_initialized():
                    try:
                        dist.barrier(timeout=timedelta(seconds=30))
                        logger.info(f"Task {task_id}: All ranks synchronized after completion")
                    except Exception as e:
                        logger.warning(f"Task {task_id}: Post-completion barrier failed: {e}")
             
            # 最后检查一次是否被取消
            if task_id in cancelled_tasks:
                logger.info(f"Task {task_id} was cancelled during processing")
                continue
            
            # 🔥 新增：计算总耗时
            total_time = datetime.now() - status_dict[task_id]["start_time"]
            
            status_dict[task_id]["status"] = TaskStatus.SUCCEED
            status_dict[task_id]["result_url"] = result
            status_dict[task_id]["updated_at"] = datetime.now().isoformat()
            status_dict[task_id]["elapsed_time"] = str(total_time)  # 🔥 新增
            logger.info(f"✅ Task {task_id} completed successfully in {total_time}")
            
        except Exception as e:
            # 🔥 新增：异常时也记录耗时
            if "start_time" in status_dict[task_id]:
                total_time = datetime.now() - status_dict[task_id]["start_time"]
                status_dict[task_id]["elapsed_time"] = str(total_time)
            
            if task_id in cancelled_tasks:
                logger.info(f"❌ Task {task_id} cancelled during processing")
            else:
                status_dict[task_id]["status"] = TaskStatus.FAILED
                status_dict[task_id]["error"] = str(e)
                status_dict[task_id]["updated_at"] = datetime.now().isoformat()
                logger.error(f"💥 Task {task_id} failed: {e}")

def main():
    global pipeline
    rank, local_rank, world_size = init_distributed()  # 🔥 修复：调用init_distributed
    
    logger.info(f"Rank {rank}: Starting I2V API service (world_size={world_size})")
    
    # 🔥 所有rank都创建pipeline（对齐本地generate.py）
    pipeline = create_pipeline()
    logger.info(f"Rank {rank}: Pipeline created successfully")
    
    if rank == 0:
        # 🔥 修复：创建app对象
        app = create_app()
        
        # rank 0运行FastAPI服务 + 工作循环
        logger.info("Rank 0: Starting FastAPI server...")
        import threading
        
        # 🔥 在后台线程启动任务处理循环
        task_thread = threading.Thread(target=task_processing_loop, daemon=True)
        task_thread.start()
        
        # 🔥 在后台线程启动工作循环
        worker_thread = threading.Thread(target=distributed_worker_loop, daemon=True)
        worker_thread.start()
        
        uvicorn.run(app, host="0.0.0.0", port=8088, log_level="info")
    else:
        # 🔥 其他rank运行工作循环
        logger.info(f"Rank {rank}: Starting distributed worker...")
        distributed_worker_loop()

def task_processing_loop():
    """任务处理循环（只在rank 0运行）"""
    logger.info("Task processing loop started")
    while True:
        try:
            process_tasks()  # 处理任务队列
            time.sleep(0.1)  # 防止CPU占用过高
        except KeyboardInterrupt:
            logger.info("Task processing loop interrupted")
            break
        except Exception as e:
            logger.error(f"Task processing error: {e}")
            time.sleep(1)

def distributed_worker_loop():
    """分布式工作循环 - 所有rank都参与"""
    rank = int(os.environ.get("RANK", 0))
    
    logger.info(f"Rank {rank}: Worker ready for distributed tasks")
    
    if rank == 0:
        logger.info("Rank 0: Main worker loop handled by task processing")
        return
    else:
        logger.info(f"Rank {rank}: Waiting for distributed tasks...")
        
        import torch.distributed as dist
        while True:
            try:
                if dist.is_initialized():
                    # 🔥 关键：等待rank 0的任务广播
                    task_data = [None]
                    dist.broadcast_object_list(task_data, src=0)
                    
                    if task_data[0] is not None:
                        if task_data[0] == "SHUTDOWN":  # 🔥 添加：关闭信号
                            logger.info(f"Rank {rank}: Received shutdown signal")
                            break
                            
                        request, task_id = task_data[0]['request'], task_data[0]['task_id']
                        logger.info(f"Rank {rank}: Received task {task_id}")
                        
                        try:
                            # 🔥 关键：其他rank也调用generate_video
                            pipeline.generate_video(request, task_id)
                            logger.info(f"Rank {rank}: Task {task_id} completed successfully")
                            
                        except Exception as e:
                            logger.error(f"Rank {rank}: Task {task_id} failed: {e}")
                            
                        # 🔥 添加：任务完成后的同步
                        try:
                            dist.barrier(timeout=timedelta(seconds=30))
                            logger.info(f"Rank {rank}: Post-task barrier completed")
                        except Exception as e:
                            logger.warning(f"Rank {rank}: Post-task barrier failed: {e}")
                            
                else:
                    time.sleep(0.1)
                    
            except KeyboardInterrupt:
                logger.info(f"Rank {rank}: Received interrupt")
                break
            except Exception as e:
                logger.error(f"Rank {rank}: Worker error: {e}")
                time.sleep(0.1)
                
        # 🔥 添加：退出时的清理
        logger.info(f"Rank {rank}: Worker loop exiting, cleaning up...")
        try:
            if dist.is_initialized():
                dist.barrier(timeout=timedelta(seconds=10))
                logger.info(f"Rank {rank}: Final cleanup barrier completed")
        except Exception as e:
            logger.warning(f"Rank {rank}: Final cleanup failed: {e}")

if __name__ == "__main__":
    main()