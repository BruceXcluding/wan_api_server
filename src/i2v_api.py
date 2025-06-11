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
        
        # 检查是否被取消
        if task_id in cancelled_tasks:
            logger.info(f"Task {task_id} was cancelled, skipping")
            continue
            
        try:
            logger.info(f"🚀 Processing task: {task_id}")
            status_dict[task_id]["status"] = TaskStatus.RUNNING
            status_dict[task_id]["updated_at"] = datetime.now().isoformat()
            status_dict[task_id]["start_time"] = datetime.now()  # 🔥 新增：记录开始时间
            
            # 🔥 修改：增强的进度回调
            def progress_callback(progress, stage="Processing"):
                if task_id in cancelled_tasks:
                    raise Exception("Task was cancelled")
                status_dict[task_id]["progress"] = progress
                status_dict[task_id]["current_stage"] = stage  # 🔥 新增：当前阶段
                
                # 🔥 新增：详细日志
                elapsed = datetime.now() - status_dict[task_id]["start_time"]
                logger.info(f"📊 Task {task_id}: {stage} ({progress:.1f}%) - Elapsed: {elapsed}")
                return progress
            
            result = pipeline.generate_video(request, task_id, progress_callback=progress_callback)
            
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
    
    # 初始化分布式
    rank, local_rank, world_size = init_distributed()
    
    # 创建pipeline
    pipeline = create_pipeline()
    logger.info(f"Rank {rank}: Pipeline created successfully")
    
    # 只有rank 0运行API服务器
    if rank == 0:
        app = create_app()
        
        # 启动后台任务处理线程
        def task_worker():
            while True:
                process_tasks()
                time.sleep(0.1)
        
        threading.Thread(target=task_worker, daemon=True).start()
        
        # 启动API服务器
        logger.info("Starting API server on rank 0...")
        uvicorn.run(app, host="0.0.0.0", port=8088)
    else:
        # 其他rank等待并参与分布式计算
        logger.info(f"Rank {rank}: Waiting for distributed tasks...")
        while True:
            time.sleep(1)

if __name__ == "__main__":
    main()