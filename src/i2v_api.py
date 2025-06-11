import os
import sys
import torch
import torch.distributed as dist

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

import logging
import uuid
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
    
    if world_size > 1:
        # 设置设备
        if npu_available:
            import torch_npu
            torch.npu.set_device(local_rank)
            logger.info(f"Rank {rank}: Using NPU device {local_rank}")
        elif torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            logger.info(f"Rank {rank}: Using CUDA device {local_rank}")
        
        # 初始化分布式进程组
        backend = "hccl" if npu_available else "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
        logger.info(f"Rank {rank}: Distributed initialized with {backend}")
    
    return rank, local_rank, world_size

def create_app():
    """创建FastAPI应用"""
    app = FastAPI(title="Multi-GPU I2V Generation API")
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
    
    os.makedirs("generated_videos", exist_ok=True)
    app.mount("/videos", StaticFiles(directory="generated_videos"), name="videos")

    @app.post("/submit", response_model=VideoSubmitResponse)
    async def submit(request: VideoSubmitRequest):
        task_id = f"req_{uuid.uuid4().hex[:16]}"
        status_dict[task_id] = {"status": TaskStatus.PENDING, "result_url": "", "error": ""}
        task_queue.append((task_id, request))
        logger.info(f"Task submitted: {task_id}")
        return VideoSubmitResponse(requestId=task_id, status=TaskStatus.PENDING, message="任务已提交", estimated_time=30)

    @app.post("/batch_submit", response_model=List[VideoSubmitResponse])
    async def batch_submit(requests: List[VideoSubmitRequest]):
        responses = []
        for req in requests:
            task_id = f"req_{uuid.uuid4().hex[:16]}"
            status_dict[task_id] = {"status": TaskStatus.PENDING, "result_url": "", "error": ""}
            task_queue.append((task_id, req))
            responses.append(VideoSubmitResponse(requestId=task_id, status=TaskStatus.PENDING, message="任务已提交", estimated_time=30))
        return responses

    @app.get("/status/{task_id}", response_model=VideoStatusResponse)
    async def status(task_id: str):
        s = status_dict.get(task_id)
        if not s:
            raise HTTPException(status_code=404, detail="Task not found")
        return VideoStatusResponse(
            requestId=task_id,
            status=s.get("status", TaskStatus.PENDING),
            progress=s.get("progress", 0),
            message=s.get("error", "") if s.get("status") == TaskStatus.FAILED else "任务处理中",
            created_at=s.get("created_at", ""),
            updated_at=s.get("updated_at", ""),
            results=s.get("results"),
            reason=s.get("reason"),
            elapsed_time=s.get("elapsed_time"),
        )

    @app.get("/health")
    async def health():
        dtype, dcount, backend = detect_device()
        return {"device_type": dtype, "device_count": dcount, "backend": backend, "queue_size": len(task_queue), "tasks": len(status_dict)}

    return app

def process_tasks():
    """处理任务队列"""
    while task_queue:
        task_id, request = task_queue.pop(0)
        try:
            logger.info(f"Processing task: {task_id}")
            status_dict[task_id]["status"] = TaskStatus.RUNNING
            
            result = pipeline.generate_video(request, task_id)
            
            status_dict[task_id]["status"] = TaskStatus.SUCCEED
            status_dict[task_id]["result_url"] = result
            logger.info(f"Task {task_id} completed successfully")
            
        except Exception as e:
            status_dict[task_id]["status"] = TaskStatus.FAILED
            status_dict[task_id]["error"] = str(e)
            logger.error(f"Task {task_id} failed: {e}")

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
        
        # 启动后台任务处理
        import threading
        def task_worker():
            while True:
                process_tasks()
                import time
                time.sleep(0.1)
        
        threading.Thread(target=task_worker, daemon=True).start()
        
        # 启动API服务器
        uvicorn.run(app, host="0.0.0.0", port=8088)
    else:
        # 其他rank等待并参与分布式计算
        while True:
            import time
            time.sleep(1)

if __name__ == "__main__":
    main()