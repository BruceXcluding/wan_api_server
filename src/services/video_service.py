import time
import uuid
import logging
from typing import Dict, Any, Optional
from multiprocessing.managers import DictProxy
from pathlib import Path
from ..schemas.video import VideoSubmitRequest, TaskStatus, VideoResults

logger = logging.getLogger(__name__)

class VideoService:
    """视频生成服务（适合多进程队列架构）"""

    def __init__(self, queue, status_dict: DictProxy):
        self.queue = queue
        self.status: Dict[str, Dict[str, Any]] = status_dict
        self.task_timeout = 3600  # 1小时超时

    def submit_video_task(self, request: VideoSubmitRequest) -> str:
        """提交视频生成任务"""
        task_id = uuid.uuid4().hex
        self.status[task_id] = {
            "id": task_id,
            "status": TaskStatus.IN_QUEUE,
            "request": request.dict(),
            "created_at": time.time(),
            "updated_at": time.time(),
            "progress": 0.0,
            "results": None,
            "reason": None
        }
        self.queue.put((task_id, request.dict()))
        logger.info(f"Task {task_id} submitted to queue")
        return task_id

    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """获取任务状态"""
        return self.status.get(task_id)

    def cancel_task(self, task_id: str) -> bool:
        """取消任务"""
        task = self.status.get(task_id)
        if not task or task["status"] in [TaskStatus.SUCCEED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
            return False
        task["status"] = TaskStatus.CANCELLED
        task["updated_at"] = time.time()
        task["reason"] = "User cancelled"
        logger.info(f"Task {task_id} cancelled by user")
        return True

    def cleanup_expired_tasks(self) -> int:
        """清理过期任务"""
        current_time = time.time()
        expired_tasks = []
        for task_id, task in list(self.status.items()):
            if (task["status"] in [TaskStatus.SUCCEED, TaskStatus.FAILED, TaskStatus.CANCELLED] and
                current_time - task["updated_at"] > 86400):
                expired_tasks.append(task_id)
            elif (task["status"] == TaskStatus.IN_PROGRESS and
                  current_time - task["updated_at"] > self.task_timeout):
                task["status"] = TaskStatus.FAILED
                task["reason"] = "Task timeout"
                task["updated_at"] = current_time
                expired_tasks.append(task_id)
        for task_id in expired_tasks:
            del self.status[task_id]
        if expired_tasks:
            logger.info(f"Cleaned up {len(expired_tasks)} expired tasks")
        return len(expired_tasks)

    def get_service_stats(self) -> Dict[str, Any]:
        """获取服务统计信息"""
        stats = {
            "total_tasks": len(self.status),
            "status_breakdown": {}
        }
        for status in TaskStatus:
            stats["status_breakdown"][status.value] = len([
                t for t in self.status.values() if t["status"] == status
            ])
        return stats