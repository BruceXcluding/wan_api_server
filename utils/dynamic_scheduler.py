import torch
import torch.distributed as dist
import logging
import time
import threading
from typing import List, Dict, Optional, Set
from dataclasses import dataclass
from .load_monitor import LoadMonitor

logger = logging.getLogger(__name__)

@dataclass
class TaskResourcePlan:
    """任务资源计划"""
    task_id: str
    required_gpus: int
    assigned_ranks: List[int]
    complexity_level: str  # "simple", "medium", "complex"
    estimated_time: float

class DynamicGPUScheduler:
    """动态GPU调度器"""
    
    def __init__(self, world_size: int, device_type: str = "npu"):
        self.world_size = world_size
        self.device_type = device_type
        self.load_monitor = LoadMonitor(world_size, device_type)
        
        # 资源管理
        self.available_ranks: Set[int] = set(range(world_size))
        self.active_tasks: Dict[str, TaskResourcePlan] = {}
        self.task_queue: List[TaskResourcePlan] = []
        self.task_lock = threading.Lock()
        self.gpu_status = {i: "available" for i in range(world_size)}
        self.pending_tasks = {}
        self.running = False
        
        logger.info(f"🎯 Dynamic GPU Scheduler initialized: {world_size} {device_type.upper()} devices")

    def start(self):
        """启动调度器"""
        self.load_monitor.start()
        logger.info("Dynamic GPU scheduler started")
    
    def stop(self):
        """停止调度器"""
        self.load_monitor.stop()
        logger.info("Dynamic GPU scheduler stopped")
    

    def estimate_task_requirements(self, request) -> TaskResourcePlan:
        """估算任务资源需求"""

        # 🔥 获取分配策略
        import os
        strategy = os.environ.get("GPU_ALLOCATION_STRATEGY", "adaptive")

        # 解析任务参数
        image_size = getattr(request, 'image_size', '512*512')
        frame_num = getattr(request, 'frame_num', 81)
        sample_steps = getattr(request, 'sample_steps', 30)

        # 解析分辨率
        if '*' in image_size:
            h, w = map(int, image_size.split('*'))
        elif 'x' in image_size:
            w, h = map(int, image_size.split('x'))
        else:
            h, w = 512, 512

        # 计算复杂度分数
        total_pixels = h * w * frame_num
        complexity_score = total_pixels * sample_steps / 1000000

        # 🔥 根据集群规模动态计算GPU分配
        if strategy == "conservative":
            # 保守策略：更少GPU，更多并发
            if complexity_score < 800:
                required_gpus = max(1, self.world_size // 8)  # 1/8集群大小，最少1张
                complexity_level = "simple"
                estimated_time = 45
            elif complexity_score < 2000:
                required_gpus = max(1, self.world_size // 4)  # 1/4集群大小
                complexity_level = "medium"
                estimated_time = 80
            else:
                required_gpus = max(2, self.world_size // 2)  # 1/2集群大小，最少2张
                complexity_level = "complex"
                estimated_time = 150

        elif strategy == "aggressive":
            # 激进策略：更多GPU，更快完成
            if complexity_score < 1500:
                required_gpus = max(2, self.world_size // 2)  # 1/2集群大小，最少2张
                complexity_level = "simple"
                estimated_time = 20
            else:
                required_gpus = self.world_size  # 全部GPU
                complexity_level = "complex"
                estimated_time = 90

        else:  # adaptive（默认）
            # 自适应策略：平衡GPU使用和并发
            if complexity_score < 800:
                required_gpus = max(1, self.world_size // 4)  # 1/4集群大小，最少1张
                complexity_level = "simple"
                estimated_time = 30
            elif complexity_score < 2500:
                required_gpus = max(2, self.world_size // 2)  # 1/2集群大小，最少2张
                complexity_level = "medium"
                estimated_time = 60
            else:
                required_gpus = self.world_size  # 全部GPU
                complexity_level = "complex"
                estimated_time = 120

        # 🔥 确保不超过集群大小
        required_gpus = min(required_gpus, self.world_size)

        logger.info(f"Strategy: {strategy}, Cluster: {self.world_size} GPUs, Complexity: {complexity_score:.1f}, Allocated: {required_gpus} GPUs")

        return TaskResourcePlan(
            task_id="",
            required_gpus=required_gpus,
            assigned_ranks=[],
            complexity_level=complexity_level,
            estimated_time=estimated_time
        )

    def try_allocate_gpus(self, plan: TaskResourcePlan) -> Optional[List[int]]:
        """尝试分配GPU"""
        
        if len(self.available_ranks) < plan.required_gpus:
            logger.info(f"Insufficient GPUs: need {plan.required_gpus}, available {len(self.available_ranks)}")
            return None
        
        # 🔥 选择负载最低的GPU
        load_status = self.load_monitor.get_load_status()
        available_in_status = [r for r in load_status["available"] if r in self.available_ranks]
        busy_in_status = [r for r in load_status["busy"] if r in self.available_ranks]
        
        # 优先使用空闲GPU
        candidate_ranks = available_in_status + busy_in_status
        
        if len(candidate_ranks) >= plan.required_gpus:
            selected_ranks = candidate_ranks[:plan.required_gpus]
            
            # 🔥 分配GPU
            for rank in selected_ranks:
                self.available_ranks.remove(rank)
            
            plan.assigned_ranks = selected_ranks
            self.active_tasks[plan.task_id] = plan
            
            logger.info(f"Task {plan.task_id}: Allocated {plan.required_gpus} GPUs: {selected_ranks}")
            return selected_ranks
        
        return None

    def find_available_gpus(self, required_count: int) -> Optional[List[int]]:
        """查找可用的GPU"""
        available_list = [rank for rank, status in self.gpu_status.items() if status == "available"]

        if len(available_list) >= required_count:
            return available_list[:required_count]
        else:
            return None


    def release_gpus(self, task_id: str):
        """释放GPU资源"""
        if task_id in self.active_tasks:
            plan = self.active_tasks.pop(task_id)

            # 🔥 更新两种状态记录
            for rank in plan.assigned_ranks:
                self.gpu_status[rank] = "available"
                self.available_ranks.add(rank)

            logger.info(f"Task {task_id}: Released GPUs {plan.assigned_ranks}")

            # 🔥 尝试启动队列中的任务
            self._try_start_queued_tasks()

            # 🔥 清理等待队列中的已完成任务
            if task_id in self.pending_tasks:
                del self.pending_tasks[task_id]

    def _try_start_queued_tasks(self):
        """尝试启动队列中的任务"""
        scheduled_tasks = []

        for plan in self.task_queue[:]:  # 复制列表避免修改时出错
            available_ranks = self.find_available_gpus(plan.required_gpus)

            if available_ranks is not None:
                # 分配GPU
                plan.assigned_ranks = available_ranks
                self.active_tasks[plan.task_id] = plan

                # 更新状态
                for rank in available_ranks:
                    self.gpu_status[rank] = "busy"
                    self.available_ranks.discard(rank)

                scheduled_tasks.append(plan)
                logger.info(f"Started queued task {plan.task_id} with GPUs {available_ranks}")

        # 移除已调度的任务
        for plan in scheduled_tasks:
            self.task_queue.remove(plan)
            if plan.task_id in self.pending_tasks:
                del self.pending_tasks[plan.task_id]

    def schedule_task(self, task_id: str, request) -> Optional[List[int]]:
        """调度任务到最佳GPU组合"""
        with self.task_lock:
            # 估算资源需求
            plan = self.estimate_task_requirements(request)
            plan.task_id = task_id

            logger.info(f"🎯 Task {task_id}: Requires {plan.required_gpus} GPUs ({plan.complexity_level})")

            # 查找可用GPU
            available_ranks = self.find_available_gpus(plan.required_gpus)

            if available_ranks is None:
                # 没有足够的GPU，加入等待队列
                self.pending_tasks[task_id] = plan
                self.task_queue.append(plan)  # 🔥 同时添加到两个队列保持兼容
                logger.warning(f"⏳ Task {task_id}: Insufficient GPUs, queued for later (needs {plan.required_gpus})")
                return None

            # 分配GPU
            plan.assigned_ranks = available_ranks
            self.active_tasks[task_id] = plan

            # 🔥 更新两种状态记录
            for rank in available_ranks:
                self.gpu_status[rank] = "busy"
                self.available_ranks.discard(rank)  # 从可用集合中移除

            logger.info(f"✅ Task {task_id}: Allocated {len(available_ranks)} GPUs: {available_ranks}")
            return available_ranks

    def get_scheduler_status(self) -> Dict:
        """获取调度器状态"""
        with self.task_lock:
            available_gpus = [rank for rank, status in self.gpu_status.items() if status == "available"]
            busy_gpus = [rank for rank, status in self.gpu_status.items() if status == "busy"]
            
            return {
                "dynamic_scheduling": True,
                "world_size": self.world_size,
                "device_type": self.device_type,
                "total_gpus": self.world_size,
                "available_gpus": len(available_gpus),
                "busy_gpus": len(busy_gpus),
                "active_tasks": len(self.active_tasks),
                "queued_tasks": len(self.task_queue),
                "pending_tasks": len(self.pending_tasks),
                "available_ranks": available_gpus,
                "busy_ranks": busy_gpus,
                "running": getattr(self, 'running', False),
                "gpu_status": dict(self.gpu_status),
                "active_task_details": {
                    task_id: {
                        "assigned_ranks": plan.assigned_ranks,
                        "complexity": plan.complexity_level,
                        "required_gpus": plan.required_gpus,
                        "estimated_time": plan.estimated_time
                    }
                    for task_id, plan in self.active_tasks.items()
                }
            }