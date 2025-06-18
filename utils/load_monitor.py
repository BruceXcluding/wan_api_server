import asyncio
import time
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass
import threading

try:
    import torch
    import torch_npu
    NPU_AVAILABLE = True
except ImportError:
    NPU_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class DeviceLoad:
    """设备负载信息"""
    rank: int
    memory_used_gb: float
    memory_total_gb: float
    utilization: float  # 0-1
    temperature: Optional[float] = None
    last_update: float = 0.0
    
    @property
    def load_score(self) -> float:
        """负载评分 (0-1，越低越好)"""
        return self.utilization

class LoadMonitor:
    """轻量级负载监控器"""
    
    def __init__(self, world_size: int, device_type: str = "npu"):
        self.world_size = world_size
        self.device_type = device_type
        self.device_loads: Dict[int, DeviceLoad] = {}
        self.monitoring = False
        self._monitor_thread = None
        self._lock = threading.Lock()
        
    def start(self, interval: float = 10.0):
        """启动监控"""
        if self.monitoring:
            return
            
        self.monitoring = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop, 
            args=(interval,),
            daemon=True
        )
        self._monitor_thread.start()
        logger.info(f"Load monitor started for {self.world_size} {self.device_type} devices")
    
    def stop(self):
        """停止监控"""
        self.monitoring = False
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=5)
    
    def _monitor_loop(self, interval: float):
        """监控循环"""
        while self.monitoring:
            try:
                self._collect_metrics()
                time.sleep(interval)
            except Exception as e:
                logger.warning(f"Load monitoring error: {e}")
                time.sleep(interval)
    
    def _collect_metrics(self):
        """收集设备指标"""
        for rank in range(self.world_size):
            try:
                load = self._get_device_load(rank)
                with self._lock:
                    self.device_loads[rank] = load
            except Exception as e:
                logger.warning(f"Failed to get load for rank {rank}: {e}")
    
    def _get_device_load(self, rank: int) -> DeviceLoad:
        """获取单个设备负载"""
        try:
            if self.device_type == "npu" and NPU_AVAILABLE:
                memory_used = torch_npu.npu.memory_allocated(rank) / (1024**3)
                memory_total = 32.0  # NPU通常32GB
                utilization = memory_used / memory_total
                
            elif self.device_type == "cuda" and torch.cuda.is_available():
                memory_used = torch.cuda.memory_allocated(rank) / (1024**3)
                props = torch.cuda.get_device_properties(rank)
                memory_total = props.total_memory / (1024**3)
                utilization = memory_used / memory_total
                
            else:
                # 模拟数据
                memory_used = 8.0
                memory_total = 32.0
                utilization = 0.25
            
            return DeviceLoad(
                rank=rank,
                memory_used_gb=memory_used,
                memory_total_gb=memory_total,
                utilization=utilization,
                last_update=time.time()
            )
            
        except Exception as e:
            logger.warning(f"Error getting device {rank} load: {e}")
            return DeviceLoad(
                rank=rank,
                memory_used_gb=0.0,
                memory_total_gb=32.0,
                utilization=0.0,
                last_update=time.time()
            )
    
    def get_current_loads(self) -> Dict[int, DeviceLoad]:
        """获取当前负载"""
        with self._lock:
            return self.device_loads.copy()
    
    def get_best_ranks(self, count: int = None) -> List[int]:
        """获取负载最低的ranks"""
        if count is None:
            count = self.world_size
        
        with self._lock:
            if not self.device_loads:
                return list(range(min(count, self.world_size)))
            
            # 按负载评分排序
            sorted_ranks = sorted(
                self.device_loads.keys(),
                key=lambda r: self.device_loads[r].load_score
            )
            
            return sorted_ranks[:count]
    
    def get_load_status(self) -> Dict[str, List[int]]:
        """获取负载状态分类"""
        available = []  # < 0.3
        busy = []       # 0.3-0.7
        overloaded = [] # > 0.7
        
        with self._lock:
            for rank, load in self.device_loads.items():
                if load.load_score < 0.3:
                    available.append(rank)
                elif load.load_score < 0.7:
                    busy.append(rank)
                else:
                    overloaded.append(rank)
        
        return {
            "available": available,
            "busy": busy,
            "overloaded": overloaded
        }