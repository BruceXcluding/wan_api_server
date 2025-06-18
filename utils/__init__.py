"""
外层工具包 (utils/)
项目级通用工具函数
"""

from .device_detector import detect_device
from .load_monitor import LoadMonitor, DeviceLoad  
from .dynamic_scheduler import DynamicGPUScheduler, TaskResourcePlan

__all__ = [
    'detect_device',
    'LoadMonitor', 
    'DeviceLoad',
    'DynamicGPUScheduler',
    'TaskResourcePlan'
]

__version__ = "1.0.0"