from abc import ABC, abstractmethod
import torch
from typing import List, Union
from task_type import VisionTaskType, TextTaskType

class BaseMetrics(ABC):
    def __init__(self, supported_tasks: Union[List[VisionTaskType], List[TextTaskType]]) -> None:
        super().__init__()
        self.supported_tasks = supported_tasks
