from abc import ABC, abstractmethod
import torch
from typing import List, Union
from task_type import VisionTaskType, TextTaskType

from torcheval.metrics import Metric
class BaseMetrics(ABC):
    metric: Metric

    def __init__(self) -> None:
        super().__init__()

    def add(self, preds: torch.Tensor, gts: torch.Tensor):
        self.metric.update(preds.long(), gts.long())

    def compute(self):
        return self.metric.compute().item()
    
    def reset(self):
        return self.metric.reset()