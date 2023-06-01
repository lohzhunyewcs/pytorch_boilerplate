import torch
from torcheval.metrics.aggregation import auc
from .base import BaseMetrics

class AUC(BaseMetrics):
    __name__ = "AUC"
    def __init__(self) -> None:
        super().__init__()
        self.metric = auc.AUC()

    def compute(self):
        return self.metric.compute()[0].item()