from typing import Any, Union
from .base import BaseMetrics
from torcheval.metrics import BinaryRecall, MulticlassRecall


class Recall(BaseMetrics):
    task_choices = ['binary', 'multiclass']
    task_to_metric: dict[str, Union[BinaryRecall, MulticlassRecall]] = {
        "binary": BinaryRecall,
        "multiclass": MulticlassRecall,
    }
    __name__ = "Precision"
    def __init__(
            self, num_class: int,
            task: str=None
    ) -> None:
        super().__init__()
        if task is None:
            if num_class > 1:
                task = "multiclass"
            else:
                task = "binary"
        assert task in Recall.task_choices
        self.metric = Recall.task_to_metric[task]()