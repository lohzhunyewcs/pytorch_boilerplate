from typing import Any, Union
from torcheval.metrics import BinaryAccuracy, MulticlassAccuracy, MultilabelAccuracy
from .base import BaseMetrics

# class Precision:
#     task_choices = ['binary', 'multiclass', 'multilabel']
#     __name__ = "Precision"
#     def __init__(
#             self, num_class: int,
#             threshold: float=0.5,
#             average: str="micro",
#             ignore_index: int=None,
#             task: str=None
#     ) -> None:
#         if task is None:
#             if num_class > 1:
#                 task = "multiclass"
#             else:
#                 task = "binary"
#         assert task in Precision.task_choices
#         self.precision = tm.Precision(
#             task=task, threshold=threshold, num_classes=num_class,
#             average=average,ignore_index=ignore_index,
#         )

#     def __call__(self, pred_classes, target_classes) -> Any:
#         return self.precision(pred_classes, target_classes)

class Precision(BaseMetrics):
    task_choices = ['binary', 'multiclass', 'multilabel']
    task_to_metric: dict[str, Union[BinaryAccuracy, MulticlassAccuracy, MultilabelAccuracy]] = {
        "binary": BinaryAccuracy,
        "multiclass": MulticlassAccuracy,
        "multilabel": MultilabelAccuracy
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
        assert task in Precision.task_choices
        self.metric = Precision.task_to_metric[task]()

    