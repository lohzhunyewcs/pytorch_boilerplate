from abc import ABC, abstractmethod
import torch

class BaseImageClassification(ABC):

    def __call__(self, image: torch.Tensor) -> None:
        output = self.model()
