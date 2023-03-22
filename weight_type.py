from enum import Enum

class WeightType(Enum):
    RandomInit = 1
    Pretrained = 2
    OwnWeights = 3