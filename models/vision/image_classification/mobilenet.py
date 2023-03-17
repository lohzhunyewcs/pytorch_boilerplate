import torchvision
import torch
import os
from enum import Enum
from weight_type import WeightType


class MobileNetModelsType(Enum):
    # mobilenet_v1 = 1
    mobilenet_v2 = 2
    mobilenet_v3_small = 3
    mobilenet_v3_large = 3

def load_model(model_type: MobileNetModelsType, weight_type: WeightType, weights_path: str = None, num_class: int=None)\
    -> torch.nn.Module:
    # if model_type == MobileNetModels.mobilenet_v1:
    #     model = torchvision.models.mobilenet.
    if model_type == MobileNetModelsType.mobilenet_v2:
        model_func = torchvision.models.mobilenet_v2
    elif model_type == MobileNetModelsType.mobilenet_v3_small:
        model_func = torchvision.models.mobilenet_v3_small
    elif model_type == MobileNetModelsType.mobilenet_v3_large:
        model_func = torchvision.models.mobilenet_v3_large
    else:
        raise Exception
    if weight_type in (WeightType.RandomInit, WeightType.OwnWeights):
        model = model_func()
    elif weight_type == WeightType.Pretrained:
        model = model_func(weights='DEFAULT')
    else:
        raise NotImplementedError
    
    if num_class is not None:
        # # Change output layer to match output classification class
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = torch.nn.Linear(in_features, out_features=num_class)

    # If it's ownweights, load weights
    if weight_type == WeightType.OwnWeights:
        model.load_state_dict(torch.load(weights_path), strict=False)

    return model



# class MobileNet:
#     def __init__(
#             self, model_type: int, weight_type: WeightType,
#             weight_path: str =None
#         ):
        
        
#         self.model = 