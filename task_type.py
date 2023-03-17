from enum import Enum

class VisionTaskType(Enum):
    ImageClassification = 1
    SemanticSegmentation = 2
    InstanceSegmentation = 3
    ObjectDetection = 4

class TextTaskType(Enum):
    TextClassification = 1