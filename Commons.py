from enum import Enum
import os

class ModelsStrings(Enum):
    mask_rcnn = 'mask-rcnn'
    yolo = 'yolo'
    ssd = 'ssd'

    def __str__(self):
        return self.value

class Display(Enum):
    opencv = 'opencv'
    plt = 'plt'

    def __str__(self):
        return self.value

class Skipping(Enum):
    lost_frames = 'lost_frames'
    empty = 'empty'
    thread = 'thread'

    def __str__(self):
        return self.value


class ObjectDetectionType(Enum):
    tomato = 'tomato'
    apple = 'apple'

    def __str__(self):
        return self.value


def dir_path(string):
    if os.path.exists(string):
        return string
    else:
        raise NotADirectoryError(string)