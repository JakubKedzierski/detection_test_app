import abc
from Commons import Display


class DetectionModule(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def detect(self, frame, plot, image_depth=[], analyze_depth=True, display=Display.plt):
        raise NotImplementedError