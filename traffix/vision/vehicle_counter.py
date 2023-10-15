import numpy
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
from ultralytics.engine.results import Results


class VehicleCounter:
    def __init__(self):
        
        self.__count = 0

        self.__model = YOLO()

    def process_frame(self, frame: numpy.ndarray):
        pass

    def get_last_count(self):
        return self.__count()
