import numpy as np
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
from ultralytics.engine.results import Results


class VehicleCounter:
    def __init__(self, yolo_model=""):
        
        self._count = 0

        self._model = YOLO(yolo_model)


    def process_frame(self, frame: np.ndarray) -> Results:
        results = self._model.track(frame, persist=True)
        return results


    def get_last_count(self):
        return self._count()
