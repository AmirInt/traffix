import os
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
from ultralytics.engine.results import Results

from traffix.vision.image_retriever import ImageRetriever


class VehicleCounter:
    def __init__(self, yolo_model: os.PathLike = "", roi: tuple = ()):
        
        if len(roi) > 0 and \
                (len(roi) != 2 or \
                (type(roi[0]) != tuple and \
                 type(roi[1]) != tuple)):
            raise ValueError("ROI not properly set")

        self._count = 0

        self._model = YOLO(yolo_model)
        
        self._roi = roi

        self._running = False


    def process_frame(self, frame: np.ndarray) -> Results:
        if frame is None:
            print("Vehicle counter: skipping invalid data...")
            return None
    
        # Crop the frame if ROI specified
        if len(self._roi) > 0:
            frame = frame[
                self._roi[0][0]:self._roi[0][1],
                self._roi[1][0]:self._roi[1][1],
                :
            ]

        results = self._model.track(frame, persist=True)
        return results


    def process_source(self, source: ImageRetriever):
        self._running = True

        try:
            while self._running:
                results = self.process_frame(source.get_last_frame())
                if results is not None:
                    source.display_frame(results[0].plot())
        except KeyboardInterrupt:
            print("Finishing")

    def get_last_count(self):
        return self._count()


    def stop(self):
        self._running = False