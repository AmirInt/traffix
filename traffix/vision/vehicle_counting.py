import os
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
from ultralytics.engine.results import Results
from ultralytics.engine.results import Boxes
from traffix.vision.image_retrieving import ImageRetriever


class Detection:
    def __init__(
            self,
            id: int,
            x_pos: int,
            y_pos: int,
            x_dir: int = 0,
            y_dir: int = 0):
        
        self._id = id
        self._x_pos = x_pos
        self._y_pos = y_pos
        self._avg_x_dir = x_dir
        self._avg_y_dir = y_dir
        self._comp_count = 0
        

    def compare(self, other: "Detection"):
        
        self._avg_x_dir = self._x_pos - other._x_pos + other._avg_x_dir * other._comp_count
        self._avg_y_dir = self._y_pos - other._y_pos + other._avg_y_dir * other._comp_count

        self._comp_count = other._comp_count + 1

        self._avg_x_dir /= self._comp_count
        self._avg_y_dir /= self._comp_count

    
    def get_avg_dir(self) -> tuple:
        return (self._avg_x_dir, self._avg_y_dir)
        



class VehicleCounter:
    def __init__(
            self,
            yolo_model: os.PathLike,
            roi: tuple,
            target_classes: set = {},
            min_dir_prob: float = 0.5) -> None:
        
        if len(roi) != 2 or len(roi[0]) != 2 or len(roi[1]) != 2:
            raise ValueError("ROI not properly set")

        self._count = 0
        self._model = YOLO(yolo_model)
        self._annotator = Annotator(np.ndarray(shape=(1, 1, 1)))
        self._roi = roi
        self._target_classes = target_classes
        self._min_dir_prob = min_dir_prob
        self._running = False
        self._track_history = []


    def process_frame(self, frame: np.ndarray) -> Results:
        if frame is None:
            print("Skipping invalid data...")
            return None
        
        results = self._model.track(
            frame[
                self._roi[0][0]:self._roi[0][1],
                self._roi[1][0]:self._roi[1][1],
                :],
            persist=True,
            verbose=False)

        return results


    def filter_class(self, boxes: Boxes) -> Boxes:
        if len(self._target_classes) == 0:
            return boxes

        filtered_boxes = Boxes(
            np.array(
                [box.data.numpy() for box in boxes if self._model.names[int(box.cls)] in self._target_classes]
                ),
            boxes.orig_shape
            )
        
        return filtered_boxes


    def filter_direction(self, boxes: Boxes) -> Boxes:
        pass


    def filter_results(self, boxes: Boxes) -> Boxes:
        pass


    def process_source(self, source: ImageRetriever) -> None:
        self._running = True

        try:
            while self._running:
                frame = source.get_last_frame()
                results = self.process_frame(frame)
                if results is not None:
                    
                    #self._annotator.fromarray(frame)
                    
                    if source._name == "south":
                        print("New result:")
                        print(len(results[0].boxes.xywh))
                        print(len(self.filter_class(results[0].boxes)))

                    source.display_frame(results[0].plot())
        except KeyboardInterrupt:
            print("Finishing...")


    def get_last_count(self) -> int:
        return self._count()


    def stop(self) -> None:
        self._running = False