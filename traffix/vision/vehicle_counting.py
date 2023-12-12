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
            detection_id: int,
            x_pos: int,
            y_pos: int,
            x_dir: int = 0,
            y_dir: int = 0):
        
        self._id = detection_id
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

    
    def get_id(self) -> int:
        return self._id


    def get_avg_dir(self) -> tuple:
        return (self._avg_x_dir, self._avg_y_dir)
    

    def get_comp_count(self) -> int:
        return self._comp_count




class VehicleCounter:
    def __init__(
            self,
            yolo_model: os.PathLike,
            roi: tuple,
            target_classes: set = {}) -> None:
        
        if len(roi) != 2 or len(roi[0]) != 2 or len(roi[1]) != 2:
            raise ValueError("ROI not properly set")

        self._model = YOLO(yolo_model)
        self._annotator = Annotator(np.ndarray(shape=(1, 1, 1)))
        self._roi = roi
        self._target_classes = target_classes
        self._running = False
        self._track_history = {}
        self._current_vehicle_count = 0
        self._passing_vehicles_id_set = set({})


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
                [box.data for box in boxes if self._model.names[int(box.cls)] in self._target_classes]),
                boxes.orig_shape)
        
        return filtered_boxes


    def filter_direction(self, boxes: Boxes) -> Boxes:
        track_history = {}
        filtered_boxes = []

        for box in boxes:
            detection = Detection(
                int(box.id),
                int((box.xyxy[0][0] + box.xyxy[0][2]) / 2),
                int((box.xyxy[0][1] + box.xyxy[0][3]) / 2))
            try:
                detection.compare(self._track_history[detection.get_id()])
            except KeyError:
                pass

            track_history[detection.get_id()] = detection
            
            if detection.get_avg_dir()[1] > 0:
                filtered_boxes.append(box.data)

        self._track_history = track_history

        if len(filtered_boxes) == 0:
            return Boxes(np.ndarray((0, 7), dtype=Boxes), boxes.orig_shape)

        return Boxes(np.array(filtered_boxes), boxes.orig_shape)


    def filter_results(self, boxes: Boxes) -> Boxes:
        return self.filter_direction(self.filter_class(boxes))


    def process_source(self, source: ImageRetriever) -> None:
        self._running = True

        try:
            while self._running:
                frame = source.get_last_frame()
                results = self.process_frame(frame)
                if results is not None:
                    filtered_boxes = self.filter_results(results[0].boxes)

                    self._passing_vehicles_id_set.update([int(box.id) for box in results[0].boxes])

                    self._current_vehicle_count = len(filtered_boxes)

                    self._annotator.im = frame
                    for box in filtered_boxes:
                        self._annotator.box_label(
                            box.xyxy[0],
                            self._model.names[int(box.cls)])
                    
                    source.display_frame(self._annotator.result())

        except KeyboardInterrupt:
            print("Finishing...")


    def get_current_count(self) -> int:
        return self._current_vehicle_count


    def get_last_count(self) -> int:
        return len(self._passing_vehicles_id_set)


    def clear_count(self) -> None:
        self._passing_vehicles_id_set.clear()


    def stop(self) -> None:
        self._running = False
