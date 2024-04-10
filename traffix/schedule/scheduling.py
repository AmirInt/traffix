import numpy as np
import time
from threading import Thread
from traffix.utils.interface import Interface, DisplayCanvas




class Scheduler:
    def __init__(self,
                 interface: Interface,
                 paths: list,
                 schedule_period: float,
                 history_depth: int = 672,
                 prediction_depth: int = 24) -> None:
        self._paths = paths
        self._schedule_period = schedule_period
        self._traffic_counts = np.zeros((len(paths), 2), dtype=float)
        self._timings = np.zeros(len(paths), dtype=float)
        self._history_depth = history_depth
        self._prediction_depth = prediction_depth
        self._interface = interface


    def get_traffic_counts(self, path_idx: int, line_idx: int, line: tuple) -> None:

        current_vehicle_count = line[0].get_last_count()
        line[1].append_data_point(line[0].get_last_count())
        line[0].clear_count()
        history = line[1].get_range((-self._history_depth,))
        history = line[1].scale_data(history)
        history = line[1].convert_to_batch(history).transpose()
        history = line[1].convert_to_torch(history)
        prediction = line[2].predict(history)
        prediction = line[1].reverse_scale_data(prediction)

        avg_prediction = np.sum(prediction) / self._prediction_depth

        self._interface.get_display_canvas(path_idx * 2 + line_idx).update(
            estimated_flow=int(np.round(avg_prediction)))

        self._traffic_counts[path_idx][0] += current_vehicle_count
        self._traffic_counts[path_idx][1] += avg_prediction


    def calculate_timings(self) -> None:
        self._traffic_counts.fill(0.0)
            
        for path_idx, path in enumerate(self._paths):
            
            for line_idx, line in enumerate(path):
                    
                self.get_traffic_counts(path_idx, line_idx, line)

        total_count = np.sum(self._traffic_counts)

        self._timings = np.sum(self._traffic_counts, axis=1) / total_count

        self._timings *= self._schedule_period

        self._timings = np.round(self._timings)
        
        for path_idx in range(len(self._paths)):
            self._interface.get_sum_canvas(path_idx).update(
                current_flow=int(self._traffic_counts[path_idx, 0]),
                estimated_flow=int(np.round(self._traffic_counts[path_idx, 1])),
                green_light_time=self._timings[path_idx])


    def run_schedule(self) -> None:
        while True:
            calculator_thread = Thread(target=self.calculate_timings)
            calculator_thread.daemon = True
            calculator_thread.start()

            time.sleep(self._schedule_period)
