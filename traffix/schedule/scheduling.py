import numpy as np
import time
from threading import Thread




class Scheduler:
    def __init__(self,
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


    def get_traffic_counts(self, idx: int, line: tuple) -> None:

        current_vehicle_count = line[0].get_current_count()
        line[1].append_data_point(line[0].get_last_count())
        line[0].clear_count()
        history = line[1].get_range((-self._history_depth,))
        history = line[1].scale_data(history)
        history = line[1].convert_to_batch(history).transpose()
        history = line[1].convert_to_torch(history)
        prediction = line[2].predict(history)
        prediction = line[1].reverse_scale_data(prediction)

        avg_prediction = np.sum(prediction) / self._prediction_depth

        self._traffic_counts[idx][0] += current_vehicle_count
        self._traffic_counts[idx][1] += avg_prediction


    def calculate_timings(self) -> None:
        idx = 0
        self._traffic_counts.fill(0.0)
            
        for path in self._paths:
            
            for line in path:
                    
                self.get_traffic_counts(idx, line)

            idx += 1
        
        print("Current and predicted vehicle count:")
        print(self._traffic_counts)
        
        total_count = np.sum(self._traffic_counts)
        print("Total count:", total_count)

        self._timings = np.sum(self._traffic_counts, axis=1) / total_count

        self._timings *= self._schedule_period

        self._timings = np.round(self._timings)

        print("Schedule:")
        print(self._timings)


    def run_schedule(self) -> None:
        while True:
            calculator_thread = Thread(target=self.calculate_timings)
            calculator_thread.daemon = True
            calculator_thread.start()

            time.sleep(self._schedule_period)
