import numpy as np
import cv2 as cv

from os import PathLike
from datetime import datetime


class ImageRetriever:
    def __init__(self, name: str = "source", source: PathLike = "", fps: float = 30.0):

        self._name = name

        if source == "camera":
            self._source = cv.VideoCapture(0)
        else:
            self._source = cv.VideoCapture(source)

        if not self._source.isOpened():
            raise RuntimeError("Could not access video source")

        self._last_frame: np.ndarray = None

        self._fps = fps

        self._running = False

    
    def play_source(self, display=False):
        self._running = True

        try:
            legit_wait_time = int(1.0 / self._fps * 1000)
            wait_time = legit_wait_time
            frames_read = 0
            start_time = datetime.now()
            while self._running:
                ret, frame = self._source.read()

                if not ret:
                    raise RuntimeError("Could not receive stream frame.")
                
                self._last_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

                if display:
                    cv.imshow(self._name, self._last_frame)
                
                if cv.waitKey(wait_time) == ord('q'):
                    break
                
                frames_read += 1
                time_delta = (datetime.now() - start_time).total_seconds()
                current_fps = float(frames_read) / time_delta

                wait_time = legit_wait_time + 10 * int(current_fps - self._fps)
                if wait_time <= 0:
                    wait_time = 1

        except (KeyboardInterrupt, InterruptedError, RuntimeError, cv.error) as e:
            print(e)
            print("Exiting...")
            self.stop()
            print("Done")
            

    def get_last_frame(self) -> np.ndarray:
        return self._last_frame


    def display_frame(self, frame: np.ndarray):
        cv.imshow(self._name, cv.cvtColor(frame, cv.COLOR_RGB2BGR))
        if cv.waitKey(1) == ord('q'):
            self.stop()


    def stop(self):
        self._running = False
        self._source.release()
        cv.destroyAllWindows()