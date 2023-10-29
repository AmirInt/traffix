import numpy as np
import cv2 as cv

from datetime import datetime


class ImageRetriever:
    def __init__(self, source="", fps: float = 30.0):

        if source == "camera":
            self._source = cv.VideoCapture(0)
        else:
            self._source = cv.VideoCapture(source)

        if not self._source.isOpened():
            raise RuntimeError("Could not access video source")

        cv.namedWindow("source", cv.WINDOW_AUTOSIZE)

        self._last_frame: np.ndarray = None

        self._fps = fps

    
    def play_source(self, display=False):

        try:
            legit_wait_time = int(1.0 / self._fps * 1000)
            wait_time = legit_wait_time
            frames_read = 0
            start_time = datetime.now()
            while True:
                ret, frame = self._source.read()

                if not ret:
                    raise RuntimeError("Could not receive stream frame. Exiting...")
                
                self._last_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

                if display:
                    cv.imshow("source", self._last_frame)
                
                if cv.waitKey(wait_time) == ord('q'):
                    break
                
                frames_read += 1
                time_delta = (datetime.now() - start_time).total_seconds()
                current_fps = float(frames_read) / time_delta

                wait_time = legit_wait_time + 10 * int(current_fps - self._fps)
                if wait_time <= 0:
                    wait_time = 1

        except KeyboardInterrupt:
            print("Exiting...")
            self.stop()
            

    def get_last_frame(self) -> np.ndarray:
        return self._last_frame


    def display_frame(self, frame: np.ndarray):
        cv.imshow("source", cv.cvtColor(frame, cv.COLOR_RGB2BGR))
        cv.waitKey(1)


    def stop(self):
        self._source.release()
        cv.destroyAllWindows()