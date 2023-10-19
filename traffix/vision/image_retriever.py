import numpy as np
import cv2 as cv

from datetime import datetime
from pympler.asizeof import asizeof


class ImageRetriever:
    def __init__(self, source="", fps=30.0):
        if source == "":
            self.__source = cv.VideoCapture(0)
        else:
            self.__source = cv.VideoCapture(source)

        if not self.__source.isOpened():
            print("Could not access video source")
            exit()

        cv.namedWindow("source", cv.WINDOW_AUTOSIZE)

        self.__last_frame: np.ndarray

        self.__fps = fps

    
    def play_source(self, display=False):

        try:
            legit_wait_time = int(1.0 / self.__fps * 1000)
            wait_time = legit_wait_time
            frames_read = 0
            start_time = datetime.now()
            while True:
                ret, self.__last_frame = self.__source.read()

                if not ret:
                    print("Could not receive stream frame. Exiting...")
                    break
                
                self.__last_frame = cv.cvtColor(self.__last_frame, cv.COLOR_BGR2RGB)

                if display:
                    cv.imshow("source", self.__last_frame)

                if cv.waitKey(wait_time) == ord('q'):
                    break
                
                frames_read += 1
                time_delta = (datetime.now() - start_time).total_seconds()
                current_fps = float(frames_read) / time_delta

                wait_time = legit_wait_time + 10 * int(current_fps - self.__fps)
                if wait_time <= 0:
                    wait_time = 1

        except KeyboardInterrupt:
            print("Exiting...")
            self.stop()
            

    def get_last_frame(self):
        return self.__last_frame


    def display_frame(self, frame: np.ndarray):
        cv.imshow("source", frame)
        cv.waitKey(1)


    def stop(self):
        self.__source.release()
        cv.destroyAllWindows()