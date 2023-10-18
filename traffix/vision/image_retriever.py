import os
import sys
import numpy as np
import cv2 as cv

from datetime import datetime
from pympler.asizeof import asizeof


class ImageRetriever:
    def __init__(self, source="", history_size=2e+8, fps=30.0):
        if source == "":
            self.__source = cv.VideoCapture(0)
        else:
            self.__source = cv.VideoCapture(source)

        if not self.__source.isOpened():
            print("Could not access video source")
            exit()

        cv.namedWindow("source", cv.WINDOW_AUTOSIZE)

        self.__frames = []

        self.__history_size = int(history_size)

        self.__fps = fps

    
    def play_source(self, display=False):

        try:
            legit_wait_time = int(1.0 / self.__fps * 1000)
            wait_time = legit_wait_time
            frames_read = 0
            start_time = datetime.now()
            while True:
                ret, frame = self.__source.read()

                if not ret:
                    print("Could not receive stream frame. Exiting...")
                    break
                
                rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

                if display:
                    cv.imshow("source", rgb_frame)

                self.append_to_history(frame)

                if cv.waitKey(wait_time) == ord('q'):
                    break
                
                frames_read += 1
                time_delta = (datetime.now() - start_time).total_seconds()
                current_fps = float(frames_read) / time_delta

                wait_time = legit_wait_time + 10 * int(current_fps - self.__fps)
                if wait_time <= 0:
                    wait_time = 3

        except KeyboardInterrupt:
            print("Exiting...")
            self.stop()


    def append_to_history(self, frame: np.ndarray):
        self.__frames.append(frame)
        while asizeof(self.__frames) > self.__history_size:
            self.__frames.pop(0)


    def set_history_size(self, history_size):
        self.__history_size = history_size


    def get_last_frame(self):
        return self.__frames[-1]


    def get_all_frames(self):
        return self.__frames


    def stop(self):
        self.__source.release()
        cv.destroyAllWindows()