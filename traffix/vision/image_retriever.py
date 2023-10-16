import sys
import numpy as np
import cv2 as cv


class ImageRetriever:
    def __init__(self, source="", history_size=2e+8):
        if source == "":
            self.__source = cv.VideoCapture(0)
        else:
            self.__source = cv.VideoCapture(source)

        if not self.__source.isOpened():
            print("Could not access video source")
            exit()

        cv.namedWindow("source", cv.WINDOW_AUTOSIZE)

        self.__frames = []

        self.__history_size = history_size

    
    def play_source(self, display=False):

        try:
            while True:
                ret, frame = self.__source.read()

                if not ret:
                    print("Could not receive stream frame. Exiting...")
                    break
                
                rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

                if display:
                    cv.imshow("source", rgb_frame)

                if cv.waitKey(1) == ord('q'):
                    break

        except KeyboardInterrupt:
            self.stop()


    def append_to_history(self, frame: np.ndarray):
        self.__frames.append(frame)
        while sys.getsizeof(self.__frames) > self.__history_size:
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