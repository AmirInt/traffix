import os
import sys
import cv2 as cv


def make_video(src_dir: os.PathLike, dest_dir: os.PathLike, duration: float, video_name: str):
    images = [img for img in os.listdir(src_dir)]

    frame = cv.imread(os.path.join(src_dir, images[0]))

    height, width, channels = frame.shape

    fps = len(images) / duration

    video_writer = cv.VideoWriter(os.path.join(dest_dir, video_name), 0, fps, (width, height))

    for img in images:
        video_writer.write(cv.imread(os.path.join(src_dir, img)))

    video_writer.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    if sys.argv[1] == "make_video":
        make_video(sys.argv[2], sys.argv[3], float(sys.argv[4]), sys.argv[5])
