import sys
import threading

from traffix.vision.image_retriever import ImageRetriever


if __name__ == "__main__":
    video = ImageRetriever(source=sys.argv[1], fps=float(sys.argv[2]))

    video.play_source(True)

    video.stop()