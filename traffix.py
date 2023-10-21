import sys
import threading

from traffix.vision.image_retriever import ImageRetriever
from traffix.vision.vehicle_counter import VehicleCounter


if __name__ == "__main__":
    video = ImageRetriever(source=sys.argv[1], fps=float(sys.argv[2]))

    vehicle_counter = VehicleCounter("yolo_weights/yolov8x.pt")

    video_thread = threading.Thread(group=None, target=video.play_source)

    video_thread.start()

    try:
        while True:
            results = vehicle_counter.process_frame(video.get_last_frame())
            video.display_frame(results[0].plot())
    except KeyboardInterrupt:
        print("Finishing")
    
    except:
        print("Errors occurred. Exitting")

    video_thread.join(1.0)

    video.stop()