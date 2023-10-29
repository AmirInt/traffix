import yaml

from threading import Thread
from traffix.vision.image_retriever import ImageRetriever
from traffix.vision.vehicle_counter import VehicleCounter


if __name__ == "__main__":

    with open("config.yaml") as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
            exit()
    
    videos = []
    video_threads = []
    vehicle_counters = []
    vehicle_counter_threads = []
    
    for view in config["views"]:
        
        videos.append(ImageRetriever(view["name"], view["source"], float(view["fps"])))

        video_threads.append(Thread(group=None, target=videos[-1].play_source))
        
        roi = ((view["roi"]["y1"], view["roi"]["y2"]), (view["roi"]["x1"], view["roi"]["x2"]))
        
        vehicle_counters.append(VehicleCounter(config["vehicle_counter"]["yolo_weights"], roi))

    #video = ImageRetriever(source=sys.argv[1], fps=float(sys.argv[2]))

    #vehicle_counter = VehicleCounter("yolo_weights/yolov8x.pt", ((0, 300), (0, 200)))

    #video_thread = threading.Thread(group=None, target=video.play_source)

    #video_thread.start()

    #try:
    #    while True:
    #        results = vehicle_counter.process_frame(video.get_last_frame())
    #        if results is not None:
    #            video.display_frame(results[0].plot())
    #except KeyboardInterrupt:
    #    print("Finishing")
    
    #video_thread.join(1.0)

    #video.stop()