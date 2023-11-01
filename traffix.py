import yaml
import torch
import numpy as np
from threading import Thread
from time import sleep
from traffix.vision.image_retrieving import ImageRetriever
from traffix.vision.vehicle_counting import VehicleCounter


if __name__ == "__main__":
    
    try:
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

            video_threads.append(Thread(target=videos[-1].play_source))
        
            roi = ((view["roi"]["y1"], view["roi"]["y2"]), (view["roi"]["x1"], view["roi"]["x2"]))
        
            vehicle_counters.append(
                VehicleCounter(
                    config["vehicle_counter"]["yolo_weights"],
                    roi,
                    config["vehicle_counter"]["target_classes"]
                    )
                )

            vehicle_counter_threads.append(Thread(target=vehicle_counters[-1].process_source, args=[videos[-1]]))
        
            vehicle_counter_threads[-1].daemon = True
            vehicle_counter_threads[-1].start()
        

        for video_thread in video_threads:
            video_thread.daemon = True
            video_thread.start()
    
        while True:
            sleep(1.0)
    except KeyboardInterrupt:
        print("Stopping vehicle counters...")
        
        for vehicle_counter in vehicle_counters:
            vehicle_counter.stop()

        sleep(1.0)
        
        print("Stopping video sources...")

        for video in videos:
            video.stop()
