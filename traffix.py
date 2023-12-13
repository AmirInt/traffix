import yaml
from threading import Thread
from time import sleep
from traffix.vision.image_retrieving import ImageRetriever
from traffix.vision.vehicle_counting import VehicleCounter
from traffix.utils.time_series_utils import DataProcessor
from traffix.prediction.time_series import Predictor
from traffix.schedule.scheduling import Scheduler




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
        traffic_count_recorders = []
        traffic_predictors = []
    
        # Initialise video sources and traffic viewers
        for view in config["views"]:
        
            videos.append(
                ImageRetriever(
                    view["name"],
                    view["source"],
                    float(view["fps"])))

            video_threads.append(Thread(target=videos[-1].play_source))
        
            roi = ((view["roi"]["y1"], view["roi"]["y2"]),
                  (view["roi"]["x1"], view["roi"]["x2"]))
        
            vehicle_counters.append(
                VehicleCounter(
                    config["vehicle_counter"]["yolo_weights"],
                    roi,
                    config["vehicle_counter"]["target_classes"]))

            vehicle_counter_threads.append(
                Thread(target=vehicle_counters[-1].process_source,
                       args=[videos[-1]]))
        
        # Initialise traffic count data and predictors
        history_depth = config["traffic_predictor"]["history_depth"]
        prediction_depth = config["traffic_predictor"]["prediction_depth"]
        hidden_size= config["traffic_predictor"]["hidden_size"]
        num_stacked_layers = config["traffic_predictor"]["num_stacked_layers"]

        for predictor in config["traffic_predictors"]:
            traffic_count_recorders.append(DataProcessor(
                data_csv=predictor["initial_data"],
                n_lookback=history_depth,
                n_predict=prediction_depth,
                index_title="index",
                vehicle_count_title="vehicle_count"))
            
            traffic_count_recorders[-1].fit_self()

            traffic_predictors.append(Predictor(
                1,
                output_size=prediction_depth,
                hidden_size=hidden_size,
                num_stacked_layers=num_stacked_layers,
                n_lookback=history_depth,
                n_predict=prediction_depth))

            traffic_predictors[-1].load_model(predictor["model_weights"])

        # Initialise the scheduler module
        paths = []
        for i in range(config["scheduler"]["paths"]):
            path = []
            for line_idx in config["scheduler"][f"path_{i}"]:
                line = (vehicle_counters[line_idx],
                        traffic_count_recorders[line_idx],
                        traffic_predictors[line_idx])
                path.append(line)
            paths.append(path)

        scheduler = Scheduler(paths,
                              config["scheduler"]["schedule_period"],
                              history_depth,
                              prediction_depth)

        for video_thread in video_threads:
            video_thread.daemon = True
            video_thread.start()

        for vehicle_counter_thread in vehicle_counter_threads:
            vehicle_counter_thread.daemon = True
            vehicle_counter_thread.start()

        scheduler.run_schedule()

    except KeyboardInterrupt:
        print("Stopping vehicle counters...")
        
        for vehicle_counter in vehicle_counters:
            vehicle_counter.stop()

        sleep(1.0)
        
        print("Stopping video sources...")

        for video in videos:
            video.stop()
