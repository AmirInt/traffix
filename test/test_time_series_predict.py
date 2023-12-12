import sys
import matplotlib.pyplot as plt

sys.path.append("D:\\projects\\traffix\\")
import traffix.prediction.time_series as time_series
from traffix.utils.time_series_utils import DataProcessor


data_csv = "test\\north.csv"
model_filename = "test\\model_8_1.torch"
model_hidden_size = 8
n_lookback = 672
n_predict = 24
index_title = "index"
vehicle_count_title: str = "vehicle_count"

if __name__ == "__main__":

    # Getting and preprocessing data
    data_processor = DataProcessor(data_csv,
                                   n_lookback,
                                   n_predict,
                                   index_title,
                                   vehicle_count_title)

    data_processor.fit_self()

    input_data = data_processor.get_range((27000 - 672, 27000))

    input_data = data_processor.scale_data(input_data)

    input_data = data_processor.convert_to_batch(input_data)

    input_data = data_processor.convert_to_torch(input_data.transpose())

    predictor = time_series.Predictor()

    predictor.load_model(model_filename)

    predicted = predictor.predict(input_data)

    predicted = data_processor.reverse_scale_data(predicted)

    actual = data_processor.get_range((27000, 27024)).transpose()

    #plt.plot(predictor, label="Predicted")
    plt.plot(actual[0], label="Actual")
    plt.plot(predicted[0], label="Predicted")
    plt.legend()
    plt.show()
