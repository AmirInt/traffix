import sys
import matplotlib.pyplot as plt

sys.path.append("D:\\projects\\traffix\\")
import traffix.prediction.time_series as time_series


data_csv = "test\\north.csv"
model_filename = "test\\model_8_1.torch"
model_hidden_size = 8

if __name__ == "__main__":
    predictor = time_series.Predictor(data_csv,
                                      model_filename,
                                      hidden_size=model_hidden_size)

    predictor.load_model(model_filename)

    predicted = predictor.predict(27000)

    actual = predictor.get_y_range((27000, 27024))

    #plt.plot(predictor, label="Predicted")
    plt.plot(actual[0], label="Actual")
    plt.plot(predicted[0], label="Predicted")
    plt.show()
