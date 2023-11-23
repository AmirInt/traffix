import sys
import matplotlib.pyplot as plt

sys.path.append("D:\\projects\\traffix\\")
import traffix.prediction.time_series as time_series


data_csv = "north.csv"
model_filename = "model_8_1_trained.torch"
model_hidden_size = 8

if __name__ == "__main__":
    predictor = time_series.Predictor(data_csv,
                                      model_filename,
                                      hidden_size=model_hidden_size)

    train_loss, val_loss = predictor.train_model()

    predicted = predictor.predict(27000)

    actual = predictor.get_y_range((27000, 27024))

    plt.plot(train_loss, label="Train Loss")
    plt.plot(val_loss, label="Validation Loss")
    plt.show()
    
    plt.plot(actual[0], label="Actual")
    plt.plot(predicted[0], label="Predicted")
    plt.show()