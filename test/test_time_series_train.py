import sys
import matplotlib.pyplot as plt

sys.path.append("D:\\projects\\traffix\\")
import traffix.prediction.time_series as time_series
from traffix.utils.time_series_utils import DataProcessor


data_csv = "test/north.csv"
model_filename = "test/model_8_1_trained.torch"
n_lookback = 672
n_predict = 24
index_title = "index"
vehicle_count_title: str = "vehicle_count"
batch_size: int = 512
train_ratio: float = 0.9
                    

if __name__ == "__main__":
    
    # Getting and preprocessing data
    data_processor = DataProcessor(data_csv,
                                   n_lookback,
                                   n_predict,
                                   index_title,
                                   vehicle_count_title)

        
    lstm_train_dataset = data_processor.shift_dataset_for_lstm_train()
    fit_dataset = data_processor.fit_scale(lstm_train_dataset)
    X, y = data_processor.extract_x_y(fit_dataset)
        
    train_loader, test_loader = data_processor.get_train_datasets(
        X,
        y,
        batch_size,
        train_ratio)
        
    predictor = time_series.Predictor()

    train_loss, val_loss = predictor.train_model(model_filename, train_loader, test_loader)

    input_data = data_processor.get_x_range(X, (27000 - 672, 27000))

    predicted = predictor.predict(input_data)

    predicted = data_processor.reverse_scale_data(predicted)

    actual = data_processor.get_y_range(y, (27000, 27001)).to("cpu").numpy()

    actual = data_processor.reverse_scale_data(actual)

    plt.plot(train_loss, label="Train Loss")
    plt.plot(val_loss, label="Validation Loss")
    plt.legend()
    plt.show()
    
    plt.plot(actual[0], label="Actual")
    plt.plot(predicted[0], label="Predicted")
    plt.legend()
    plt.show()