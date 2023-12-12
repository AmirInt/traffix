import os
import numpy as np
import pandas as pd
import torch

from copy import deepcopy
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader
from torch.utils.data import Dataset




class TimeSeriesDataset(Dataset):
    def __init__(self, X: torch.Tensor, y: torch.Tensor) -> None:
        self.X = X
        self.y = y

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i) -> tuple:
        return self.X[i], self.y[i]




class DataProcessor:
    def __init__(self,
                 data_csv: os.PathLike,
                 n_lookback: int,
                 n_predict: int,
                 index_title: str,
                 vehicle_count_title: str) -> None:
        
        self._df = pd.read_csv(data_csv)
        self._n_lookback = n_lookback
        self._n_predict = n_predict
        self._index_title = index_title
        self._vehicle_count_title = vehicle_count_title

        print("DataProcessor loaded")


    def shift_dataset_for_lstm_train(self) -> pd.DataFrame:

        shifted_df = deepcopy(self._df)

        shifted_df.set_index(self._index_title, inplace=True)

        for i in range(1, self._n_predict):
            shifted_df[self._vehicle_count_title + f"(t+{i})"] = shifted_df[self._vehicle_count_title].shift(-i)

        for i in range(self._n_lookback, 0, -1):
            shifted_df[self._vehicle_count_title + f"(t-{i})"] = shifted_df[self._vehicle_count_title].shift(i)

        shifted_df.dropna(inplace=True)
        
        print("DataProcessor prepared dataset for LSTM")

        return shifted_df


    def fit_self(self) -> None:
        data_frame = deepcopy(self._df)

        data_frame.set_index(self._index_title, inplace=True)

        data_np = data_frame.to_numpy()

        self._scaler = MinMaxScaler(feature_range=(-1, 1))

        self._scaler.fit(data_np)


    def fit_scale(self, data: pd.DataFrame) -> np.ndarray:
        data_np = data.to_numpy()
        self._scaler = MinMaxScaler(feature_range=(-1, 1))
        data_np = self._scaler.fit_transform(data_np)

        print("DataProcessor scaled data")

        return data_np


    def extract_x_y(self, dataset: np.ndarray) -> tuple:
        data_tn = torch.tensor(dataset).float()
        
        X = data_tn[:, self._n_predict:].reshape((-1, self._n_lookback, 1))
        y = data_tn[:, :self._n_predict].reshape((-1, self._n_predict, 1))
        
        print("DataProcessor produced X and y tensors")

        return X, y


    def get_train_datasets(self,
                           X: torch.Tensor,
                           y: torch.Tensor,
                           batch_size: int,
                           train_ratio: float) -> tuple:

        split_index = int(train_ratio * len(X))
        
        X_train = X[:split_index]
        y_train = y[:split_index]
        X_test = X[split_index:]
        y_test = y[split_index:]

        train_dataset = TimeSeriesDataset(X_train, y_train)
        test_dataset = TimeSeriesDataset(X_test, y_test)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        print("DataProcessor prepared train and test loaders")

        return train_loader, test_loader


    def scale_data(self, data: np.ndarray) -> np.ndarray:
        return self._scaler.transform(data)


    def reverse_scale_data(self, data: np.ndarray) -> np.ndarray:
        padded = np.zeros((data.shape[0], self._n_lookback + self._n_predict))
        padded[:, self._n_lookback:] = data[:, :, 0]
        return self._scaler.inverse_transform(padded)[:, self._n_lookback:]


    def convert_to_batch(self, data: np.ndarray) -> np.ndarray:
        return np.expand_dims(data, axis=0)


    def convert_to_torch(self, data: np.ndarray) -> torch.Tensor:
        return torch.tensor(data).float()

    
    def get_x_range(self, X: torch.Tensor, x_range: tuple) -> torch.Tensor:
        return X[x_range[0]:x_range[1]]


    def get_y_range(self, y: torch.Tensor, y_range: tuple) -> torch.Tensor:
        return y[y_range[0]:y_range[1]]


    def get_range(self, range: tuple) -> np.ndarray:
        data_range = self._df[self._vehicle_count_title][range[0]:range[1]].to_numpy()

        data_range = data_range.reshape(-1, 1)

        return data_range


    def append_data_point(self, new_data_point: float) -> None:
        self._df.loc[self._df.shape[0]] = [self._df.shape[0], new_data_point]