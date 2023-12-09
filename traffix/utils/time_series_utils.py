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


    def prepare_dataset_for_lstm(self) -> None:

        self._shifted_df = deepcopy(self._df)

        self._shifted_df.set_index(self._index_title, inplace=True)

        for i in range(1, self._n_predict):
            self._shifted_df[self._vehicle_count_title + f"(t+{i})"] = self._shifted_df[self._vehicle_count_title].shift(-i)

        for i in range(self._n_lookback, 0, -1):
            self._shifted_df[self._vehicle_count_title + f"(t-{i})"] = self._shifted_df[self._vehicle_count_title].shift(i)

        self._shifted_df.dropna(inplace=True)
        
        print("DataProcessor prepared dataset for LSTM")


    def scale_data(self) -> None:
        self._data_np = self._shifted_df.to_numpy()
        self._scaler = MinMaxScaler(feature_range=(-1, 1))
        self._data_np = self._scaler.fit_transform(self._data_np)

        print("DataProcessor scaled data")

    def extract_x_y(self) -> None:
        data_tn = torch.tensor(self._data_np).float()
        
        self._X = data_tn[:, self._n_predict:].reshape((-1, self._n_lookback, 1))
        self._y = data_tn[:, :self._n_predict].reshape((-1, self._n_predict, 1))
        
        print("DataProcessor produced X and y tensors")


    def get_train_datasets(self,
                           batch_size: int,
                           train_ratio: float) -> tuple:

        split_index = int(train_ratio * len(self._X))
        
        X_train = self._X[:split_index]
        y_train = self._y[:split_index]
        X_test = self._X[split_index:]
        y_test = self._y[split_index:]

        train_dataset = TimeSeriesDataset(X_train, y_train)
        test_dataset = TimeSeriesDataset(X_test, y_test)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        print("DataProcessor prepared train and test loaders")

        return train_loader, test_loader

    
    def get_x_range(self, x_range: tuple) -> torch.Tensor:
        return self._X[x_range[0]:x_range[1]]


    def get_y_range(self, y_range: tuple) -> torch.Tensor:
        return self._y[y_range[0]:y_range[1]]


    def reverse_scale_data(self, data: np.ndarray) -> np.ndarray:
        padded = np.zeros((data.shape[0], self._n_lookback + self._n_predict))
        padded[:, self._n_lookback:] = data[:, :, 0]
        return self._scaler.inverse_transform(padded)[:, self._n_lookback:]

