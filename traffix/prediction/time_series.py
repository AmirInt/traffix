import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from operator import index
from copy import deepcopy
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset
from torch.utils.data import DataLoader




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


    def prepare_dataset_for_lstm(self) -> None:

        self._shifted_df = deepcopy(self._df)

        self._shifted_df.set_index(self._index_title, inplace=True)

        for i in range(1, self._n_predict):
            self._shifted_df[self._vehicle_count_title + f"(t+{i})"] = self._shifted_df[self._vehicle_count_title].shift(-i)

        for i in range(self._n_lookback, 0, -1):
            self._shifted_df[self._vehicle_count_title + f"(t-{i})"] = self._shifted_df[self._vehicle_count_title].shift(i)

        self._shifted_df.dropna(inplace=True)


    def scale_data(self) -> None:
        self._data_np = self._shifted_df.to_numpy()
        self._scaler = MinMaxScaler(feature_range=(-1, 1))
        self._data_np = self._scaler.fit_transform(self._data_np)


    def tensor_data(self) -> None:
        self._data_tn = torch.tensor(self._data_np).float()


    def get_train_datasets(self,
                           batch_size: int,
                           train_ratio: float) -> tuple:

        split_index = train_ratio * len(self._data_tn)
        X_train = self._data_tn[:split_index, self._n_lookback:, :]
        y_train = self._data_tn[:split_index, :self._n_lookback, :]
        X_test = self._data_tn[split_index:, self._n_lookback:, :]
        y_test = self._data_tn[split_index:, :self._n_lookback, :]

        train_dataset = TimeSeriesDataset(X_train, y_train)
        test_dataset = TimeSeriesDataset(X_test, y_test)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, test_loader

    
    def get_data(self, index: int) -> torch.Tensor:
        return self._data_tn[index:index + 1, :, :]


    def reverse_scale_data(self, data: np.ndarray) -> np.ndarray:
        padded = np.zeros((data.shape[0], self._n_lookback + self._n_predict))
        padded[:, self._n_lookback:] = data[:, :, 0]
        return self._scaler.inverse_transform(padded)[:, self._n_lookback:]



class LSTM(nn.Module):
    def __init__(self,
                 input_size: int,
                 output_size: int,
                 hidden_size: int,
                 num_stacked_layers: int,
                 device: str) -> None:
        
        super().__init__()
        self.hidden_size = hidden_size
        self.num_stacked_layers = num_stacked_layers

        self.lstm = nn.LSTM(input_size,
                            hidden_size,
                            num_stacked_layers,
                            batch_first=True)

        self.fc = nn.Linear(hidden_size, output_size)

        self.device = device


    def forward(self, x) -> torch.Tensor:
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_stacked_layers,
                         batch_size,
                         self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_stacked_layers,
                         batch_size,
                         self.hidden_size).to(self.device)

        out, _ = self.lstm(x, (h0, c0))

        out = self.fc(out[:, -1, :])

        out = out.reshape((*out.shape, 1))

        return out




class Predictor:
    def __init__(self,
                 data_csv_file: os.PathLike,
                 model_filename: os.PathLike,
                 input_size: int = 1,
                 output_size: int = 24,
                 hidden_size: int = 8,
                 num_stacked_layers: int = 1,
                 n_lookback: int = 672,
                 n_predict: int = 24,
                 index_title: str = "index",
                 vehicle_count_title: str = "vehicle_count") -> None:

        self._device = "cuda:0" if torch.cuda.is_available() else "cpu"

        self._model_filename = model_filename

        self._output_size = output_size

        self._n_lookback = n_lookback
        self._n_predict = n_predict

        self._lstm = LSTM(input_size,
                          output_size,
                          hidden_size,
                          num_stacked_layers,
                          self._device)
        
        self._lstm.to(self._device)

        # Getting and preprocessing data
        self._data_processor = DataProcessor(data_csv_file,
                                             n_lookback,
                                             n_predict,
                                             index_title,
                                             vehicle_count_title)

        self._data_processor.prepare_dataset_for_lstm()
        self._data_processor.scale_data()
        self._data_processor.tensor_data()

        


    def train_model(self,
                    batch_size: int = 512,
                    train_ratio: float = 0.9,
                    learning_rate: float = 0.004,
                    num_epochs: int = 30) -> tuple:
        
        train_loader, test_loader = self._data_processor.get_train_datasets(
            batch_size,
            train_ratio)
        
        # Set training parameters
        loss_function = nn.MSELoss()
        optimiser = torch.optim.Adam(self._lstm.parameters(), lr=learning_rate)
        best_val_loss = 100.0

        train_losses = []
        validation_losses = []

        for epoch in range(num_epochs):
            # Train
            self._lstm.train(True)
            running_loss = 0.0
            print(f"Epoch: {epoch + 1}")

            for batch in train_loader:
                x_batch, y_batch = batch[0].to(self._device), batch[1].to(self._device)
                output = self._lstm(x_batch)
                loss = loss_function(output, y_batch)
                running_loss += loss.item()
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()

            print(f"Train loss: {running_loss / len(train_loader)}")

            train_losses.append(running_loss / len(train_loader))

            # Validate
            self._lstm.train(False)
            running_loss = 0.0

            for batch in test_loader:
                x_batch, y_batch = batch[0].to(self._device), batch[1].to(self._device)

                with torch.no_grad():
                    output = self._lstm(x_batch)
                    loss = loss_function(output, y_batch)
                    running_loss += loss.item()

            avg_loss_across_batches = running_loss / len(test_loader)

            validation_losses.append(avg_loss_across_batches)

            if avg_loss_across_batches < best_val_loss:
                torch.save(self._lstm.state_dict(), self._model_filename)
                best_val_loss = avg_loss_across_batches

            print(f"Val loss: {avg_loss_across_batches}")
            print("************************************")
            print()
        
        return train_losses, validation_losses
    
    def load_model(self, model_file:os.PathLike) -> None:
        self._lstm.load_state_dict(torch.load(model_file, map_location=self._device))
        self._lstm.to(self._device)
        self._lstm.eval()


    def predict(self, index: int) -> np.ndarray:
        with torch.no_grad():
            predicted = self._lstm(
                self._data_processor.get_data(index).to(self._device)).to("cpu").numpy()

        return self._data_processor.reverse_scale_data(predicted)