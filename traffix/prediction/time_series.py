import os
import numpy as np
import torch
import torch.nn as nn

from traffix.utils.time_series_utils import DataLoader



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


    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
                 input_size: int = 1,
                 output_size: int = 24,
                 hidden_size: int = 8,
                 num_stacked_layers: int = 1,
                 n_lookback: int = 672,
                 n_predict: int = 24) -> None:

        self._device = "cuda:0" if torch.cuda.is_available() else "cpu"

        print(f"Predictor device: {self._device}")

        self._output_size = output_size

        self._lstm = LSTM(input_size,
                          output_size,
                          hidden_size,
                          num_stacked_layers,
                          self._device)
        
        self._lstm.to(self._device)

        print("Predictor loaded")

        
    def train_model(self,
                    model_filename: os.PathLike,
                    train_loader: DataLoader,
                    test_loader: DataLoader,
                    learning_rate: float = 0.004,
                    num_epochs: int = 30) -> tuple:
        
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
                torch.save(self._lstm.state_dict(), model_filename)
                best_val_loss = avg_loss_across_batches

            print(f"Val loss: {avg_loss_across_batches}")
            print("************************************")
            print()
        
        print("Predictor trained model")

        return train_losses, validation_losses
    

    def load_model(self, model_file:os.PathLike) -> None:
        self._lstm.load_state_dict(torch.load(model_file, map_location=self._device))
        self._lstm.to(self._device)
        self._lstm.eval()
        
        print("Predictor loaded model specs from file")


    def predict(self, input_data: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            predicted = self._lstm(input_data.to(self._device)).to("cpu").numpy()

        return predicted
