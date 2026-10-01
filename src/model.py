import torch
import torch.nn as nn


class DCCRNBasic(nn.Module):

    def __init__(self):
        super().__init__()

        # CNN encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        # Recurrent layer
        self.gru = nn.GRU(
            input_size=32 * 257,
            hidden_size=256,
            num_layers=1,
            batch_first=True
        )

        # Convert GRU output back to frequency representation
        self.rnn_projection = nn.Linear(
            256,
            32 * 257
        )

        # CNN decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.Conv2d(16, 2, kernel_size=3, padding=1)
        )

    def forward(self, x):

        # x:
        # [batch, 2, frequency, time]

        x = self.encoder(x)

        # Current shape:
        # [batch, 32, frequency, time]

        batch, channels, frequency, time = x.shape

        # Move time to sequence dimension
        x = x.permute(0, 3, 1, 2)

        # [batch, time, channels, frequency]
        x = x.reshape(
            batch,
            time,
            channels * frequency
        )

        # GRU
        x, _ = self.gru(x)

        # Project back
        x = self.rnn_projection(x)

        # Restore CNN shape
        x = x.reshape(
            batch,
            time,
            channels,
            frequency
        )

        x = x.permute(0, 2, 3, 1)

        # Decoder
        x = self.decoder(x)

        return x