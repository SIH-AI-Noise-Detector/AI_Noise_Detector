import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import NoiseDataset
from model import DCCRNBasic


# -----------------------------
# Device
# -----------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# -----------------------------
# Dataset
# -----------------------------

train_dataset = NoiseDataset(
    "data/processed/splits/train.csv"
)

validation_dataset = NoiseDataset(
    "data/processed/splits/validation.csv"
)


# -----------------------------
# DataLoader
# -----------------------------

train_loader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True,
    num_workers=0
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=4,
    shuffle=False,
    num_workers=0
)


# -----------------------------
# Model
# -----------------------------

model = DCCRNBasic().to(device)

print("Model loaded.")


# -----------------------------
# STFT settings
# -----------------------------

N_FFT = 512
HOP_LENGTH = 128

window = torch.hann_window(
    N_FFT,
    device=device
)


def make_stft(audio):

    stft = torch.stft(
        audio.squeeze(1),
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        window=window,
        return_complex=True
    )

    # Real + imaginary channels
    stft = torch.stack(
        [stft.real, stft.imag],
        dim=1
    )

    return stft


# -----------------------------
# Loss and optimizer
# -----------------------------

criterion = nn.L1Loss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


# -----------------------------
# Training
# -----------------------------

EPOCHS = 5

for epoch in range(EPOCHS):

    model.train()

    total_loss = 0.0

    for batch_no, (noisy, clean) in enumerate(train_loader):

        noisy = noisy.to(device)
        clean = clean.to(device)

        # Convert audio to STFT
        noisy_stft = make_stft(noisy)
        clean_stft = make_stft(clean)

        # Model prediction
        predicted_stft = model(noisy_stft)

        # Compare predicted clean STFT with actual clean STFT
        loss = criterion(
            predicted_stft,
            clean_stft
        )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        if (batch_no + 1) % 20 == 0:
            print(
                f"Epoch [{epoch + 1}/{EPOCHS}] "
                f"Batch [{batch_no + 1}/{len(train_loader)}] "
                f"Loss: {loss.item():.4f}"
            )

    average_loss = total_loss / len(train_loader)

    print(
        f"\nEpoch {epoch + 1} complete "
        f"| Average Loss: {average_loss:.4f}\n"
    )


# -----------------------------
# Save model
# -----------------------------

torch.save(
    model.state_dict(),
    "models/dccrn_basic.pth"
)

print("Training complete.")
print("Model saved to models/dccrn_basic.pth")