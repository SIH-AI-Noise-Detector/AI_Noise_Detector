import torch
from model import DCCRNBasic


# Create fake complex STFT data
batch_size = 1
frequency_bins = 257
time_frames = 126

# 2 channels:
# channel 0 = real part
# channel 1 = imaginary part

x = torch.randn(
    batch_size,
    2,
    frequency_bins,
    time_frames
)


# Create model
model = DCCRNBasic()

# Run the model
output = model(x)


print("Input shape:", x.shape)
print("Output shape:", output.shape)