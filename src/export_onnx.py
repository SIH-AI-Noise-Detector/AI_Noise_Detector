import torch
from model import DCCRNBasic


# -----------------------------
# Device
# -----------------------------

device = torch.device("cpu")


# -----------------------------
# Load trained model
# -----------------------------

model = DCCRNBasic().to(device)

model.load_state_dict(
    torch.load(
        "models/dccrn_basic.pth",
        map_location=device
    )
)

model.eval()

print("Trained model loaded.")


# -----------------------------
# Example input
# -----------------------------

# 2 seconds of audio at 16 kHz
# STFT -> 257 frequency bins
# Approx. 251 time frames

dummy_input = torch.randn(
    1,
    2,
    257,
    251
).to(device)


# -----------------------------
# Export
# -----------------------------

output_file = "models/dccrn_basic.onnx"

torch.onnx.export(
    model,
    dummy_input,
    output_file,
    input_names=["noisy_stft"],
    output_names=["enhanced_stft"],
    opset_version=17,
    dynamo=False
)

print("\nONNX export complete.")
print("Saved to:", output_file)