import torch
import soundfile as sf
from pathlib import Path

from model import DCCRNBasic


# -----------------------------
# Device
# -----------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# -----------------------------
# Load model
# -----------------------------

model = DCCRNBasic().to(device)

model.load_state_dict(
    torch.load(
        "models/dccrn_basic.pth",
        map_location=device
    )
)

model.eval()

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


# -----------------------------
# Input/output
# -----------------------------

input_file = sorted(
    Path("data/processed/noisy").glob("*.wav")
)[0]

output_file = Path(
    "outputs/enhanced.wav"
)

output_file.parent.mkdir(
    parents=True,
    exist_ok=True
)


# -----------------------------
# Load audio
# -----------------------------

audio, sample_rate = sf.read(
    input_file,
    dtype="float32"
)

audio = torch.from_numpy(audio).float()

print("Input:", input_file)
print("Sample rate:", sample_rate)
print("Samples:", len(audio))


# -----------------------------
# STFT
# -----------------------------

audio_gpu = audio.unsqueeze(0).to(device)

stft = torch.stft(
    audio_gpu,
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
    window=window,
    return_complex=True
)

model_input = torch.stack(
    [stft.real, stft.imag],
    dim=1
)


# -----------------------------
# Model inference
# -----------------------------

with torch.no_grad():

    enhanced = model(model_input)


# -----------------------------
# Convert back to complex
# -----------------------------

enhanced_complex = torch.complex(
    enhanced[:, 0],
    enhanced[:, 1]
)


# -----------------------------
# ISTFT
# -----------------------------

enhanced_audio = torch.istft(
    enhanced_complex,
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
    window=window,
    length=len(audio)
)


# -----------------------------
# Save
# -----------------------------

enhanced_audio = (
    enhanced_audio
    .squeeze(0)
    .cpu()
    .numpy()
)

sf.write(
    output_file,
    enhanced_audio,
    sample_rate
)

print("\nEnhanced audio saved to:")
print(output_file)