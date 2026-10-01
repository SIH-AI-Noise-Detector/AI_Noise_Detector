import torch
import soundfile as sf

from model import DCCRNBasic


N_FFT = 512
HOP_LENGTH = 128
SAMPLE_RATE = 16000

MODEL_PATH = "models/dccrn_basic.pth"
INPUT_AUDIO = "data/processed/noisy/sample_0000_snr-5.wav"
OUTPUT_AUDIO = "outputs/demo_enhanced.wav"

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("====================================")
print("AI NOISE CANCELLATION DEMO")
print("====================================")
print("Device:", device)

print("\nLoading model...")

model = DCCRNBasic().to(device)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

model.load_state_dict(checkpoint)
model.eval()

print("Model loaded successfully.")

print("\nLoading noisy audio...")

audio, sr = sf.read(
    INPUT_AUDIO,
    dtype="float32"
)

if sr != SAMPLE_RATE:
    raise ValueError(
        f"Expected 16 kHz audio, got {sr} Hz"
    )

audio = torch.from_numpy(audio).float().to(device)

print("Applying STFT...")

window = torch.hann_window(
    N_FFT,
    device=device
)

stft = torch.stft(
    audio,
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
    window=window,
    return_complex=True
)

model_input = torch.stack(
    [
        stft.real,
        stft.imag
    ],
    dim=0
)

model_input = model_input.unsqueeze(0)

print("Running AI noise suppression...")

with torch.no_grad():
    enhanced = model(model_input)

enhanced_complex = torch.complex(
    enhanced[:, 0],
    enhanced[:, 1]
)

print("Reconstructing enhanced audio...")

enhanced_audio = torch.istft(
    enhanced_complex,
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
    window=window,
    length=len(audio)
)

enhanced_audio = (
    enhanced_audio
    .squeeze(0)
    .cpu()
    .numpy()
)

sf.write(
    OUTPUT_AUDIO,
    enhanced_audio,
    SAMPLE_RATE
)

print("\n====================================")
print("DEMO COMPLETE")
print("====================================")
print("Input :", INPUT_AUDIO)
print("Output:", OUTPUT_AUDIO)
print("Status: SUCCESS")
