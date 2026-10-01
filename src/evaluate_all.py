import torch
import soundfile as sf
import numpy as np
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
# SNR function
# -----------------------------

def calculate_snr(clean, signal):

    noise = clean - signal

    signal_power = np.mean(clean ** 2)
    noise_power = np.mean(noise ** 2)

    if noise_power == 0:
        return float("inf")

    return 10 * np.log10(
        signal_power / noise_power
    )


# -----------------------------
# Test files
# -----------------------------

test_file = Path(
    "data/processed/splits/test.csv"
)

import csv

pairs = []

with open(test_file, "r") as f:

    reader = csv.DictReader(f)

    for row in reader:
        pairs.append(
            (row["noisy"], row["clean"])
        )


print("Test samples:", len(pairs))


# -----------------------------
# Evaluate
# -----------------------------

noisy_snrs = []
enhanced_snrs = []
improvements = []


for i, (noisy_path, clean_path) in enumerate(pairs):

    clean, clean_sr = sf.read(
        clean_path,
        dtype="float32"
    )

    noisy, noisy_sr = sf.read(
        noisy_path,
        dtype="float32"
    )

    # Use 2-second segment
    length = min(
        len(clean),
        len(noisy)
    )

    clean = clean[:length]
    noisy = noisy[:length]

    segment_length = 32000

    if length >= segment_length:

        clean = clean[:segment_length]
        noisy = noisy[:segment_length]

    else:

        padding = segment_length - length

        clean = np.pad(
            clean,
            (0, padding)
        )

        noisy = np.pad(
            noisy,
            (0, padding)
        )


    # -------------------------
    # STFT
    # -------------------------

    noisy_tensor = torch.from_numpy(
        noisy
    ).float().unsqueeze(0).to(device)

    stft = torch.stft(
        noisy_tensor,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        window=window,
        return_complex=True
    )

    model_input = torch.stack(
        [stft.real, stft.imag],
        dim=1
    )


    # -------------------------
    # DCCRN
    # -------------------------

    with torch.no_grad():

        enhanced = model(
            model_input
        )


    # -------------------------
    # Convert back
    # -------------------------

    enhanced_complex = torch.complex(
        enhanced[:, 0],
        enhanced[:, 1]
    )


    enhanced_audio = torch.istft(
        enhanced_complex,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        window=window,
        length=segment_length
    )


    enhanced_audio = (
        enhanced_audio
        .squeeze(0)
        .cpu()
        .numpy()
    )


    # -------------------------
    # Calculate SNR
    # -------------------------

    noisy_snr = calculate_snr(
        clean,
        noisy
    )

    enhanced_snr = calculate_snr(
        clean,
        enhanced_audio
    )

    improvement = (
        enhanced_snr - noisy_snr
    )

    noisy_snrs.append(noisy_snr)
    enhanced_snrs.append(enhanced_snr)
    improvements.append(improvement)


    if (i + 1) % 10 == 0:

        print(
            f"Processed "
            f"{i + 1}/{len(pairs)}"
        )


# -----------------------------
# Final results
# -----------------------------

print("\n========== TEST RESULTS ==========")

print(
    f"Average noisy SNR     : "
    f"{np.mean(noisy_snrs):.2f} dB"
)

print(
    f"Average enhanced SNR  : "
    f"{np.mean(enhanced_snrs):.2f} dB"
)

print(
    f"Average SNR improvement: "
    f"{np.mean(improvements):.2f} dB"
)

print(
    f"Best improvement      : "
    f"{np.max(improvements):.2f} dB"
)

print(
    f"Worst improvement     : "
    f"{np.min(improvements):.2f} dB"
)

print("==================================")