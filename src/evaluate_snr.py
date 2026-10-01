import csv
from pathlib import Path

import numpy as np
import soundfile as sf
import torch


# =========================
# SETTINGS
# =========================

N_FFT = 512
HOP_LENGTH = 128
SEGMENT_LENGTH = 32000
SAMPLE_RATE = 16000

MODEL_PATH = "models/dccrn_basic.pth"
TEST_CSV = "data/processed/splits/test.csv"

OUTPUT_CSV = "outputs/snr_evaluation.csv"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# =========================
# LOAD MODEL
# =========================

from model import DCCRNBasic


print("Using device:", DEVICE)

model = DCCRNBasic().to(DEVICE)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

model.load_state_dict(checkpoint)

model.eval()

print("Model loaded successfully.")


# =========================
# STFT SETTINGS
# =========================

window = torch.hann_window(
    N_FFT,
    device=DEVICE
)


# =========================
# SNR FUNCTION
# =========================

def calculate_snr(clean, signal):

    noise = clean - signal

    signal_power = np.mean(clean ** 2)
    noise_power = np.mean(noise ** 2)

    if noise_power == 0:
        return float("inf")

    return 10 * np.log10(
        signal_power / noise_power
    )


# =========================
# LOAD TEST DATA
# =========================

with open(TEST_CSV, "r") as f:

    reader = csv.DictReader(f)

    rows = list(reader)


print("Test samples:", len(rows))


# =========================
# OUTPUT DIRECTORY
# =========================

Path("outputs").mkdir(
    exist_ok=True
)


results = []


# =========================
# EVALUATION
# =========================

with torch.no_grad():

    for index, row in enumerate(rows):

        clean_path = row["clean"].replace("\\", "/")
        noisy_path = row["noisy"].replace("\\", "/")

        clean_audio, clean_sr = sf.read(
            clean_path,
            dtype="float32"
        )

        noisy_audio, noisy_sr = sf.read(
            noisy_path,
            dtype="float32"
        )

        if clean_sr != SAMPLE_RATE:
            raise ValueError(
                f"Wrong clean sample rate: {clean_sr}"
            )

        if noisy_sr != SAMPLE_RATE:
            raise ValueError(
                f"Wrong noisy sample rate: {noisy_sr}"
            )

        # Make both exactly 2 seconds
        clean_audio = clean_audio[:SEGMENT_LENGTH]
        noisy_audio = noisy_audio[:SEGMENT_LENGTH]

        if len(clean_audio) < SEGMENT_LENGTH:

            clean_audio = np.pad(
                clean_audio,
                (0, SEGMENT_LENGTH - len(clean_audio))
            )

        if len(noisy_audio) < SEGMENT_LENGTH:

            noisy_audio = np.pad(
                noisy_audio,
                (0, SEGMENT_LENGTH - len(noisy_audio))
            )

        # Convert to torch
        noisy_tensor = torch.from_numpy(
            noisy_audio
        ).float().to(DEVICE)

        # STFT
        noisy_stft = torch.stft(
            noisy_tensor,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            window=window,
            return_complex=True
        )

        # Real + imaginary
        model_input = torch.stack(
            [
                noisy_stft.real,
                noisy_stft.imag
            ],
            dim=0
        )

        # [2, F, T]
        model_input = model_input.unsqueeze(0)

        # Model inference
        enhanced_stft = model(
            model_input
        )

        # Convert model output back to complex
        enhanced_complex = torch.complex(
            enhanced_stft[:, 0],
            enhanced_stft[:, 1]
        )

        # ISTFT
        enhanced_audio = torch.istft(
            enhanced_complex,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            window=window,
            length=SEGMENT_LENGTH
        )

        enhanced_audio = (
            enhanced_audio.squeeze(0)
            .cpu()
            .numpy()
        )

        # Calculate SNRs
        noisy_snr = calculate_snr(
            clean_audio,
            noisy_audio
        )

        enhanced_snr = calculate_snr(
            clean_audio,
            enhanced_audio
        )

        improvement = (
            enhanced_snr - noisy_snr
        )

        results.append(
            {
                "sample": index,
                "noisy_snr": noisy_snr,
                "enhanced_snr": enhanced_snr,
                "snr_improvement": improvement
            }
        )

        print(
            f"[{index + 1}/{len(rows)}] "
            f"Noisy SNR: {noisy_snr:.2f} dB | "
            f"Enhanced SNR: {enhanced_snr:.2f} dB | "
            f"Improvement: {improvement:.2f} dB"
        )


# =========================
# SUMMARY
# =========================

noisy_snrs = np.array(
    [r["noisy_snr"] for r in results]
)

enhanced_snrs = np.array(
    [r["enhanced_snr"] for r in results]
)

improvements = np.array(
    [r["snr_improvement"] for r in results]
)


print("\n==============================")
print("FINAL SNR RESULTS")
print("==============================")

print(
    f"Average noisy SNR: "
    f"{np.mean(noisy_snrs):.2f} dB"
)

print(
    f"Average enhanced SNR: "
    f"{np.mean(enhanced_snrs):.2f} dB"
)

print(
    f"Average SNR improvement: "
    f"{np.mean(improvements):.2f} dB"
)

print(
    f"Best improvement: "
    f"{np.max(improvements):.2f} dB"
)

print(
    f"Worst improvement: "
    f"{np.min(improvements):.2f} dB"
)


# =========================
# SAVE RESULTS
# =========================

with open(
    OUTPUT_CSV,
    "w",
    newline=""
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "sample",
            "noisy_snr",
            "enhanced_snr",
            "snr_improvement"
        ]
    )

    writer.writeheader()

    writer.writerows(results)


print(
    f"\nResults saved to: {OUTPUT_CSV}"
)
