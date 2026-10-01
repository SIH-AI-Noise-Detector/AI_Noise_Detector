import random
from pathlib import Path

import librosa
import soundfile as sf
import numpy as np


# -----------------------------
# Paths
# -----------------------------

CLEAN_PATH = Path("data/raw/LibriSpeech/train-clean-5")
NOISE_PATH = Path("data/noise/selected")

OUTPUT_CLEAN = Path("data/processed/clean")
OUTPUT_NOISY = Path("data/processed/noisy")


# -----------------------------
# Settings
# -----------------------------

SAMPLE_RATE = 16000

# Start with only 1000 files for testing
NUM_SAMPLES = 1000

# Different noise strengths
SNR_LEVELS = [-5, 0, 5, 10]


# -----------------------------
# Create output folders
# -----------------------------

OUTPUT_CLEAN.mkdir(parents=True, exist_ok=True)
OUTPUT_NOISY.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Find audio files
# -----------------------------

clean_files = list(CLEAN_PATH.rglob("*.flac"))
noise_files = list(NOISE_PATH.rglob("*.wav"))


print("Clean speech files found:", len(clean_files))
print("Noise files found:", len(noise_files))


if len(clean_files) == 0:
    raise RuntimeError("No clean speech files found.")

if len(noise_files) == 0:
    raise RuntimeError("No noise files found.")


# -----------------------------
# Mixing function
# -----------------------------

def mix_at_snr(clean, noise, snr_db):

    # Make noise the same length as speech
    if len(noise) < len(clean):
        repetitions = int(np.ceil(len(clean) / len(noise)))
        noise = np.tile(noise, repetitions)

    noise = noise[:len(clean)]

    # Calculate power
    clean_power = np.mean(clean ** 2)
    noise_power = np.mean(noise ** 2)

    if noise_power == 0:
        return clean.copy()

    # Required noise power for desired SNR
    target_noise_power = clean_power / (10 ** (snr_db / 10))

    scale = np.sqrt(target_noise_power / noise_power)

    noise = noise * scale

    noisy = clean + noise

    # Prevent clipping
    max_value = np.max(np.abs(noisy))

    if max_value > 0.99:
        noisy = noisy / max_value * 0.99

    return noisy


# -----------------------------
# Generate samples
# -----------------------------

print("\nCreating test dataset...\n")

for i in range(NUM_SAMPLES):

    clean_file = random.choice(clean_files)
    noise_file = random.choice(noise_files)
    snr = random.choice(SNR_LEVELS)

    clean, _ = librosa.load(
        clean_file,
        sr=SAMPLE_RATE,
        mono=True
    )

    noise, _ = librosa.load(
        noise_file,
        sr=SAMPLE_RATE,
        mono=True
    )

    noisy = mix_at_snr(clean, noise, snr)

    filename = f"sample_{i:04d}_snr{snr}.wav"

    clean_output = OUTPUT_CLEAN / filename
    noisy_output = OUTPUT_NOISY / filename

    sf.write(clean_output, clean, SAMPLE_RATE)
    sf.write(noisy_output, noisy, SAMPLE_RATE)

    print(
        f"{i + 1:02d}/20 | "
        f"SNR: {snr:>3} dB | "
        f"Noise: {noise_file.name}"
    )


print("\nDONE.")
print("Clean files:", OUTPUT_CLEAN)
print("Noisy files:", OUTPUT_NOISY)