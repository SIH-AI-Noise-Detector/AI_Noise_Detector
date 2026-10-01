import numpy as np
import soundfile as sf
from pathlib import Path


CLEAN_DIR = Path("data/processed/clean")
NOISY_DIR = Path("data/processed/noisy")
ENHANCED_FILE = Path("outputs/enhanced.wav")


# Use the same first sample used by enhance.py
noisy_file = sorted(NOISY_DIR.glob("*.wav"))[0]
clean_file = sorted(CLEAN_DIR.glob("*.wav"))[0]


# Load audio
clean, sr_clean = sf.read(clean_file, dtype="float32")
noisy, sr_noisy = sf.read(noisy_file, dtype="float32")
enhanced, sr_enhanced = sf.read(ENHANCED_FILE, dtype="float32")


# Make lengths equal
length = min(
    len(clean),
    len(noisy),
    len(enhanced)
)

clean = clean[:length]
noisy = noisy[:length]
enhanced = enhanced[:length]


def calculate_snr(clean, signal):

    noise = clean - signal

    signal_power = np.mean(clean ** 2)
    noise_power = np.mean(noise ** 2)

    if noise_power == 0:
        return float("inf")

    return 10 * np.log10(
        signal_power / noise_power
    )


noisy_snr = calculate_snr(
    clean,
    noisy
)

enhanced_snr = calculate_snr(
    clean,
    enhanced
)

improvement = enhanced_snr - noisy_snr


print("\n========== RESULTS ==========")

print(f"Original noisy SNR   : {noisy_snr:.2f} dB")
print(f"Enhanced speech SNR  : {enhanced_snr:.2f} dB")
print(f"SNR improvement      : {improvement:.2f} dB")

print("=============================")