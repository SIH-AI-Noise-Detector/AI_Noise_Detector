import csv
from pathlib import Path

import numpy as np
import soundfile as sf
import torch


N_FFT = 512
HOP_LENGTH = 128
SEGMENT_LENGTH = 32000

OUTPUT_DIR = Path("data/calibration_npy")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

window = torch.hann_window(N_FFT)

with open("data/processed/splits/train.csv", "r") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

rows = rows[:100]

for i, row in enumerate(rows):

    audio_path = row["noisy"].replace("\\", "/")

    audio, sr = sf.read(
        audio_path,
        dtype="float32"
    )

    if sr != 16000:
        raise ValueError(
            f"Wrong sample rate: {sr}"
        )

    audio = torch.from_numpy(audio).float()

    if len(audio) >= SEGMENT_LENGTH:
        audio = audio[:SEGMENT_LENGTH]
    else:
        padding = SEGMENT_LENGTH - len(audio)

        audio = torch.nn.functional.pad(
            audio,
            (0, padding)
        )

    stft = torch.stft(
        audio,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        window=window,
        return_complex=True
    )

    stft = torch.stack(
        [stft.real, stft.imag],
        dim=0
    )

    stft = stft.unsqueeze(0)

    output_file = OUTPUT_DIR / f"calib_{i:03d}.npy"

    np.save(
        output_file,
        stft.numpy().astype(np.float32)
    )

print(
    "Created calibration tensors:",
    len(rows)
)

print(
    "Shape:",
    tuple(stft.shape)
)

print(
    "Saved to:",
    OUTPUT_DIR
)
