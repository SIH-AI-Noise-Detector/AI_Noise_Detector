from pathlib import Path
import csv
import random

import torch
import soundfile as sf
from torch.utils.data import Dataset


class NoiseDataset(Dataset):

    def __init__(self, csv_file, segment_seconds=2):

        self.csv_file = Path(csv_file)
        self.segment_length = 16000 * segment_seconds

        self.pairs = []

        with open(self.csv_file, "r") as f:
            reader = csv.DictReader(f)

            for row in reader:
                self.pairs.append(
                    (row["noisy"], row["clean"])
                )

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):

        noisy_path, clean_path = self.pairs[index]

        noisy, noisy_sr = sf.read(noisy_path, dtype="float32")
        clean, clean_sr = sf.read(clean_path, dtype="float32")

        if noisy_sr != 16000 or clean_sr != 16000:
            raise ValueError("Audio must be sampled at 16000 Hz")

        noisy = torch.from_numpy(noisy).float()
        clean = torch.from_numpy(clean).float()

        # Make sure both have the same length
        length = min(len(noisy), len(clean))

        noisy = noisy[:length]
        clean = clean[:length]

        # Random 2-second crop
        if length >= self.segment_length:

            start = random.randint(
                0,
                length - self.segment_length
            )

            noisy = noisy[start:start + self.segment_length]
            clean = clean[start:start + self.segment_length]

        else:

            padding = self.segment_length - length

            noisy = torch.nn.functional.pad(
                noisy,
                (0, padding)
            )

            clean = torch.nn.functional.pad(
                clean,
                (0, padding)
            )

        # Add channel dimension
        noisy = noisy.unsqueeze(0)
        clean = clean.unsqueeze(0)

        return noisy, clean


if __name__ == "__main__":

    dataset = NoiseDataset(
        "data/processed/splits/train.csv"
    )

    print("Training samples:", len(dataset))

    noisy, clean = dataset[0]

    print("Noisy shape:", noisy.shape)
    print("Clean shape:", clean.shape)