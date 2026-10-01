from pathlib import Path
import random
import csv

CLEAN_DIR = Path("data/processed/clean")
NOISY_DIR = Path("data/processed/noisy")

OUTPUT_DIR = Path("data/processed/splits")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

clean_files = sorted(CLEAN_DIR.glob("*.wav"))
noisy_files = sorted(NOISY_DIR.glob("*.wav"))

if len(clean_files) != len(noisy_files):
    raise ValueError(
        f"Mismatch: {len(clean_files)} clean files and "
        f"{len(noisy_files)} noisy files"
    )

pairs = list(zip(noisy_files, clean_files))

random.seed(42)
random.shuffle(pairs)

total = len(pairs)

train_end = int(total * 0.8)
val_end = int(total * 0.9)

train_pairs = pairs[:train_end]
val_pairs = pairs[train_end:val_end]
test_pairs = pairs[val_end:]

def save_split(name, data):

    output_file = OUTPUT_DIR / f"{name}.csv"

    with open(output_file, "w", newline="") as f:

        writer = csv.writer(f)

        writer.writerow(["noisy", "clean"])

        for noisy, clean in data:
            writer.writerow([str(noisy), str(clean)])

    print(f"{name}: {len(data)} pairs")


save_split("train", train_pairs)
save_split("validation", val_pairs)
save_split("test", test_pairs)

print("\nDataset split complete.")