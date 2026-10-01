import csv
from pathlib import Path

OUTPUT = Path("data/calibration_dataset.txt")
OUTPUT.parent.mkdir(parents=True, exist_ok=True)

with open("data/processed/splits/train.csv", "r") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

# Use 100 representative training samples
rows = rows[:100]

with open(OUTPUT, "w") as f:
    for row in rows:
        f.write(str(Path(row["noisy"]).resolve()) + "\n")

print("Calibration samples:", len(rows))
print("Saved to:", OUTPUT)
