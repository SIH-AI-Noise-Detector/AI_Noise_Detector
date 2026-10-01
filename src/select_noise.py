import pandas as pd
from pathlib import Path
import shutil


CSV_PATH = Path("data/noise/ESC-50-master/meta/esc50.csv")
AUDIO_PATH = Path("data/noise/ESC-50-master/audio")
OUTPUT_PATH = Path("data/noise/selected")


SELECTED_CATEGORIES = [
    "helicopter",
    "engine",
    "siren",
    "airplane",
    "train",
    "chainsaw",
    "fireworks",
    "car_horn"
]


df = pd.read_csv(CSV_PATH)

selected = df[
    df["category"].isin(SELECTED_CATEGORIES)
]

print("Selected recordings:", len(selected))


for category in SELECTED_CATEGORIES:

    category_files = selected[
        selected["category"] == category
    ]

    category_output = OUTPUT_PATH / category

    category_output.mkdir(
        parents=True,
        exist_ok=True
    )

    for filename in category_files["filename"]:

        source = AUDIO_PATH / filename

        destination = category_output / filename

        shutil.copy2(
            source,
            destination
        )

    print(
        category,
        ":",
        len(category_files),
        "files copied"
    )


print("\nNoise selection complete.")