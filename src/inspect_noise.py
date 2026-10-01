import pandas as pd

csv_path = "data/noise/ESC-50-master/meta/esc50.csv"

df = pd.read_csv(csv_path)

print("Number of recordings:", len(df))

print("\nAvailable sound categories:")
print(df["category"].unique())