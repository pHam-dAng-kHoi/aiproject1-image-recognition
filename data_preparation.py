from sklearn.model_selection import train_test_split
import pandas as pd
import os

images = sorted(os.listdir("data/images"))
labels = sorted(os.listdir("data/labels"))

data_df = pd.DataFrame(columns=["image_path", "label_path"])

data_df = pd.DataFrame({
    "image_path": [f"data/images/{img}" for img in images],
    "label_path": [f"data/labels/{lbl}" for lbl in labels]
})

train_df, val_df = train_test_split(data_df, test_size=0.3, random_state=42)

data_df.to_csv("data/CSVs/dataset.csv", index=False)
train_df.to_csv("data/CSVs/train_df.csv", index=False)
val_df.to_csv("data/CSVs/val_df.csv", index=False)

print(data_df)
