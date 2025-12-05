import pandas as pd

# Load YOLO training log
df = pd.read_csv("runs/detect/train5/results.csv")

# Identify best epoch by highest mAP50-95
best_idx = df["metrics/mAP50-95(B)"].idxmax()
best_row = df.loc[best_idx]

print("Best Epoch:")
print(f"Epoch: {int(best_row['epoch'])}")
print(f"mAP50-95: {best_row['metrics/mAP50-95(B)']}")
print(f"mAP50: {best_row['metrics/mAP50(B)']}")
print(f"Precision: {best_row['metrics/precision(B)']}")
print(f"Recall: {best_row['metrics/recall(B)']}")
