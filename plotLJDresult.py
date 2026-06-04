import pandas as pd
import matplotlib.pyplot as plt

csv_path = "runs/detect/train7/results.csv"


df = pd.read_csv(csv_path)


epochs = df["epoch"]
map50 = df["metrics/mAP50(B)"]
map5095 = df["metrics/mAP50-95(B)"]


plt.figure(figsize=(8, 5))

plt.plot(epochs, map50, label="mAP50", linewidth=2)
plt.plot(epochs, map5095, label="mAP50-95", linewidth=2)

plt.xlabel("Epoch")
plt.ylabel("Performance")
plt.title("Performance vs Epoch")
plt.legend()
plt.grid(True)

plt.savefig("performance_curve.png", dpi=300)

plt.show()