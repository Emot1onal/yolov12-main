import pandas as pd

df = pd.read_csv("runs/detect/exp2_sum_all/results.csv")

best = df.loc[df["metrics/mAP50(B)"].idxmax()]

print(best)