import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

csv_paths = [
    Path("eval_results/policy_vs_zarr_eval.csv"),
    Path("eval_results_2/policy_vs_zarr_eval.csv"),
]

dfs = []
for path in csv_paths:
    if path.exists():
        df = pd.read_csv(path)
        df["source"] = path.parent.name
        dfs.append(df)

df = pd.concat(dfs, ignore_index=True)

out = Path("eval_threshold_analysis")
out.mkdir(exist_ok=True)

# Histograms
for metric in ["mae", "rmse", "l2", "max_abs"]:
    plt.figure(figsize=(10, 6))
    for source, sub in df.groupby("source"):
        plt.hist(sub[metric], bins=50, alpha=0.5, label=source)
    plt.xlabel(metric)
    plt.ylabel("Count")
    plt.title(f"{metric.upper()} Distribution")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(out / f"{metric}_histogram.png", dpi=150, bbox_inches="tight")
    plt.close()

# Threshold pass/fail table
thresholds = [0.006, 0.007, 0.008, 0.009, 0.01]
rows = []

for source, sub in df.groupby("source"):
    for t in thresholds:
        good = int((sub["mae"] <= t).sum())
        bad = int((sub["mae"] > t).sum())
        total = len(sub)

        rows.append({
            "source": source,
            "mae_threshold": t,
            "good_predictions": good,
            "bad_predictions": bad,
            "pass_rate": good / total,
            "total": total,
        })

threshold_df = pd.DataFrame(rows)
threshold_df.to_csv(out / "threshold_pass_fail.csv", index=False)

print(threshold_df)
print(f"\nSaved plots and table to: {out}")