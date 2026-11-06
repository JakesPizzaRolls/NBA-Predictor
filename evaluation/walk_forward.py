import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--features", default="features_regular.csv")
parser.add_argument("--start", type=int, default=2018)
parser.add_argument("--end", type=int, default=2021)
args = parser.parse_args()

df = pd.read_csv(args.features, low_memory=False)

if "season" not in df.columns:
    raise ValueError("Dataset needs a season column.")

df["season"] = pd.to_numeric(df["season"], errors="raise").astype(int)

output_dir = Path("walk_forward_results")
output_dir.mkdir(exist_ok=True)

results = []

for val_year in range(args.start, args.end + 1):
    test_year = val_year + 1

    # Remap chronological seasons to the fixed split expected by train.py:
    # train < 2021, validation == 2021, test == 2022.
    subset = df[df["season"] <= test_year].copy()

    train_mask = subset["season"] < val_year
    val_mask = subset["season"] == val_year
    test_mask = subset["season"] == test_year

    if min(train_mask.sum(), val_mask.sum(), test_mask.sum()) == 0:
        print(f"Skipping {val_year}/{test_year}: missing games")
        continue

    subset.loc[train_mask, "season"] = 2020
    subset.loc[val_mask, "season"] = 2021
    subset.loc[test_mask, "season"] = 2022

    data_path = output_dir / f"fold_{val_year}_{test_year}.csv"
    model_path = output_dir / f"model_{val_year}_{test_year}.pt"

    subset.to_csv(data_path, index=False)

    print(f"\n=== Validation {val_year} | Test {test_year} ===", flush=True)

    subprocess.run(
        [
            sys.executable, "train.py",
            "--features", str(data_path),
            "--out", str(model_path)
        ],
        check=True
    )

    metrics_path = model_path.with_suffix(".metrics.json")

    with open(metrics_path) as f:
        metrics = json.load(f)

    results.append({
        "validation_season": val_year,
        "test_season": test_year,
        "accuracy": metrics["test"]["accuracy"],
        "roc_auc": metrics["test"]["roc_auc"],
        "log_loss": metrics["test"]["log_loss"],
        "brier": metrics["test"]["brier"],
        "games": metrics["test_games"]
    })

    pd.DataFrame(results).to_csv(
        output_dir / "summary.csv", index=False
    )

print("\n=== WALK-FORWARD SUMMARY ===")

if results:
    summary = pd.DataFrame(results)
    print(summary.to_string(index=False))

    print("\nAverage metrics:")
    print(summary[
        ["accuracy", "roc_auc", "log_loss", "brier"]
    ].mean().to_string())
else:
    print("No valid folds.")
