from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, roc_auc_score,
    log_loss, brier_score_loss
)

folder = Path("model_comparison")
rows = []

for year in [2019, 2020, 2021]:
    predictions = {}

    for architecture in ["large_nn", "small_nn"]:
        frames = []

        for seed in [1, 2, 3]:
            path = folder / (
                f"{architecture}_{year}_seed{seed}.predictions.csv"
            )

            df = pd.read_csv(path)
            df = df[["game_id", "home_win", "pred_home_win"]]
            df = df.set_index("game_id").sort_index()
            frames.append(df)

        index = frames[0].index

        if any(not f.index.equals(index) for f in frames):
            raise ValueError("Game IDs do not match")

        y = frames[0]["home_win"].to_numpy()

        if any(
            not np.array_equal(y, f["home_win"].to_numpy())
            for f in frames
        ):
            raise ValueError("Labels do not match")

        predictions[architecture] = np.mean(
            [f["pred_home_win"].to_numpy() for f in frames],
            axis=0
        )

    combinations = {
        "large_ensemble": predictions["large_nn"],
        "small_ensemble": predictions["small_nn"],
        "combined_ensemble": (
            predictions["large_nn"] + predictions["small_nn"]
        ) / 2,
    }

    for name, prob in combinations.items():
        rows.append({
            "season": year,
            "model": name,
            "accuracy": accuracy_score(y, prob >= 0.5),
            "roc_auc": roc_auc_score(y, prob),
            "log_loss": log_loss(y, prob),
            "brier": brier_score_loss(y, prob),
        })

result = pd.DataFrame(rows)

print("\n=== ENSEMBLE RESULTS ===")
print(result.to_string(index=False))

print("\n=== AVERAGES ===")
print(
    result.groupby("model")[
        ["accuracy", "roc_auc", "log_loss", "brier"]
    ].mean().to_string()
)
