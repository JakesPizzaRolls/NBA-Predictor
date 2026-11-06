import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn

class Net(nn.Module):
    def __init__(self, n, dropout=0.25):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(n, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.layers(x).squeeze(-1)

def predict(data):
    root = Path(__file__).resolve().parent

    prep = joblib.load(root / "nn_preprocessing.joblib")
    logistic = joblib.load(root / "logistic.joblib")

    features = prep["features"]

    missing = [c for c in features if c not in data.columns]
    if missing:
        raise ValueError(
            f"Missing {len(missing)} features: {missing[:10]}"
        )

    X = data[features].copy()

    for col in features:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    if X.isna().all(axis=1).any():
        raise ValueError("At least one row has no usable features")

    Xnn = prep["scaler"].transform(
        prep["imputer"].transform(X)
    ).astype("float32")

    nn_probs = []

    for seed in [1, 2, 3]:
        checkpoint = torch.load(
            root / f"nn_seed_{seed}.pt",
            map_location="cpu",
            weights_only=True
        )

        if checkpoint["features"] != features:
            raise ValueError("Model feature mismatch")

        model = Net(checkpoint["input_dim"])
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()

        with torch.no_grad():
            p = torch.sigmoid(
                model(torch.from_numpy(Xnn))
            ).numpy()

        nn_probs.append(p)

    p_nn = np.mean(nn_probs, axis=0)
    p_logistic = logistic.predict_proba(X)[:, 1]

    p_final = 0.5 * p_nn + 0.5 * p_logistic

    result = data[
        [c for c in ["game_id", "date", "home_team_id",
                     "away_team_id"] if c in data.columns]
    ].copy()

    result["home_win_probability"] = p_final
    result["away_win_probability"] = 1 - p_final
    result["predicted_winner"] = np.where(
        p_final >= 0.5, "HOME", "AWAY"
    )
    result["confidence"] = np.maximum(p_final, 1 - p_final)

    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="predictions.csv")
    args = parser.parse_args()

    data = pd.read_csv(args.input)
    result = predict(data)
    result.to_csv(args.output, index=False)

    print(result.to_string(index=False))
    print(f"\nSaved {args.output}")
