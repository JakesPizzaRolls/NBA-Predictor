import json
import random
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn

from train import Net

ROOT = Path("nba_predictor_v1")
ROOT.mkdir(exist_ok=True)

df = pd.read_csv("features_elo_v2_only.csv", low_memory=False)
df = df.sort_values(["date", "game_id"]).reset_index(drop=True)

excluded = {
    "game_id", "date", "season", "home_win",
    "home_team_id", "away_team_id"
}

features = [
    c for c in df.columns
    if c not in excluded
    and pd.api.types.is_numeric_dtype(df[c])
    and df.loc[df.season < 2022, c].notna().any()
]

assert len(features) == 177, (
    f"Expected 177 features; found {len(features)}"
)
assert df.season.max() == 2022

train = df[df.season < 2022]
validation = df[df.season == 2022]

print("=== FINAL MODEL TRAINING ===", flush=True)
print("Total games:", len(df), flush=True)
print("Features:", len(features), flush=True)

# Determine NN training duration using the latest season.
imputer = SimpleImputer(
    strategy="median", keep_empty_features=True
)
scaler = StandardScaler()

X_train = scaler.fit_transform(
    imputer.fit_transform(train[features])
).astype("float32")

X_val = scaler.transform(
    imputer.transform(validation[features])
).astype("float32")

y_train = train.home_win.to_numpy(dtype="float32")
y_val = validation.home_win.to_numpy(dtype="float32")

def fit_nn(X, y, seed, epochs, Xv=None, yv=None):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    model = Net(X.shape[1])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.0007,
        weight_decay=0.01
    )
    criterion = nn.BCEWithLogitsLoss()

    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.from_numpy(X),
            torch.from_numpy(y)
        ),
        batch_size=256,
        shuffle=True
    )

    best_loss = float("inf")
    best_epoch = 1
    stale = 0

    for epoch in range(epochs):
        model.train()

        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(
                model.parameters(), 1.0
            )
            optimizer.step()

        if Xv is not None:
            model.eval()
            with torch.no_grad():
                val_loss = criterion(
                    model(torch.from_numpy(Xv)),
                    torch.from_numpy(yv)
                ).item()

            if val_loss < best_loss - 1e-5:
                best_loss = val_loss
                best_epoch = epoch + 1
                stale = 0
            else:
                stale += 1

            if stale >= 12:
                break

    return model, best_epoch

# Use 2022 only to choose training duration.
# Then refit from scratch on ALL available games.
selected_epochs = {}

for seed in [1, 2, 3]:
    _, best_epoch = fit_nn(
        X_train, y_train, seed, 80,
        X_train if False else X_val, y_val
    )
    selected_epochs[seed] = best_epoch
    print(
        f"Seed {seed}: selected {best_epoch} epochs",
        flush=True
    )

# Final preprocessing is fitted on all available games.
final_imputer = SimpleImputer(
    strategy="median", keep_empty_features=True
)
final_scaler = StandardScaler()

X_all = final_scaler.fit_transform(
    final_imputer.fit_transform(df[features])
).astype("float32")

y_all = df.home_win.to_numpy(dtype="float32")

joblib.dump(
    {
        "imputer": final_imputer,
        "scaler": final_scaler,
        "features": features
    },
    ROOT / "nn_preprocessing.joblib"
)

for seed, epochs in selected_epochs.items():
    model, _ = fit_nn(
        X_all, y_all, seed, epochs
    )

    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dim": len(features),
            "features": features,
            "seed": seed,
            "epochs": epochs
        },
        ROOT / f"nn_seed_{seed}.pt"
    )
    print(f"Saved neural network seed {seed}", flush=True)

# Logistic regression: fit on all available games.
logistic = make_pipeline(
    SimpleImputer(
        strategy="median", keep_empty_features=True
    ),
    StandardScaler(),
    LogisticRegression(
        C=0.1,
        max_iter=2000
    )
)

logistic.fit(df[features], df.home_win)
joblib.dump(logistic, ROOT / "logistic.joblib")

metadata = {
    "version": "1.0",
    "architecture": "50% logistic + 50% mean of 3 NNs",
    "feature_count": len(features),
    "training_games": len(df),
    "last_training_game": str(df.date.max()),
    "nn_epochs": selected_epochs,
    "historical_accuracy": 0.64272,
    "historical_log_loss": 0.63334,
    "note": (
        "Historical metrics describe earlier rolling-season "
        "experiments, not this refitted production model."
    )
}

(ROOT / "metadata.json").write_text(
    json.dumps(metadata, indent=2)
)

shutil.copy2("train.py", ROOT / "train_architecture.py")
shutil.copy2(
    "ensemble_results.csv",
    ROOT / "ensemble_results.csv"
)
shutil.copy2(
    "xgboost_comparison.csv",
    ROOT / "xgboost_comparison.csv"
)

print("\n=== PRODUCTION MODEL SAVED ===")
print(json.dumps(metadata, indent=2))
