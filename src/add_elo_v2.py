import argparse
import zipfile
from collections import defaultdict

import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--features", default="features.csv")
parser.add_argument("--archive", default="archive (7).zip")
parser.add_argument("--output", default="features_regular_v2.csv")
parser.add_argument("--regular-only", action="store_true")
args = parser.parse_args()

# Load existing pregame features
features = pd.read_csv(args.features, low_memory=False)

# Load historical final scores
with zipfile.ZipFile(args.archive) as z:
    games = pd.read_csv(
        z.open("csv/game.csv"),
        usecols=["game_id", "pts_home", "pts_away"],
        low_memory=False
    )

games = games.drop_duplicates("game_id")

df = features.merge(
    games,
    on="game_id",
    how="left",
    validate="one_to_one"
)

if df[["pts_home", "pts_away"]].isna().any().any():
    raise ValueError("Missing game scores. Cannot calculate Elo.")

df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["date", "game_id"])

ratings = defaultdict(lambda: 1500.0)
last_season = {}

results = []

for date, day in df.groupby("date", sort=True):
    updates = []

    for row in day.itertuples():
        home = int(row.home_team_id)
        away = int(row.away_team_id)
        season = int(row.season)

        for team in (home, away):
            previous = last_season.get(team)

            if previous is None:
                last_season[team] = season
            elif season > previous:
                ratings[team] = (
                    1500 + 0.75 * (ratings[team] - 1500)
                )
                last_season[team] = season
            elif season < previous:
                raise ValueError("Unexpected season ordering")

        home_elo = ratings[home]
        away_elo = ratings[away]

        expected = 1 / (
            1 + 10 ** (
                -(home_elo + 70 - away_elo) / 400
            )
        )

        margin = float(row.pts_home - row.pts_away)

        actual = (
            1.0 if margin > 0
            else 0.0 if margin < 0
            else 0.5
        )

        mov_multiplier = min(
            2.5,
            1 + 0.35 * np.log1p(abs(margin))
        )

        change = 20 * mov_multiplier * (actual - expected)

        results.append({
            "game_id": row.game_id,
            "elo_v2_home": home_elo,
            "elo_v2_away": away_elo,
            "elo_v2_diff": home_elo - away_elo,
            "elo_v2_home_prob": expected
        })

        updates.append((home, away, change))

    # Apply results only after all same-day features
    # have been calculated.
    for home, away, change in updates:
        ratings[home] += change
        ratings[away] -= change

elo = pd.DataFrame(results)

output = features.merge(
    elo,
    on="game_id",
    how="left",
    validate="one_to_one"
)

if args.regular_only:
    output = output[
        output["game_type_code"] == 2
    ]

output.to_csv(args.output, index=False)

print("Success!")
print("Games:", len(output))
print("New features:", [
    "elo_v2_home",
    "elo_v2_away",
    "elo_v2_diff",
    "elo_v2_home_prob"
])
print("Saved:", args.output)
