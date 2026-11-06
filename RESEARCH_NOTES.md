# NBA full-information neural network

A runnable research prototype for the **entire** `archive (7).zip` NBA dataset, including historical game data and streaming extraction from the 2.26 GB play-by-play CSV. **Elo resets to 1500 at the start of each season** (no offseason carryover). Game histories and player histories can carry over, but only observations dated before tipoff are used.

## Install on macOS

```bash
cd nba_full_information
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run on full archive

```bash
python build_features.py --data "$HOME/Downloads/archive (7).zip" --out features.csv --cache player_events.sqlite
python train.py --features features.csv --out nba_full_model.pt
```

**First command may take a long time** because it reads billions of bytes of compressed play-by-play. The SQLite cache is reused on subsequent runs. Disk space: allow multiple GB for cache and intermediates. The script is CPU-compatible on a Mac.

### Quick smoke test with the small ZIP (no PBP)

```bash
python build_features.py --data nba_small.zip --out quick_features.csv --no-pbp
python train.py --features quick_features.csv --out quick_model.pt
```

The quick test intentionally has **no player data**. Full training requires the original ZIP containing `csv/play_by_play.csv`.

## Inputs

- `game.csv`: all available historical games and team box score metrics, rolling 5/10/20 games, efficiency, win rates, rest and back-to-back, past-only Elo.
- `play_by_play.csv`: streaming event-based approximate player per-game scoring, shooting, assists, rebounds, turnovers, participation, and historical roster proxy; aggregated only from previous games. This is *not* a complete, validated official player box-score reconstruction.
- Game season type as contextual feature.

`common_player_info.csv`, `draft_history.csv`, `draft_combine_stats.csv`, `team_info_common.csv`, and `team_details.csv` are **not directly included** because many are current snapshots, incomplete historical records, or contain fields not safe to join to past games without time-aware validation. `inactive_players.csv` is excluded because actual pregame publication timing is not known. `game_info`, `game_summary`, `line_score`, `other_stats`, and `officials` contain postgame or otherwise unknown-at-tipoff data and are excluded from same-game inputs. This is **all usable verified pregame information**, not literally every field in the archive.

## Validation

Train: 1946–47 through 2020–21; validation: 2021–22; test: 2022–23, with no random game split. Change `--min-season` to `1946` to train on the **full historical range** (older data has many missing fields, imputed from training only). Default begins in 1996 because play-by-play coverage begins later than team game history. All available seasons are still processed for history. `--validation-season` and `--test-season` are configurable.

Model: MLP 256→128→64→1, GELU, LayerNorm, dropout, AdamW, BCEWithLogitsLoss, early stopping on validation loss. Outputs `.pt`, `.preprocessing.joblib`, `.metrics.json`, `.predictions.csv`.

## Caveats

- Historical rosters are inferred from player events in prior games (45-day lookback), not verified active rosters or injury reports. Players who did not record a tracked event can be missed. Traded players may appear on their previous team until updated events arrive. This is a proxy, not a reliable real-world future game roster.
- Event descriptions and player2 roles vary over NBA eras. Derived player stats are approximate and require comparison to official box scores before serious use.
- `play_by_play.csv` is assumed to be grouped by game ID for streaming aggregation. If game IDs are interleaved, the PBP cache needs a different upsert aggregation strategy.
- A single season-wide Elo reset is used; if season IDs are unexpectedly nonchronological, verify reset order. Elo updates are applied only after all games on the same date are featured.
- Not all archived games are regular-season games; playoff/preseason games may be included, depending on the archive. The season-type feature distinguishes them.
- No real test metric is claimed for the **full PBP model** until it runs on your Mac with the original archive.
