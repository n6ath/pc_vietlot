# Walk-forward Vietlott model

The local walk-forward predictor is separate from the existing analyzer. It uses
the existing CSV histories and a separate multi-output Random Forest model for
each of 5/35, 6/45, and 6/55. Only main numbers are modeled and scored.

From the first 200 chronological draws, it predicts draw 201. For every later
draw, it fits on labeled examples available strictly before that draw, records
the selected numbers and actual result, then advances the training cutoff by
one draw. The saved next-draw model is retrained on the complete available
history. Feature rows contain prior-draw indicators, rolling frequencies, and
time since each number last appeared; no feature uses the target or a future
draw.

Install scikit-learn if it is not already available, then run from the project
root:

```powershell
python -m analysis.walk_forward
```

Choose one game with `--game 5_35`, `--game 6_45`, or `--game 6_55`. Outputs
are written under `results/walk_forward/<game>/`: `walk_forward.csv` contains
each prediction, actual result, and hit count; `training_history.csv` records
the training cutoff for each fit; `performance.json` contains aggregate hit
counts and rates; `model.joblib` stores the final model; and
`next_prediction.json` stores its prediction for the next draw.

Load the saved model without retraining from the command line with
`python -m analysis.walk_forward --game 6_45 --predict-only`, or call
`analysis.walk_forward.predict_next_draw("6_45")` from Python. A saved model
whose history no longer matches the CSV is rejected; retrain before using it.
Results are experimental rankings, not guarantees of lottery outcomes.

## Strategy benchmark

Run `python -m analysis.walk_forward_benchmark` to compare seeded random
combinations, historical-frequency ranking, longest-gap ranking, and the
existing Random Forest on the same targets from draw position 201 onward.
Outputs are written to `results/walk_forward/benchmark/`: the Markdown report,
machine-readable JSON (including exact baseline distributions and tests), and
a per-target CSV of predictions and hit counts. Use `--game 5_35`,
`--game 6_45`, or `--game 6_55` to benchmark one game.

The benchmark tests cumulative hits against the exact hypergeometric random
baseline and adjusts the four per-game p-values with Holm-Bonferroni. A
significant two-sided difference can indicate worse as well as better results;
it is not proof of future predictive performance.
