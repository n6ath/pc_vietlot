# Predictor review

## Scope

`predictor.py` trains one Random Forest classifier per main number and ranks numbers for the next draw. It is useful as an exploratory ranking component, but it is not a ticket-probability model.

## Conceptual weaknesses

- Main-number models are trained independently, so they do not model the fixed-size combination constraint or dependencies between numbers.
- `predict_proba` is an empirical classifier score, not a calibrated lottery probability. The output must be described as an experimental ranking score.
- RF-2 samples from a ranked pool; it is ticket diversification, not a second independently validated forecast.
- Historical lottery draws are designed to be independent. Frequency, gap, pair, and RF signals may describe past variation without improving future odds.

## Leakage and evaluation risks

- Full-history prediction without a chronological holdout is in-sample ranking and cannot demonstrate forward performance.
- Any feature that includes the target draw, future draws, or statistics calculated after the target timestamp would leak information.
- A fair evaluation must use walk-forward training: train only on draws before each target draw, then score the next draw.
- The dedicated `analysis/random_forest.py` layer records chronological validation metadata and avoids future feature values.

## Overfitting risks

- Many trees, feature choices, class weighting, and repeated candidate selection can overfit a finite history.
- Searching weights or ticket combinations against the same history used for evaluation inflates results.
- Metrics should be reported with sample counts and compared with a random baseline; null is preferable to invented metrics.

## Data and game assumptions

- The analyzer's game-specific CSV validation controls number count and ranges.
- `special_number` is a separate field. It may equal a main number under the current game contract and must not be rejected for that reason.
- The predictor's generic fallback reader is convenient for legacy CSV layouts, but production analysis should use the validated crawler schema.

## Performance

- A separate classifier per number is computationally more expensive than one shared model.
- `n_jobs=-1` can consume all available CPU cores; deployments may want a configurable worker limit.
- Repeated model fitting during walk-forward backtests can be expensive. Cache features or use a bounded evaluation interval for interactive runs.

## Recommendations

1. Treat RF values as ranking features only.
2. Keep chronological walk-forward evaluation separate from final full-history scoring.
3. Report baseline, sample count, and uncertainty for every metric.
4. Use the modular analysis layer so RF can later be replaced without changing the C# JSON contract.

python predictor.py ^
--csv data\lotto535.csv ^
--main-max 35 ^
--main-n 5

python predictor.py ^
--csv data\mega645.csv ^
--main-max 45 ^
--main-n 6

python predictor.py ^
--csv data\power655.csv ^
--main-max 55 ^
--main-n 6
