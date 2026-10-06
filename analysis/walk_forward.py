"""Leakage-free, walk-forward Random Forest prediction for Vietlott games."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from analyzer import BASE_DIR, GAME_CONFIGS, load_draws
from analysis.reporting import export_json

INITIAL_TRAINING_DRAWS = 200
FEATURE_WARMUP_DRAWS = 50
FEATURE_WINDOWS = (10, 30, 100)
MODEL_PARAMETERS = {
    "n_estimators": 16,
    "max_depth": 8,
    "min_samples_leaf": 3,
    "max_features": "sqrt",
    "random_state": 42,
    "n_jobs": 1,
}

PREDICTION_FIELDS = [
    "target_position",
    "target_draw_number",
    "target_draw_datetime",
    "training_draw_count",
    "predicted_numbers",
    "actual_numbers",
    "hits",
]
TRAINING_FIELDS = [
    "phase",
    "target_position",
    "target_draw_number",
    "training_draw_count",
    "first_training_draw",
    "last_training_draw",
    "first_supervised_target",
    "last_supervised_target",
    "training_examples",
    "trees",
    "random_state",
]


def _load_draws(game_type: str, data_dir: Path) -> list[dict[str, Any]]:
    if game_type not in GAME_CONFIGS:
        raise ValueError(f"Unknown game type: {game_type}")
    _, max_number, main_count, filename = GAME_CONFIGS[game_type]
    draws, warnings = load_draws(data_dir / filename, game_type, main_count, max_number)
    if warnings:
        raise ValueError(
            f"Invalid history for {game_type}: " + "; ".join(warnings[:10])
        )
    if len(draws) < INITIAL_TRAINING_DRAWS:
        raise ValueError(
            f"{game_type} needs at least {INITIAL_TRAINING_DRAWS} valid draws; "
            f"found {len(draws)}."
        )
    return draws


def _encode_draws(
    draws: list[dict[str, Any]], max_number: int
) -> list[list[int]]:
    return [
        [int(number in draw["numbers"]) for number in range(1, max_number + 1)]
        for draw in draws
    ]


def _feature_rows(
    draws: list[dict[str, Any]], max_number: int
) -> list[list[float]]:
    """Build features for every draw position using only earlier draws."""
    encoded = _encode_draws(draws, max_number)
    last_seen = [-1] * max_number
    rows: list[list[float]] = []

    for index in range(len(draws) + 1):
        history = encoded[:index]
        features: list[float] = list(history[-1]) if history else [0.0] * max_number
        for window in FEATURE_WINDOWS:
            recent = history[-window:]
            denominator = max(1, len(recent))
            features.extend(
                sum(row[number] for row in recent) / denominator
                for number in range(max_number)
            )
        features.extend(
            min(index - previous if previous >= 0 else index + 1, 100) / 100
            for previous in last_seen
        )
        rows.append(features)

        if index < len(draws):
            for number, present in enumerate(encoded[index]):
                if present:
                    last_seen[number] = index
    return rows


def _training_matrices(
    draws: list[dict[str, Any]], max_number: int, target_index: int
) -> tuple[list[list[float]], list[list[int]]]:
    """Return supervised rows whose outcomes are all known before target_index."""
    if target_index < INITIAL_TRAINING_DRAWS or target_index > len(draws):
        raise ValueError("target_index must follow at least 200 available draws.")
    features = _feature_rows(draws[:target_index], max_number)
    encoded = _encode_draws(draws[:target_index], max_number)
    return (
        features[FEATURE_WARMUP_DRAWS:target_index],
        encoded[FEATURE_WARMUP_DRAWS:target_index],
    )


def _new_model() -> Any:
    try:
        from sklearn.ensemble import RandomForestClassifier
    except ImportError as exc:
        raise RuntimeError(
            "scikit-learn is required for walk-forward prediction."
        ) from exc
    return RandomForestClassifier(**MODEL_PARAMETERS)


def _positive_scores(model: Any, features: list[float], max_number: int) -> list[float]:
    probabilities = model.predict_proba([features])
    scores: list[float] = []
    for index in range(max_number):
        classes = model.classes_[index]
        positive_class = next(
            (class_index for class_index, value in enumerate(classes) if value == 1),
            None,
        )
        scores.append(
            float(probabilities[index][0][positive_class])
            if positive_class is not None
            else 0.0
        )
    return scores


def _rank_numbers(scores: list[float], main_count: int) -> list[int]:
    return sorted(
        range(1, len(scores) + 1),
        key=lambda number: (-scores[number - 1], number),
    )[:main_count]


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _training_record(
    phase: str, target_index: int, draws: list[dict[str, Any]]
) -> dict[str, Any]:
    training_count = target_index
    return {
        "phase": phase,
        "target_position": target_index + 1,
        "target_draw_number": (
            draws[target_index]["draw_number"] if target_index < len(draws) else ""
        ),
        "training_draw_count": training_count,
        "first_training_draw": draws[0]["draw_number"],
        "last_training_draw": draws[target_index - 1]["draw_number"],
        "first_supervised_target": draws[FEATURE_WARMUP_DRAWS]["draw_number"],
        "last_supervised_target": draws[target_index - 1]["draw_number"],
        "training_examples": target_index - FEATURE_WARMUP_DRAWS,
        "trees": MODEL_PARAMETERS["n_estimators"],
        "random_state": MODEL_PARAMETERS["random_state"],
    }


def _score(predicted: list[int], actual: list[int]) -> int:
    return len(set(predicted).intersection(actual))


def train_and_evaluate_game(
    game_type: str,
    data_dir: Path = BASE_DIR / "data",
    output_dir: Path = BASE_DIR / "results" / "walk_forward",
) -> dict[str, Any]:
    """Evaluate all draws after the 200-draw warm-up, then save a final model."""
    name, max_number, main_count, _ = GAME_CONFIGS[game_type]
    draws = _load_draws(game_type, data_dir)
    features = _feature_rows(draws, max_number)
    encoded = _encode_draws(draws, max_number)
    prediction_rows: list[dict[str, Any]] = []
    training_rows: list[dict[str, Any]] = []

    for target_index in range(INITIAL_TRAINING_DRAWS, len(draws)):
        model = _new_model()
        model.fit(
            features[FEATURE_WARMUP_DRAWS:target_index],
            encoded[FEATURE_WARMUP_DRAWS:target_index],
        )
        predicted_scores = _positive_scores(
            model, features[target_index], max_number
        )
        predicted = _rank_numbers(predicted_scores, main_count)
        actual = draws[target_index]["numbers"]
        prediction_rows.append(
            {
                "target_position": target_index + 1,
                "target_draw_number": draws[target_index]["draw_number"],
                "target_draw_datetime": draws[target_index]["draw_datetime"],
                "training_draw_count": target_index,
                "predicted_numbers": " ".join(map(str, predicted)),
                "actual_numbers": " ".join(map(str, actual)),
                "hits": _score(predicted, actual),
            }
        )
        training_rows.append(_training_record("walk_forward", target_index, draws))

    final_model = _new_model()
    final_model.fit(
        features[FEATURE_WARMUP_DRAWS:len(draws)],
        encoded[FEATURE_WARMUP_DRAWS:len(draws)],
    )
    training_rows.append(_training_record("next_draw_model", len(draws), draws))

    game_output = output_dir / game_type
    game_output.mkdir(parents=True, exist_ok=True)
    artifact = {
        "game_type": game_type,
        "model": final_model,
        "max_number": max_number,
        "main_count": main_count,
        "training_draws": len(draws),
        "latest_draw_number": draws[-1]["draw_number"],
        "latest_draw_datetime": draws[-1]["draw_datetime"],
    }
    try:
        import joblib
    except ImportError as exc:
        raise RuntimeError("joblib is required to save the trained model.") from exc
    joblib.dump(artifact, game_output / "model.joblib", compress=3)

    _write_csv(game_output / "walk_forward.csv", PREDICTION_FIELDS, prediction_rows)
    _write_csv(game_output / "training_history.csv", TRAINING_FIELDS, training_rows)

    hit_counts = {f"{hits}/{main_count}": 0 for hits in range(main_count + 1)}
    for row in prediction_rows:
        hit_counts[f"{row['hits']}/{main_count}"] += 1
    evaluated = len(prediction_rows)
    performance = {
        "game_type": game_type,
        "game_name": name,
        "model": "RandomForestClassifier (multi-output)",
        "model_parameters": MODEL_PARAMETERS,
        "initial_training_draws": INITIAL_TRAINING_DRAWS,
        "first_evaluated_position": INITIAL_TRAINING_DRAWS + 1,
        "historical_draws": len(draws),
        "evaluation_draws": evaluated,
        "hit_count_distribution": hit_counts,
        "hit_count_rates": {
            label: (count / evaluated if evaluated else None)
            for label, count in hit_counts.items()
        },
        "average_hits": (
            sum(row["hits"] for row in prediction_rows) / evaluated
            if evaluated
            else None
        ),
        "random_baseline_expected_hits": main_count * main_count / max_number,
        "latest_training_draw": {
            "draw_number": draws[-1]["draw_number"],
            "draw_datetime": draws[-1]["draw_datetime"],
        },
        "warnings": [
            "Lottery draws are random; these experimental scores do not guarantee future results.",
            "Predictions use main numbers only; special numbers are excluded.",
        ],
    }
    export_json(performance, game_output / "performance.json")
    next_prediction = predict_next_draw(game_type, data_dir, output_dir)
    export_json(next_prediction, game_output / "next_prediction.json")
    return performance


def predict_next_draw(
    game_type: str,
    data_dir: Path = BASE_DIR / "data",
    output_dir: Path = BASE_DIR / "results" / "walk_forward",
) -> dict[str, Any]:
    """Use the latest saved model to rank numbers for the next draw."""
    import joblib

    draws = _load_draws(game_type, data_dir)
    game_output = output_dir / game_type
    artifact = joblib.load(game_output / "model.joblib")
    if (
        artifact["training_draws"] != len(draws)
        or artifact["latest_draw_number"] != draws[-1]["draw_number"]
        or artifact["latest_draw_datetime"] != draws[-1]["draw_datetime"]
    ):
        raise RuntimeError(
            f"The saved {game_type} model is stale; run walk-forward training again."
        )
    features = _feature_rows(draws, artifact["max_number"])
    scores = _positive_scores(
        artifact["model"], features[-1], artifact["max_number"]
    )
    return {
        "game_type": game_type,
        "predicted_draw_position": len(draws) + 1,
        "based_on_draw_number": draws[-1]["draw_number"],
        "based_on_draw_datetime": draws[-1]["draw_datetime"],
        "training_draws": len(draws),
        "predicted_numbers": _rank_numbers(scores, artifact["main_count"]),
        "number_scores": {
            str(number): round(score, 6)
            for number, score in enumerate(scores, 1)
        },
        "warning": "Experimental ranking only; lottery results are random.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game", choices=[*GAME_CONFIGS, "all"], default="all")
    parser.add_argument("--data-dir", type=Path, default=BASE_DIR / "data")
    parser.add_argument(
        "--output-dir", type=Path, default=BASE_DIR / "results" / "walk_forward"
    )
    parser.add_argument(
        "--predict-only",
        action="store_true",
        help="Load the latest saved model instead of retraining.",
    )
    args = parser.parse_args()
    games = GAME_CONFIGS if args.game == "all" else {args.game: GAME_CONFIGS[args.game]}
    for game_type in games:
        if args.predict_only:
            result = predict_next_draw(game_type, args.data_dir, args.output_dir)
            print(f"{game_type}: {result['predicted_numbers']}")
        else:
            result = train_and_evaluate_game(
                game_type, args.data_dir, args.output_dir
            )
            print(
                f"{game_type}: evaluated {result['evaluation_draws']} draws; "
                f"average hits={result['average_hits']}"
            )


if __name__ == "__main__":
    main()
