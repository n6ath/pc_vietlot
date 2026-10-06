"""Compare walk-forward lottery-number strategies on shared target draws."""

from __future__ import annotations

import argparse
import csv
import math
import random
from pathlib import Path
from typing import Any

from analyzer import BASE_DIR, GAME_CONFIGS
from analysis.reporting import export_json
from analysis.walk_forward import (
    FEATURE_WARMUP_DRAWS,
    INITIAL_TRAINING_DRAWS,
    MODEL_PARAMETERS,
    _encode_draws,
    _feature_rows,
    _load_draws,
    _new_model,
    _positive_scores,
    _rank_numbers,
)

METHODS = (
    ("random", "Random combination"),
    ("frequency", "Historical frequency"),
    ("gap", "Longest-gap / recency"),
    ("random_forest", "Current Random Forest"),
)
RANDOM_SEED = 20260926
PREDICTION_FIELDS = [
    "game_type",
    "target_position",
    "target_draw_number",
    "training_draw_count",
    "method",
    "predicted_numbers",
    "actual_numbers",
    "hits",
]


def _rank_frequency(
    history: list[dict[str, Any]], max_number: int, main_count: int
) -> list[int]:
    frequencies = [0] * max_number
    for draw in history:
        for number in draw["numbers"]:
            frequencies[number - 1] += 1
    return sorted(
        range(1, max_number + 1),
        key=lambda number: (-frequencies[number - 1], number),
    )[:main_count]


def _rank_gap(
    history: list[dict[str, Any]], max_number: int, main_count: int
) -> list[int]:
    last_seen = [-1] * max_number
    for index, draw in enumerate(history):
        for number in draw["numbers"]:
            last_seen[number - 1] = index
    gaps = [
        len(history) - 1 - previous if previous >= 0 else len(history)
        for previous in last_seen
    ]
    return sorted(
        range(1, max_number + 1),
        key=lambda number: (-gaps[number - 1], number),
    )[:main_count]


def _hypergeometric_pmf(population: int, selected: int) -> list[float]:
    denominator = math.comb(population, selected)
    return [
        (
            math.comb(selected, hits)
            * math.comb(population - selected, selected - hits)
            / denominator
        )
        if 0 <= selected - hits <= population - selected
        else 0.0
        for hits in range(selected + 1)
    ]


def _random_total_distribution(
    evaluation_draws: int, population: int, selected: int
) -> list[float]:
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("NumPy is required for exact baseline calculations.") from exc

    per_draw = np.asarray(
        _hypergeometric_pmf(population, selected), dtype=float
    )
    distribution = np.asarray([1.0])
    for _ in range(evaluation_draws):
        distribution = np.convolve(distribution, per_draw)
    return distribution.tolist()


def _two_sided_exact_p(
    observed_total: int, total_distribution: list[float]
) -> float:
    mean = sum(
        total * probability
        for total, probability in enumerate(total_distribution)
    )
    distance = abs(observed_total - mean)
    return min(
        1.0,
        sum(
            probability
            for total, probability in enumerate(total_distribution)
            if abs(total - mean) + 1e-10 >= distance
        ),
    )


def _holm_adjust(p_values: list[float]) -> list[float]:
    order = sorted(range(len(p_values)), key=p_values.__getitem__)
    adjusted = [0.0] * len(p_values)
    running_max = 0.0
    count = len(p_values)
    for rank, index in enumerate(order):
        running_max = max(running_max, (count - rank) * p_values[index])
        adjusted[index] = min(1.0, running_max)
    return adjusted


def _write_predictions(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PREDICTION_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _summarize_method(
    hits: list[int],
    main_count: int,
    random_expected_hits_per_draw: float,
    random_total_distribution: list[float],
) -> dict[str, Any]:
    evaluation_draws = len(hits)
    hit_distribution = {
        f"{count}/{main_count}": hits.count(count)
        for count in range(main_count + 1)
    }
    cumulative_hits = sum(hits)
    expected_random_total_hits = evaluation_draws * random_expected_hits_per_draw
    return {
        "evaluation_draws": evaluation_draws,
        "average_hits_per_draw": cumulative_hits / evaluation_draws,
        "cumulative_total_hits": cumulative_hits,
        "hit_count_distribution": hit_distribution,
        "random_baseline_expected_hits_per_draw": random_expected_hits_per_draw,
        "random_baseline_expected_total_hits": expected_random_total_hits,
        "difference_from_random_expected_total": (
            cumulative_hits - expected_random_total_hits
        ),
        "exact_two_sided_p_value": _two_sided_exact_p(
            cumulative_hits, random_total_distribution
        ),
    }


def benchmark_game(
    game_type: str, data_dir: Path = BASE_DIR / "data"
) -> dict[str, Any]:
    name, max_number, main_count, _ = GAME_CONFIGS[game_type]
    draws = _load_draws(game_type, data_dir)
    if len(draws) <= INITIAL_TRAINING_DRAWS:
        raise ValueError(f"{game_type} has no draws after the 200-draw warm-up.")

    features = _feature_rows(draws, max_number)
    encoded = _encode_draws(draws, max_number)
    random_generator = random.Random(RANDOM_SEED + list(GAME_CONFIGS).index(game_type))
    hits_by_method = {key: [] for key, _ in METHODS}
    prediction_rows: list[dict[str, Any]] = []
    target_draw_numbers: list[str] = []

    for target_index in range(INITIAL_TRAINING_DRAWS, len(draws)):
        history = draws[:target_index]
        actual = set(draws[target_index]["numbers"])

        random_numbers = sorted(random_generator.sample(range(1, max_number + 1), main_count))
        frequency_numbers = _rank_frequency(history, max_number, main_count)
        gap_numbers = _rank_gap(history, max_number, main_count)

        model = _new_model()
        model.fit(
            features[FEATURE_WARMUP_DRAWS:target_index],
            encoded[FEATURE_WARMUP_DRAWS:target_index],
        )
        scores = _positive_scores(model, features[target_index], max_number)
        forest_numbers = _rank_numbers(scores, main_count)

        combinations = {
            "random": random_numbers,
            "frequency": frequency_numbers,
            "gap": gap_numbers,
            "random_forest": forest_numbers,
        }
        target_draw_numbers.append(draws[target_index]["draw_number"])
        for method, numbers in combinations.items():
            method_hits = len(set(numbers).intersection(actual))
            hits_by_method[method].append(method_hits)
            prediction_rows.append(
                {
                    "game_type": game_type,
                    "target_position": target_index + 1,
                    "target_draw_number": draws[target_index]["draw_number"],
                    "training_draw_count": target_index,
                    "method": method,
                    "predicted_numbers": " ".join(map(str, numbers)),
                    "actual_numbers": " ".join(map(str, sorted(actual))),
                    "hits": method_hits,
                }
            )

    evaluation_draws = len(draws) - INITIAL_TRAINING_DRAWS
    random_pmf = _hypergeometric_pmf(max_number, main_count)
    expected_distribution = [
        evaluation_draws * probability for probability in random_pmf
    ]
    random_expected_hits_per_draw = sum(
        hits * probability for hits, probability in enumerate(random_pmf)
    )
    total_distribution = _random_total_distribution(
        evaluation_draws, max_number, main_count
    )
    methods = {
        method: _summarize_method(
            hits_by_method[method],
            main_count,
            random_expected_hits_per_draw,
            total_distribution,
        )
        for method, _ in METHODS
    }
    return {
        "game_type": game_type,
        "game_name": name,
        "population_size": max_number,
        "numbers_per_draw": main_count,
        "initial_training_draws": INITIAL_TRAINING_DRAWS,
        "first_target_position": INITIAL_TRAINING_DRAWS + 1,
        "last_target_position": len(draws),
        "historical_draws": len(draws),
        "evaluation_draws": evaluation_draws,
        "target_draw_numbers": target_draw_numbers,
        "same_targets_for_all_methods": True,
        "random_seed": RANDOM_SEED + list(GAME_CONFIGS).index(game_type),
        "random_hit_count_pmf_per_draw": {
            f"{count}/{main_count}": probability
            for count, probability in enumerate(random_pmf)
        },
        "random_expected_hit_count_distribution": {
            f"{count}/{main_count}": expected
            for count, expected in enumerate(expected_distribution)
        },
        "inference": {
            "test": "Exact two-sided tail probability for cumulative hits under "
            "the convolution of per-draw Hypergeometric(N, k, k) distributions.",
            "multiple_comparison_adjustment": (
                "Holm-Bonferroni across all strategies and games in this run."
            ),
            "interpretation": (
                "Evidence is assessed at adjusted p < 0.05; this is not proof "
                "of future predictive performance."
            ),
        },
        "methods": methods,
        "strategy_definitions": {
            "random": "Uniformly sample k distinct numbers from 1..N using the "
            "recorded fixed seed.",
            "frequency": "Choose the k numbers with the largest counts over "
            "all draws strictly before the target; ties favor lower numbers.",
            "gap": "Choose the k numbers with the longest time since last "
            "appearance over draws strictly before the target; never-seen "
            "numbers have gap equal to history length; ties favor lower numbers.",
            "random_forest": {
                "model": "The existing walk-forward multi-output Random Forest",
                "parameters": MODEL_PARAMETERS,
                "training": "Refit independently for every target, using the "
                "same past-only feature and label cutoff as the existing model.",
            },
        },
        "warnings": [
            "Lottery results are assumed to be independent uniform combinations "
            "under the random-baseline null.",
            "The tests measure historical deviation from that null and do not "
            "establish stable future predictive signal.",
        ],
        "_prediction_rows": prediction_rows,
    }


def _render_report(results: dict[str, Any]) -> str:
    lines = [
        "# Vietlott walk-forward strategy benchmark",
        "",
        "All methods are scored on the same chronological target draws, beginning "
        "at position 201. Each strategy sees only draws before its target. Random "
        "baseline expectations use the exact Hypergeometric(N, k, k) distribution.",
        "",
        "Exact p-values are two-sided tests of cumulative hits against the "
        "convolved per-draw hypergeometric null; Holm-Bonferroni adjustment is "
        "applied across all method/game comparisons in this report.",
        "",
    ]
    for game_type, result in results.items():
        main_count = result["numbers_per_draw"]
        lines += [
            f"## {result['game_name']} ({game_type})",
            "",
            f"Draws: {result['historical_draws']}; evaluation targets: "
            f"{result['first_target_position']}–{result['last_target_position']} "
            f"({result['evaluation_draws']}).",
            "",
            "| Method | Avg hits/draw | Total hits | Difference vs random expected | "
            f"Hit distribution (0/{main_count}–{main_count}/{main_count}) | "
            "Exact p | Holm p |",
            "|---|---:|---:|---:|---|---:|---:|",
        ]
        for method, label in METHODS:
            summary = result["methods"][method]
            distribution = ", ".join(
                f"{hit}: {summary['hit_count_distribution'][f'{hit}/{main_count}']}"
                for hit in range(main_count + 1)
            )
            lines.append(
                f"| {label} | {summary['average_hits_per_draw']:.4f} | "
                f"{summary['cumulative_total_hits']} | "
                f"{summary['difference_from_random_expected_total']:+.2f} | "
                f"{distribution} | "
                f"{summary['exact_two_sided_p_value']:.4f} | "
                f"{summary['holm_adjusted_p_value_across_all_comparisons']:.4f} |"
            )
        baseline = result["random_expected_hit_count_distribution"]
        baseline_distribution = ", ".join(
            f"{hit}: {baseline[f'{hit}/{main_count}']:.2f}"
            for hit in range(main_count + 1)
        )
        lines += [
            "",
            f"Random baseline mean: "
            f"{result['methods']['random']['random_baseline_expected_hits_per_draw']:.4f} "
            f"hits/draw ({result['methods']['random']['random_baseline_expected_total_hits']:.2f} "
            f"over this evaluation). Expected hit distribution (not observed): "
            f"{baseline_distribution}.",
            "",
            "No method is designated predictive unless its Holm-adjusted p-value "
            "is below 0.05; even then, independent confirmation would be needed.",
            "",
        ]
    return "\n".join(lines)


def run_benchmark(
    data_dir: Path = BASE_DIR / "data",
    output_dir: Path = BASE_DIR / "results" / "walk_forward" / "benchmark",
    selected_games: list[str] | None = None,
) -> dict[str, Any]:
    games = selected_games or list(GAME_CONFIGS)
    unknown = set(games) - set(GAME_CONFIGS)
    if unknown:
        raise ValueError(f"Unknown games: {sorted(unknown)}")

    results: dict[str, Any] = {}
    all_predictions: list[dict[str, Any]] = []
    for game_type in games:
        result = benchmark_game(game_type, data_dir)
        all_predictions.extend(result.pop("_prediction_rows"))
        results[game_type] = result

    comparisons = [
        (game_type, method)
        for game_type in games
        for method, _ in METHODS
    ]
    adjusted = _holm_adjust([
        results[game_type]["methods"][method]["exact_two_sided_p_value"]
        for game_type, method in comparisons
    ])
    for (game_type, method), p_value in zip(comparisons, adjusted):
        summary = results[game_type]["methods"][method]
        summary["holm_adjusted_p_value_across_all_comparisons"] = p_value
        summary["evidence_of_difference_from_random_at_0_05"] = p_value < 0.05

    output_dir.mkdir(parents=True, exist_ok=True)
    export_json(
        {
            "benchmark": "walk_forward_strategy_comparison",
            "methods": {key: label for key, label in METHODS},
            "games": results,
        },
        output_dir / "benchmark_results.json",
    )
    _write_predictions(output_dir / "predictions.csv", all_predictions)
    (output_dir / "benchmark_report.md").write_text(
        _render_report(results), encoding="utf-8"
    )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game", choices=[*GAME_CONFIGS, "all"], default="all")
    parser.add_argument("--data-dir", type=Path, default=BASE_DIR / "data")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=BASE_DIR / "results" / "walk_forward" / "benchmark",
    )
    args = parser.parse_args()
    games = list(GAME_CONFIGS) if args.game == "all" else [args.game]
    results = run_benchmark(args.data_dir, args.output_dir, games)
    for game_type, result in results.items():
        print(
            f"{game_type}: {result['evaluation_draws']} targets; "
            + ", ".join(
                f"{label}={result['methods'][key]['average_hits_per_draw']:.4f}"
                for key, label in METHODS
            )
        )
    print(f"Report: {args.output_dir / 'benchmark_report.md'}")


if __name__ == "__main__":
    main()
