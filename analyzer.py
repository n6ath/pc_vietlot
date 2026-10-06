"""Orchestrate modular Vietlott analysis and export stable artifacts."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
from typing import Any

from analysis import (
    backtest,
    clustering,
    covering_design,
    ensemble,
    entropy,
    frequency,
    gap,
    genetic_algorithm,
    kelly,
    pair,
    random_forest,
    rolling,
    scoring,
    spatial_pattern,
    structure,
    wheel_system,
)
from analysis.reporting import JSON_WARNINGS, export_reports

BASE_DIR = Path(__file__).resolve().parent
GAME_CONFIGS = {
    "5_35": ("Lotto 5/35", 35, 5, "lotto535.csv"),
    "6_45": ("Mega 6/45", 45, 6, "mega645.csv"),
    "6_55": ("Power 6/55", 55, 6, "power655.csv"),
}

def load_draws(path: Path, game_type: str, main_count: int, max_number: int) -> tuple[list[dict[str, Any]], list[str]]:
    """Validate schema and history without modifying source data."""
    required = {"id", "game_type", "draw_number", "draw_datetime", *(f"n{i}" for i in range(1, main_count + 1))}
    if not path.exists():
        return [], [f"Missing CSV: {path}"]
    draws, warnings, ids, keys = [], [], set(), set()
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = required - set(reader.fieldnames or [])
        if missing:
            return [], [f"Missing required columns: {sorted(missing)}"]
        previous = ""
        for line, row in enumerate(reader, 2):
            key = (row["game_type"], row["draw_number"])
            values = [row.get(f"n{i}", "") for i in range(1, main_count + 1)]
            try:
                numbers = [int(v) for v in values]
                datetime.strptime(row["draw_datetime"], "%Y-%m-%d %H:%M:%S")
                if row["game_type"] != game_type or any(n < 1 or n > max_number for n in numbers) or len(set(numbers)) != main_count:
                    raise ValueError
            except (TypeError, ValueError):
                warnings.append(f"Invalid draw row {line}")
                continue
            if row["id"] in ids or key in keys:
                warnings.append(f"Duplicate id/draw at row {line}")
                continue
            if previous and row["draw_datetime"] < previous:
                warnings.append(f"Non-chronological row {line}")
            previous = row["draw_datetime"]
            ids.add(row["id"]); keys.add(key)
            draws.append({"draw_number": row["draw_number"], "draw_datetime": row["draw_datetime"], "numbers": sorted(numbers)})
    draws.sort(key=lambda d: (d["draw_datetime"], d["draw_number"]))
    return draws, warnings

def analyze_game(game_type: str, data_dir: Path, results_dir: Path, top: int) -> dict[str, Any]:
    name, max_number, main_count, filename = GAME_CONFIGS[game_type]
    draws, validation_warnings = load_draws(data_dir / filename, game_type, main_count, max_number)
    freq = frequency.analyze(draws, max_number)
    gaps = gap.analyze(draws, max_number)
    pairs = pair.analyze(draws)
    structures = structure.analyze(draws, max_number)
    rolling_result = rolling.analyze(draws, max_number)
    statistical = scoring.number_scores(freq, gaps)
    rf = random_forest.analyze(draws, max_number, main_count)
    combined = ensemble.combine(statistical, rf)
    candidate_scores = combined["number_scores"]
    candidates = scoring.candidates(candidate_scores, main_count, max_number, pairs, top=top)
    for item in candidates:
        item["ml_score"] = round(sum(rf["scores"].get(str(n), 0) for n in item["numbers"]) / main_count, 6)
        item["score"] = round(.7 * item["statistical_score"] + .3 * item["ml_score"], 6)
    entropy_result = entropy.analyze(draws, max_number)
    spatial_result = spatial_pattern.analyze(draws, max_number)
    ga_result = genetic_algorithm.optimize(
        max_number=max_number,
        main_count=main_count,
        number_scores=candidate_scores,
        pair_strength=pairs.get("pair_strength"),
        entropy_scores=entropy_result["entropy_per_number"],
        component_scores={
            number: {
                "frequency": item["frequency"],
                "gap": item["gap"],
                "ml": rf["scores"].get(number, 0.0),
            }
            for number, item in statistical.items()
        },
    )
    selected_numbers = sorted(
        (int(number) for number in candidate_scores),
        key=lambda number: (-candidate_scores[str(number)], number),
    )[: max(main_count, min(12, max_number))]
    wheel_result = wheel_system.analyze(selected_numbers, main_count, mode="abbreviated")
    covering_result = covering_design.generate(selected_numbers, main_count, mode="full", max_tickets=1000)
    kelly_result = kelly.calculate(
        bankroll=0.0 if False else 1.0,
        ticket_price=1.0,
        user_defined_probability=None,
        expected_return=None,
    )
    bt = backtest.run(draws, max_number, main_count)
    result = {
        "game_type": game_type, "game_name_vi": name,
        "analysis_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "historical_draws": len(draws),
        "date_range": {"first": draws[0]["draw_datetime"] if draws else None, "last": draws[-1]["draw_datetime"] if draws else None},
        "latest_draw_number": draws[-1]["draw_number"] if draws else None,
        "statistics": {**freq, **gaps, **pairs, **structures, "rolling": rolling_result},
        "machine_learning": {"random_forest": rf, **clustering.analyze(draws, max_number)},
        "ensemble": combined, "predictions": candidates, "backtest": bt,
        "entropy": entropy_result,
        "spatial_pattern": spatial_result,
        "genetic_algorithm": ga_result,
        "covering_design": covering_result,
        "wheel_system": wheel_result,
        "kelly": kelly_result,
        "warnings": JSON_WARNINGS + validation_warnings,
    }
    export_reports(result, results_dir / game_type)
    return result

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game", choices=[*GAME_CONFIGS, "all"], default="all")
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--data-dir", type=Path, default=BASE_DIR / "data")
    parser.add_argument("--results-dir", type=Path, default=BASE_DIR / "results")
    args = parser.parse_args()
    games = GAME_CONFIGS if args.game == "all" else {args.game: GAME_CONFIGS[args.game]}
    for game_type in games:
        result = analyze_game(game_type, args.data_dir, args.results_dir, max(1, args.top))
        print(f"{game_type}: {result['historical_draws']} draws -> {args.results_dir / game_type}")

if __name__ == "__main__":
    main()
