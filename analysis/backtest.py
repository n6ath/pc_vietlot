"""Chronological backtesting without future-data access."""
from typing import Any

def run(draws: list[dict[str, Any]], max_number: int, main_count: int, min_history: int = 60) -> dict[str, Any]:
    if len(draws) <= min_history:
        return {"status": "insufficient_history", "random_baseline": None, "statistical_model": None, "random_forest": None, "ensemble": None, "explanation": "Không đủ điểm đánh giá walk-forward."}
    evaluated = len(draws) - min_history
    baseline_hits = main_count * main_count / max_number
    return {"status": "evaluated", "evaluation_draws": evaluated,
            "random_baseline": {"average_hits": baseline_hits},
            "statistical_model": {"average_hits": None, "note": "Cần candidate policy cố định để tính."},
            "random_forest": {"average_hits": None, "note": "Metrics được để null khi chưa có walk-forward predictions."},
            "ensemble": {"average_hits": None, "note": "Metrics được để null khi chưa có walk-forward predictions."}}
