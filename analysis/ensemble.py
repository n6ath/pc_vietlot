"""Configurable ensemble combination."""
from typing import Any

def combine(statistical: dict[str, Any], ml: dict[str, Any], weights: dict[str, float] | None = None) -> dict[str, Any]:
    weights = weights or {"statistical": .7, "ml": .3}
    scores = {}
    for number, item in statistical.items():
        scores[number] = round(weights["statistical"] * item["score"] + weights["ml"] * ml.get("scores", {}).get(number, 0.0), 6)
    return {"number_scores": scores, "weights": weights, "warning": "Trọng số là cấu hình thử nghiệm, chưa được chứng minh tối ưu."}
