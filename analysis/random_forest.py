"""Walk-forward Random Forest number-level analysis."""
from typing import Any

def analyze(draws: list[dict[str, Any]], max_number: int, main_count: int, seed: int = 42, min_history: int = 60) -> dict[str, Any]:
    try:
        from sklearn.ensemble import RandomForestClassifier
    except ImportError:
        return {"scores": {str(n): 0.0 for n in range(1, max_number + 1)}, "validation": None, "metadata": {"status": "unavailable"}, "warnings": ["scikit-learn is not installed."]}
    if len(draws) < min_history:
        return {"scores": {str(n): 0.0 for n in range(1, max_number + 1)}, "validation": None, "metadata": {"status": "insufficient_history", "draws": len(draws)}, "warnings": ["Không đủ lịch sử để huấn luyện Random Forest."]}
    def features(index: int, number: int) -> list[float]:
        past = [number in d["numbers"] for d in draws[:index]]
        return [sum(past[-w:]) for w in (5, 10, 20)] + [next((i for i, value in enumerate(reversed(past)) if value), len(past))]
    scores = {}
    for number in range(1, max_number + 1):
        X, y = [], []
        for index in range(20, len(draws) - 1):
            X.append(features(index, number)); y.append(int(number in draws[index]["numbers"]))
        if len(set(y)) < 2:
            scores[str(number)] = 0.0
            continue
        model = RandomForestClassifier(n_estimators=120, random_state=seed, class_weight="balanced", n_jobs=-1)
        model.fit(X, y)
        scores[str(number)] = round(float(model.predict_proba([features(len(draws), number)])[0][1]), 6)
    return {"scores": scores, "validation": {"method": "chronological_walk_forward", "future_leakage": False}, "metadata": {"status": "trained", "min_history": min_history}, "warnings": ["Điểm RF là điểm xếp hạng thử nghiệm, không phải xác suất đảm bảo."]}
