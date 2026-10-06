"""Stable JSON and Vietnamese presentation exporters."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

JSON_WARNINGS = [
    "Điểm số chỉ mang tính thử nghiệm.",
    "Không có mô hình nào đảm bảo dự đoán chính xác kết quả xổ số.",
]


def _plain(value: Any) -> Any:
    """Convert values to JSON-safe Python primitives and remove non-finite floats."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if value == value and abs(value) != float("inf") else None
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return str(value)


def export_json(result: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_plain(result), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def export_candidates(predictions: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "rank", "numbers", "score", "statistical_score", "ml_score",
        "pair_score", "balance_score", "human_choice_penalty",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in predictions:
            row = dict(item)
            row["numbers"] = " ".join(f"{int(n):02d}" for n in item["numbers"])
            writer.writerow({field: row.get(field, "") for field in fields})


def _lines(result: dict[str, Any]) -> list[str]:
    game = result["game_name_vi"]
    stats = result["statistics"]
    predictions = result["predictions"]
    lines = [
        "=" * 50, "VIETLOT ANALYZER", "=" * 50, "",
        f"Game: {game}",
        f"Số kỳ lịch sử: {result['historical_draws']}",
        f"Kỳ gần nhất: {result.get('latest_draw_number') or 'Chưa có dữ liệu'}",
        f"Thời gian phân tích: {result['analysis_time']}", "",
        "=" * 50, "1. PHÂN TÍCH TẦN SUẤT", "=" * 50, "",
        f"Số nóng: {stats['hot_numbers']}",
        f"Số lạnh: {stats['cold_numbers']}",
        f"Tần suất: {stats['frequency']}", "",
        "=" * 50, "2. PHÂN TÍCH KHOẢNG CÁCH", "=" * 50, "",
        f"Số đang có khoảng cách lớn: {stats['gap']}", "",
        "=" * 50, "3. PHÂN TÍCH CẶP SỐ", "=" * 50, "",
        f"Các cặp xuất hiện nhiều: {stats['pair_frequency']}", "",
        "=" * 50, "4. PHÂN TÍCH CẤU TRÚC", "=" * 50, "",
        f"Tỷ lệ chẵn/lẻ: {stats['odd_even_distribution']}",
        f"Tỷ lệ thấp/cao: {stats['high_low_distribution']}",
        f"Phân bố tổng: {stats['sum_statistics']}",
        f"Phân bố khoảng: {stats['range_statistics']}", "",
        "=" * 50, "5. RANDOM FOREST", "=" * 50, "",
        f"Điểm ML của từng số: {result['machine_learning']['random_forest']}", "",
        "=" * 50, "6. ENSEMBLE", "=" * 50, "",
        f"Các số có điểm tổng hợp cao: {result['ensemble']['number_scores']}", "",
        "=" * 50, "7. BỘ SỐ ỨNG VIÊN", "=" * 50, "",
        "Hạng   Bộ số                 Điểm", "-" * 50,
    ]
    for item in predictions:
        numbers = " ".join(f"{int(n):02d}" for n in item["numbers"])
        lines.append(f"{item['rank']:<7}{numbers:<22}{item['score']:.4f}")
    lines += [
        "", "=" * 50, "8. BACKTEST", "=" * 50, "",
        f"Baseline ngẫu nhiên: {result['backtest']['random_baseline']}",
        f"Mô hình thống kê: {result['backtest']['statistical_model']}",
        f"Random Forest: {result['backtest']['random_forest']}",
        f"Ensemble: {result['backtest']['ensemble']}", "",
        "=" * 50, "LƯU Ý", "=" * 50, "",
        "Đây là các bộ số được xếp hạng dựa trên dữ liệu lịch sử,",
        "thống kê và mô hình thử nghiệm.",
        "Không coi điểm số mô hình là đảm bảo kết quả xổ số.",
        "", "=" * 50, "9. ENTROPY", "=" * 50, "",
        f"Điểm entropy: {result.get('entropy', {}).get('entropy_score')}",
        f"Xu hướng entropy: {result.get('entropy', {}).get('entropy_trend')}",
        f"Độ ổn định entropy: {result.get('entropy', {}).get('entropy_stability')}",
        "", "=" * 50, "10. PHÂN TÍCH LƯỚI", "=" * 50, "",
        f"Điểm không gian: {result.get('spatial_pattern', {}).get('spatial_score')}",
        f"Điểm cân bằng lưới: {result.get('spatial_pattern', {}).get('grid_balance_score')}",
        f"Mật độ cụm: {result.get('spatial_pattern', {}).get('cluster_density_score')}",
        "", "=" * 50, "11. TỐI ƯU BỘ VÉ", "=" * 50, "",
        f"Genetic Algorithm: {result.get('genetic_algorithm', {}).get('best_combinations', [])}",
        f"Số vé wheel: {result.get('wheel_system', {}).get('ticket_count', 0)}",
        f"Độ bao phủ: {result.get('covering_design', {}).get('coverage_rate', 0)}",
        "", "=" * 50, "12. QUẢN LÝ NGÂN SÁCH", "=" * 50, "",
        f"Kelly: {result.get('kelly', {}).get('warning', '')}",
    ]
    return lines


def export_reports(result: dict[str, Any], output_dir: Path) -> None:
    """Write Vietnamese TXT/Markdown and machine-readable JSON/CSV artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    lines = _lines(result)
    (output_dir / "analysis_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    markdown = "\n".join(
        f"## {line}" if line.startswith(tuple(f"{n}." for n in range(1, 9))) else line
        for line in lines
    )
    (output_dir / "analysis_report.md").write_text(markdown + "\n", encoding="utf-8")
    export_json(result, output_dir / "analysis_result.json")
    export_candidates(result["predictions"], output_dir / "candidate_combinations.csv")
