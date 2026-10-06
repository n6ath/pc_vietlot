"""Bankroll management using user-supplied assumptions only."""
from __future__ import annotations

from typing import Any


def calculate(
    bankroll: float,
    ticket_price: float,
    user_defined_probability: float | None,
    expected_return: float | None,
    fractional_kelly: float = 0.25,
) -> dict[str, Any]:
    if bankroll <= 0 or ticket_price <= 0:
        raise ValueError("bankroll and ticket_price must be positive")
    if user_defined_probability is None or expected_return is None:
        return {"suggested_ticket_count": 0, "suggested_spending": 0.0, "bankroll_risk": 0.0,
                "kelly_fraction": None, "status": "assumptions_required",
                "warning": "Cần xác suất và lợi nhuận kỳ vọng do người dùng cung cấp; không tự suy đoán từ lịch sử."}
    if not 0 < user_defined_probability < 1 or expected_return <= 0:
        raise ValueError("probability must be in (0, 1) and expected_return must be positive")
    odds = expected_return / ticket_price
    kelly = max(0.0, (odds * user_defined_probability - (1 - user_defined_probability)) / odds)
    applied = min(1.0, kelly * fractional_kelly)
    spending = bankroll * applied
    return {"suggested_ticket_count": int(spending // ticket_price),
            "suggested_spending": round(spending, 2), "bankroll_risk": round(applied, 6),
            "kelly_fraction": round(kelly, 6), "fractional_kelly": fractional_kelly,
            "status": "calculated",
            "warning": "Kelly chỉ quản lý ngân sách theo giả định đầu vào, không tăng xác suất trúng."}
