#!/usr/bin/env python3
"""
predictor.py — Universal RandomForest (CSV-only) predictor for lottery-style games.
What it does
- Reads a game's CSV draw history directly (no dependence on your engine pools).
- Trains RandomForestClassifier models to estimate the probability each number will appear next draw.
- Produces:
    * RF-1: Top-N numbers by probability (most likely)
    * RF-2: Diversified RF (sampled from top pool; differs from RF-1 by >=2 numbers by default)
Designed to be robust across many CSV formats:
- Columns like n1..n5 / num1..num5 / ball1..ball5
- Any 5 numeric columns (fallback)
- A single column containing a hyphen/space/comma separated list of numbers (fallback heuristic)
Optional bonus support
- If a bonus column exists (e.g., megaball/powerball/starball), you can pass bonus_max and it will
  also train a bonus RF and return bonus prediction.
Requirements
- pandas
- scikit-learn
- numpy
If scikit-learn isn't installed, this module returns None predictions (safe failure).
Usage (import)
    from predictor import UniversalRFPredictor
    rf = UniversalRFPredictor()
    out = rf.predict(csv_path, main_max=52, main_n=5, bonus_max=10, bonus_n=1)
    print(out["rf1_numbers"], out.get("rf1_bonus"))
    print(out["rf2_numbers"], out.get("rf2_bonus"))
Usage (CLI)
    python predictor.py --csv "E:\\data\\la_results.csv" --main-max 52 --main-n 5 --bonus-max 10 --bonus-n 1
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

# ----------------------------- helpers -----------------------------

_SPLIT_RE = re.compile(r"[^0-9]+")

def _safe_int(x: Any) -> Optional[int]:
    try:
        if x is None:
            return None
        if isinstance(x, bool):
            return None
        return int(x)
    except Exception:
        try:
            s = str(x).strip()
            if not s:
                return None
            return int(float(s))
        except Exception:
            return None

def _dedupe(seq: Sequence[Any]) -> List[int]:
    out: List[int] = []
    seen = set()
    for v in (seq or []):
        iv = _safe_int(v)
        if iv is None:
            continue
        if iv in seen:
            continue
        seen.add(iv)
        out.append(iv)
    return out

def _md5_seed(s: str, fallback: int = 1337) -> int:
    try:
        return int(hashlib.md5(s.encode("utf-8")).hexdigest()[:8], 16)
    except Exception:
        return fallback

def _clip(nums: Sequence[int], lo: int, hi: int) -> List[int]:
    out = []
    for n in nums:
        if lo <= n <= hi:
            out.append(n)
    return out

# ----------------------------- CSV parsing -----------------------------

@dataclass
class ParsedHistory:
    draws: List[List[int]]            # list of main number lists
    bonus: List[Optional[int]]        # list of bonus values aligned with draws

class CSVHistoryReader:
    """Best-effort reader for many draw-history CSV layouts."""

    DEFAULT_MAIN_PATTERNS = [
        ["n1", "n2", "n3", "n4", "n5", "n6"],
        ["num1", "num2", "num3", "num4", "num5"],
        ["ball1", "ball2", "ball3", "ball4", "ball5"],
        ["w1", "w2", "w3", "w4", "w5"],
    ]

    DEFAULT_BONUS_CANDIDATES = [
        "mb", "megaball", "mega_ball",
        "pb", "powerball", "power_ball",
        "sb", "star", "starball", "star_ball",
        "bonus", "bonusball", "bonus_ball", "special_number",
        "luckyball", "lucky_ball",
    ]

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def read(self, csv_path: str, main_n: int = 5) -> ParsedHistory:
        try:
            import pandas as pd  # type: ignore
        except Exception:
            return ParsedHistory(draws=[], bonus=[])

        try:
            df = pd.read_csv(csv_path)
        except Exception:
            return ParsedHistory(draws=[], bonus=[])

        # 1) Find main columns using common patterns
        main_cols: Optional[List[str]] = None
        for pat in self.DEFAULT_MAIN_PATTERNS:
            if all(c in df.columns for c in pat[:main_n]):
                main_cols = pat[:main_n]
                break

        # 2) If not found, choose first `main_n` numeric columns
        if main_cols is None:
            numeric_cols = []
            for c in df.columns:
                try:
                    kind = getattr(df[c].dtype, "kind", "")
                except Exception:
                    kind = ""
                if kind in ("i", "u", "f"):
                    numeric_cols.append(c)
            if len(numeric_cols) >= main_n:
                main_cols = numeric_cols[:main_n]

        # 3) If still not found, look for a single column that contains a list of numbers
        list_col: Optional[str] = None
        if main_cols is None:
            for c in df.columns:
                if df[c].dtype == object:
                    # check if it looks like "1-2-3-4-5" or "1 2 3 4 5"
                    sample = df[c].dropna().astype(str).head(10).tolist()
                    hits = 0
                    for s in sample:
                        parts = [p for p in _SPLIT_RE.split(s) if p]
                        if len(parts) >= main_n and all(p.isdigit() for p in parts[:main_n]):
                            hits += 1
                    if hits >= max(1, len(sample) // 2):
                        list_col = c
                        break

        # Bonus column detection
        bonus_col: Optional[str] = None
        lower_cols = {str(c).lower(): c for c in df.columns}
        for cand in self.DEFAULT_BONUS_CANDIDATES:
            if cand in lower_cols:
                bonus_col = lower_cols[cand]
                break

        draws: List[List[int]] = []
        bonus: List[Optional[int]] = []

        if main_cols is not None:
            # Iterate over the complete row so optional bonus/special columns
            # remain available alongside the main-number columns.
            for _, row in df.iterrows():
                nums = []
                ok = True
                for c in main_cols:
                    iv = _safe_int(row[c])
                    if iv is None:
                        ok = False
                        break
                    nums.append(iv)
                if ok and len(nums) == main_n:
                    draws.append(nums)
                    bonus.append(_safe_int(row[bonus_col]) if bonus_col else None)

        elif list_col is not None:
            for _, row in df[[list_col]].iterrows():
                s = str(row[list_col])
                parts = [p for p in _SPLIT_RE.split(s) if p]
                if len(parts) < main_n:
                    continue
                nums = []
                ok = True
                for p in parts[:main_n]:
                    iv = _safe_int(p)
                    if iv is None:
                        ok = False
                        break
                    nums.append(iv)
                if ok and len(nums) == main_n:
                    draws.append(nums)
                    # bonus may also be embedded later; ignore here (None)
                    bonus.append(None)
        else:
            # Could not parse
            return ParsedHistory(draws=[], bonus=[])

        return ParsedHistory(draws=draws, bonus=bonus)

# ----------------------------- RF core -----------------------------

class UniversalRFPredictor:
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.reader = CSVHistoryReader(verbose=verbose)

    def _build_features(self, draws: List[List[int]], universe_max: int, lookback: int = 12):
        """Build per-number time-series features and next-step feature vector."""
        import numpy as np  # type: ignore

        if len(draws) < (lookback + 5):
            return {}

        appears = {n: [0] * len(draws) for n in range(1, universe_max + 1)}
        for t, d in enumerate(draws):
            s = set(d)
            for n in s:
                if 1 <= n <= universe_max:
                    appears[n][t] = 1

        def recent_count(arr, t, w):
            return int(sum(arr[max(0, t - w):t]))

        def gap_since(arr, t):
            for k in range(1, t + 1):
                if arr[t - k] == 1:
                    return k
            return t

        feats = {}
        for n in range(1, universe_max + 1):
            arr = appears[n]
            X, y = [], []
            for t in range(lookback, len(draws)):
                f = [
                    recent_count(arr, t, 5),
                    recent_count(arr, t, 10),
                    gap_since(arr, t),
                    arr[t - 1],
                    arr[t - 2] if t - 2 >= 0 else 0,
                    arr[t - 3] if t - 3 >= 0 else 0,
                ]
                X.append(f)
                y.append(arr[t])

            t = len(draws)
            last_f = [
                recent_count(arr, t, 5),
                recent_count(arr, t, 10),
                gap_since(arr, t),
                arr[t - 1],
                arr[t - 2] if t - 2 >= 0 else 0,
                arr[t - 3] if t - 3 >= 0 else 0,
            ]
            feats[n] = (np.asarray(X, float), np.asarray(y, int), np.asarray(last_f, float))
        return feats

    def _rank_numbers(self, draws: List[List[int]], universe_max: int, seed: int) -> List[Tuple[int, float]]:
        try:
            from sklearn.ensemble import RandomForestClassifier  # type: ignore
            import numpy as np  # type: ignore
        except Exception:
            return []

        feats = self._build_features(draws, universe_max, lookback=12)
        if not feats:
            return []

        probs: List[Tuple[int, float]] = []
        for n, (X, y, last_f) in feats.items():
            if int(y.sum()) < 5 or int((1 - y).sum()) < 5:
                continue
            try:
                clf = RandomForestClassifier(
                    n_estimators=240,
                    max_depth=9,
                    random_state=seed,
                    class_weight="balanced",
                    n_jobs=-1
                )
                clf.fit(X, y)
                p = float(clf.predict_proba(last_f.reshape(1, -1))[0][1])
                probs.append((n, p))
            except Exception:
                continue

        probs.sort(key=lambda t: t[1], reverse=True)
        return probs

    def _rank_bonus(self, bonus_series: List[Optional[int]], bonus_max: int, seed: int) -> List[Tuple[int, float]]:
        """Simple RF for bonus (single categorical per draw)."""
        try:
            from sklearn.ensemble import RandomForestClassifier  # type: ignore
            import numpy as np  # type: ignore
        except Exception:
            return []

        # Need enough bonus observations
        b = [_safe_int(x) for x in bonus_series]
        if sum(1 for x in b if x is not None) < 40:
            return []

        # Build appearance series for each bonus value 1..bonus_max
        T = len(b)

        def recent_count(arr, t, w):
            return int(sum(arr[max(0, t - w):t]))

        def gap_since(arr, t):
            for k in range(1, t + 1):
                if arr[t - k] == 1:
                    return k
            return t

        probs: List[Tuple[int, float]] = []
        for val in range(1, bonus_max + 1):
            arr = [1 if _safe_int(b[t]) == val else 0 for t in range(T)]
            X, y = [], []
            lookback = 10
            for t in range(lookback, T):
                f = [
                    recent_count(arr, t, 5),
                    recent_count(arr, t, 10),
                    gap_since(arr, t),
                    arr[t - 1],
                    arr[t - 2] if t - 2 >= 0 else 0,
                    arr[t - 3] if t - 3 >= 0 else 0,
                ]
                X.append(f)
                y.append(arr[t])
            if len(X) < 30:
                continue
            X = np.asarray(X, float)
            y = np.asarray(y, int)
            if int(y.sum()) < 3 or int((1 - y).sum()) < 3:
                continue
            last_f = np.asarray([
                recent_count(arr, T, 5),
                recent_count(arr, T, 10),
                gap_since(arr, T),
                arr[T - 1],
                arr[T - 2] if T - 2 >= 0 else 0,
                arr[T - 3] if T - 3 >= 0 else 0,
            ], float).reshape(1, -1)

            try:
                clf = RandomForestClassifier(
                    n_estimators=200,
                    max_depth=8,
                    random_state=seed,
                    class_weight="balanced",
                    n_jobs=-1
                )
                clf.fit(X, y)
                p = float(clf.predict_proba(last_f)[0][1])
                probs.append((val, p))
            except Exception:
                continue

        probs.sort(key=lambda t: t[1], reverse=True)
        return probs

    def _pick_rf1_rf2(self, probs: List[Tuple[int, float]], main_n: int, min_diff: int = 2) -> Tuple[List[int], Optional[List[int]]]:
        if not probs or len(probs) < main_n:
            return [], None

        rf1 = [n for n, _ in probs[:main_n]]

        # diversified sampling from top pool
        pool = probs[:max(12, main_n * 3)]
        nums = [n for n, _ in pool]
        weights = [max(1e-9, p) for _, p in pool]

        seed = _md5_seed(",".join(map(str, rf1)) + "|" + ",".join(map(str, nums[:10])))
        rng = random.Random(seed)

        def sample_ticket() -> List[int]:
            chosen = []
            remaining = list(zip(nums, weights))
            for _ in range(main_n):
                total = sum(w for _, w in remaining)
                if total <= 0:
                    break
                r = rng.random() * total
                cum = 0.0
                pick_idx = 0
                for i, (n, w) in enumerate(remaining):
                    cum += w
                    if cum >= r:
                        pick_idx = i
                        break
                n_pick, _ = remaining.pop(pick_idx)
                chosen.append(n_pick)
            return sorted(_dedupe(chosen))

        rf2: Optional[List[int]] = None
        for _ in range(60):
            cand = sample_ticket()
            if len(cand) != main_n:
                continue
            overlap = len(set(cand) & set(rf1))
            # require at least `min_diff` numbers different
            if overlap <= (main_n - min_diff):
                rf2 = cand
                break

        # fallback: if diversification fails, return None (honest)
        return sorted(rf1), sorted(rf2) if rf2 else None

    def predict(
        self,
        csv_path: str,
        main_max: int,
        main_n: int = 5,
        bonus_max: Optional[int] = None,
        bonus_n: int = 0,
        seed_key: str = "",
        min_diff: int = 2,
        min_draws: int = 60,
    ) -> Dict[str, Any]:
        """Return RF predictions from CSV history."""
        out: Dict[str, Any] = {
            "ok": False,
            "rf1_numbers": [],
            "rf2_numbers": None,
            "rf1_bonus": None,
            "rf2_bonus": None,
            "reason": "",
        }

        if not csv_path or not os.path.exists(csv_path):
            out["reason"] = "csv_missing"
            return out

        hist = self.reader.read(csv_path, main_n=main_n)
        draws = hist.draws
        if len(draws) < min_draws:
            out["reason"] = f"too_few_draws:{len(draws)}"
            return out

        # clip to valid range defensively
        draws = [_clip(_dedupe(d), 1, int(main_max))[:main_n] for d in draws if d]
        draws = [d for d in draws if len(d) == main_n]
        if len(draws) < min_draws:
            out["reason"] = f"too_few_valid_draws:{len(draws)}"
            return out

        seed = _md5_seed(seed_key or (csv_path + "|" + str(main_max) + "|" + str(main_n)))
        probs = self._rank_numbers(draws, int(main_max), seed=seed)
        if not probs:
            out["reason"] = "rf_rank_empty"
            return out

        rf1, rf2 = self._pick_rf1_rf2(probs, main_n=int(main_n), min_diff=int(min_diff))
        if not rf1:
            out["reason"] = "rf1_empty"
            return out

        out["rf1_numbers"] = rf1
        out["rf2_numbers"] = rf2
        out["ok"] = True

        # Bonus prediction (optional)
        if bonus_max and bonus_n and bonus_n > 0:
            bprobs = self._rank_bonus(hist.bonus, int(bonus_max), seed=seed)
            if bprobs:
                out["rf1_bonus"] = bprobs[0][0]
                # For bonus, "diversification" isn't meaningful; if rf2 exists, reuse rf1_bonus
                out["rf2_bonus"] = out["rf1_bonus"]

        return out


# ----------------------------- CLI -----------------------------

def main():
    p = argparse.ArgumentParser(description="Universal RF predictor (CSV-only).")
    p.add_argument("--csv", required=True, help="Path to draw-history CSV")
    p.add_argument("--main-max", required=True, type=int, help="Max main number (e.g., 52)")
    p.add_argument("--main-n", default=5, type=int, help="Count of main numbers per draw")
    p.add_argument("--bonus-max", default=None, type=int, help="Max bonus number (optional)")
    p.add_argument("--bonus-n", default=0, type=int, help="Bonus count (0 or 1)")
    p.add_argument("--min-draws", default=60, type=int, help="Minimum draws required")
    p.add_argument("--min-diff", default=2, type=int, help="RF-2 min number-diff vs RF-1")
    p.add_argument("--seed-key", default="", help="Seed key for reproducibility")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    rf = UniversalRFPredictor(verbose=args.verbose)
    out = rf.predict(
        csv_path=args.csv,
        main_max=args.main_max,
        main_n=args.main_n,
        bonus_max=args.bonus_max,
        bonus_n=args.bonus_n,
        seed_key=args.seed_key,
        min_diff=args.min_diff,
        min_draws=args.min_draws,
    )
    print(out)

    if out.get("ok"):
        print("RF-1:", out["rf1_numbers"], "BONUS:", out.get("rf1_bonus"))
        print("RF-2:", out.get("rf2_numbers"), "BONUS:", out.get("rf2_bonus"))
    else:
        print("Not OK:", out.get("reason"))

if __name__ == "__main__":
    main()
