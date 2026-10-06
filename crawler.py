"""Crawl and maintain complete historical Vietlott result CSV files."""

from __future__ import annotations

import argparse
import csv
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import requests
from bs4 import BeautifulSoup
from requests import Response
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


@dataclass(frozen=True)
class GameConfig:
    game_type: str
    url_pattern: str
    main_count: int
    main_max: int
    has_special: bool
    first_draw: int
    draws_per_day: int
    output_name: str


GAME_CONFIGS = {
    "5_35": GameConfig(
        "5_35", "https://kqxso.top/vietlot/power535/{draw_number}",
        5, 35, True, 1, 2, "lotto535.csv",
    ),
    "6_45": GameConfig(
        "6_45", "https://kqxso.top/vietlot/power645/{draw_number}",
        6, 45, False, 198, 1, "mega645.csv",
    ),
    "6_55": GameConfig(
        "6_55", "https://kqxso.top/vietlot/power655/{draw_number}",
        6, 55, True, 1, 1, "power655.csv",
    ),
}

SOURCE_NAME = "kqxso.top"
REQUEST_TIMEOUT_SECONDS = 20
REQUEST_RETRIES = 2
REQUEST_DELAY_SECONDS = 1.0
MAX_CONSECUTIVE_FAILURES = 30
CSV_FIELDNAMES = [
    "id", "game_type", "draw_number", "draw_datetime",
    "n1", "n2", "n3", "n4", "n5", "n6",
    "special_number", "created_at", "source",
]
BASE_DIR = Path(__file__).resolve().parent
DEBUG_HTML_DIR = BASE_DIR / "data" / "crawler_debug"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0 Safari/537.36"
)


def build_draw_url(config: GameConfig, draw_number: int) -> str:
    """Build a source URL using its five-digit draw-number convention."""
    if draw_number < 1:
        raise ValueError("draw_number must be positive")
    return config.url_pattern.format(draw_number=f"{draw_number:05d}")


def create_session() -> requests.Session:
    """Create a session with realistic headers and bounded retries."""
    session = requests.Session()
    session.headers.update({
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "vi,en-US;q=0.8,en;q=0.6",
    })
    retry = Retry(
        total=REQUEST_RETRIES,
        connect=REQUEST_RETRIES,
        read=REQUEST_RETRIES,
        status=REQUEST_RETRIES,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def fetch_page(session: requests.Session, config: GameConfig, draw_number: int) -> Response | None:
    """Fetch one draw page; malformed or unavailable pages are reported to the caller."""
    url = build_draw_url(config, draw_number)
    print(f"[FETCH] {url}")
    try:
        response = session.get(url, timeout=REQUEST_TIMEOUT_SECONDS)
    except requests.exceptions.RequestException as exc:
        print(f"[WARN] Draw {draw_number:05d} request failed: {exc}")
        return None
    if response.status_code >= 400 or not response.text.strip():
        print(f"[WARN] Draw {draw_number:05d} unavailable (HTTP {response.status_code})")
        return None
    return response


def parse_draw_number(soup: BeautifulSoup) -> str:
    text = soup.get_text(" ", strip=True)
    match = re.search(r"Kỳ\s*#\s*(\d+)", text, flags=re.IGNORECASE)
    if not match:
        raise ValueError("missing draw number")
    return match.group(1).zfill(5)


def parse_draw_datetime(soup: BeautifulSoup) -> str:
    """Parse the source date and time without losing the twice-daily 5/35 time."""
    element = soup.select_one("p.text-sm.font-semibold.text-white\\/90")
    text = element.get_text(" ", strip=True) if element else soup.get_text(" ", strip=True)
    date_match = re.search(r"(\d{1,2}/\d{1,2}/\d{4})", text)
    if not date_match:
        raise ValueError("missing draw date")
    time_match = re.search(r"(\d{1,2}:\d{2})(?::(\d{2}))?", text)
    time_text = time_match.group(0) if time_match else "00:00:00"
    if time_match and time_match.group(2) is None:
        time_text += ":00"
    try:
        parsed = datetime.strptime(
            f"{date_match.group(1)} {time_text}", "%d/%m/%Y %H:%M:%S"
        )
    except ValueError as exc:
        raise ValueError(f"invalid draw date/time: {text}") from exc
    return parsed.strftime("%Y-%m-%d %H:%M:%S")


def parse_numbers(soup: BeautifulSoup, config: GameConfig) -> tuple[list[int], int | None]:
    """Extract and validate the configured main and special balls."""
    candidates: list[list[str]] = []
    for container in soup.find_all("div"):
        spans = container.find_all(
            "span", class_=lambda value: value and "tabular-nums" in value
        )
        values = [span.get_text(strip=True) for span in spans]
        if len(values) in (config.main_count, config.main_count + 1) and all(
            re.fullmatch(r"\d{1,2}", value) for value in values
        ):
            candidates.append(values)
    if not candidates:
        raise ValueError("missing lottery number result area")
    values = max(candidates, key=len)
    if len(values) < config.main_count:
        raise ValueError("missing main numbers")
    numbers = [int(value) for value in values[:config.main_count]]
    special = int(values[config.main_count]) if len(values) > config.main_count else None
    if len(set(numbers)) != config.main_count:
        raise ValueError(f"duplicate main numbers: {numbers}")
    if any(number < 1 or number > config.main_max for number in numbers):
        raise ValueError(f"main number outside 1-{config.main_max}: {numbers}")
    if config.has_special:
        if special is None or not 1 <= special <= config.main_max:
            raise ValueError("missing or invalid special number")
    else:
        special = None
    return sorted(numbers), special


def parse_draw_page(html: str, config: GameConfig) -> dict[str, Any]:
    """Parse one page into a validated, CSV-ready record."""
    if not html.strip():
        raise ValueError("missing HTML")
    soup = BeautifulSoup(html, "html.parser")
    numbers, special = parse_numbers(soup, config)
    record: dict[str, Any] = {
        "game_type": config.game_type,
        "draw_number": parse_draw_number(soup),
        "draw_datetime": parse_draw_datetime(soup),
        "n1": numbers[0], "n2": numbers[1], "n3": numbers[2],
        "n4": numbers[3], "n5": numbers[4],
        "n6": numbers[5] if config.main_count == 6 else "",
        "special_number": special if special is not None else "",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source": SOURCE_NAME,
    }
    return record


def validate_row(row: dict[str, Any], config: GameConfig) -> None:
    """Validate required fields and game-specific number layout."""
    if row.get("game_type") != config.game_type:
        raise ValueError(f"wrong game_type in {config.output_name}")
    if not re.fullmatch(r"\d+", str(row.get("draw_number", ""))):
        raise ValueError("invalid draw_number")
    try:
        datetime.strptime(str(row.get("draw_datetime", "")), "%Y-%m-%d %H:%M:%S")
    except ValueError as exc:
        raise ValueError("invalid draw_datetime") from exc
    numbers = []
    for index in range(1, config.main_count + 1):
        value = row.get(f"n{index}", "")
        if not str(value).isdigit() or not 1 <= int(value) <= config.main_max:
            raise ValueError(f"invalid n{index}")
        numbers.append(int(value))
    if len(set(numbers)) != config.main_count:
        raise ValueError("duplicate main numbers")
    if config.main_count == 5 and row.get("n6", "") not in ("", None):
        raise ValueError("5/35 n6 must be empty")
    special = row.get("special_number", "")
    if config.has_special:
        if not str(special).isdigit() or not 1 <= int(special) <= config.main_max:
            raise ValueError("invalid special_number")
    elif special not in ("", None):
        raise ValueError("6/45 special_number must be empty")


def load_existing_csv(csv_path: Path, config: GameConfig) -> list[dict[str, str]]:
    """Load existing rows, accepting the old bonus column only for migration."""
    if not csv_path.exists():
        return []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {csv_path}")
        missing = [field for field in CSV_FIELDNAMES if field not in reader.fieldnames
                   and not (field == "special_number" and "bonus_number" in reader.fieldnames)]
        if missing:
            raise ValueError(f"CSV is missing required columns: {missing}")
        rows: list[dict[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for raw_row in reader:
            row = {field: raw_row.get(field, "") for field in CSV_FIELDNAMES}
            if not row["special_number"] and "bonus_number" in reader.fieldnames:
                row["special_number"] = raw_row.get("bonus_number", "")
            validate_row(row, config)
            key = (row["game_type"], row["draw_number"])
            if key in seen:
                raise ValueError(f"duplicate draw in {csv_path}: {key[1]}")
            seen.add(key)
            rows.append(row)
        return rows


def save_csv(records: list[dict[str, Any]], csv_path: Path, config: GameConfig) -> tuple[int, int]:
    """Validate, merge by draw identity, and rewrite rows in chronological order."""
    existing = load_existing_csv(csv_path, config)
    rows = existing[:]
    keys = {(row["game_type"], row["draw_number"]) for row in rows}
    seen_datetimes = {row["draw_datetime"] for row in rows}
    skipped = 0
    for record in records:
        validate_row(record, config)
        key = (record["game_type"], record["draw_number"])
        # Draw number is the stable identity, so rerunning a crawl never appends it twice.
        if key in keys:
            skipped += 1
            continue
        if record["draw_datetime"] in seen_datetimes:
            raise ValueError(f"duplicate draw_datetime: {record['draw_datetime']}")
        keys.add(key)
        seen_datetimes.add(record["draw_datetime"])
        rows.append(record)
    rows.sort(key=lambda row: (row["draw_datetime"], row["draw_number"]))
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        for row_id, row in enumerate(rows, start=1):
            writer.writerow({field: row_id if field == "id" else row.get(field, "")
                             for field in CSV_FIELDNAMES})
    return skipped, len(rows)


def crawl_game(session: requests.Session, config: GameConfig) -> tuple[list[dict[str, Any]], int, int]:
    """Crawl every available draw, stopping only after a long unavailable tail."""
    draw_number = config.first_draw
    consecutive_failures = 0
    requested = failed = 0
    records: list[dict[str, Any]] = []
    while consecutive_failures < MAX_CONSECUTIVE_FAILURES:
        requested += 1
        response = fetch_page(session, config, draw_number)
        if response is None:
            # A long unavailable tail marks the end of the source's historical range.
            failed += 1
            consecutive_failures += 1
            draw_number += 1
            time.sleep(REQUEST_DELAY_SECONDS)
            continue
        try:
            record = parse_draw_page(response.text, config)
            if int(record["draw_number"]) != draw_number:
                raise ValueError("page draw number does not match requested draw")
            records.append(record)
            consecutive_failures = 0
            print(f"[OK] {config.game_type} {record['draw_number']} {record['draw_datetime']}")
        except ValueError as exc:
            failed += 1
            consecutive_failures += 1
            DEBUG_HTML_DIR.mkdir(parents=True, exist_ok=True)
            (DEBUG_HTML_DIR / f"{config.game_type}_{draw_number:05d}.html").write_text(
                response.text[:3000], encoding="utf-8"
            )
            print(f"[WARN] Invalid draw {draw_number:05d}: {exc}")
        draw_number += 1
        time.sleep(REQUEST_DELAY_SECONDS)
    return records, requested, failed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game", choices=[*GAME_CONFIGS, "all"], default="all")
    parser.add_argument("--output-dir", type=Path, default=BASE_DIR / "data")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configs = GAME_CONFIGS.values() if args.game == "all" else (GAME_CONFIGS[args.game],)
    session = create_session()
    try:
        for config in configs:
            print(f"Starting {config.game_type} ({config.draws_per_day} draw(s)/day)")
            records, requested, failed = crawl_game(session, config)
            output_path = args.output_dir / config.output_name
            skipped, total = save_csv(records, output_path, config)
            print(
                f"Saved {len(records) - skipped} new {config.game_type} draws to {output_path}; "
                f"total={total}, requested={requested}, failed={failed}, duplicates={skipped}"
            )
    finally:
        session.close()


if __name__ == "__main__":
    main()
