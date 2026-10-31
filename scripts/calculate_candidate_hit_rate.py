#!/usr/bin/env python3
"""Calculate how often item_id appears in candidate_item_id for test CSV files."""

from __future__ import annotations

import argparse
import ast
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate hit rate of ground-truth item_id in candidate_item_id lists."
    )
    parser.add_argument(
        "paths",
        nargs="*",
        default=["data"],
        help="CSV file(s) or directory/directories to scan. Defaults to data/.",
    )
    parser.add_argument(
        "--pattern",
        default="test.csv",
        help="Filename pattern to scan inside directories. Defaults to test.csv.",
    )
    parser.add_argument(
        "--gt-col",
        default="item_id",
        help="Ground-truth item column. Defaults to item_id.",
    )
    parser.add_argument(
        "--candidate-col",
        default="candidate_item_id",
        help="Candidate item list column. Defaults to candidate_item_id.",
    )
    parser.add_argument(
        "--k",
        type=int,
        nargs="*",
        default=[],
        help="Optional top-k hit rates to report, e.g. --k 1 5 10.",
    )
    return parser.parse_args()


def normalize_id(value: object) -> str:
    return str(value).strip()


def read_candidate_list(raw_value: str, csv_path: Path, row_number: int) -> list[str]:
    try:
        parsed = ast.literal_eval(raw_value)
    except (SyntaxError, ValueError) as exc:
        raise ValueError(
            f"{csv_path}:{row_number}: cannot parse candidate list: {raw_value!r}"
        ) from exc

    if not isinstance(parsed, list):
        raise ValueError(f"{csv_path}:{row_number}: candidate value is not a list")

    return [normalize_id(item) for item in parsed]


def find_csv_files(paths: list[str], pattern: str) -> list[Path]:
    csv_files: list[Path] = []
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_file():
            csv_files.append(path)
        elif path.is_dir():
            csv_files.extend(sorted(path.rglob(pattern)))
        else:
            raise FileNotFoundError(f"Path does not exist: {path}")
    return sorted(set(csv_files))


def calculate_file(
    csv_path: Path, gt_col: str, candidate_col: str, topks: list[int]
) -> dict[str, float | int | str]:
    total = 0
    hits = 0
    topk_hits = {k: 0 for k in topks}

    with csv_path.open(newline="", encoding="utf-8") as file_obj:
        reader = csv.DictReader(file_obj)
        if reader.fieldnames is None:
            raise ValueError(f"{csv_path}: missing CSV header")

        missing = [col for col in (gt_col, candidate_col) if col not in reader.fieldnames]
        if missing:
            raise ValueError(f"{csv_path}: missing column(s): {', '.join(missing)}")

        for row_number, row in enumerate(reader, start=2):
            gt_item = normalize_id(row[gt_col])
            candidates = read_candidate_list(row[candidate_col], csv_path, row_number)

            total += 1
            if gt_item in candidates:
                hits += 1
            for k in topks:
                if gt_item in candidates[:k]:
                    topk_hits[k] += 1

    result: dict[str, float | int | str] = {
        "file": str(csv_path),
        "total": total,
        "hits": hits,
        "hit_rate": hits / total if total else 0.0,
    }
    for k in topks:
        result[f"HR@{k}"] = topk_hits[k] / total if total else 0.0
    return result


def print_result(result: dict[str, float | int | str], topks: list[int]) -> None:
    parts = [
        str(result["file"]),
        f"hits={result['hits']}/{result['total']}",
        f"hit_rate={result['hit_rate']:.6f}",
    ]
    for k in topks:
        parts.append(f"HR@{k}={result[f'HR@{k}']:.6f}")
    print(" | ".join(parts))


def main() -> None:
    args = parse_args()
    topks = sorted(set(k for k in args.k if k > 0))
    csv_files = find_csv_files(args.paths, args.pattern)

    if not csv_files:
        raise SystemExit("No matching CSV files found.")

    all_total = 0
    all_hits = 0
    all_topk_hits = {k: 0.0 for k in topks}

    for csv_path in csv_files:
        result = calculate_file(csv_path, args.gt_col, args.candidate_col, topks)
        print_result(result, topks)

        total = int(result["total"])
        all_total += total
        all_hits += int(result["hits"])
        for k in topks:
            all_topk_hits[k] += float(result[f"HR@{k}"]) * total

    if len(csv_files) > 1:
        aggregate: dict[str, float | int | str] = {
            "file": "ALL",
            "total": all_total,
            "hits": all_hits,
            "hit_rate": all_hits / all_total if all_total else 0.0,
        }
        for k in topks:
            aggregate[f"HR@{k}"] = all_topk_hits[k] / all_total if all_total else 0.0
        print_result(aggregate, topks)


if __name__ == "__main__":
    main()
