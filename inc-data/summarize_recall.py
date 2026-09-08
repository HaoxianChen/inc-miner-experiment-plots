#!/usr/bin/env python3
"""Summarize recall for one method, grouped by dataset.

The script accepts either an experiment directory containing ``results.csv``
files or a flat, exported result CSV such as ``results-sep-8.csv``. For an
exported CSV, dataset names are inferred from ``csv_path`` because its
``dataset`` column can be blank.

Examples:
  python summarize_recall.py --root /path/to/f5_mcorr_v228
  python summarize_recall.py --csv results-sep-8.csv
  python summarize_recall.py --csv results-sep-8.csv --variant pincminer \
      --output /tmp/pincminer_recall_detail.csv
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
import re
from typing import Iterable


DATASET_RE = re.compile(
    r"(?:^|[^a-z0-9])(adult|ncvoter|inspection|dblp|aminer)(?:_ml\d+)?(?:$|[^a-z0-9])",
    re.I,
)
RECALL_FIELDS = ("recall", "recall_plus", "recall_minus")


def number(value: str | None) -> float | None:
    """Return a numeric CSV value, or None for an unavailable value."""
    if value is None or value.strip().lower() in {"", "nan", "na", "n/a"}:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def dataset_name(row: dict[str, str], source: Path | None) -> str:
    """Infer a canonical dataset name from explicit metadata or a source path."""
    candidates = [row.get("dataset", ""), row.get("csv_path", "")]
    if source is not None:
        candidates.append(str(source))
    for candidate in candidates:
        match = DATASET_RE.search(candidate.lower())
        if match:
            return match.group(1).lower()
    return "unknown"


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def input_rows(args: argparse.Namespace) -> Iterable[tuple[Path, dict[str, str]]]:
    if args.csv:
        for row in load_csv(args.csv):
            yield args.csv, row
        return
    for path in sorted(args.root.rglob("results.csv")):
        for row in load_csv(path):
            yield path, row


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.6f}"


def summarize(records: list[dict[str, object]]) -> dict[str, object]:
    values = {
        field: [record[field] for record in records if record[field] is not None]
        for field in RECALL_FIELDS
    }
    meaningful_plus = [
        record for record in records
        if record["recall_plus"] is not None and record["gt_new_rules"] > 0
    ]
    total_new = sum(record["gt_new_rules"] for record in meaningful_plus)
    weighted_plus = (
        sum(record["recall_plus"] * record["gt_new_rules"] for record in meaningful_plus)
        / total_new
        if total_new else None
    )
    return {
        "rows": len(records),
        "recall_n": len(values["recall"]),
        "recall_mean": mean(values["recall"]),
        "recall_min": min(values["recall"], default=None),
        "recall_below_one": sum(value < 1 for value in values["recall"]),
        "recall_plus_n": len(values["recall_plus"]),
        "recall_plus_mean": mean(values["recall_plus"]),
        "recall_plus_min": min(values["recall_plus"], default=None),
        "plus_meaningful_n": len(meaningful_plus),
        "plus_meaningful_mean": mean([record["recall_plus"] for record in meaningful_plus]),
        "plus_meaningful_min": min(
            [record["recall_plus"] for record in meaningful_plus], default=None
        ),
        "plus_meaningful_weighted": weighted_plus,
        "missing_recall": len(records) - len(values["recall"]),
    }


def print_summary(name: str, summary: dict[str, object]) -> None:
    print(
        f"{name}: rows={summary['rows']}, recall n={summary['recall_n']}, "
        f"mean={fmt(summary['recall_mean'])}, min={fmt(summary['recall_min'])}, "
        f"below 1={summary['recall_below_one']}, missing={summary['missing_recall']}"
    )
    print(
        f"  recall_plus: n={summary['recall_plus_n']}, "
        f"mean={fmt(summary['recall_plus_mean'])}, min={fmt(summary['recall_plus_min'])}; "
        f"with gt_new_rules>0: n={summary['plus_meaningful_n']}, "
        f"mean={fmt(summary['plus_meaningful_mean'])}, "
        f"min={fmt(summary['plus_meaningful_min'])}, "
        f"weighted={fmt(summary['plus_meaningful_weighted'])}"
    )


def write_detail(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "dataset", "source", "panel", "slot", "round", "variant", *RECALL_FIELDS,
        "gt_new_rules", "gt_invalid_rules",
    ]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    print(f"wrote {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--root", type=Path,
                        help="Experiment directory containing results.csv files")
    source.add_argument("--csv", type=Path, help="Flat exported results CSV")
    ap.add_argument("--variant", default="pincminer", help="Method to summarize")
    ap.add_argument("--output", type=Path, help="Optional per-record CSV output")
    args = ap.parse_args()

    records: list[dict[str, object]] = []
    for source_path, row in input_rows(args):
        if row.get("variant") != args.variant:
            continue
        records.append({
            "dataset": dataset_name(row, source_path),
            "source": row.get("csv_path") or str(source_path),
            "panel": row.get("panel", ""),
            "slot": row.get("slot", ""),
            "round": row.get("round", ""),
            "variant": args.variant,
            **{field: number(row.get(field)) for field in RECALL_FIELDS},
            "gt_new_rules": number(row.get("gt_new_rules")) or 0.0,
            "gt_invalid_rules": number(row.get("gt_invalid_rules")) or 0.0,
        })
    if not records:
        raise SystemExit(f"No rows found for variant={args.variant!r}")

    if args.output:
        write_detail(args.output, records)
    print(f"variant={args.variant}; records={len(records)}")
    print_summary("ALL", summarize(records))
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in records:
        grouped[str(record["dataset"])].append(record)
    for dataset in sorted(grouped):
        print_summary(dataset, summarize(grouped[dataset]))


if __name__ == "__main__":
    main()
