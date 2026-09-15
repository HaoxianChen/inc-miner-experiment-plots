#!/usr/bin/env python3
"""Build results-sep-15.csv from results-sep-14.csv plus new_fg (v306).

Replaces paper panels:
  f <- new_fg/data/ncvoter_f  (NCVoter pure add)
  g <- new_fg/data/ncvoter_g  (NCVoter pure delete)
h/i stay on results-sep-14. DBLP new_fg sweeps are unused.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path

SLOT_X = {
    "add1": "1", "add5": "5", "add15": "15", "add20": "20", "add30": "30",
    "del1": "1", "del5": "5", "del15": "15", "del20": "20", "del30": "30",
}
PANEL_MAP = (
    ("f", "ncvoter_f", "ncvoter", "add"),
    ("g", "ncvoter_g", "ncvoter", "delete"),
)
REPLACE_PANELS = {panel for panel, *_ in PANEL_MAP}
EXPORTED_AT = "2026-09-15T00:00:00"


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        return list(reader.fieldnames or []), list(reader)


def load_slot_rows(root: Path, sweep: str) -> list[tuple[str, Path, dict[str, str]]]:
    out = []
    for slot_dir in sorted((root / sweep).iterdir()):
        csv_path = slot_dir / "results.csv"
        if not csv_path.is_file():
            continue
        with csv_path.open(newline="") as stream:
            for row in csv.DictReader(stream):
                out.append((slot_dir.name, csv_path, row))
    return out


def convert(
    panel: str,
    dataset: str,
    update_type: str,
    slot: str,
    src: dict[str, str],
    csv_path: Path,
    fields: list[str],
) -> dict[str, str]:
    inc = src.get("inc_runtime_s", "")
    batch = src.get("batch_runtime_s", "")
    try:
        ratio = "" if not inc or not batch else str(float(inc) / float(batch))
    except ValueError:
        ratio = ""
    row = {field: "" for field in fields}
    row.update({key: value for key, value in src.items() if key in row})
    row["panel"] = panel
    row["slot"] = slot
    row["x"] = SLOT_X.get(slot, src.get("round", "0"))
    row["round"] = src.get("round", "0")
    row["dataset"] = dataset
    row["update_type"] = src.get("update_type") or update_type
    row["sweep"] = "new_fg_v306"
    row["source_kind"] = "new_fg"
    row["csv_path"] = str(csv_path)
    row["exported_at"] = EXPORTED_AT
    row["batch_plot_s"] = batch
    row["inc_over_batch_plot"] = ratio
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=Path("results-sep-14.csv"))
    parser.add_argument("--new-fg", type=Path, default=Path("new_fg/data"))
    parser.add_argument("--output", type=Path, default=Path("results-sep-15.csv"))
    args = parser.parse_args()
    fields, rows = read_csv(args.base)
    kept = [row for row in rows if row.get("panel") not in REPLACE_PANELS]
    added: list[dict[str, str]] = []
    for panel, sweep, dataset, update_type in PANEL_MAP:
        for slot, path, src in load_slot_rows(args.new_fg, sweep):
            added.append(convert(panel, dataset, update_type, slot, src, path, fields))
    extra_keys = []
    for row in added:
        for key in row:
            if key not in fields:
                extra_keys.append(key)
    # preserve sep-14 column order
    out_fields = fields
    out_rows = kept + added
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=out_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(out_rows)
    print(
        f"wrote {args.output} ({len(out_rows)} rows; "
        f"kept {len(kept)}, replaced panels {sorted(REPLACE_PANELS)} with {len(added)})"
    )


if __name__ == "__main__":
    main()
