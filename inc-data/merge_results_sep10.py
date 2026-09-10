#!/usr/bin/env python3
"""Build results-sep-10.csv from the sep-8 export plus engineer supplements.

Inputs (same directory by default):
  results-sep-8.csv
  Figure5_abc_seed_std.csv   -- 5-seed means/std for panels a,b,c
  Figure5_qrs_recall.csv     -- DBLP delete and NCVoter mixed/add-dom recall

IncDC/3DC on panels h (insert) and i (delete) come from IncDC-0401.xlsx
and 3DC-0401.xlsx (NCVoter, 1/5/15/20/30%). Times are converted from ms
to seconds. OutOfMemory is stored as 5000 s, matching the paper plot
convention. IncDC has no delete runs.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

ABC_OVERLAY = (
    "inc_runtime_s", "inc_runtime_s_std", "sk_mb", "sk_mb_std",
    "recall", "recall_std", "recall_plus", "recall_plus_std",
    "n_sketch_seeds", "sketch_seeds",
)
STD_FIELDS = (
    "inc_runtime_s_std", "sk_mb_std", "recall_std", "recall_plus_std",
    "n_sketch_seeds", "sketch_seeds",
)
OOM_SECONDS = 5000.0
RATIO_SLOT = {0.01: "1", 0.05: "5", 0.15: "15", 0.20: "20", 0.30: "30"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def overlay_abc(sep8: list[dict[str, str]], abc: list[dict[str, str]]) -> None:
    index = {(row["panel"], row["slot"], row["variant"]): row for row in sep8}
    for src in abc:
        key = (src["panel"], src["slot"], src["variant"])
        row = index.get(key)
        if row is None:
            row = {"panel": src["panel"], "slot": src["slot"], "variant": src["variant"],
                   "round": "0", "x": src.get("x", "")}
            sep8.append(row)
            index[key] = row
        for field in ABC_OVERLAY:
            if src.get(field, "") != "":
                row[field] = src[field]
        row["csv_path"] = src.get("source", "")
        row["source_kind"] = "seed-agg"
        row["sweep"] = "paper_supp_recall_sketch"


def append_qrs(sep8: list[dict[str, str]], qrs: list[dict[str, str]]) -> None:
    dataset = {"q": "dblp", "r": "ncvoter", "s": "ncvoter"}
    update = {"q": "delete", "r": "mix", "s": "mix"}
    for src in qrs:
        panel = src["panel"]
        row = {
            "panel": panel,
            "slot": src["slot"],
            "x": src.get("x", ""),
            "variant": src["variant"],
            "round": "0",
            "recall": src.get("recall", ""),
            "recall_plus": src.get("recall_plus", ""),
            "recall_minus": src.get("recall_minus", ""),
            "tp": src.get("tp", ""),
            "fp": src.get("fp", ""),
            "fn": src.get("fn", ""),
            "gt_new_rules": src.get("gt_new_rules", ""),
            "gt_invalid_rules": src.get("gt_invalid_rules", ""),
            "inc_runtime_s": src.get("inc_runtime_s", ""),
            "dataset": dataset.get(panel, ""),
            "update_type": update.get(panel, ""),
            "csv_path": src.get("source", ""),
            "source_kind": "qrs-recall",
            "sweep": src.get("source", "").split("/sweeps/")[-1].split("/")[0]
            if "/sweeps/" in src.get("source", "") else "",
        }
        sep8.append(row)


def _xlsx_seconds(ms, note: object) -> float:
    text = "" if note is None or (isinstance(note, float) and note != note) else str(note)
    if "outofmemory" in text.lower() or ms is None or (isinstance(ms, float) and ms != ms):
        return OOM_SECONDS
    return float(ms) / 1000.0


def append_dc_0401(sep8: list[dict[str, str]], incdc: Path, dc3: Path) -> None:
    import pandas as pd

    specs = (
        (incdc, "adult varying |ΔD+|", "incdc", "h", "add"),
        (dc3, "ncvoter varying |ΔD+|", "dc3", "h", "add"),
        (dc3, "ncvoter varying |ΔD-|", "dc3", "i", "del"),
    )
    for path, sheet, variant, panel, prefix in specs:
        frame = pd.read_excel(path, sheet)
        for _, item in frame.iterrows():
            ratio = round(float(item["增量比例"]), 2)
            tick = RATIO_SLOT[ratio]
            slot = f"{prefix}{tick}"
            seconds = _xlsx_seconds(item["耗时（ms）"], item.get("备注"))
            sep8.append({
                "panel": panel,
                "slot": slot,
                "x": tick,
                "variant": variant,
                "round": "0",
                "inc_runtime_s": str(seconds),
                "dataset": "ncvoter",
                "update_type": "add" if prefix == "add" else "delete",
                "source_kind": "xlsx-0401-dc",
                "csv_path": f"{path.name}:{sheet}",
                "sweep": "xlsx-0401-incdc-3dc",
            })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sep8", type=Path, default=Path("results-sep-8.csv"))
    parser.add_argument("--abc", type=Path, default=Path("Figure5_abc_seed_std.csv"))
    parser.add_argument("--qrs", type=Path, default=Path("Figure5_qrs_recall.csv"))
    parser.add_argument("--incdc", type=Path, default=Path("IncDC-0401.xlsx"))
    parser.add_argument("--dc3", type=Path, default=Path("3DC-0401.xlsx"))
    parser.add_argument("--output", type=Path, default=Path("results-sep-10.csv"))
    args = parser.parse_args()
    with args.sep8.open(newline="") as stream:
        reader = csv.DictReader(stream)
        header = list(reader.fieldnames or [])
        rows = list(reader)
    overlay_abc(rows, read_csv(args.abc))
    append_qrs(rows, read_csv(args.qrs))
    append_dc_0401(rows, args.incdc, args.dc3)
    extra = [field for field in STD_FIELDS if field not in header]
    fields = header + extra
    write_csv(args.output, rows, fields)
    print(f"wrote {args.output} ({len(rows)} rows, {len(fields)} columns)")


if __name__ == "__main__":
    main()
