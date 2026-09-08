#!/usr/bin/env python3
"""Generate the September 8 paper plots from one exported result CSV.

Usage:
  python plot_results_sep8.py --csv results-sep-8.csv
  python plot_results_sep8.py --csv results-sep-8.csv --output plots-results-sep-8

The output names follow ``<panel>-<dataset>-vary-<parameter>.pdf``.  Each
runtime panel omits Naive and has no local legend; see ``legend.pdf``.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path
import tempfile
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "sep8-matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MaxNLocator


FONT_SIZE = 18
TICK_LABEL_SIZE = 20
MARKER_SIZE = 12
FIGSIZE = (6.4, 4.8)

METHODS = {
    "pincminer": ("PIncMiner", "C0", "o"),
    "batch": ("BatchMiner", "C1", "s"),
    "nocs": (r"PIncMiner$_{\mathsf{noCS}}$", "C4", "v"),
    "noaux": (r"PIncMiner$_{\mathsf{noAux}}$", "C5", "p"),
    "staticcorr": (r"PIncMiner$_{\mathsf{staticCorr}}$", "C6", "X"),
}
METHOD_ORDER = tuple(METHODS)


def configure_style() -> None:
    plt.rcParams.update({
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{sansmath}\sansmath",
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Avant Garde", "Computer Modern Sans serif"],
        "font.size": FONT_SIZE,
        "axes.labelsize": FONT_SIZE,
        "axes.titlesize": FONT_SIZE,
        "xtick.labelsize": TICK_LABEL_SIZE,
        "ytick.labelsize": TICK_LABEL_SIZE,
        "legend.fontsize": FONT_SIZE,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def number(row: dict[str, str], field: str) -> float | None:
    value = row.get(field, "")
    if value.strip().lower() in {"", "nan", "na", "n/a"}:
        return None
    return float(value)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def panel_rows(rows: Iterable[dict[str, str]], panel: str) -> list[dict[str, str]]:
    selected = [row for row in rows if row.get("panel") == panel]
    if not selected:
        raise ValueError(f"panel {panel!r} has no rows")
    return selected


def one(rows: Iterable[dict[str, str]], **match: str) -> dict[str, str] | None:
    found = [row for row in rows if all(row.get(key) == value for key, value in match.items())]
    if len(found) > 1:
        raise ValueError(f"duplicate CSV rows for {match}: {len(found)}")
    return found[0] if found else None


def finish(fig: plt.Figure, ax: plt.Axes, output: Path, ylabel: str, log_y: bool = False) -> None:
    ax.set_xlabel("")
    ax.set_ylabel(ylabel, fontsize=22)
    ax.tick_params(axis="both", labelsize=TICK_LABEL_SIZE)
    if log_y:
        ax.set_yscale("log")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {output}")


def plot_method(ax: plt.Axes, xs: list[float], ys: list[float], method: str) -> None:
    label, color, marker = METHODS[method]
    ax.plot(xs, ys, color=color, marker=marker, ms=MARKER_SIZE,
            markeredgewidth=4, linewidth=1.8, linestyle="-", label=label)


def categorical_runtime(
    rows: list[dict[str, str]], slots: tuple[str, ...], labels: tuple[str, ...],
    output: Path, *, rotation: float = 0,
) -> None:
    fig, ax = plt.subplots(figsize=FIGSIZE)
    xs = list(range(len(slots)))
    plotted = 0
    for method in METHOD_ORDER:
        variant = "pincminer" if method == "batch" else method
        field = "batch_runtime_s" if method == "batch" else "inc_runtime_s"
        ys: list[float] = []
        for slot in slots:
            row = one(rows, slot=slot, variant=variant)
            value = number(row, field) if row else None
            ys.append(math.nan if value is None else value)
        if all(math.isnan(value) for value in ys):
            continue
        plot_method(ax, xs, ys, method)
        plotted += 1
    if not plotted:
        raise ValueError("no runtime values available")
    ax.set_xticks(xs, labels, rotation=rotation, ha="right" if rotation else "center")
    finish(fig, ax, output, "Running Time (s)", log_y=True)


def recall_by_depth(rows: list[dict[str, str]], output: Path) -> None:
    slots = ("h1", "h3", "h5", "h7", "h9")
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for method in ("pincminer", "nocs", "staticcorr"):
        ys = []
        for slot in slots:
            row = one(rows, slot=slot, variant=method)
            value = number(row, "recall") if row else None
            ys.append(math.nan if value is None else value)
        if not all(math.isnan(value) for value in ys):
            plot_method(ax, [1, 3, 5, 7, 9], ys, method)
    ax.set_xticks([1, 3, 5, 7, 9])
    finite = [value for line in ax.lines for value in line.get_ydata() if not math.isnan(value)]
    ax.set_ylim(max(0.0, min(finite) - 0.01), 1.005)
    finish(fig, ax, output, "Recall")


def sketch_runtime_memory(rows: list[dict[str, str]], panel: str, output: Path) -> None:
    slots = (("w65536", "w262144", "w1048576", "w4194304", "w16777216")
             if panel == "a" else ("h1", "h3", "h5", "h7", "h9"))
    xs = ([16, 18, 20, 22, 24] if panel == "a" else [1, 3, 5, 7, 9])
    labels = ([rf"$2^{{{x}}}$" for x in xs] if panel == "a" else [str(x) for x in xs])
    selected = [one(rows, slot=slot, variant="pincminer") for slot in slots]
    runtime = [number(row, "inc_runtime_s") for row in selected]
    memory = [number(row, "peak_mem_mb") for row in selected]
    if any(value is None for value in runtime + memory):
        raise ValueError(f"panel {panel} lacks PIncMiner runtime or memory")
    fig, ax1 = plt.subplots(figsize=FIGSIZE)
    ax2 = ax1.twinx()
    line1 = ax1.plot(xs, runtime, color="tab:blue", marker="o", ms=MARKER_SIZE,
                     markeredgewidth=4, linewidth=1.8, label="Runtime")[0]
    line2 = ax2.plot(xs, memory, color="tab:red", marker="s", ms=MARKER_SIZE,
                     markeredgewidth=4, linewidth=1.8, label="Memory")[0]
    ax1.set_xlabel("")
    ax1.set_ylabel("Runtime (s)", color="tab:blue", fontsize=22)
    ax2.set_ylabel("Memory (MB)", color="tab:red", fontsize=22)
    ax1.tick_params(axis="y", labelcolor="tab:blue", labelsize=TICK_LABEL_SIZE)
    ax2.tick_params(axis="y", labelcolor="tab:red", labelsize=TICK_LABEL_SIZE)
    ax1.tick_params(axis="x", labelsize=TICK_LABEL_SIZE)
    ax1.set_xticks(xs, labels)
    ax1.legend([line1, line2], ["Runtime", "Memory"], loc="upper center",
               bbox_to_anchor=(0.5, 1.15), ncol=2, frameon=True)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {output}")


def repeated_runtime(rows: list[dict[str, str]], output: Path) -> None:
    rounds = list(range(10))
    categorical_runtime(rows, tuple(f"r{round_}" for round_ in rounds),
                        tuple(str(round_) for round_ in rounds), output)


def aff_runtime(rows: list[dict[str, str]], output: Path) -> None:
    points = []
    for row in rows:
        if row.get("variant") != "pincminer":
            continue
        x = number(row, "aff_total")
        y = number(row, "inc_runtime_s")
        if x is not None and y is not None:
            points.append((x, y))
    if not points:
        raise ValueError("panel m lacks AFF/runtime values")
    points.sort()
    fig, ax = plt.subplots(figsize=FIGSIZE)
    plot_method(ax, [x for x, _ in points], [y for _, y in points], "pincminer")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.1e}"))
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    finish(fig, ax, output, "Running Time (s)", log_y=True)


def shared_legend(output: Path) -> None:
    fig = plt.figure(figsize=(8, 0.6))
    handles = [
        Line2D([], [], color=METHODS[method][1], marker=METHODS[method][2],
               markersize=8, markeredgewidth=4, linestyle="-", label=METHODS[method][0])
        for method in METHOD_ORDER
    ]
    fig.legend(handles=handles, ncol=len(handles), loc="center", frameon=True)
    plt.axis("off")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {output}")


def generate(rows: list[dict[str, str]], output: Path) -> None:
    percent = (r"$1\%$", r"$5\%$", r"$15\%$", r"$20\%$", r"$30\%$")
    jobs = [
        ("legend", lambda: shared_legend(output / "legend.pdf")),
        ("a", lambda: sketch_runtime_memory(panel_rows(rows, "a"), "a", output / "a-dblp-vary-w.pdf")),
        ("b", lambda: sketch_runtime_memory(panel_rows(rows, "b"), "b", output / "b-dblp-vary-h.pdf")),
        ("c", lambda: recall_by_depth(panel_rows(rows, "c"), output / "c-adult-vary-h-recall.pdf")),
        ("d", lambda: categorical_runtime(panel_rows(rows, "d"), ("a1d1", "a5d2", "a15d3", "a20d4", "a30d5"), (r"$(1\%,1\%)$", r"$(5\%,2\%)$", r"$(15\%,3\%)$", r"$(20\%,4\%)$", r"$(30\%,5\%)$"), output / "d-dblp-vary-add-dominant.pdf", rotation=15)),
        ("e", lambda: categorical_runtime(panel_rows(rows, "e"), ("a1d1", "a2d5", "a3d15", "a4d20", "a5d30"), (r"$(1\%,1\%)$", r"$(2\%,5\%)$", r"$(3\%,15\%)$", r"$(4\%,20\%)$", r"$(5\%,30\%)$"), output / "e-dblp-vary-delete-dominant.pdf", rotation=15)),
        ("f", lambda: categorical_runtime(panel_rows(rows, "f"), ("add1", "add5", "add15", "add20", "add30"), percent, output / "f-dblp-vary-add.pdf")),
        ("g", lambda: categorical_runtime(panel_rows(rows, "g"), ("del1", "del5", "del10", "del15", "del20", "del25", "del30"), (r"$1\%$", r"$5\%$", r"$10\%$", r"$15\%$", r"$20\%$", r"$25\%$", r"$30\%$"), output / "g-ncvoter-vary-delete.pdf")),
        ("h", lambda: categorical_runtime(panel_rows(rows, "h"), ("add1", "add5", "add15", "add20", "add30"), percent, output / "h-ncvoter-vary-add.pdf")),
        ("i", lambda: categorical_runtime(panel_rows(rows, "i"), ("del1", "del5", "del15", "del20", "del30"), percent, output / "i-ncvoter-ml40-vary-delete.pdf")),
        ("j", lambda: categorical_runtime(panel_rows(rows, "j"), ("add1", "add5", "add10", "add15", "add20", "add25", "add30"), (r"$1\%$", r"$5\%$", r"$10\%$", r"$15\%$", r"$20\%$", r"$25\%$", r"$30\%$"), output / "j-ncvoter-vary-add-sigma.pdf")),
        ("k", lambda: repeated_runtime(panel_rows(rows, "k"), output / "k-ncvoter-repeated-updates.pdf")),
        ("l", lambda: categorical_runtime(panel_rows(rows, "l"), ("d0.2", "d0.4", "d0.6", "d0.8", "d1.0"), (r"$20\%$", r"$40\%$", r"$60\%$", r"$80\%$", r"$100\%$"), output / "l-dblp-vary-dataset-size.pdf")),
        ("m", lambda: aff_runtime(panel_rows(rows, "m"), output / "m-ncvoter-vary-aff.pdf")),
        ("n", lambda: categorical_runtime(panel_rows(rows, "n"), ("p20", "p25", "p30", "p35", "p40"), ("20", "25", "30", "35", "40"), output / "n-ncvoter-vary-p0.pdf")),
        ("o", lambda: categorical_runtime(panel_rows(rows, "o"), ("sig1e-6", "sig1e-5", "sig1e-4", "sig1e-3", "sig1e-2"), (r"$10^{-6}$", r"$10^{-5}$", r"$10^{-4}$", r"$10^{-3}$", r"$10^{-2}$"), output / "o-inspection-vary-sigma.pdf")),
        ("p", lambda: categorical_runtime(panel_rows(rows, "p"), ("conf0.7", "conf0.8", "conf0.9", "conf0.95"), ("0.7", "0.8", "0.9", "0.95"), output / "p-inspection-vary-delta.pdf")),
    ]
    for name, job in jobs:
        try:
            job()
        except ValueError as error:
            print(f"skipped {name}: {error}")
    print(f"done: {len(list(output.glob('*.pdf')))} PDFs in {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    configure_style()
    output = args.output or args.csv.resolve().parent / "plots-results-sep-8"
    generate(read_rows(args.csv), output)


if __name__ == "__main__":
    main()
