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
    "incdc": ("IncDC", "C2", "^"),
    "dc3": ("3DC", "C3", "d"),
    "nocs": (r"PIncMiner$_{\mathsf{noCS}}$", "C4", "v"),
    "noaux": (r"PIncMiner$_{\mathsf{noAux}}$", "C5", "p"),
    "staticcorr": (r"PIncMiner$_{\mathsf{staticCorr}}$", "C6", "X"),
}
METHOD_ORDER = tuple(METHODS)
LEGEND_ORDER = METHOD_ORDER
BASELINE_ORDER = ("batch", "incdc", "dc3", "nocs", "noaux", "staticcorr")
# Distinct from METHODS markers (o, s, v, p, X, ^, d).
EXPECTED_MARKER = "D"
PLAIN_NAMES = {
    "pincminer": "PIncMiner",
    "batch": "BatchMiner",
    "incdc": "IncDC",
    "dc3": "3DC",
    "nocs": "PIncMiner_noCS",
    "noaux": "PIncMiner_noAux",
    "staticcorr": "PIncMiner_staticCorr",
}
ERRORBAR_CAPSIZE = 4
OOM_SECONDS = 5000.0
DC_METHODS = frozenset({"incdc", "dc3"})

# Panels d-p, using the same slots as the corresponding plots.
SPEEDUP_PANELS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("d", ("a1d1", "a5d2", "a15d3", "a20d4", "a30d5")),
    ("e", ("a1d1", "a2d5", "a3d15", "a4d20", "a5d30")),
    ("f", ("add1", "add5", "add15", "add20", "add30")),
    ("g", ("del1", "del5", "del10", "del15", "del20", "del25", "del30")),
    ("h", ("add1", "add5", "add15", "add20", "add30")),
    ("i", ("del1", "del5", "del15", "del20", "del30")),
    ("j", ()),  # breakdown plot; F/S/E/R/C, not method lines
    ("k", tuple(f"r{round_}" for round_ in range(10))),
    ("l", ("d0.2", "d0.4", "d0.6", "d0.8", "d1.0")),
    ("m", ()),  # AFF plot draws only PIncMiner
    ("n", ("p20", "p25", "p30", "p35", "p40")),
    ("o", ("sig1e-6", "sig1e-5", "sig1e-4", "sig1e-3", "sig1e-2")),
    ("p", ("conf0.7", "conf0.8", "conf0.9", "conf0.95")),
)


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


def finish(fig: plt.Figure, ax: plt.Axes, output: Path, ylabel: str,
           log_y: bool = False, tight: bool = True) -> None:
    ax.set_xlabel("")
    ax.set_ylabel(ylabel, fontsize=22)
    ax.tick_params(axis="both", labelsize=TICK_LABEL_SIZE)
    if log_y:
        ax.set_yscale("log")
    if tight:
        fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {output}")


def is_dc_oom(method: str, value: float | None) -> bool:
    """IncDC/3DC store OutOfMemory as 5000 s in the CSV; do not plot that value."""
    return method in DC_METHODS and value is not None and value >= OOM_SECONDS


def plot_method(ax: plt.Axes, xs: list[float], ys: list[float], method: str) -> None:
    label, color, marker = METHODS[method]
    pairs = [(x, y) for x, y in zip(xs, ys) if not math.isnan(y)]
    if not pairs:
        return
    xs, ys = [x for x, _ in pairs], [y for _, y in pairs]
    ax.plot(xs, ys, color=color, marker=marker, ms=MARKER_SIZE,
            markeredgewidth=4, linewidth=1.8, linestyle="-", label=label)


def _axis_break_marks(ax_top: plt.Axes, ax_bot: plt.Axes) -> None:
    kwargs = dict(color="k", clip_on=False, linewidth=1.0)
    d = 0.012
    ax_bot.plot((-d, +d), (1 - d, 1 + d), transform=ax_bot.transAxes, **kwargs)
    ax_bot.plot((1 - d, 1 + d), (1 - d, 1 + d), transform=ax_bot.transAxes, **kwargs)
    ax_top.plot((-d, +d), (-d, +d), transform=ax_top.transAxes, **kwargs)
    ax_top.plot((1 - d, 1 + d), (-d, +d), transform=ax_top.transAxes, **kwargs)


def categorical_runtime(
    rows: list[dict[str, str]], slots: tuple[str, ...], labels: tuple[str, ...],
    output: Path, *, rotation: float = 0, annotate_oom: bool = False,
) -> None:
    xs = list(range(len(slots)))
    series: list[tuple[str, list[float]]] = []
    oom_series: list[tuple[str, list[float]]] = []
    for method in METHOD_ORDER:
        variant = "pincminer" if method == "batch" else method
        field = "batch_runtime_s" if method == "batch" else "inc_runtime_s"
        ys: list[float] = []
        oom_ys: list[float] = []
        for slot in slots:
            row = one(rows, slot=slot, variant=variant)
            value = number(row, field) if row else None
            if is_dc_oom(method, value):
                ys.append(math.nan)
                oom_ys.append(1.0)
            else:
                ys.append(math.nan if value is None else value)
                oom_ys.append(math.nan)
        if not all(math.isnan(value) for value in ys):
            series.append((method, ys))
        if not all(math.isnan(value) for value in oom_ys):
            oom_series.append((method, oom_ys))
    if not series:
        raise ValueError("no runtime values available")
    if annotate_oom and oom_series:
        fig, (ax_to, ax) = plt.subplots(
            2, 1, sharex=True, figsize=FIGSIZE,
            gridspec_kw={"height_ratios": [1.05, 3.4], "hspace": 0.06},
        )
        for method, ys in series:
            plot_method(ax, xs, ys, method)
        for method, ys in oom_series:
            plot_method(ax_to, xs, ys, method)
        ax_to.set_ylim(0.35, 1.65)
        ax_to.set_yticks([1.0])
        ax_to.set_yticklabels(["TO"])
        ax_to.tick_params(axis="x", bottom=False, labelbottom=False, labelsize=TICK_LABEL_SIZE)
        ax_to.tick_params(axis="y", labelsize=TICK_LABEL_SIZE)
        ax_to.spines["bottom"].set_visible(False)
        ax.spines["top"].set_visible(False)
        _axis_break_marks(ax_to, ax)
        ax.set_xticks(xs, labels, rotation=rotation, ha="right" if rotation else "center")
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="both", labelsize=TICK_LABEL_SIZE)
        ax.set_yscale("log")
        fig.supylabel("Running Time (s)", fontsize=22)
        fig.subplots_adjust(left=0.18, right=0.97, top=0.97, bottom=0.12)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"saved: {output}")
        return
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for method, ys in series:
        plot_method(ax, xs, ys, method)
    ax.set_xticks(xs, labels, rotation=rotation, ha="right" if rotation else "center")
    finish(fig, ax, output, "Running Time (s)", log_y=True)


def expected_recall(depth: int) -> float:
    """Lemma 1: beta = Theta(e^{-h}), so expected recall is 1-beta = 1-e^{-h}.

    Width w only sets the Count-Sketch error eps (w = Theta(eps^{-2})); it does not
    enter the recall bound. Panel c holds w = 4096 and varies h.
    """
    return 1.0 - math.exp(-depth)


def recall_by_depth(rows: list[dict[str, str]], output: Path) -> None:
    slots = ("h1", "h3", "h5", "h7", "h9")
    depths = [1, 3, 5, 7, 9]
    actual: list[float] = []
    yerr: list[float] = []
    for slot in slots:
        row = one(rows, slot=slot, variant="pincminer")
        value = number(row, "recall") if row else None
        if value is None:
            raise ValueError(f"panel c lacks PIncMiner recall for {slot}")
        actual.append(value)
        yerr.append(number(row, "recall_std") or 0.0)
    predicted = [expected_recall(depth) for depth in depths]
    fig, ax = plt.subplots(figsize=FIGSIZE)
    _, color, marker = METHODS["pincminer"]
    ax.errorbar(depths, actual, yerr=yerr, color=color, marker=marker, ms=MARKER_SIZE,
                markeredgewidth=4, linewidth=1.8, linestyle="-", capsize=ERRORBAR_CAPSIZE,
                label=METHODS["pincminer"][0])
    ax.plot(depths, predicted, color="C1", marker=EXPECTED_MARKER, ms=MARKER_SIZE,
            markeredgewidth=4, linewidth=1.8, linestyle="-", label="Expected")
    ax.set_xticks(depths)
    ax.set_ylim(max(0.0, min(actual + predicted) - 0.03), 1.01)
    ax.set_xlabel("")
    ax.set_ylabel("Recall", fontsize=22)
    ax.tick_params(axis="both", labelsize=TICK_LABEL_SIZE)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=True,
               bbox_to_anchor=(0.55, 0.98))
    fig.subplots_adjust(top=0.85, bottom=0.12, left=0.16, right=0.96)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {output}")


def breakdown_runtime(rows: list[dict[str, str]], output: Path) -> None:
    """PIncMiner cost breakdown vs insert ratio (paper fig-breakdown)."""
    slots = ("add1", "add5", "add15", "add20", "add30")
    labels = (r"$1\%$", r"$5\%$", r"$15\%$", r"$20\%$", r"$30\%$")
    components = (
        ("F", "breakdown_F_s", "C0", "P"),
        ("S", "breakdown_S_s", "C1", "X"),
        ("E", "breakdown_E_s", "C2", "h"),
        ("R", "breakdown_R_s", "C3", "H"),
        ("C", "breakdown_C_s", "C4", "D"),
    )
    fig, ax = plt.subplots(figsize=FIGSIZE)
    xs = list(range(len(slots)))
    plotted = []
    for label, field, color, marker in components:
        ys: list[float] = []
        for slot in slots:
            row = one(rows, slot=slot, variant="pincminer")
            value = number(row, field) if row else None
            ys.append(math.nan if value is None else value)
        if all(math.isnan(value) or value <= 0 for value in ys):
            continue
        ax.plot(xs, ys, color=color, marker=marker, ms=MARKER_SIZE,
                markeredgewidth=4, linewidth=1.8, linestyle="-", label=label)
        plotted.append(label)
    if not plotted:
        raise ValueError("no breakdown values available")
    ax.set_xticks(xs, labels)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15),
              ncol=len(plotted), frameon=True)
    finish(fig, ax, output, "Running Time (s)", log_y=True)


def sketch_runtime_memory(rows: list[dict[str, str]], panel: str, output: Path) -> None:
    slots = (("w65536", "w262144", "w1048576", "w4194304", "w16777216")
             if panel == "a" else ("h1", "h3", "h5", "h7", "h9"))
    xs = ([16, 18, 20, 22, 24] if panel == "a" else [1, 3, 5, 7, 9])
    labels = ([rf"$2^{{{x}}}$" for x in xs] if panel == "a" else [str(x) for x in xs])
    selected = [one(rows, slot=slot, variant="pincminer") for slot in slots]
    runtime = [number(row, "inc_runtime_s") if row else None for row in selected]
    memory = []
    for row in selected:
        mem = number(row, "peak_mem_mb") if row else None
        if mem is None and row is not None:
            mem = number(row, "sk_mb")
        memory.append(mem)
    yerr = [(number(row, "inc_runtime_s_std") or 0.0) if row else 0.0 for row in selected]
    if any(value is None for value in runtime + memory):
        raise ValueError(f"panel {panel} lacks PIncMiner runtime or memory")
    fig, ax1 = plt.subplots(figsize=FIGSIZE)
    ax2 = ax1.twinx()
    line1 = ax1.errorbar(xs, runtime, yerr=yerr, color="tab:blue", marker="o",
                         ms=MARKER_SIZE, markeredgewidth=4, linewidth=1.8,
                         capsize=ERRORBAR_CAPSIZE, label="Runtime")
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
        join = number(row, "aff_join_s")
        relabel = number(row, "aff_relabel_s")
        if x is None or join is None or relabel is None:
            continue
        points.append((x, join + relabel))
    if not points:
        raise ValueError("panel m lacks AFF join/relabel values")
    points.sort()
    fig, ax = plt.subplots(figsize=FIGSIZE)
    plot_method(ax, [x for x, _ in points], [y for _, y in points], "pincminer")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.1e}"))
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    finish(fig, ax, output, "Running Time (s)")


def slot_times(
    rows: list[dict[str, str]], slots: tuple[str, ...], method: str,
) -> list[float | None]:
    """Times matching the plotted lines: BatchMiner from the PIncMiner row."""
    variant = "pincminer" if method == "batch" else method
    field = "batch_runtime_s" if method == "batch" else "inc_runtime_s"
    values: list[float | None] = []
    for slot in slots:
        row = one(rows, slot=slot, variant=variant)
        value = number(row, field) if row else None
        values.append(None if is_dc_oom(method, value) else value)
    return values


def point_speedups(pinc: list[float | None], baseline: list[float | None]) -> list[float]:
    ratios: list[float] = []
    for ours, theirs in zip(pinc, baseline):
        if ours is None or theirs is None or ours <= 0:
            continue
        ratios.append(theirs / ours)
    return ratios


def format_speedup(value: float) -> str:
    return f"{value:.2f}x"


def report_speedups(rows: list[dict[str, str]], output: Path) -> None:
    """PIncMiner vs each plotted line on panels d-p: arithmetic mean and max."""
    lines = [
        "PIncMiner speedup vs each plotted line (panels d-p)",
        "speedup = baseline_time / pincminer inc_runtime_s",
        "BatchMiner uses batch_runtime_s on the pincminer row",
        "mean is the arithmetic mean over plotted x-points (panel k: 10 rounds)",
        "IncDC/3DC OOM (5000 s placeholder) is excluded from the plot and from speedups",
        "",
        f"{'panel':<6} {'vs':<22} {'n':>3} {'mean':>10} {'max':>10} {'n<1':>4}",
        "-" * 60,
    ]
    for panel, slots in SPEEDUP_PANELS:
        selected = [row for row in rows if row.get("panel") == panel]
        if not selected:
            lines.append(f"{panel:<6} {'(no rows)':<22}")
            continue
        if panel == "j":
            lines.append(f"{panel:<6} {'(F/S/E/R/C breakdown)':<22}")
            continue
        if panel == "m" or not slots:
            lines.append(f"{panel:<6} {'(only PIncMiner plotted)':<22}")
            continue
        pinc = slot_times(selected, slots, "pincminer")
        any_baseline = False
        for method in BASELINE_ORDER:
            baseline = slot_times(selected, slots, method)
            ratios = point_speedups(pinc, baseline)
            if not ratios:
                continue
            any_baseline = True
            mean = sum(ratios) / len(ratios)
            maximum = max(ratios)
            slower = sum(1 for value in ratios if value < 1)
            lines.append(
                f"{panel:<6} {PLAIN_NAMES[method]:<22} {len(ratios):>3} "
                f"{format_speedup(mean):>10} {format_speedup(maximum):>10} {slower:>4}"
            )
        if not any_baseline:
            lines.append(f"{panel:<6} {'(no other plotted line)':<22}")
    text = "\n".join(lines) + "\n"
    output.mkdir(parents=True, exist_ok=True)
    table_path = output / "speedup-table-d-p.txt"
    table_path.write_text(text)
    print(text, end="")
    print(f"saved: {table_path}")


def shared_legend(output: Path) -> None:
    styles = METHODS
    fig = plt.figure(figsize=(12, 0.6))
    handles = [
        Line2D([], [], color=styles[method][1], marker=styles[method][2],
               markersize=8, markeredgewidth=4, linestyle="-", label=styles[method][0])
        for method in LEGEND_ORDER
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
        ("h", lambda: categorical_runtime(panel_rows(rows, "h"), ("add1", "add5", "add15", "add20", "add30"), percent, output / "h-ncvoter-vary-add.pdf", annotate_oom=True)),
        ("i", lambda: categorical_runtime(panel_rows(rows, "i"), ("del1", "del5", "del15", "del20", "del30"), percent, output / "i-ncvoter-ml40-vary-delete.pdf", annotate_oom=True)),
        ("j", lambda: breakdown_runtime(panel_rows(rows, "f"), output / "j-dblp-add-breakdown.pdf")),
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
    report_speedups(rows, output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--speedup-only", action="store_true",
                        help="Print the d-p speedup table without generating PDFs")
    args = parser.parse_args()
    output = args.output or args.csv.resolve().parent / "plots-results-sep-8"
    rows = read_rows(args.csv)
    if args.speedup_only:
        report_speedups(rows, output)
        return
    configure_style()
    generate(rows, output)


if __name__ == "__main__":
    main()
