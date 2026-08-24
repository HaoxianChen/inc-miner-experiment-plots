#!/usr/bin/env python3
"""Generate paper-style plots from an f5_mcorr experiment directory.

The input root is expected to contain the named f5_mcorr sweep directories.
Missing sweep directories are reported and skipped, so the same script can be
reused for later partial iterations by changing --root.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import tempfile
from pathlib import Path
from typing import Sequence

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "f5-mcorr-matplotlib-cache")
)

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
    "pincminer": {
        "label": "PIncMiner",
        "color": "C0",
        "marker": "o",
    },
    "batch": {
        "label": "BatchMiner",
        "color": "C1",
        "marker": "s",
    },
    "nocs": {
        "label": r"PIncMiner$_{\mathsf{noCS}}$",
        "color": "C4",
        "marker": "v",
    },
    "noaux": {
        "label": r"PIncMiner$_{\mathsf{noAux}}$",
        "color": "C5",
        "marker": "p",
    },
    "staticcorr": {
        "label": r"PIncMiner$_{\mathsf{staticCorr}}$",
        "color": "C6",
        "marker": "X",
    },
    "naive": {
        "label": r"PIncMiner$_{\mathsf{Naive}}$",
        "color": "C7",
        "marker": "P",
    },
}

DEFAULT_METHODS = ("pincminer", "batch", "nocs", "noaux", "staticcorr")


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "text.usetex": True,
            "text.latex.preamble": r"\usepackage{sansmath}\sansmath",
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Helvetica",
                "Avant Garde",
                "Computer Modern Sans serif",
            ],
            "font.size": FONT_SIZE,
            "axes.labelsize": FONT_SIZE,
            "axes.titlesize": FONT_SIZE,
            "xtick.labelsize": TICK_LABEL_SIZE,
            "ytick.labelsize": TICK_LABEL_SIZE,
            "legend.fontsize": FONT_SIZE,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def as_float(row: dict[str, str], field: str) -> float:
    value = row.get(field, "")
    if value in ("", "nan", "NaN"):
        raise ValueError(f"missing numeric field {field!r}")
    return float(value)


def one_row(rows: Sequence[dict[str, str]], variant: str, source: Path) -> dict[str, str]:
    selected = [row for row in rows if row.get("variant") == variant]
    if len(selected) != 1:
        raise ValueError(
            f"expected one {variant!r} row in {source}, found {len(selected)}"
        )
    return selected[0]


def plot_line(ax: plt.Axes, xs: Sequence[float], ys: Sequence[float], method: str) -> None:
    style = METHODS[method]
    ax.plot(
        xs,
        ys,
        color=style["color"],
        marker=style["marker"],
        ms=MARKER_SIZE,
        markeredgewidth=4,
        linewidth=1.8,
        linestyle="-",
        label=style["label"],
    )


def finish(
    fig: plt.Figure,
    ax: plt.Axes,
    output: Path,
    *,
    ylabel: str,
    log_y: bool = False,
    grid: bool = False,
) -> None:
    ax.set_xlabel("")
    ax.set_ylabel(ylabel, fontsize=22)
    ax.tick_params(axis="both", labelsize=TICK_LABEL_SIZE)
    if log_y:
        ax.set_yscale("log")
    if grid:
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {output}")


def load_sweep(
    base: Path, slots: Sequence[str]
) -> list[tuple[str, Path, list[dict[str, str]]]]:
    loaded = []
    for slot in slots:
        source = base / slot / "results.csv"
        if not source.exists():
            raise FileNotFoundError(source)
        loaded.append((slot, source, read_csv(source)))
    return loaded


def runtime_sweep(
    base: Path,
    slots: Sequence[str],
    labels: Sequence[str],
    output: Path,
    methods: Sequence[str],
    tick_rotation: float = 0,
) -> None:
    loaded = load_sweep(base, slots)
    xs = list(range(len(slots)))
    present = {row.get("variant") for _, _, rows in loaded for row in rows}
    for method in methods:
        source_variant = "pincminer" if method == "batch" else method
        if source_variant not in present:
            print(f"warning: skipping unavailable variant {method} in {base}")
            continue
        field = "batch_runtime_s" if method == "batch" else "inc_runtime_s"
        ys = [
            as_float(one_row(rows, source_variant, source), field)
            for _, source, rows in loaded
        ]
        if "fig" not in locals():
            fig, ax = plt.subplots(figsize=FIGSIZE)
        plot_line(ax, xs, ys, method)
    if "fig" not in locals():
        raise ValueError(f"no plottable methods in {base}")
    ax.set_xticks(
        xs,
        labels,
        rotation=tick_rotation,
        ha="right" if tick_rotation else "center",
    )
    finish(fig, ax, output, ylabel="Running Time (s)", log_y=True)


def union_runtime(
    source: Path,
    slots: Sequence[str],
    labels: Sequence[str],
    output: Path,
    methods: Sequence[str],
    tick_rotation: float = 0,
) -> None:
    rows = read_csv(source)
    xs = list(range(len(slots)))
    available = {row["variant"] for row in rows}
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for method in methods:
        source_variant = "pincminer" if method == "batch" else method
        if source_variant not in available:
            print(f"warning: skipping unavailable variant {method} in {source}")
            continue
        field = "batch_runtime_s" if method == "batch" else "inc_runtime_s"
        ys = []
        for slot in slots:
            selected = [
                row
                for row in rows
                if row["slot"] == slot and row["variant"] == source_variant
            ]
            if len(selected) != 1:
                raise ValueError(
                    f"expected one {source_variant} row for {slot} in {source}"
                )
            ys.append(as_float(selected[0], field))
        plot_line(ax, xs, ys, method)
    ax.set_xticks(
        xs,
        labels,
        rotation=tick_rotation,
        ha="right" if tick_rotation else "center",
    )
    finish(fig, ax, output, ylabel="Running Time (s)", log_y=True)


def width_runtime_memory(root: Path, output: Path) -> None:
    widths = [2**16, 2**18, 2**20, 2**22, 2**24]
    slots = [f"w{width}" for width in widths]
    loaded = load_sweep(root / "a_adult_width", slots)
    rows = [one_row(data, "pincminer", source) for _, source, data in loaded]
    runtime = [as_float(row, "inc_runtime_s") for row in rows]
    memory = [as_float(row, "peak_mem_mb") for row in rows]
    exponents = [int(math.log2(width)) for width in widths]

    fig, ax1 = plt.subplots(figsize=FIGSIZE)
    ax2 = ax1.twinx()
    line1 = ax1.plot(
        exponents,
        runtime,
        color="tab:blue",
        marker="o",
        ms=MARKER_SIZE,
        markeredgewidth=4,
        label="Runtime",
    )[0]
    line2 = ax2.plot(
        exponents,
        memory,
        color="tab:red",
        marker="s",
        ms=MARKER_SIZE,
        markeredgewidth=4,
        label="Memory",
    )[0]
    ax1.set_xlabel("")
    ax1.set_ylabel("Runtime (s)", color="tab:blue", fontsize=22)
    ax2.set_ylabel("Memory (MB)", color="tab:red", fontsize=22)
    ax1.tick_params(axis="y", labelcolor="tab:blue", labelsize=TICK_LABEL_SIZE)
    ax2.tick_params(axis="y", labelcolor="tab:red", labelsize=TICK_LABEL_SIZE)
    ax1.tick_params(axis="x", labelsize=TICK_LABEL_SIZE)
    ax1.set_xticks(exponents, [rf"$2^{{{value}}}$" for value in exponents])
    ax1.legend(
        [line1, line2],
        ["Runtime", "Memory"],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.15),
        ncol=2,
        frameon=True,
    )
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {output}")


def depth_runtime_memory(root: Path, output: Path) -> None:
    depths = [1, 3, 5, 7, 9]
    loaded = load_sweep(root / "b_adult_depth", [f"h{depth}" for depth in depths])
    rows = [one_row(data, "pincminer", source) for _, source, data in loaded]
    runtime = [as_float(row, "inc_runtime_s") for row in rows]
    memory = [as_float(row, "peak_mem_mb") for row in rows]

    fig, ax1 = plt.subplots(figsize=FIGSIZE)
    ax2 = ax1.twinx()
    line1 = ax1.plot(
        depths,
        runtime,
        color="tab:blue",
        marker="o",
        ms=MARKER_SIZE,
        markeredgewidth=4,
        label="Runtime",
    )[0]
    line2 = ax2.plot(
        depths,
        memory,
        color="tab:red",
        marker="s",
        ms=MARKER_SIZE,
        markeredgewidth=4,
        label="Memory",
    )[0]
    ax1.set_xlabel("")
    ax1.set_ylabel("Runtime (s)", color="tab:blue", fontsize=22)
    ax2.set_ylabel("Memory (MB)", color="tab:red", fontsize=22)
    ax1.tick_params(axis="y", labelcolor="tab:blue", labelsize=TICK_LABEL_SIZE)
    ax2.tick_params(axis="y", labelcolor="tab:red", labelsize=TICK_LABEL_SIZE)
    ax1.tick_params(axis="x", labelsize=TICK_LABEL_SIZE)
    ax1.set_xticks(depths)
    ax1.legend(
        [line1, line2],
        ["Runtime", "Memory"],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.15),
        ncol=2,
        frameon=True,
    )
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {output}")


def depth_recall(root: Path, output: Path) -> None:
    depths = [1, 3, 5, 7, 9]
    loaded = load_sweep(root / "b_adult_depth", [f"h{depth}" for depth in depths])
    rows = [one_row(data, "pincminer", source) for _, source, data in loaded]
    actual = [as_float(row, "recall") for row in rows]
    expected = [1.0 - math.exp(-depth) for depth in depths]

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(
        depths,
        actual,
        color="C0",
        marker="o",
        ms=MARKER_SIZE,
        markeredgewidth=4,
        label="Actual",
    )
    ax.plot(
        depths,
        expected,
        color="C1",
        marker="s",
        ms=MARKER_SIZE,
        markeredgewidth=4,
        label="Expected",
    )
    ax.set_xticks(depths)
    ax.set_ylim(max(0.0, min(expected + actual) - 0.03), 1.01)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.15),
        ncol=2,
        frameon=True,
    )
    finish(fig, ax, output, ylabel="Recall")


def repeated_plots(
    root: Path, output_dir: Path, methods: Sequence[str]
) -> None:
    datasets = ("adult_ml40", "inspection_ml", "ncvoter_ml40")
    for dataset in datasets:
        source = root / "k_repeated" / dataset / "r10" / "results.csv"
        rows = read_csv(source)
        rounds = sorted({int(float(row["round"])) for row in rows})
        fig, ax = plt.subplots(figsize=FIGSIZE)
        available = {row["variant"] for row in rows}
        for method in methods:
            source_variant = "pincminer" if method == "batch" else method
            if source_variant not in available:
                continue
            field = "batch_runtime_s" if method == "batch" else "inc_runtime_s"
            selected = sorted(
                (row for row in rows if row["variant"] == source_variant),
                key=lambda row: int(float(row["round"])),
            )
            plot_line(ax, rounds, [as_float(row, field) for row in selected], method)
        ax.set_xticks(rounds)
        finish(
            fig,
            ax,
            output_dir / f"{dataset}_repeated_updates_runtime.pdf",
            ylabel="Running Time (s)",
            log_y=True,
        )

        fig, ax = plt.subplots(figsize=FIGSIZE)
        for method in methods:
            if method == "batch" or method not in available:
                continue
            selected = sorted(
                (row for row in rows if row["variant"] == method),
                key=lambda row: int(float(row["round"])),
            )
            plot_line(ax, rounds, [as_float(row, "recall") for row in selected], method)
        ax.set_xticks(rounds)
        ax.set_ylim(0.9, 1.005)
        finish(
            fig,
            ax,
            output_dir / f"{dataset}_repeated_updates_recall.pdf",
            ylabel="Recall",
        )


def aff_plot(root: Path, dataset: str, output: Path) -> None:
    slots = ("add1", "add5", "add15", "add20", "add30")
    xs = []
    ys = []
    for slot in slots:
        directory = root / "m_aff" / dataset / slot
        probe_rows = read_csv(directory / "aff_probe.csv")
        if len(probe_rows) != 1:
            raise ValueError(
                f"expected one AFF probe row in {directory / 'aff_probe.csv'}"
            )
        probe = probe_rows[0]
        result = one_row(read_csv(directory / "results.csv"), "pincminer", directory / "results.csv")
        xs.append(as_float(probe, "total"))
        ys.append(as_float(result, "inc_runtime_s"))
    ordered = sorted(zip(xs, ys))
    ordered_x = [point[0] for point in ordered]
    ordered_y = [point[1] for point in ordered]
    fig, ax = plt.subplots(figsize=FIGSIZE)
    plot_line(ax, ordered_x, ordered_y, "pincminer")
    ax.set_xlim(min(ordered_x) * 0.9, max(ordered_x) * 1.05)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.1e}"))
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    finish(fig, ax, output, ylabel="Running Time (s)")


def breakdown_plot(root: Path, output: Path) -> None:
    slots = ("add1", "add5", "add15", "add20", "add30")
    labels = (r"$1\%$", r"$5\%$", r"$15\%$", r"$20\%$", r"$30\%$")
    loaded = load_sweep(root / "f_dblp_add", slots)
    rows = [one_row(data, "pincminer", source) for _, source, data in loaded]
    components = (
        ("F", "breakdown_F_s", "C0", "P"),
        ("S", "breakdown_S_s", "C1", "X"),
        ("E", "breakdown_E_s", "C2", "h"),
        ("R", "breakdown_R_s", "C3", "H"),
        ("C", "breakdown_C_s", "C4", "D"),
    )
    fig, ax = plt.subplots(figsize=FIGSIZE)
    plotted = []
    for label, field, color, marker in components:
        ys = [as_float(row, field) for row in rows]
        if not any(value > 0 for value in ys):
            print(f"warning: breakdown component {label} is zero at every point")
            continue
        ax.plot(
            range(len(slots)),
            ys,
            color=color,
            marker=marker,
            ms=MARKER_SIZE,
            markeredgewidth=4,
            linewidth=1.8,
            label=label,
        )
        plotted.append(label)
    ax.set_xticks(range(len(slots)), labels)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.15),
        ncol=max(1, len(plotted)),
        frameon=True,
    )
    finish(fig, ax, output, ylabel="Running Time (s)", log_y=True)


def save_shared_legend(output: Path, methods: Sequence[str]) -> None:
    fig = plt.figure(figsize=(8, 0.6))
    handles = [
        Line2D(
            [],
            [],
            color=METHODS[method]["color"],
            marker=METHODS[method]["marker"],
            markersize=8,
            markeredgewidth=4,
            linestyle="-",
            label=METHODS[method]["label"],
        )
        for method in methods
    ]
    fig.legend(
        handles=handles,
        ncol=len(handles),
        loc="center",
        frameon=True,
    )
    plt.axis("off")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {output}")


def generate(root: Path, output: Path, include_naive: bool) -> None:
    methods = DEFAULT_METHODS + (("naive",) if include_naive else ())
    percent = (r"$1\%$", r"$5\%$", r"$15\%$", r"$20\%$", r"$30\%$")
    delete_dominant_slots = ("a1d1", "a2d5", "a3d15", "a4d20", "a5d30")
    delete_dominant_labels = (
        r"$(1\%,1\%)$",
        r"$(2\%,5\%)$",
        r"$(3\%,15\%)$",
        r"$(4\%,20\%)$",
        r"$(5\%,30\%)$",
    )

    jobs = [
        ("shared legend", lambda: save_shared_legend(output / "legend.pdf", methods)),
        ("Adult width", lambda: width_runtime_memory(root, output / "adult_width_runtime_memory.pdf")),
        ("Adult depth runtime", lambda: depth_runtime_memory(root, output / "adult_depth_runtime_memory.pdf")),
        ("Adult depth recall", lambda: depth_recall(root, output / "adult_depth_recall.pdf")),
        (
            "NCVoter mixed updates",
            lambda: runtime_sweep(
                root / "d_ncvoter_mixed",
                delete_dominant_slots,
                delete_dominant_labels,
                output / "ncvoter_mixed_delete_dominant_runtime.pdf",
                methods,
                15,
            ),
        ),
        (
            "DBLP mixed updates",
            lambda: union_runtime(
                root / "e_dblp_union" / "union_summary.csv",
                delete_dominant_slots,
                delete_dominant_labels,
                output / "dblp_mixed_delete_dominant_runtime.pdf",
                methods,
                15,
            ),
        ),
        (
            "DBLP additions",
            lambda: runtime_sweep(
                root / "f_dblp_add",
                ("add1", "add5", "add15", "add20", "add30"),
                percent,
                output / "dblp_add_runtime.pdf",
                methods,
            ),
        ),
        (
            "NCVoter additions",
            lambda: runtime_sweep(
                root / "extra_add" / "ncvoter_ml40",
                ("add1", "add5", "add15", "add20", "add30"),
                percent,
                output / "ncvoter_add_runtime.pdf",
                methods,
            ),
        ),
        (
            "NCVoter deletions",
            lambda: runtime_sweep(
                root / "g_ncvoter_delete",
                ("del1", "del5", "del15", "del20", "del30"),
                percent,
                output / "ncvoter_delete_runtime.pdf",
                methods,
            ),
        ),
        ("DBLP breakdown", lambda: breakdown_plot(root, output / "dblp_add_breakdown.pdf")),
        ("repeated updates", lambda: repeated_plots(root, output, methods)),
        (
            "Adult dataset size",
            lambda: runtime_sweep(
                root / "l_adult_dsize",
                ("d0.2", "d0.4", "d0.6", "d0.8", "d1.0"),
                (r"$20\%$", r"$40\%$", r"$60\%$", r"$80\%$", r"$100\%$"),
                output / "adult_dataset_size_runtime.pdf",
                methods,
            ),
        ),
        ("Inspection AFF", lambda: aff_plot(root, "inspection_ml", output / "inspection_aff_runtime.pdf")),
        ("NCVoter AFF", lambda: aff_plot(root, "ncvoter_ml40", output / "ncvoter_aff_runtime.pdf")),
        (
            "NCVoter P0",
            lambda: runtime_sweep(
                root / "n_ncvoter_p0",
                (
                    "ncvoter_ml20/p20",
                    "ncvoter_ml25/p25",
                    "ncvoter_ml30/p30",
                    "ncvoter_ml35/p35",
                    "ncvoter_ml40/p40",
                ),
                ("20", "25", "30", "35", "40"),
                output / "ncvoter_p0_runtime.pdf",
                methods,
            ),
        ),
        (
            "NCVoter sigma",
            lambda: runtime_sweep(
                root / "o_ncvoter_sigma",
                ("sig1e-6", "sig1e-5", "sig1e-4", "sig1e-3", "sig1e-2"),
                (r"$10^{-6}$", r"$10^{-5}$", r"$10^{-4}$", r"$10^{-3}$", r"$10^{-2}$"),
                output / "ncvoter_sigma_runtime.pdf",
                methods,
            ),
        ),
        (
            "NCVoter delta",
            lambda: runtime_sweep(
                root / "p_ncvoter_delta",
                ("conf0.7", "conf0.8", "conf0.9", "conf0.95"),
                ("0.7", "0.8", "0.9", "0.95"),
                output / "ncvoter_delta_runtime.pdf",
                methods,
            ),
        ),
    ]

    skipped = 0
    for name, job in jobs:
        try:
            job()
        except FileNotFoundError as error:
            skipped += 1
            print(f"skipped {name}: missing {error.filename or error}")
    count = len(list(output.glob("*.pdf"))) if output.exists() else 0
    print(f"done: {count} PDFs in {output}; skipped {skipped} plot groups")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="experiment-result directory")
    parser.add_argument(
        "--output",
        type=Path,
        help="output directory (default: <root>/generated_plots)",
    )
    parser.add_argument(
        "--include-naive",
        action="store_true",
        help="include the naive variant in runtime and repeated-update plots",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_matplotlib()
    root = args.root.resolve()
    output = args.output.resolve() if args.output else root / "generated_plots"
    generate(root, output, args.include_naive)


if __name__ == "__main__":
    main()
