#!/usr/bin/env python3
"""Plot Adult F1 scores for IncMiner and DCFinder while varying eta."""

from pathlib import Path

import matplotlib.pyplot as plt


plt.rcParams["figure.figsize"] = [4, 3]
plt.rcParams["font.size"] = 16
plt.rcParams["lines.markersize"] = 10

colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

eta = [0.5, 0.6, 0.7, 0.8, 0.9]
incminer_f1 = [0.7439, 0.6994, 0.7039, 0.7217, 0.7224]
dcfinder_f1 = [0.7059, 0.6015, 0.5355, 0.5695, 0.4807]

fig, ax = plt.subplots()

ax.plot(
    eta,
    incminer_f1,
    color=colors[0],
    linestyle="-",
    marker="s",
    markerfacecolor="none",
    markeredgewidth=1.5,
)
ax.plot(
    eta,
    dcfinder_f1,
    color=colors[5],
    linestyle="-",
    marker="x",
    markeredgewidth=1.0,
)

ax.set_ylabel("F1")
ax.set_xticks(eta)

fig.tight_layout()

output_path = Path("plots/incminer_adult_f1_varying_eta.pdf")
output_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(output_path, bbox_inches="tight")
plt.close(fig)

print(f"Saved {output_path}")
