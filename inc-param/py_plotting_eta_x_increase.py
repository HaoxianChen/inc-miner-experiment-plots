#!/usr/bin/env python3
"""
Regenerate plots/exp1_hospital_eta_X_increase.pdf
Matches old notebook cell 19 structure exactly.
Main data:  result_pearson.xlsx "Δ𝜂 > 0 in X"
IApriori:   result_fp_growth.xlsx "Δ𝜂 > 0 in X", col gen rule time(s)
DCFinder:   flat timeout line at top of top panel
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

plt.rcParams['figure.figsize'] = [4, 3]
plt.rcParams['font.size'] = 16
plt.rcParams['lines.markersize'] = 10

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

colorMap = {
    "batch":       colors[1],
    "IncMiner":    colors[0],
    "IncMinerNS":  colors[4],
    "IncMinerNoO": colors[2],
    "IApriori":    colors[3],
    "DCFinder":    colors[5],
}
markerMap = {
    "batch":       '^',
    "IncMiner":    's',
    "IncMinerNS":  '*',
    "IncMinerNoO": 'v',
    "IApriori":    '+',
    "DCFinder":    'x',
}
markerStyleMap = {
    "batch":       {'markerfacecolor': 'none',  'markeredgewidth': 1.0},
    "IncMiner":    {'markerfacecolor': 'none',  'markeredgewidth': 1.5},
    "IncMinerNS":  {'markerfacecolor': 'none',  'markeredgewidth': 1.0, 'markersize': 10},
    "IncMinerNoO": {'markerfacecolor': 'none',  'markeredgewidth': 1.0, 'markersize': 10},
    "IApriori":    {'markerfacecolor': 'auto',  'markeredgewidth': 1.5},
    "DCFinder":    {'markerfacecolor': 'auto',  'markeredgewidth': 1.0},
}
legendName = {
    "batch":       "BatchMiner",
    "IncMiner":    r"$IncMiner^\approx_\Omega$",
    "IncMinerNS":  r"$IncMiner_{NS}$",
    "IncMinerNoO": r"$IncMiner_{-\Omega}$",
    "IApriori":    r"$IApriori_\Theta$",
    "DCFinder":    "DCFinder",
}
ylabel_time = 'Running Time (s)'

# =============================================================================
# Load data
# =============================================================================

tab = 'Δ𝜂 > 0 in X'

df = pd.read_excel('result_pearson.xlsx', sheet_name=tab)
df['etaDiff'] = df['new eta'] - df['old eta']
df['time'] = df['Mining time'] / 1000   # ms -> s
groups = df.groupby('Baseline')

X = groups.get_group('IncMiner')['etaDiff'].values
N = len(X)
xtick_labels = [f"+{v:.2f}" for v in X]

df_iapriori = pd.read_excel('result_fp_growth.xlsx', sheet_name=tab)

top_labels    = ['batch', 'IncMinerNS']
bottom_labels = ['IncMiner', 'IncMinerNoO']

# =============================================================================
# Figure: two vertically stacked boxes, height_ratios [1, 3]
# =============================================================================

fig, (ax_top, ax_bottom) = plt.subplots(
    2, 1,
    sharex=True,
    gridspec_kw={'height_ratios': [1, 3]},
)

def plot_line(ax, Y, label):
    ax.plot(
        range(N), Y, label=legendName[label],
        marker=markerMap[label], color=colorMap[label],
        markerfacecolor=markerStyleMap[label].get('markerfacecolor', 'auto'),
        markeredgewidth=markerStyleMap[label].get('markeredgewidth', 1.0),
        markersize=markerStyleMap[label].get('markersize', 10),
    )

# --- Top box: batch + IncMinerNS first ---
for label in top_labels:
    plot_line(ax_top, groups.get_group(label)['time'].values, label)

# Compute timeout from top-panel data (matches old script's global_max_time pattern)
global_max_time = max(groups.get_group(l)['time'].max() for l in top_labels)
timeout_value = global_max_time * 1.1

# DCFinder: flat line at timeout_value (timed out)
plot_line(ax_top, [timeout_value] * N, "DCFinder")

# TO tick on top panel (matches old script exactly)
ax_top.set_yticks(list(ax_top.get_yticks())[:3] + [timeout_value])
ax_top.set_yticklabels(
    [str(int(tick)) if tick <= global_max_time else 'TO'
     for tick in ax_top.get_yticks()[:4]]
)

ymin, ymax = ax_top.get_ylim()
ax_top.set_ylim(ymin * 0.95, ymax * 1.05)

# --- Bottom box: IncMiner + IncMinerNoO + IApriori ---
for label in bottom_labels:
    plot_line(ax_bottom, groups.get_group(label)['time'].values, label)
plot_line(ax_bottom, df_iapriori['gen rule time(s)'].values, "IApriori")

# x-axis ticks
plt.xticks(range(N), xtick_labels)
fig.text(0.01, 0.5, ylabel_time, va='center', rotation='vertical')

# Broken-axis visual cues
ax_top.spines['bottom'].set_visible(False)
ax_bottom.spines['top'].set_visible(False)
ax_top.tick_params(axis='x', which='both', bottom=False)
d = 0.015
for ax, y_pos in [(ax_top, (-d, +d)), (ax_bottom, (1 - d, 1 + d))]:
    kw = dict(transform=ax.transAxes, color='k', clip_on=False, lw=1)
    ax.plot((-d, +d), y_pos, **kw)
    ax.plot((1 - d, 1 + d), y_pos, **kw)

plt.tight_layout()
fig.subplots_adjust(hspace=0.08)

os.makedirs('plots', exist_ok=True)
plt.savefig('plots/exp1_hospital_eta_X_increase.pdf', bbox_inches='tight')
plt.close()
print("Saved plots/exp1_hospital_eta_X_increase.pdf")
