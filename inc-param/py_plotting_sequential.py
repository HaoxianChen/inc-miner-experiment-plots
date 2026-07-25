#!/usr/bin/env python3
"""
Sequential cutoff updates plot: IncMiner_Omega vs BatchMiner cumulative runtime.
Data from sequential_update_0724.md — Adult dataset, 4 rounds (eta: 0.9->0.8->0.7->0.6->0.5).
"""

import matplotlib.pyplot as plt
import numpy as np
import os

plt.rcParams['figure.figsize'] = [4, 3]
plt.rcParams['font.size'] = 16
plt.rcParams['lines.markersize'] = 10

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

colorMap = {
    "batch":    colors[1],
    "IncMiner": colors[0],
}
markerMap = {
    "batch":    '^',
    "IncMiner": 's',
}
markerStyleMap = {
    "batch":    {'markerfacecolor': 'none', 'markeredgewidth': 1.0},
    "IncMiner": {'markerfacecolor': 'none', 'markeredgewidth': 1.5},
}
legendName = {
    "batch":    "BatchMiner",
    "IncMiner": r"$IncMiner^\approx_\Omega$",
}

# =============================================================================
# Data (ms -> s)
# Source: sequential_update_0724.md — online cumulative only
# =============================================================================

rounds = [0, 1, 2, 3, 4]

# End-to-end cumulative including initialization (sections 4 & 5)
incdiso_cum  = [578429/1000, 579985/1000, 581720/1000, 583217/1000, 585594/1000]
batchdis_cum = [465283/1000, 958296/1000, 1451839/1000, 1945187/1000, 2422810/1000]

# =============================================================================
# Plot
# =============================================================================

plt.figure()

plt.plot(rounds, incdiso_cum,
         color=colorMap["IncMiner"], marker=markerMap["IncMiner"],
         markerfacecolor=markerStyleMap["IncMiner"]['markerfacecolor'],
         markeredgewidth=markerStyleMap["IncMiner"]['markeredgewidth'],
         ls='-')

plt.plot(rounds, batchdis_cum,
         color=colorMap["batch"], marker=markerMap["batch"],
         markerfacecolor=markerStyleMap["batch"]['markerfacecolor'],
         markeredgewidth=markerStyleMap["batch"]['markeredgewidth'],
         ls='-')

import matplotlib.ticker as ticker

plt.yscale('log')
plt.xlabel('Update Round')
plt.ylabel('Total Time (s)')
plt.xticks(rounds, ['Init', '1', '2', '3', '4'])

ax = plt.gca()
ax.set_yticks([500, 1000, 2000])
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
ax.yaxis.set_minor_formatter(ticker.NullFormatter())
# No legend — use global legend figure

plt.tight_layout()
os.makedirs('plots', exist_ok=True)
plt.savefig('plots/adult_sequential_updates.pdf', dpi=200, bbox_inches='tight')
plt.close()

print("Saved plots/adult_sequential_updates.pdf")
