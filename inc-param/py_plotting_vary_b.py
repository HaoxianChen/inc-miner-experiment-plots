#!/usr/bin/env python3
"""
Runtime and storage while varying discretization granularity b (Adult dataset).
Data from vary-b_0724_1.md. B&W-print-safe: line style + marker differentiation only.
"""

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import os

plt.rcParams['font.size'] = 16
plt.rcParams['lines.markersize'] = 8

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
c0, c1 = colors[0], colors[1]

b            = [5, 10, 20]
runtime_s    = [869/1000, 775/1000, 764/1000]
storage_mb   = [39232/1e3, 63752/1e3, 112792/1e3]

# One plot with a shared b axis and separate runtime/storage y-axes.
fig, ax_runtime = plt.subplots(figsize=(4, 3))

runtime_line = ax_runtime.plot(
    b, runtime_s,
    color=c0, ls='-', marker='s',
    markerfacecolor='none', markeredgewidth=1.5,
    label='Runtime',
)
ax_runtime.set_xticks(b)
ax_runtime.set_ylabel('Running Time (s)', color=c0, fontsize=16)
ax_runtime.tick_params(axis='x', labelsize=16)
ax_runtime.tick_params(axis='y', labelcolor=c0, labelsize=16)
ax_runtime.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

ax_storage = ax_runtime.twinx()
storage_line = ax_storage.plot(
    b, storage_mb,
    color=c1, ls='--', marker='D',
    markerfacecolor='none', markeredgewidth=1.5,
    label='Storage',
)
ax_storage.set_ylabel('Storage (MB)', color=c1, fontsize=16)
ax_storage.tick_params(axis='y', labelcolor=c1, labelsize=16)

lines = runtime_line + storage_line
ax_storage.legend(
    lines, [line.get_label() for line in lines],
    fontsize=16, loc='upper center', facecolor='white', framealpha=1.0,
)

os.makedirs('plots', exist_ok=True)
plt.savefig('plots/exp3_vary_b.pdf', dpi=200, bbox_inches='tight')
plt.close()
print("Saved plots/exp3_vary_b.pdf")
