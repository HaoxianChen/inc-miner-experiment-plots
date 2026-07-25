#!/usr/bin/env python3
"""
Two-panel stacked figure: varying discretization granularity b (Adult dataset).
Data from vary-b_0724_1.md. B&W-print-safe: line style + marker differentiation only.
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os

plt.rcParams['font.size'] = 14
plt.rcParams['lines.markersize'] = 8

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
c0, c1, c2, c3 = colors[0], colors[1], colors[2], colors[3]

b            = [5, 10, 20]
runtime_s    = [869/1000, 775/1000, 764/1000]
ree_count    = [19, 0, 0]
quant_error  = [1/5, 1/10, 1/20]
storage_mb   = [39232/1e3, 63752/1e3, 112792/1e3]

# Two stacked panels sharing x-axis; each panel ~1.2in tall
fig, (ax_a, ax_b) = plt.subplots(2, 1, sharex=True,
                                   figsize=(3.2, 2.6),
                                   gridspec_kw={'hspace': 0.12})

# ---- Panel (a): runtime (left) and re-evaluated REE count (right) ----------

lns1 = ax_a.plot(b, runtime_s,
                 color=c0, ls='-', marker='s',
                 markerfacecolor='none', markeredgewidth=1.5,
                 label='Runtime (s)')
ax_a.set_ylabel('Runtime (s)', color=c0, fontsize=12)
ax_a.tick_params(axis='y', labelcolor=c0, labelsize=11)

ax_a2 = ax_a.twinx()
lns2 = ax_a2.plot(b, ree_count,
                  color=c1, ls='--', marker='^',
                  markerfacecolor='none', markeredgewidth=1.5,
                  label='Re-eval. REEs')
ax_a2.set_ylabel('Re-eval. REEs', color=c1, fontsize=12)
ax_a2.tick_params(axis='y', labelcolor=c1, labelsize=11)
ax_a2.yaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=3))

# Combined legend inside panel (a)
lns = lns1 + lns2
ax_a.legend(lns, [l.get_label() for l in lns],
            fontsize=9, loc='upper right', framealpha=0.8)

# ---- Panel (b): quantization error (left) and storage (right) --------------

lns3 = ax_b.plot(b, quant_error,
                 color=c2, ls='-', marker='o',
                 markerfacecolor='none', markeredgewidth=1.5,
                 label='Quant. Error')
ax_b.set_ylabel('Quant. Error', color=c2, fontsize=12)
ax_b.tick_params(axis='y', labelcolor=c2, labelsize=11)
ax_b.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

ax_b2 = ax_b.twinx()
lns4 = ax_b2.plot(b, storage_mb,
                  color=c3, ls=':', marker='D',
                  markerfacecolor='none', markeredgewidth=1.5,
                  label='Storage (KB)')
ax_b2.set_ylabel('Storage (KB)', color=c3, fontsize=12)
ax_b2.tick_params(axis='y', labelcolor=c3, labelsize=11)
ax_b2.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.0f'))

lns = lns3 + lns4
ax_b.legend(lns, [l.get_label() for l in lns],
            fontsize=9, loc='center right', framealpha=0.8)

ax_b.set_xlabel('$b$', fontsize=14)
ax_b.set_xticks(b)
ax_b.tick_params(axis='x', labelsize=11)

os.makedirs('plots', exist_ok=True)
plt.savefig('plots/exp3_vary_b.pdf', dpi=200, bbox_inches='tight')
plt.close()
print("Saved plots/exp3_vary_b.pdf")
