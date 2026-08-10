#!/usr/bin/env python3
"""
Alternative sequential updates plot: per-round incremental mining time only (no init).
Data from sequential_update_0724.md — Adult dataset, rounds 1–4.
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os

plt.rcParams['figure.figsize'] = [4, 3]
plt.rcParams['font.size'] = 16
plt.rcParams['lines.markersize'] = 10

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

# Cumulative online times (ms -> s), no init
rounds = [1, 2, 3, 4]

incminer_per_round   = [1526/1000, 3231/1000, 4701/1000, 7040/1000]
batchminer_per_round = [493013/1000, 986556/1000, 1479904/1000, 1957527/1000]

plt.figure()

plt.plot(rounds, incminer_per_round,
         color=colors[0], marker='s',
         markerfacecolor='none', markeredgewidth=1.5,
         ls='-')

plt.plot(rounds, batchminer_per_round,
         color=colors[1], marker='^',
         markerfacecolor='none', markeredgewidth=1.0,
         ls='-')

plt.yscale('log')
plt.xlabel('Update Round')
plt.ylabel('Total Time (s)')
plt.xticks(rounds)

ax = plt.gca()
ax.yaxis.set_minor_formatter(ticker.NullFormatter())

plt.tight_layout()
os.makedirs('plots', exist_ok=True)
plt.savefig('plots/adult_sequential_incr.pdf', dpi=200, bbox_inches='tight')
plt.close()
print("Saved plots/adult_sequential_incr.pdf")
