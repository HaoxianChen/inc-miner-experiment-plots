#!/usr/bin/env python3
"""
Plot incMiner runtime vs |AFF| (affected area size).
Source: result_pearson.xlsx, sheet "AFF-incminertime".
"""

import matplotlib.pyplot as plt
import pandas as pd
import os

plt.rcParams['figure.figsize'] = [4, 3]
plt.rcParams['font.size'] = 16
plt.rcParams['lines.markersize'] = 10

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

df = pd.read_excel('result_pearson.xlsx', sheet_name='AFF-incminertime')
df = df.sort_values('affectedAreaSize')

x = df['affectedAreaSize']
y = df['incMinerRuntimeMsMean(ms)'] / 1000  # ms -> s

plt.figure()
plt.plot(x, y,
         color=colors[0], marker='s',
         markerfacecolor='none', markeredgewidth=1.5,
         ls='-')

plt.ylabel('Running Time (s)')
plt.tight_layout()

os.makedirs('plots', exist_ok=True)
plt.savefig('plots/aff_incminer_runtime.pdf', dpi=200, bbox_inches='tight')
plt.close()
print("Saved plots/aff_incminer_runtime.pdf")
