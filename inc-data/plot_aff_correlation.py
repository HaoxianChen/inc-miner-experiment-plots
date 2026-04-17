#!/usr/bin/env python3
"""
Plot correlation between AFF (Additional Feature Workload) and IncMiner runtime
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as pyplot

# =============================================================================
# Data
# =============================================================================

# Basic workload (70% data)
basic_workload = 3151814982206

# Incremental workloads
increment_labels = ["1%", "5%", "15%", "20%", "30%"]
increment_workloads = [
    3228027420682,  # 1% (71%)
    3603260717024,  # 5% (75%)
    4627439142368,  # 15% (85%)
    5189031630648,  # 20% (90%)
    6407682385578,  # 30% (100%)
]

# IncMiner runtime from figure c (DBLP varying Delta D+)
incminer_runtime = [2.74, 7.22, 24.96, 32.59, 51.51]  # seconds

# =============================================================================
# Compute AFF
# =============================================================================

aff = np.array(increment_workloads) - basic_workload

print("=" * 70)
print("AFF (Additional Feature Workload) Calculation")
print("=" * 70)
print(f"Basic workload (70% data):  {basic_workload:,}")
print()
for label, workload, a, rt in zip(increment_labels, increment_workloads, aff, incminer_runtime):
    print(f"Increment {label}:")
    print(f"  Total workload:  {workload:>18,}")
    print(f"  AFF:             {a:>18,}")
    print(f"  IncMiner time:   {rt:>14.2f} s")
    print()

# =============================================================================
# Correlation analysis
# =============================================================================

print("=" * 70)
print("Correlation Analysis: AFF vs IncMiner Runtime")
print("=" * 70)

# Manual Pearson correlation calculation
def pearson_correlation(x, y):
    n = len(x)
    mean_x = sum(x) / n
    mean_y = sum(y) / n

    numerator = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    denominator_x = (sum((xi - mean_x)**2 for xi in x))**0.5
    denominator_y = (sum((yi - mean_y)**2 for yi in y))**0.5

    return numerator / (denominator_x * denominator_y)

def linear_regression(x, y):
    n = len(x)
    mean_x = sum(x) / n
    mean_y = sum(y) / n

    numerator = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    denominator = sum((xi - mean_x)**2 for xi in x)

    slope = numerator / denominator
    intercept = mean_y - slope * mean_x

    # R-squared
    y_pred = [slope * xi + intercept for xi in x]
    ss_total = sum((yi - mean_y)**2 for yi in y)
    ss_residual = sum((yi - yp)**2 for yi, yp in zip(y, y_pred))
    r_squared = 1 - (ss_residual / ss_total) if ss_total != 0 else 0

    return slope, intercept, r_squared

pearson_r = pearson_correlation(aff, incminer_runtime)
print(f"Pearson correlation:  r = {pearson_r:.4f}")

slope, intercept, r_squared = linear_regression(aff, incminer_runtime)
print(f"\nLinear regression:")
print(f"  Slope:       {slope:.2e} s/unit")
print(f"  Intercept:   {intercept:.2f} s")
print(f"  R-squared:   {r_squared:.4f}")
print(f"  Equation:    y = {slope:.2e}x + {intercept:.2f}")
print()

# =============================================================================
# Plotting
# =============================================================================

pyplot.rcParams.update({
    'font.size': 16,
    'axes.labelsize': 16,
    'axes.titlesize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'text.usetex': True,
    'font.family': 'sans-serif',
})

fig, (ax1, ax2) = pyplot.subplots(1, 2, figsize=(14, 5.5))

# -------------------------------------------------------------------------
# Plot 1: Side-by-side bar + line plot
# -------------------------------------------------------------------------

x = np.arange(len(increment_labels))
width = 0.35

ax1.bar(x - width/2, aff / 1e12, width, label='AFF', color='C0', alpha=0.7, edgecolor='k', linewidth=1)
ax1_twin = ax1.twinx()
ax1_twin.plot(x, incminer_runtime, label='IncMiner', color='C1', marker='o', ms=12,
              markeredgewidth=2, markeredgecolor='k', linewidth=3)

ax1.set_xlabel('Increment ($\\Delta D^+$)')
ax1.set_ylabel('AFF ($\\times 10^{12}$ units)', color='C0')
ax1_twin.set_ylabel('IncMiner Runtime (s)', color='C1')
ax1.set_xticks(x)
ax1.set_xticklabels(increment_labels)
ax1.tick_params(axis='y', labelcolor='C0')
ax1_twin.tick_params(axis='y', labelcolor='C1')

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax1_twin.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', framealpha=0.9)
ax1.grid(True, alpha=0.2, axis='y')

# -------------------------------------------------------------------------
# Plot 2: Scatter plot with regression line
# -------------------------------------------------------------------------

aff_normalized = aff / 1e12

# Scatter points
scatter = ax2.scatter(aff_normalized, incminer_runtime, s=150, c='C0', alpha=0.9,
                      edgecolors='k', linewidth=2, zorder=5)

# Regression line
x_reg = np.linspace(min(aff_normalized) * 0.9, max(aff_normalized) * 1.1, 100)
y_reg = slope * (x_reg * 1e12) + intercept
ax2.plot(x_reg, y_reg, 'C1--', linewidth=3,
         label=f'$y = {slope:.2e}x + {intercept:.2f}$\n$R^2 = {r_squared:.4f}$')

ax2.set_xlabel('AFF ($\\times 10^{12}$ units)')
ax2.set_ylabel('IncMiner Runtime (s)')
ax2.legend(loc='upper left', framealpha=0.9)
ax2.grid(True, alpha=0.3)

# Add labels to points
for i, (a, rt, label) in enumerate(zip(aff_normalized, incminer_runtime, increment_labels)):
    ax2.annotate(label, (a, rt), xytext=(8, 8), textcoords='offset points',
                 fontsize=14, bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

# Add correlation text
ax2.text(0.05, 0.95, f'Pearson $r = {pearson_r:.4f}$', transform=ax2.transAxes,
         fontsize=14, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

pyplot.tight_layout()

os.makedirs('figures', exist_ok=True)
pyplot.savefig('figures/aff_correlation.pdf', dpi=200, bbox_inches='tight')
print("\nPlot saved to figures/aff_correlation.pdf")

pyplot.close()
